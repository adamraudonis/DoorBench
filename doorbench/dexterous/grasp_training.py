"""Local tactile reflex curriculum from a supplied near-handle pose.

This skill initializes at the handle and holds body/arm motor targets. It cannot
count as a complete door task, even if all fingers establish opposing contact.
"""
import json
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from .environment import DexterousDoorEnv
from .contact_audit import lever_contacts


class GraspSkillEnv(gym.Env):
    def __init__(self,door,robot,seed_pose,preload_json,horizon=150):
        robot=Path(robot);self.sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()))
        m=self.sim.m
        self.seed_pose=np.load(seed_pose)['qpos'];self.preload=np.array(json.loads(Path(preload_json).read_text())['thumb_delta_and_curl_rad'])
        local={m.actuator(i).name:k for k,i in enumerate(self.sim.actuators)}
        self.thumb=[local['robot/rh_A_THJ'+str(i)] for i in (5,4,3,2,1)]
        self.curl=[local['robot/rh_A_'+finger+'J0'] for finger in ('FF','MF','RF','LF')]
        joints=[j for j in self.sim.joints if m.joint(j).name.startswith('robot/rh_')]
        self.qadr=m.jnt_qposadr[joints];self.vadr=m.jnt_dofadr[joints]
        self.taxels=np.concatenate([np.arange(m.sensor_adr[i],m.sensor_adr[i]+m.sensor_dim[i]) for i in sorted(self.sim.tactile_sensor_ids) if m.sensor(i).name.startswith('robot/rh_')])
        self.horizon=horizon;self.previous=np.zeros(6);self.steps=0;self.held=0
        self.action_space=spaces.Box(-1.,1.,(6,),np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(len(joints)*2+len(self.taxels)+6,),np.float32)

    def observation(self):
        d=self.sim.d
        return np.r_[d.qpos[self.qadr],np.clip(d.qvel[self.vadr]/10.,-10,10),
            np.clip(d.sensordata[self.taxels]/10.,-10,10),self.previous].astype(np.float32)

    def configuration_audit(self):
        m=self.sim.m
        joints=[j for j in self.sim.joints if m.joint(j).name.startswith('robot/rh_')]
        return {'interface_version':'h1-shadow-initialized-grasp-v1',
            'observation_dimensions':int(self.observation_space.shape[0]),
            'observation_order':['joint_position_rad','joint_velocity_divided_by_10',
                                 'tactile_force_divided_by_10','previous_six_actions'],
            'joint_order':[m.joint(j).name for j in joints],
            'tactile_sensor_order':[m.sensor(i).name for i in sorted(self.sim.tactile_sensor_ids)
                                    if m.sensor(i).name.startswith('robot/rh_')],
            'thumb_action_order':[m.actuator(self.sim.actuators[i]).name for i in self.thumb],
            'shared_curl_actuators':[m.actuator(self.sim.actuators[i]).name for i in self.curl],
            'residual_scale_rad':.2,'timestep_seconds':float(m.opt.timestep),
            'control_frequency_hz':50,'horizon_control_steps':self.horizon,
            'success_contiguous_steps':50,'min_digit_force_N':.2,
            'scope':'Initialized contact hold only; no approach, opening or traversal'}

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);s=self.sim;s.reset(images=False,randomize=False)
        s.d.qpos[:]=self.seed_pose;s.d.qvel[:]=0
        # Small hand initialization variation, never a runtime pose correction.
        s.d.qpos[self.qadr]+=self.np_random.uniform(-.002,.002,len(self.qadr))
        mujoco.mj_forward(s.m,s.d)
        self.base=s.d.actuator_length[s.actuators].copy();self.previous=np.zeros(6);self.steps=0;self.held=0
        return self.observation(),{}

    def step(self,action):
        action=np.clip(np.asarray(action),-1,1);delta=self.preload+.20*action
        controls=self.base.copy();controls[self.thumb]+=delta[:5];controls[self.curl]+=delta[5]
        self.sim.step(self.sim.normalize(np.clip(controls,self.sim.low,self.sim.high)),images=False)
        contact=lever_contacts(self.sim.m,self.sim.d,'leaf_handle_lever_col_n')
        diag=self.sim.diagnostics();forces=np.array(list(contact['digit_forces_N'].values()))
        fallen=diag['root_height_m']<.7 or diag['torso_tilt_deg']>35 or not diag['finite'] or diag['numerical_warnings']>0
        good=contact['opposed'] and diag['root_height_m']>.8 and diag['torso_tilt_deg']<12
        self.held=self.held+1 if good else 0;success=self.held>=50
        reward=float(np.minimum(forces,1).sum()+3*np.minimum(forces.min(),1)+5*bool(contact['opposed'])-.02*np.mean((action-self.previous)**2))
        if fallen:reward-=10
        self.previous=action.copy();self.steps+=1
        return self.observation(),reward,fallen or success,self.steps>=self.horizon,{
            'is_success':bool(success),'fell':bool(fallen),'opposed':bool(contact['opposed']),
            'held_steps':self.held,'digit_forces_N':contact['digit_forces_N'],**diag,
            'scope':'initialized tactile grasp skill; not a door task'}

    def close(self):self.sim.close()
