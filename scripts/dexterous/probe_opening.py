#!/usr/bin/env python3
"""Contact-driven opening teacher probe, with bounded robot motors only."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO
from doorbench.dexterous.grasp_training import GraspSkillEnv
from doorbench.dexterous.contact_audit import lever_contacts
from doorbench.dexterous.provenance import capture


def smooth(t):
    t=np.clip(t,0,1)
    return t*t*t*(10+t*(-15+6*t))


def main():
    p=argparse.ArgumentParser()
    for name in ('robot','door','seed-pose','preload-json','checkpoint','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--base-forward',type=float,default=0.)
    p.add_argument('--push-angle',type=float,default=.32)
    p.add_argument('--seconds',type=float,default=8.)
    p.add_argument('--release-angle',type=float,default=.08)
    p.add_argument('--balance-gain',type=float,default=0.)
    p.add_argument('--balance-damping',type=float,default=0.)
    p.add_argument('--balance-start',type=float,default=0.)
    p.add_argument('--stance',action='store_true')
    p.add_argument('--gate-push',action='store_true')
    p.add_argument('--lower-body',type=float,default=0.)
    p.add_argument('--forward-body',type=float,default=0.)
    p.add_argument('--push-lead',type=float,default=.08)
    p.add_argument('--stance-fast',action='store_true')
    p.add_argument('--stance-pose',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise SystemExit('Use a new output directory')
    out.mkdir(parents=True);capture(Path(__file__).resolve().parents[2],out,vars(a),timing='before_probe')
    env=GraspSkillEnv(a.door,a.robot,a.seed_pose,a.preload_json);obs,_=env.reset(seed=20000)
    s=env.sim;m,d=s.m,s.d;ik=mujoco.MjData(m);ik.qpos[:]=d.qpos
    names=['right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_yaw','rh_WRJ2','rh_WRJ1']
    joints=[m.joint('robot/'+n).id for n in names];qa=m.jnt_qposadr[joints];va=m.jnt_dofadr[joints]
    act=[list(s.actuators).index(m.actuator('robot/'+n).id) if not n.startswith('rh_') else list(s.actuators).index(m.actuator('robot/rh_A_'+n[3:]).id) for n in names]
    low=m.jnt_range[joints,0]+.015;high=m.jnt_range[joints,1]-.015
    palm=m.site('robot/rh_palm_touch').id;operator=m.body('leaf_handle').id
    hj=m.jnt_qposadr[m.joint('leaf_handle_hinge').id];dj=m.jnt_qposadr[m.joint('leaf_hinge').id]
    initial_pos=d.site_xpos[palm].copy();initial_rot=d.site_xmat[palm].reshape(3,3).copy()
    def solve(position,rotation,iterations=30):
        jp=np.zeros((3,m.nv));jr=jp.copy()
        for _ in range(iterations):
            mujoco.mj_kinematics(m,ik);mujoco.mj_comPos(m,ik)
            error=np.r_[(position-ik.site_xpos[palm])*5,
                Rotation.from_matrix(rotation@ik.site_xmat[palm].reshape(3,3).T).as_rotvec()]
            mujoco.mj_jacSite(m,ik,jp,jr,palm);jac=np.vstack([jp[:,va]*5,jr[:,va]])
            change=jac.T@np.linalg.solve(jac@jac.T+.003*np.eye(6),error)
            ik.qpos[qa]=np.clip(ik.qpos[qa]+np.clip(change,-.08,.08),low,high)
            if np.linalg.norm(error)<1e-4:break
        mujoco.mj_kinematics(m,ik)
        return float(np.linalg.norm(position-ik.site_xpos[palm]))
    # Initialization only: move the starting stance closer while keeping the palm at the handle.
    ik.qpos[s.root_qadr+1]+=a.base_forward
    initial_error=solve(initial_pos,initial_rot,150)
    d.qpos[:]=ik.qpos;d.qvel[:]=0;mujoco.mj_forward(m,d)
    env.base=d.actuator_length[s.actuators].copy();obs=env.observation()
    model=PPO.load(a.checkpoint,device='cpu')
    relative_pos=d.xmat[operator].reshape(3,3).T@(d.site_xpos[palm]-d.xpos[operator])
    relative_rot=d.xmat[operator].reshape(3,3).T@d.site_xmat[palm].reshape(3,3)
    rows=[];states={k:[] for k in ('qpos','qvel','ctrl')}
    initial_base=env.base.copy();release_time=None;release_base=None;stance=None
    initial_preload=env.preload.copy()
    finger_act=[i for i,aid in enumerate(s.actuators) if m.actuator(aid).name.startswith('robot/rh_') and 'WRJ' not in m.actuator(aid).name]
    def stance_filter(controls):
        if stance is not None:
            value,_=stance.command()
            if value is not None:controls[stance.local]=value
        return controls
    if a.stance_fast:s.motor_control_filter=stance_filter
    try:
        for step in range(round(a.seconds*50)):
            t=step/50;command_handle=.78*smooth((t-1.5)/1.5)
            command_door=a.push_angle*smooth((t-3.1)/.8)
            if a.gate_push:
                command_door=float(d.qpos[dj])+(a.push_lead if d.qpos[hj]>.60 or d.qpos[dj]>.02 else 0.)
            ik.qpos[:]=d.qpos;ik.qpos[hj]=command_handle;ik.qpos[dj]=command_door
            mujoco.mj_kinematics(m,ik)
            position=ik.xpos[operator]+ik.xmat[operator].reshape(3,3)@relative_pos
            rotation=ik.xmat[operator].reshape(3,3)@relative_rot
            ik.qpos[hj]=d.qpos[hj];ik.qpos[dj]=d.qpos[dj]
            error=solve(position,rotation)
            if release_time is None and d.qpos[dj]>=a.release_angle:
                release_time=t;release_base=env.base.copy()
            if release_time is None:
                env.base[act]=ik.qpos[qa]
                action=model.predict(obs,deterministic=True)[0]
            else:
                blend=smooth((t-release_time-.4)/.8)
                hand_blend=smooth((t-release_time)/.25)
                env.base=release_base*(1-blend)+initial_base*blend
                env.base[finger_act]*=1-hand_blend
                env.preload=initial_preload*(1-hand_blend)
                action=np.zeros(6)
            if a.balance_gain and t>=a.balance_start:
                torso=d.xmat[m.body('robot/torso_link').id].reshape(3,3)
                pitch=np.arctan2(-torso[2,0],torso[2,2])
                rate=d.sensor('robot/imu_gyro').data[1]
                for side in ('left','right'):
                    index=list(s.actuators).index(m.actuator('robot/'+side+'_ankle').id)
                    env.base[index]=initial_base[index]+a.balance_gain*pitch+a.balance_damping*rate
            stance_status=None
            if a.stance and t>=1.2:
                from doorbench.dexterous.stance import StanceController
                if a.stance_pose:
                    from doorbench.dexterous.stance import PoseStanceController as StanceController
                if stance is None:
                    stance=StanceController(s);stance.original_root=stance.target_root.copy();stance.original_height=stance.target_root[2]
                recover=smooth((t-release_time)/1.5) if release_time is not None else 0.
                stance.target_root[2]=stance.original_height-a.lower_body*smooth((t-1.5)/2.)*(1-recover)
                stance.target_root[:2]=stance.original_root[:2]+stance.target_rotation[:2,0]*a.forward_body*smooth((t-3.5)/3.)*(1-recover)
                control,stance_status=stance.command()
                if control is not None:env.base[stance.local]=control
            obs,_,_,_,info=env.step(action)
            contacts=lever_contacts(m,d,'leaf_handle_lever_col_n')
            rows.append({**info,'handle_rad':float(d.qpos[hj]),'door_rad':float(d.qpos[dj]),
                         'command_handle_rad':float(command_handle),'command_door_rad':float(command_door),
                         'ik_error_m':error,'contacts':contacts})
            rows[-1]['stance_status']=stance_status
            for key in states:states[key].append(getattr(d,key).copy())
            if info['fell']:break
        report={'scope':'Privileged near-handle opening probe; no learned approach or traversal',
            'initial_ik_error_m':initial_error,'maximum_handle_rad':max(r['handle_rad'] for r in rows),
            'maximum_door_rad':max(r['door_rad'] for r in rows),'final_door_rad':rows[-1]['door_rad'],
            'fell':rows[-1]['fell'],'steps':len(rows),'runtime_robot_pose_writes':0,'direct_door_commands':0,
            'maximum_ik_error_m':max(r['ik_error_m'] for r in rows),
            'release_time_s':release_time,'final_torso_tilt_deg':rows[-1]['torso_tilt_deg'],
            'final_root_height_m':rows[-1]['root_height_m'],
            'upright_entire_run':all(r['torso_tilt_deg']<12 and r['root_height_m']>.8 for r in rows)}
        np.savez_compressed(out/'trajectory.npz',**states)
        (out/'trace.json').write_text(json.dumps(rows)+'\n');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:env.close()


if __name__=='__main__':main()
