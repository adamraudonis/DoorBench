"""Privileged measured-state continuation; returns original robot motor forces only.

The owned authored scene is never stepped. It provides dynamics, collision
queries and a fixed foot reference for the motor controller, not plant support.
Instantiate once per attained opening endpoint and call once per physical tick.
"""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .grasp_verification import scalar_transmission_matrix
from .post_opening import plan_stow, StowRiseController
from .passage import PassageWaypoints, RobotBounds


def numeric_mapping(values, names, label):
    if not isinstance(values, dict) or set(values) != set(names):
        raise ValueError(label+' must contain exactly the declared names')
    result = np.asarray([values[name] for name in names], float)
    if result.shape != (len(names),) or not np.isfinite(result).all():
        raise ValueError('Malformed '+label)
    return result


def pose_parts(pose):
    pose = np.asarray(pose, float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-6,rtol=0):
        raise ValueError('Expected finite normalized measured xyz/wxyz pose')
    return pose[:3], Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()


def contact_contract(t, pose_time, interval, evidence, foot_loads, last_time, dt):
    if not np.isfinite([t,pose_time]).all() or t < 0 or abs(t-pose_time)>1e-7:
        raise ValueError('Measured poses and joint clock must agree')
    if last_time is not None and abs(t-last_time-dt)>1e-7:
        raise ValueError('One uninterrupted controller call per physical timestep required')
    interval=np.asarray(interval,float)
    if interval.shape!=(2,) or not np.isfinite(interval).all() or abs(interval[1]-t)>1e-7 or abs(interval[0]-max(0.,t-dt))>1e-7:
        raise ValueError('Contacts must identify the immediately preceding actual interval')
    fields={'physics_qualified','left_hand_contacts','left_hand_load_N','right_environment_contacts'}
    if not isinstance(evidence,dict) or set(evidence)!=fields:
        raise ValueError('Explicit continuation evidence contract required')
    if not isinstance(evidence['physics_qualified'],(bool,np.bool_)) or not evidence['physics_qualified']:
        raise ValueError('Active physical safety qualification is required')
    for name in ('left_hand_contacts','right_environment_contacts'):
        value=evidence[name]
        if not isinstance(value,(int,np.integer)) or isinstance(value,(bool,np.bool_)) or value<0:
            raise ValueError('Actual nonnegative hand contact counts required')
    load=evidence['left_hand_load_N'];feet=np.asarray(foot_loads,float)
    if not np.isscalar(load) or not np.isfinite(load) or load<0 or feet.shape!=(2,) or not np.isfinite(feet).all():
        raise ValueError('Finite actual hand and foot loads required')
    return interval,feet


class PostOpeningTeacher:
    """The 009 continuation with numeric inputs suitable for native or Isaac.

    ``root`` is xyz/wxyz plus world linear/angular velocity. Joint dictionaries
    use the exact imported motor-contract names. All three authored Door55
    positions and velocities use native door joint names, not semantic aliases.
    Body poses include both ankles, both palms, leaf and leaf_handle. Contact
    forces act ON every colliding hand body, in world coordinates; missing rows
    cannot silently become zero. Body pose keys may include additional bodies.

    At the first call only, supply the previous actual 61 motor forces and a
    measured outward left-panel release normal. No saved robot pose, path or
    gait history is accepted. A new geometric stow plan and landed-foot planner
    reference are built from this attained state; they may fail qualification.
    """
    def __init__(self,robot_xml,motors,body_reset,checkpoint,*,door_xml,
                 passage=True,inward_roll=.07,phase_seconds=4.,arm_gain=10.,
                 physics_dt=.002,passage_options=None,stow_profile='original-v1'):
        robot_xml,door_xml=Path(robot_xml),Path(door_xml)
        if door_xml.is_dir():door_xml=door_xml/'door.xml'
        if motors.get('hand_mechanics_profile')!='shadow-loopback-v2' or hashlib.sha256(robot_xml.read_bytes()).hexdigest()!=motors.get('source_xml_sha256'):
            raise ValueError('Exact corrected native robot and motor contract required')
        if physics_dt!=.002 or type(passage) is not bool:
            raise ValueError('This H1 adapter requires the qualified 2 ms clock and explicit passage mode')
        if stow_profile not in ('original-v1','sequential-v2'):
            raise ValueError('Unknown explicit post-opening stow profile')
        self.stow_profile=stow_profile
        scene=mujoco.MjSpec.from_file(str(door_xml));scene.memory=128*1024*1024
        scene.worldbody.add_site(name='robot_attach',pos=[0.,-1.5,0.])
        scene.attach(mujoco.MjSpec.from_file(str(robot_xml)),prefix='robot/',frame=scene.worldbody.add_frame(pos=[0.,-1.5,0.]))
        m=self.m=scene.compile();self.d=mujoco.MjData(m)
        self.names=tuple(motors['joint_names']);self.motor_names=tuple(a['name'] for a in motors['actuators'])
        if len(set(self.names))!=69 or len(self.names)!=69 or len(set(self.motor_names))!=61 or len(self.motor_names)!=61 or m.nu!=61 or len(motors.get('passive_tendons',[]))!=8:
            raise ValueError('Expected the full 69-joint / 61-motor / eight-loopback embodiment')
        self.joints=np.array([m.joint('robot/'+name).id for name in self.names]);self.qa=m.jnt_qposadr[self.joints];self.va=m.jnt_dofadr[self.joints]
        self.act=np.array([m.actuator('robot/'+name).id for name in self.motor_names]);self.caps=np.array([a['force_range'] for a in motors['actuators']])
        matrix=scalar_transmission_matrix(m,self.act,self.joints)
        checks=[(matrix,np.array([[a['terms'].get(n,0.) for n in self.names] for a in motors['actuators']])),
                (m.actuator_gainprm[self.act,0],np.array([a['kp'] for a in motors['actuators']])),
                (m.actuator_biasprm[self.act,:3],np.array([a['bias'] for a in motors['actuators']])),
                (m.actuator_ctrlrange[self.act],np.array([a['control_range'] for a in motors['actuators']])),
                (m.actuator_forcerange[self.act],self.caps)]
        if any(not np.array_equal(actual,declared) for actual,declared in checks):
            raise ValueError('Motor transmission, gains or original caps differ from the contract')
        if abs(m.opt.timestep-physics_dt)>1e-12:
            raise ValueError('Authored scene timestep differs from the qualified controller')
        # Calculator-only force convention. No active model is accepted or changed.
        m.actuator_gainprm[self.act,0]=1.;m.actuator_biasprm[self.act,:3]=0.;m.actuator_ctrlrange[self.act]=self.caps
        root=m.joint('robot/free_base').id;self.root_q=int(m.jnt_qposadr[root]);self.root_v=int(m.jnt_dofadr[root])
        self.door_names=tuple(m.joint(j).name for j in range(m.njnt) if not m.joint(j).name.startswith('robot/'))
        if set(self.door_names)!={'leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide'}:
            raise ValueError('This continuation currently supports the audited three-joint Door55 scene')
        ids=[m.joint(name).id for name in self.door_names];self.door_q=m.jnt_qposadr[ids];self.door_v=m.jnt_dofadr[ids]
        self.feet=('left_ankle_link','right_ankle_link');self.pose_names=(*self.feet,'lh_palm','rh_palm','leaf','leaf_handle')
        self.body_ids={name:m.body(name if name.startswith('leaf') else 'robot/'+name).id for name in self.pose_names}
        self.hand_bodies={m.body(m.geom_bodyid[g]).name.removeprefix('robot/') for g in range(m.ngeom)
                          if (m.geom_contype[g]|m.geom_conaffinity[g]) and m.body(m.geom_bodyid[g]).name.startswith(('robot/rh_','robot/lh_'))}
        self.sim=SimpleNamespace(m=m,d=self.d,joints=self.joints,qadr=self.qa,vadr=self.va,root_qadr=self.root_q,root_vadr=self.root_v,actuators=self.act,pelvis=m.body('robot/pelvis').id)
        self.motors=copy.deepcopy(motors);self.reset=copy.deepcopy(body_reset);self.checkpoint=Path(checkpoint)
        self.passage_enabled=passage;self.inward_roll=inward_roll;self.phase_seconds=phase_seconds;self.arm_gain=arm_gain;self.dt=physics_dt
        self.passage_options=dict(passage_options or {});self.controller=None;self.navigator=None;self.last_time=None;self.started=None;self.last_force=None;self.calls=0
        self.plan=None;self.initialization=None;self.command=None;self.amplitude=None;self.bounds=RobotBounds(m)
        self.sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (robot_xml,door_xml,self.checkpoint)}

    def force(self,t,root,joints,velocities,foot_loads,hand_forces,*,door_positions,door_velocities,
              body_poses,evidence,pose_time_s,contact_interval_s,previous_motor_forces=None,
              release_normal_world=None):
        interval,feet=contact_contract(t,pose_time_s,contact_interval_s,evidence,foot_loads,self.last_time,self.dt)
        root=np.asarray(root,float)
        if root.shape!=(13,) or not np.isfinite(root).all():raise ValueError('Expected measured 13-component root state')
        _,rotation=pose_parts(root[:7])
        q=numeric_mapping(joints,self.names,'robot positions');v=numeric_mapping(velocities,self.names,'robot velocities')
        door_q=numeric_mapping(door_positions,self.door_names,'door positions');door_v=numeric_mapping(door_velocities,self.door_names,'door velocities')
        if not isinstance(body_poses,dict) or not set(self.pose_names)<=set(body_poses):raise ValueError('Actual feet, palms and mechanism body poses are required')
        poses={name:pose_parts(body_poses[name]) for name in self.pose_names}
        loads={}
        if not isinstance(hand_forces,dict):raise ValueError('Actual hand-body force map required')
        for name,value in hand_forces.items():
            short=name.rsplit('/',1)[-1]
            if short in loads or not short.startswith(('lh_','rh_')):raise ValueError('Duplicate or unexpected hand load')
            body=self.m.body('robot/'+short).id;load=np.asarray(value,float)
            if load.shape!=(3,) or not np.isfinite(load).all():raise ValueError('Malformed world hand load')
            loads[short]=load.copy()
        if not self.hand_bodies<=set(loads):raise ValueError('Missing actual force rows for colliding hand bodies')
        m,d=self.m,self.d
        d.qpos[self.root_q:self.root_q+7]=root[:7];d.qpos[self.qa]=q;d.qpos[self.door_q]=door_q
        d.qvel[self.root_v:self.root_v+3]=root[7:10];d.qvel[self.root_v+3:self.root_v+6]=rotation.T@root[10:13]
        d.qvel[self.va]=v;d.qvel[self.door_v]=door_v;d.time=float(t)
        if self.controller is None:
            previous=np.asarray(previous_motor_forces,float)
            if previous.shape!=(61,) or not np.isfinite(previous).all() or np.any(previous<self.caps[:,0]-1e-5) or np.any(previous>self.caps[:,1]+1e-5):raise ValueError('Initial actual motor forces must match original caps/order')
            d.ctrl[self.act]=previous
        elif previous_motor_forces is not None or release_normal_world is not None:
            raise ValueError('Initialization data may not reset an active continuation')
        else:d.ctrl[self.act]=self.last_force
        mujoco.mj_forward(m,d)
        errors=[]
        for name,(position,orient) in poses.items():
            b=self.body_ids[name];errors.append([np.linalg.norm(d.xpos[b]-position),np.linalg.norm(d.xmat[b].reshape(3,3)-orient)])
        maximum=np.max(errors,axis=0)
        if maximum[0]>.003 or maximum[1]>.02:raise ValueError('Measured body poses disagree with authored dynamics/clearance frames')
        if self.controller is None:
            normal=np.asarray(release_normal_world,float)
            if normal.shape!=(3,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal),1.,atol=1e-6,rtol=0):raise ValueError('Measured outward release normal required at initialization')
            leaf_pos,leaf_rot=poses['leaf'];palm_pos,_=poses['lh_palm']
            if abs(normal@leaf_rot[:,1])<.99 or normal@(palm_pos-leaf_pos)<=0:raise ValueError('Release direction must point out of the attained left panel face')
            if min(feet)<10 or np.linalg.norm(root[7:10])>.1 or evidence['right_environment_contacts'] or door_positions['leaf_hinge']<1.2:
                raise ValueError('Initialization requires supported quiet feet, a released right hand and attained aperture >=1.2 rad')
            if self.stow_profile=='original-v1':
                self.plan=plan_stow(self.sim,self.reset,inward_roll=self.inward_roll,retreat_normal_world=normal)
            else:
                from .post_opening_route import plan_sequential_stow
                self.plan=plan_sequential_stow(self.sim,self.reset,inward_roll=self.inward_roll,
                    retreat_normal_world=normal,left_style='yaw_first',right_style='lift_yaw')
            if not self.plan['passed']:raise ValueError('Newly attained opening state has no qualified static stow route')
            self.controller=StowRiseController(self.sim,self.motors,self.plan,checkpoint=self.checkpoint,phase_seconds=self.phase_seconds,arm_gain=self.arm_gain)
            self.controller.stance.foot_positions=np.array([poses[name][0] for name in self.feet])
            self.controller.stance.foot_rotations=[poses[name][1].copy() for name in self.feet]
            self.started=float(t)
            self.navigator=PassageWaypoints(self.sim,self.controller,**self.passage_options) if self.passage_enabled else None
            self.initialization=dict(time_s=float(t),root=root.tolist(),joints=dict(joints),velocities=dict(velocities),door_positions=dict(door_positions),door_velocities=dict(door_velocities),body_poses={k:np.asarray(body_poses[k]).tolist() for k in self.pose_names},foot_loads_N=feet.tolist(),release_normal_world=normal.tolist(),previous_motor_forces=previous.tolist(),contact_interval_s=interval.tolist(),evidence=dict(evidence),sources=self.sources,scope='Actual attained endpoint; no inherited full-opening qualification')
            self.initialization['stow_profile']=self.stow_profile
        elapsed=float(t-self.started)
        if self.navigator is not None and self.calls%10==0:
            self.command,self.amplitude=self.navigator.command(elapsed)
            if self.navigator.blocked:raise ValueError('Actual aperture/shape guard blocked continued passage: '+self.navigator.stage)
        force,info=self.controller.force(elapsed,left_contacts=evidence['left_hand_contacts'],left_load=evidence['left_hand_load_N'],walking_command=self.command,walking_amplitude=self.amplitude,hand_forces=loads)
        if force.shape!=(61,) or not np.isfinite(force).all() or np.any(force<self.caps[:,0]) or np.any(force>self.caps[:,1]):raise ValueError('Invalid original-capped continuation motor force')
        self.last_force=force.copy();self.last_time=float(t);self.calls+=1
        info.update(time_s=float(t),elapsed_s=elapsed,initial_time_s=self.started,contact_interval_s=interval.tolist(),pose_time_s=float(pose_time_s),maximum_pose_position_error_m=float(maximum[0]),maximum_pose_rotation_error=float(maximum[1]),native_mirror_steps=0,privileged_teacher=True,stance_solver_failures=self.controller.solver_failures,minimum_body_y_m=float(self.bounds(d)[0][1]),passage_stage=None if self.navigator is None else self.navigator.stage,passage_completed=False if self.navigator is None else self.navigator.done)
        return force.copy(),info
