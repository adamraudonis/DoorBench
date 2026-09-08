"""Privileged waypoint steering for the proprioceptive H1 walking actor.

This teacher reads world pose/velocity and returns bounded body-frame commands.
It never writes body poses, controls a door, or applies forces directly.
"""
import hashlib,json
from collections import deque
from pathlib import Path
import numpy as np
from doorbench.dexterous.locomotion import NativeH1MotorAdapter, DEFAULT_ANGLES


def wrap_angle(angle):
    return float(np.arctan2(np.sin(angle),np.cos(angle)))


def make_door_approach(door,robot,reference,*,distance=.7,lateral=0.,yaw_offset=0.,seed=0,joint_noise=0.):
    import mujoco
    from doorbench.dexterous.environment import DexterousDoorEnv
    from doorbench.dexterous.reset import check_joint_reset
    robot=Path(robot);ref=json.loads(Path(reference).read_text());goal=np.asarray(ref['initial_root'],float)
    audit=json.loads(robot.with_suffix('.audit.json').read_text())
    if hashlib.sha256(robot.read_bytes()).hexdigest()!=audit['robot_xml_sha256']:raise ValueError('Robot differs from pinned audit')
    sim=DexterousDoorEnv(Path(door),robot,audit)
    sim.reset(seed=seed,randomize=False,images=False);m,d=sim.m,sim.d
    adapter=NativeH1MotorAdapter(m);q=sim.root_qadr;v=sim.root_vadr
    quat=goal[3:];rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,quat);R=rotation.reshape(3,3);yaw=float(np.arctan2(R[1,0],R[0,0]))
    start=goal[:2]-distance*np.array([np.cos(yaw),np.sin(yaw)])+lateral*np.array([-np.sin(yaw),np.cos(yaw)])
    d.qpos[q:q+2]=start;d.qpos[q+3:q+7]=[np.cos((yaw+yaw_offset)/2),0.,0.,np.sin((yaw+yaw_offset)/2)]
    d.qpos[adapter.qadr]=DEFAULT_ANGLES+np.random.default_rng(seed).uniform(-joint_noise,joint_noise,10)
    for side in ('left','right'):
        d.qpos[m.jnt_qposadr[m.joint('robot/'+side+'_elbow').id]]=.7
        d.qpos[m.jnt_qposadr[m.joint('robot/'+side+'_shoulder_roll').id]]=.1 if side=='left' else -.1
    check_joint_reset([m.joint(j).name for j in sim.joints],d.qpos[sim.qadr],m.jnt_range[sim.joints])
    d.qvel[:]=0.;mujoco.mj_forward(m,d)
    feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    bottoms=[]
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] not in feet or not m.geom_contype[gid]:continue
        if m.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Audit non-mesh foot before use')
        mesh=m.geom_dataid[gid];vertices=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
        bottoms.append(float((vertices@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid])[:,2].min()))
    d.qpos[q+2]+=.0001-min(bottoms);mujoco.mj_forward(m,d)
    d.ctrl[sim.actuators]=d.actuator_length[sim.actuators]
    sim.feet=feet;sim.adapter=adapter
    return sim,goal[:2].copy(),yaw


class WaypointApproach:
    def __init__(self,position,yaw,*,gain=4.,brake_prediction=.4,brake_radius=.05,max_speed=.3,brake_velocity_window=0.):
        self.position=np.asarray(position,float);self.yaw=float(yaw)
        self.gain=gain;self.brake_prediction=brake_prediction;self.brake_radius=brake_radius;self.max_speed=max_speed
        if not np.isfinite(brake_velocity_window) or brake_velocity_window<0:raise ValueError('Nonnegative finite brake averaging window required')
        self.brake_velocity_window=float(brake_velocity_window);self.position_history=deque();self.brake_velocity=np.zeros(2)
        self.velocity=np.zeros(2);self.aim_offset=np.zeros(2);self.stop_time=None;self.stops=0;self.previous_time=None
    def step(self,position,yaw,world_velocity,time_s):
        dt=.02 if self.previous_time is None else time_s-self.previous_time;self.previous_time=time_s
        self.velocity+=(1.-np.exp(-dt/.25))*(np.asarray(world_velocity)-self.velocity)
        self.brake_velocity=self.velocity.copy()
        if self.brake_velocity_window:
            self.position_history.append((float(time_s),np.array(position,copy=True)))
            cutoff=time_s-self.brake_velocity_window
            while len(self.position_history)>1 and self.position_history[1][0]<=cutoff:self.position_history.popleft()
            if self.position_history[0][0]<=cutoff and len(self.position_history)>1:
                t0,p0=self.position_history[0];t1,p1=self.position_history[1]
                past=p0+(p1-p0)*((cutoff-t0)/(t1-t0))
                self.brake_velocity=(np.asarray(position)-past)/self.brake_velocity_window
        true_error=self.position-np.asarray(position);error=true_error+self.aim_offset;angle=wrap_angle(self.yaw-yaw)
        if time_s<1:return np.zeros(3),1.,'initial'
        predicted=error-self.brake_prediction*self.brake_velocity
        if self.stop_time is None and abs((time_s%.8)-.2)<.011 and np.linalg.norm(predicted)<self.brake_radius and abs(angle)<np.deg2rad(2.):
            self.stop_time=time_s;self.stops+=1
        if self.stop_time is not None:
            if time_s-self.stop_time>4. and (np.linalg.norm(true_error)>.025 or abs(angle)>np.deg2rad(1.5)) and self.stops<4:
                self.aim_offset=np.clip(self.aim_offset+.8*true_error,-.12,.12)
                self.stop_time=None
            else:return np.zeros(3),max(0.,1.-(time_s-self.stop_time)),'brake and hold'
        R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        local=R.T@(self.gain*error-.2*self.velocity)
        command=np.r_[np.clip(local,[-.15,-.12],[self.max_speed,.12]),np.clip(1.5*angle,-.4,.4)]
        return command,1.,'waypoint approach'
