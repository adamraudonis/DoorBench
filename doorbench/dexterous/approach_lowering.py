"""Continuous motor-only approach, quiet stop and manipulation-height stance.

Privileged native development controller. It returns controls without stepping
the plant or writing any root/joint/foot pose. All physical gates remain the
caller's responsibility, including every QP failure.
"""
import numpy as np
from doorbench.dexterous.locomotion import H1WalkingPolicy,DEFAULT_ANGLES
from doorbench.dexterous.locomotion_approach import WaypointApproach
from doorbench.dexterous.locomotion_manipulation import LandedFootStanceController
from doorbench.dexterous.locomotion_posture import minimum_jerk


class ApproachLoweringController:
    def __init__(self,sim,checkpoint,goal,yaw,*,height=.8706799272,hip_spread=.07,
                 lower_seconds=4.,brake_velocity_window=.8,settle_feet=False,
                 handoff_delay=3.,yaw_weight=2.,solver_max_iter=100000):
        self.sim=sim;self.policy=H1WalkingPolicy(checkpoint)
        self.waypoint=WaypointApproach(goal,yaw,brake_velocity_window=brake_velocity_window)
        self.height=height;self.hip_spread=hip_spread;self.lower_seconds=lower_seconds;self.settle_feet=settle_feet
        self.handoff_delay=handoff_delay;self.yaw_weight=yaw_weight;self.solver_events=[]
        sim.stance_solver_settings={'max_iter':solver_max_iter}
        self.fixed=sim.d.ctrl.copy();self.target=DEFAULT_ANGLES.copy();self.steps=0
        self.stance=None;self.stance_command=None;self.solver_failures=0;self.solver_status='inactive'
        self.stance_started=None;self.initial_height=None;self.initial_feet=None
        self.stage='initial';self.body_command=np.zeros(3);self.amplitude=1.

    def command(self,foot_loads):
        s=self.sim;m,d=s.m,s.d;a=s.adapter;q=s.root_qadr;v=s.root_vadr
        R=d.xmat[s.pelvis].reshape(3,3);yaw=float(np.arctan2(R[1,0],R[0,0]))
        if self.steps%10==0 and self.stance is None:
            self.body_command,self.amplitude,self.stage=self.waypoint.step(d.qpos[q:q+2],yaw,d.qvel[v:v+2],d.time)
            blend,speed=minimum_jerk(d.time,1.,3.)
            lateral=np.array([0.,1.,0.,0.,0.,0.,-1.,0.,0.,0.])*self.hip_spread
            offset=lateral*blend;offset_velocity=lateral*speed
            self.target=self.policy.step(d.qpos[a.qadr]-offset,d.qvel[a.vadr]-offset_velocity,
                d.qvel[v+3:v+6],R.T@[0.,0.,-1.],self.body_command,d.time,phase_amplitude=self.amplitude)+offset
        controls=self.fixed.copy();controls[a.actuators]=a.command(d,self.target)
        if self.stance is None and self.waypoint.stop_time is not None and d.time-self.waypoint.stop_time>=self.handoff_delay and np.linalg.norm(d.qvel[v:v+2])<.02 and min(foot_loads)>30:
            s.stance_weights=np.r_[[200,200,1000,300,300,self.yaw_weight],np.full(10,.01)]
            self.stance=LandedFootStanceController(s);self.stance_started=float(d.time)
            self.initial_height=float(self.stance.target_root[2]);self.initial_feet=d.xpos[s.feet].copy()
        if self.stance is not None:
            elapsed=float(d.time-self.stance_started);blend,_=minimum_jerk(elapsed,0.,self.lower_seconds)
            self.stance.target_root[2]=self.initial_height+(self.height-self.initial_height)*blend
            self.stage='lowering' if elapsed<self.lower_seconds else 'low stance hold'
            self.body_command=np.zeros(3);self.amplitude=0.
            if self.steps%5==0:
                if self.settle_feet and elapsed<self.lower_seconds:self.stance.accept_physical_foot_repositioning()
                proposed,self.solver_status=self.stance.command()
                if proposed is None:
                    self.solver_failures+=1;self.solver_events.append({'time_s':float(d.time),'status':self.solver_status})
                else:self.stance_command=proposed
            if self.stance_command is not None:
                controls[self.stance.act]=np.clip(self.stance_command,m.actuator_ctrlrange[self.stance.act,0],m.actuator_ctrlrange[self.stance.act,1])
        self.steps+=1
        return controls
