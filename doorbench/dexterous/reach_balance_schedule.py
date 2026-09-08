"""Scripted joint-only prefix of a frozen acquisition route; no state feedback.

All trajectory values are static high-level commands. This module owns no robot
model, sensor evaluator, door geometry, global pose or active simulation.
"""
import numpy as np


class ReferenceReachSchedule:
    def __init__(self,goal_names,source_joint_names,path_qpos,*,stop_fraction=.45,start_s=1.,reach_seconds=8.,settle_seconds=2.):
        self.names=tuple(goal_names);source=tuple(source_joint_names);path=np.asarray(path_qpos,float)
        values=[stop_fraction,start_s,reach_seconds,settle_seconds]
        if len(set(self.names))!=len(self.names) or len(set(source))!=len(source) or not set(self.names)<=set(source):raise ValueError('Unique named joint route required')
        if path.ndim!=2 or path.shape[1]!=len(source) or len(path)<2 or not np.isfinite(path).all():raise ValueError('Finite joint path required')
        if not np.isfinite(values).all() or not 0<stop_fraction<1 or min(values[1:])<=0:raise ValueError('Declare a bounded contact-free prefix and positive durations')
        self.path=path[:,[source.index(n) for n in self.names]].copy()
        self.stop_fraction=float(stop_fraction);self.start_s=float(start_s);self.reach_seconds=float(reach_seconds);self.settle_seconds=float(settle_seconds)
        self.duration_s=self.start_s+self.reach_seconds+self.settle_seconds
        if self.duration_s>20:raise ValueError('This feasibility schedule is bounded to20s')

    def values_at_fraction(self,fraction):
        if not np.isscalar(fraction) or not np.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('Valid normalized source-path coordinate required')
        coordinate=float(fraction)*(len(self.path)-1);i=min(int(coordinate),len(self.path)-2);f=coordinate-i
        return (self.path[i]+f*(self.path[i+1]-self.path[i])).copy()

    def goals(self,t):
        if not np.isscalar(t) or not np.isfinite(t) or t<0:raise ValueError('Finite nonnegative schedule clock required')
        u=np.clip((t-self.start_s)/self.reach_seconds,0.,1.);blend=u**3*(10+u*(-15+6*u))
        return dict(zip(self.names,self.values_at_fraction(float(blend*self.stop_fraction))))

    def sampled_maximum_goal_speed(self,dt=.002):
        times=np.arange(0,self.duration_s+dt/2,dt)
        goals=np.array([[*self.goals(t).values()] for t in times])
        return float(np.max(abs(np.diff(goals,axis=0)))/dt)
