"""Static joint-only acquisition followed by an offline-planned lever press.

No world geometry, contact IDs, plant state, force labels or task success enters
this schedule. The evaluator separately rejects an unqualified acquisition.
"""
import numpy as np


class ScriptedHandleOperationSchedule:
    def __init__(self,acquisition,press_joint_names,press_path_qpos,*,press_seconds=8.,settle_seconds=3.):
        self.acquisition=acquisition;self.names=acquisition.names
        self.press_names=tuple(press_joint_names);self.press_path=np.asarray(press_path_qpos,float).copy()
        if len(set(self.press_names))!=len(self.press_names) or not set(self.press_names)<=set(self.names):raise ValueError('Unique named upper-body goals required')
        if self.press_path.ndim!=2 or len(self.press_path)<2 or self.press_path.shape[1]!=len(self.press_names) or not np.isfinite(self.press_path).all():raise ValueError('Finite static press path required')
        if not np.isfinite([press_seconds,settle_seconds]).all() or min(press_seconds,settle_seconds)<=0:raise ValueError('Positive bounded press durations required')
        self.start_s=float(acquisition.duration_s);self.press_seconds=float(press_seconds);self.settle_seconds=float(settle_seconds)
        self.duration_s=self.start_s+self.press_seconds+self.settle_seconds;self.stop_fraction=1.
        if self.duration_s>40:raise ValueError('Bounded40s component protocol required')
        self.hold=acquisition.goals(self.start_s)
        if any(abs(self.hold[n]-v)>1e-12 for n,v in zip(self.press_names,self.press_path[0])):raise ValueError('Press must start at the exact acquisition command')

    def goals(self,t):
        if not np.isscalar(t) or not np.isfinite(t) or t<0:raise ValueError('Finite nonnegative schedule clock required')
        if t<=self.start_s:return self.acquisition.goals(t)
        u=np.clip((t-self.start_s)/self.press_seconds,0.,1.);blend=np.clip(u**3*(10+u*(-15+6*u)),0.,1.)
        coordinate=blend*(len(self.press_path)-1);i=min(int(coordinate),len(self.press_path)-2);f=coordinate-i
        values=self.press_path[i]+f*(self.press_path[i+1]-self.press_path[i])
        goals=dict(self.hold);goals.update(zip(self.press_names,values));return goals

    def sampled_maximum_goal_speed(self,dt=.002):
        times=np.arange(0,self.duration_s+dt/2,dt)
        values=np.array([[*self.goals(float(t)).values()] for t in times])
        return float(np.max(abs(np.diff(values,axis=0)))/dt)
