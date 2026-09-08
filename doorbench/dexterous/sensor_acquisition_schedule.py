"""Contact-enabled scripted joint route; not a feedback grasp policy.

Only local clock and static joint numbers determine goals. All scene/contact
admission and grasp success checks belong to a separate offline/runtime evaluator.
"""
import numpy as np
from .reach_balance_schedule import ReferenceReachSchedule


class ScriptedAcquisitionSchedule:
    def __init__(self,goal_names,source_joint_names,path_qpos,*,start_s=1.,reach_seconds=16.,settle_seconds=2.):
        # Reuse the checked named-column interpolator, not its contact-free
        # prefix's timing. Exact fraction1 is explicitly allowed in this scope.
        self.route=ReferenceReachSchedule(goal_names,source_joint_names,path_qpos,
            start_s=start_s,reach_seconds=reach_seconds,settle_seconds=settle_seconds)
        self.names=self.route.names;self.path=self.route.path
        self.start_s=self.route.start_s;self.reach_seconds=self.route.reach_seconds
        self.settle_seconds=self.route.settle_seconds;self.duration_s=self.route.duration_s
        self.stop_fraction=1.

    def values_at_fraction(self,fraction):return self.route.values_at_fraction(fraction)

    def goals(self,t):
        if not np.isscalar(t) or not np.isfinite(t) or t<0:raise ValueError('Finite nonnegative schedule clock required')
        u=np.clip((t-self.start_s)/self.reach_seconds,0.,1.);blend=u**3*(10+u*(-15+6*u))
        return dict(zip(self.names,self.values_at_fraction(float(np.clip(blend,0.,1.)))))

    def sampled_maximum_goal_speed(self,dt=.002):
        times=np.arange(0,self.duration_s+dt/2,dt)
        goals=np.array([[*self.goals(t).values()] for t in times])
        return float(np.max(abs(np.diff(goals,axis=0)))/dt)
