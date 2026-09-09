"""Capture attained finger posture only after a continuous measured hold.

Uses original coupled motors, gravity compensation and a smooth motor handoff.
It has no plant setters or additional constraints; this is a privileged teacher.
"""
import numpy as np
from .attained_hand_tracking import AttainedHandTracking
from .motor_handoff import MotorHandoff


class QualifiedHandHold:
    def __init__(self,teacher):
        self.teacher=teacher;self.since=None;self.previous_time=None;self.started=None

    def force(self,t,forces,joints,velocities,*,eligible):
        if not np.isfinite(t) or type(eligible) not in (bool,np.bool_):raise ValueError('Explicit measured hold eligibility required')
        if self.previous_time is not None and t<self.previous_time:raise ValueError('Monotonic hold clock required')
        stale=self.previous_time is not None and t-self.previous_time>.05
        self.previous_time=t
        if self.started is None:
            if not eligible or stale:self.since=None
            elif self.since is None:self.since=t
            if self.since is not None and t-self.since>=.5-1e-9:
                self.hand=AttainedHandTracking(self.teacher,joints,forces)
                first,_=self.hand.force(forces,joints,velocities)
                self.handoff=MotorHandoff(forces,first,self.teacher.caps,1.)
                self.started=t
        info=dict(attained_hold_started_s=self.started)
        if self.started is None:return forces,info
        force,tracking=self.hand.force(forces,joints,velocities)
        force=self.handoff.force(force,t-self.started)
        self.teacher.last_force=force.copy()
        return force,{**tracking,**info}
