"""Advance source-bound panel segments only after an attained supported hold."""
import numpy as np


class AttainedPanelSchedule:
    def __init__(self,panels):
        if not panels:raise ValueError('At least one screened panel segment required')
        for before,after in zip(panels,panels[1:]):
            target=before.plan['final_leaf_angle_rad']
            if after.start_time<=before.start_time or not target-.02<=after.plan['initial_leaf_angle_rad']<=target+.05:
                raise ValueError('Consecutive segments must start after an attained target')
        self.panels=panels;self.index=0;self.held_since=None;self.last_time=None;self.completed=[]

    @property
    def active(self):return self.panels[self.index]

    def pending_stance_reference(self,t,stance):
        """Capture the preceding target before withdrawal resets its fallback.

        Opt-in preserves historical source replays. Capturing a reference does
        not authorize transition: advance still requires the measured hold.
        """
        if self.index+1==len(self.panels):return None
        following=self.panels[self.index+1]
        if not getattr(following,'preserve_stance_reference',False) or t<following.start_time-1e-8:return None
        return {name:getattr(stance,name).copy() for name in ('target_root','target_rotation','joint_target')}

    def advance(self,t,angle,load):
        if not np.isfinite([t,angle,load]).all():raise ValueError('Finite panel handoff required')
        if self.index+1==len(self.panels) or t<self.panels[self.index+1].start_time-1e-8:return False
        target=self.active.plan['final_leaf_angle_rad']
        if (self.held_since is None or t-self.held_since<.5-1e-8 or self.last_time is None or
                not 0<t-self.last_time<=.01 or not target-.02<=angle<=target+.05 or load<2):
            raise ValueError('Next panel segment requires a measured half-second supported target hold')
        self.completed.append(dict(index=self.index,started_s=self.active.started,handoff_s=float(t),
            supported_hold_since_s=self.held_since,target_aperture_rad=target,attained_aperture_rad=float(angle)))
        self.index+=1;self.held_since=None
        return True

    def observe(self,t,progress,angle,load):
        if not np.isfinite([t,progress,angle,load]).all():raise ValueError('Finite panel observations required')
        target=self.active.plan['final_leaf_angle_rad']
        if self.last_time is not None and not 0<t-self.last_time<=.01:raise ValueError('Consecutive panel observations required')
        ready=progress>=.999 and target-.02<=angle<=target+.05 and load>=2
        self.held_since=t if ready and self.held_since is None else self.held_since if ready else None
        self.last_time=t
