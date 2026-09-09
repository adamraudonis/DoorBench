"""Bounded privileged aperture feedback expressed only as a palm-load target."""
import numpy as np


class PanelApertureForce:
    maximum_target_N=6.
    def __init__(self):
        self.previous=None;self.velocity=0.;self.integral=0.

    def update(self,t,reference,measured,base):
        if not np.isfinite([t,reference,measured,base]).all() or t<0 or not 2<base<=4:raise ValueError('Finite aperture and supported palm target required')
        error=reference-measured;dt=0.
        if self.previous is not None:
            oldt,oldangle=self.previous;dt=t-oldt
            if not 0<dt<=.01:raise ValueError('Consecutive monotonic panel clock required')
            velocity=(measured-oldangle)/dt
            self.velocity+=dt/(.04+dt)*(velocity-self.velocity)
        candidate=float(np.clip(self.integral+2.*error*dt,0.,3.))
        raw=base+10.*error+candidate-8.*self.velocity
        if not ((raw>self.maximum_target_N and error>0) or (raw<2.05 and error<0)):self.integral=candidate
        target=float(np.clip(base+10.*error+self.integral-8.*self.velocity,2.05,self.maximum_target_N))
        self.previous=(float(t),float(measured))
        return target,dict(panel_force_profile='bounded-pi-v1',panel_force_integral_N=self.integral,panel_filtered_velocity_rad_s=self.velocity,panel_force_target_N=target)
