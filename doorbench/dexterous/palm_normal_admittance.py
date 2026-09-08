"""Slow bounded Cartesian normal-goal feedback from scalar palm load.

This changes only a target consumed by the original motors. It does not apply
forces, inspect objects, or change contact/plant parameters. The frozen first
profile is deliberately small and must pass separate geometric/physical gates.
"""
import numpy as np
from .actual_base_palm import bounded_next_velocity


class PalmNormalAdmittance:
    target_load_N=4.
    maximum_offset_m=.002
    maximum_speed_m_s=.00025
    maximum_acceleration_m_s2=.0005
    gain_m_per_N_s=.0001
    filter_time_constant_s=.05

    def __init__(self):
        self.time=None;self.offset=0.;self.velocity=0.;self.filtered_load=None

    def update(self,time_s,palm_load_N):
        if (not all(np.isscalar(value) and not isinstance(value,(bool,np.bool_)) for value in (time_s,palm_load_N))
                or not np.isfinite([time_s,palm_load_N]).all() or time_s<0 or palm_load_N<0):
            raise ValueError('Require finite nonnegative measured palm load')
        if self.time is None:
            self.time=float(time_s);self.filtered_load=float(palm_load_N)
            return self.offset,self.info(0.)
        dt=float(time_s-self.time)
        if dt<0 or (dt>0 and abs(dt-.002)>1e-9):raise ValueError('Require consecutive2ms tactile updates')
        acceleration=0.
        if dt:
            self.filtered_load+=(1-np.exp(-dt/self.filter_time_constant_s))*(palm_load_N-self.filtered_load)
            error=self.target_load_N-self.filtered_load
            desired=self.gain_m_per_N_s*np.sign(error)*max(0.,abs(error)-.1)
            low,high=bounded_next_velocity([self.offset],[self.velocity],[0.],[self.maximum_offset_m],dt,self.maximum_speed_m_s,self.maximum_acceleration_m_s2)
            next_velocity=float(np.clip(desired,low[0],high[0]))
            acceleration=(next_velocity-self.velocity)/dt
            self.offset+=.5*dt*(self.velocity+next_velocity);self.velocity=next_velocity;self.time=float(time_s)
        if not -1e-12<=self.offset<=self.maximum_offset_m+1e-12:raise ValueError('Normal target exceeded its screened envelope')
        return self.offset,self.info(acceleration)

    def info(self,acceleration):
        return dict(time_s=self.time,target_load_N=self.target_load_N,filtered_palm_load_N=self.filtered_load,
                    normal_offset_m=self.offset,normal_offset_velocity_m_s=self.velocity,
                    normal_offset_acceleration_m_s2=float(acceleration),maximum_offset_m=self.maximum_offset_m)
