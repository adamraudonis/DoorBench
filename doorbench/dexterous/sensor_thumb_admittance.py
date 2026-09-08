"""Opt-in five-thumb-goal relief; original capped force owner remains inside."""
import numpy as np
from .sensor_thumb_flexion_force import SensorThumbFlexionForceController
from .thumb_normal_admittance import RobotThumbNormalAdmittance, validate_profile


class _ThumbAdmittanceArm:
    def __init__(self, inner, calculator, tactile_slice):
        self.inner=inner;self.calculator=calculator;self.tactile_slice=tactile_slice
    def __getattr__(self,name):return getattr(self.inner,name)
    def reset_episode(self):self.calculator.reset();self.inner.reset_episode()
    def force(self,packet,*,now_s,joint_goals=None):
        extra={};goals=dict(joint_goals)
        if now_s>=23.-1e-9:
            grid=packet['tactile'][self.tactile_slice].reshape(3,2,4)
            sums=grid[:,:,1:3].sum(axis=(1,2)).astype(float)
            goals,extra=self.calculator.update(goals,packet['joint_position'],sums[[1,2,0]],max(0.,float(sums[0])),now_s=now_s)
        force,info=self.inner.force(packet,now_s=now_s,joint_goals=goals)
        return force,dict(info,**extra)


class SensorThumbAdmittanceController(SensorThumbFlexionForceController):
    def __init__(self,*args,thumb_admittance_protocol,**kwargs):
        self.thumb_admittance_protocol=validate_profile(thumb_admittance_protocol)
        super().__init__(*args,**kwargs)
        b=self.arm.balance
        self.thumb_admittance=RobotThumbNormalAdmittance(b.m,self.joint_names,self.action_names,b.matrix,self.thumb_admittance_protocol)
        self.arm=_ThumbAdmittanceArm(self.arm,self.thumb_admittance,self.slices['th'])
    def force(self,packet,*,now_s):
        force,info=super().force(packet,now_s=now_s)
        self.info=dict(info,high_level_controller='thumb_normal_admittance_v1')
        return force,dict(self.info)
