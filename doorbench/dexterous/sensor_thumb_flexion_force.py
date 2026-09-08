"""Separately declared THJ1/THJ2 pressure with original opposition posture.

Uses the qualified switching repair, unchanged local targets and original motor
caps. Handover begins only at19s through the existing two-second smooth ramp.
"""
import numpy as np
from .robot_thumb_flexion_force import RobotThumbFlexionForce
from .sensor_smooth_contact_force import SensorSmoothContactForceController


THUMB_PROTOCOL={
    'schema':'doorbench.sensor-thumb-flexion-pressure.v1',
    'start_after_s':19.,'ramp_seconds':2.,'duration_s':36.,
    'thumb_pressure_joints':['rh_THJ2','rh_THJ1'],
    'thumb_posture_joints':['rh_THJ5','rh_THJ4','rh_THJ3'],
    'original_transmission_and_caps':True,
    'other_digits':'unchanged original coupled normal-effort hierarchy',
    'local_force_targets_N':[2.,2.,2.,2.,3.],
    'contact_mode_profile':'doorbench.sensor-contact-mode-thumb-flexion.v1',
    'first19s':'same instance-specific acquisition goals and sensor feedback',
    'scope':'sensor-only state and local-touch pressure; no runtime object geometry or IDs',
}


def validate_thumb_protocol(protocol):
    if type(protocol) is not dict or set(protocol)!=set(THUMB_PROTOCOL):raise ValueError('Exact declared thumb allocation required')
    for key,wanted in THUMB_PROTOCOL.items():
        value=protocol[key]
        if type(wanted) is float:
            if type(value) not in (int,float) or not np.isfinite(value) or value!=wanted:raise ValueError('Unsupported thumb parameter: '+key)
        elif type(value) is not type(wanted) or value!=wanted:raise ValueError('Unsupported thumb parameter: '+key)
    return dict(protocol)


class SensorThumbFlexionForceController(SensorSmoothContactForceController):
    thumb_flexion_allocation=True
    def __init__(self,*args,thumb_flexion_protocol,**kwargs):
        self.thumb_flexion_protocol=validate_thumb_protocol(thumb_flexion_protocol)
        super().__init__(*args,**kwargs)
        b=self.arm.balance
        self.pad_force=RobotThumbFlexionForce(b.m,self.joint_names,self.action_names,b.matrix)

    def force(self,packet,*,now_s):
        force,info=super().force(packet,now_s=now_s)
        self.info=dict(info,high_level_controller='thumb_flexion_pressure_hierarchy_v1',
            thumb_pressure_joints=self.thumb_flexion_protocol['thumb_pressure_joints'],
            thumb_retained_posture_joints=self.thumb_flexion_protocol['thumb_posture_joints'])
        return force,dict(self.info)
