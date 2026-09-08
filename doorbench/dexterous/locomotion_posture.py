"""Motor-space posture adapter for a stopped H1 locomotion actor.

This changes only the commanded sagittal leg posture and its proprioceptive
reference. Root/foot states are never written or read here. Physical validation
is required: it is not a kinematic animation or a trained crouching policy.
"""
import numpy as np


def minimum_jerk(time_s, start_s, duration_s):
    u=np.clip((time_s-start_s)/duration_s,0.,1.)
    position=u*u*u*(10+u*(-15+6*u))
    velocity=30*u*u*(1-u)*(1-u)/duration_s if 0<u<1 else 0.
    return float(position),float(velocity)


def posture_offset(time_s, *, knee_delta=1.03):
    down,down_v=minimum_jerk(time_s,14.,4.)
    up,up_v=minimum_jerk(time_s,22.,4.)
    pattern=np.array([0.,0.,-.5,1.,-.5]*2)*knee_delta
    return pattern*(down-up),pattern*(down_v-up_v)


def transition_command(time_s):
    if time_s<1:return 'initial',np.zeros(3)
    if time_s<9:return 'approach walk',np.array([.4,0.,0.])
    if time_s<14:return 'brake and settle',np.zeros(3)
    if time_s<18:return 'lower posture',np.zeros(3)
    if time_s<22:return 'manipulation-height hold',np.zeros(3)
    if time_s<26:return 'raise posture',np.zeros(3)
    if time_s<27:return 'prepare restart',np.zeros(3)
    if time_s<32:return 'restart walk',np.array([.3,0.,0.])
    return 'final brake and settle',np.zeros(3)
