"""Bounded measured normal-load correction, without tangential pad springs.

The active plant supplies body contact forces. Geometry and Jacobians are read
from the teacher's private FK model; only original finger motor efforts change.
This privileged diagnostic is not a learned tactile policy.
"""
import mujoco
import numpy as np


def pressure_correction(target, load, inward):
    load=np.asarray(load,float);inward=np.asarray(inward,float)
    if load.shape!=(3,) or inward.shape!=(3,) or not np.isfinite(np.r_[target,load,inward]).all() or target<0 or abs(np.linalg.norm(inward)-1)>1e-6:
        raise ValueError('Finite force and unit inward normal required')
    # Contact reaction on the finger points away from the handle.
    measured=max(0.,-float(load@inward))
    return float(np.clip(.25*(target-measured),-1.,1.)),measured


class DigitPressureFeedback:
    def __init__(self,teacher):
        self.teacher=teacher
        self.jp=np.zeros((3,teacher.m.nv))

    def force(self,forces,blend,hand_loads):
        if not isinstance(hand_loads,dict):raise ValueError('Measured contact forces required')
        t=self.teacher;m,d=t.m,t.d;generalized=np.zeros(m.nv);measurements={}
        for digit,geoms in t.digit_geoms.items():
            nearest=None
            for geom in geoms:
                if not m.body(m.geom_bodyid[geom]).name.endswith('distal'):continue
                pair=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,geom,t.lever,.08,pair)
                if nearest is None or gap<nearest[0]:nearest=gap,geom,pair.copy()
            if nearest is None:raise ValueError('Distal pad geometry required')
            gap,geom,pair=nearest;delta=pair[3:]-pair[:3];distance=np.linalg.norm(delta)
            if distance<1e-7 or gap>.01:continue
            inward=delta/distance*(1 if gap>=0 else -1)
            body=int(m.geom_bodyid[geom]);name=m.body(body).name
            load=sum((np.asarray(v,float) for k,v in hand_loads.items() if k.rsplit('/',1)[-1]==name),np.zeros(3))
            correction,measured=pressure_correction(t.digit_forces[digit],load,inward)
            mujoco.mj_jac(m,d,self.jp,None,pair[:3],body)
            generalized+=self.jp.T@(inward*correction)
            measurements[digit]=dict(measured_normal_N=measured,correction_N=correction,gap_m=float(gap))
        result=np.asarray(forces,float).copy()
        result[t.fingers]+=blend*(t.finger_inverse@generalized[t.va])
        result=np.clip(result,t.caps[:,0],t.caps[:,1])
        if not np.isfinite(result).all():raise ValueError('Nonfinite pressure correction')
        return result,dict(operation_pad_control='measured-pressure-v1',digit_pressure_feedback=measurements,pad_feedback_blend=blend)
