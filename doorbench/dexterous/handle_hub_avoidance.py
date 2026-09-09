"""Bounded finger-motor avoidance of the handle hub in the private teacher model.

Privileged geometric feedback; no physical pose writes or external wrench.
"""
import mujoco
import numpy as np


def avoidance_force(gap,direction,velocity):
    direction=np.asarray(direction,float);velocity=np.asarray(velocity,float)
    if direction.shape!=(3,) or velocity.shape!=(3,) or not np.isfinite(np.r_[gap,direction,velocity]).all() or abs(np.linalg.norm(direction)-1)>1e-6:
        raise ValueError('Finite measured gap and unit outward direction required')
    if gap>=.004:return np.zeros(3)
    magnitude=float(np.clip(800*(.004-gap)-3*float(velocity@direction),0.,3.))
    return magnitude*direction


class HandleHubAvoidance:
    def __init__(self,teacher,*,include_distal=False):
        if type(include_distal) is not bool:raise ValueError('Explicit distal avoidance selection required')
        self.include_distal=include_distal
        self.teacher=teacher;m=teacher.m
        self.hub=m.geom('analytic_handle_hub').id
        self.geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name in (('rh_lfmiddle','rh_lfproximal','rh_lfdistal') if include_distal else ('rh_lfmiddle','rh_lfproximal'))]
        if not self.geoms:raise ValueError('Original little-finger collision geometry required')
        self.jp=np.zeros((3,m.nv))

    def force(self,forces,blend):
        t=self.teacher;m,d=t.m,t.d;generalized=np.zeros(m.nv);nearest=None
        for g in self.geoms:
            pair=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,self.hub,.02,pair)
            if nearest is None or gap<nearest[0]:nearest=(gap,g,pair.copy())
        gap,g,pair=nearest;delta=pair[:3]-pair[3:];distance=np.linalg.norm(delta);force=np.zeros(3)
        if distance>1e-7 and gap<.004:
            direction=delta/distance*(1 if gap>=0 else -1)
            mujoco.mj_jac(m,d,self.jp,None,pair[:3],int(m.geom_bodyid[g]))
            force=avoidance_force(gap,direction,self.jp@d.qvel)
            generalized=self.jp.T@force
        result=np.asarray(forces,float).copy()
        result[t.fingers]+=blend*(t.finger_inverse@generalized[t.va])
        result=np.clip(result,t.caps[:,0],t.caps[:,1])
        if not np.isfinite(result).all():raise ValueError('Nonfinite bounded hub-avoidance command')
        t.last_force=result.copy()
        return result,dict(hub_avoidance_profile='whole-little-finger-3N-v1' if self.include_distal else 'little-finger-3N-v1',hub_gap_m=float(gap),hub_avoidance_force_N=float(np.linalg.norm(force)),hub_avoidance_blend=float(blend))
