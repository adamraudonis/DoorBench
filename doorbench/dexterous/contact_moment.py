"""Decompose an archived contact wrench about a measured hinge axis."""
import numpy as np


def contact_moment(point,frame,wrench,anchor,axis,*,body_index):
    point,frame,wrench,anchor,axis=map(lambda x:np.asarray(x,float),(point,frame,wrench,anchor,axis))
    if (point.shape!=(3,) or frame.shape!=(3,3) or wrench.shape!=(6,) or anchor.shape!=(3,) or axis.shape!=(3,) or body_index not in (0,1)
        or not np.isfinite(np.r_[point,frame.ravel(),wrench,anchor,axis]).all()
        or not np.allclose(frame@frame.T,np.eye(3),atol=1e-6) or abs(np.linalg.det(frame)-1)>1e-6 or abs(np.linalg.norm(axis)-1)>1e-6):
        raise ValueError('Finite synchronized contact frame, wrench and unit hinge axis required')
    sign=(-1,1)[body_index]
    force=sign*frame.T@wrench[:3];normal=sign*frame[0]*wrench[0]
    normal_moment=float(np.cross(point-anchor,normal)@axis)
    tangent_moment=float(np.cross(point-anchor,force-normal)@axis)
    couple=float((sign*frame.T@wrench[3:])@axis)
    return dict(moment_about_hinge_Nm=normal_moment+tangent_moment+couple,
                normal_force_moment_Nm=normal_moment,tangential_force_moment_Nm=tangent_moment,
                contact_couple_moment_Nm=couple,force_on_body_world_N=force.tolist())
