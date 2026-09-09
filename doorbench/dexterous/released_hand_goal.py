"""Explicit released-hand goals for unstepped upright opening plans."""
import numpy as np
from scipy.spatial.transform import Rotation


def released_hand_phase(progress):
    if not np.isfinite(progress) or not -1e-12<=progress<=1+1e-12:
        raise ValueError('Bounded released-hand progress required')
    u=float(np.clip(progress,0.,1.))
    return float(np.clip(u**3*(10+u*(-15+6*u)),0.,1.))


def released_hand_goal(position,rotation,root_position,root_delta,outward,phase,*,frame='world',retreat_m=0.):
    position=np.asarray(position,float);rotation=np.asarray(rotation,float);root_position=np.asarray(root_position,float);root_delta=np.asarray(root_delta,float);outward=np.asarray(outward,float)
    if (position.shape!=(3,) or rotation.shape!=(3,3) or root_position.shape!=(3,) or root_delta.shape!=(6,) or outward.shape!=(3,) or
        not np.isfinite(np.r_[position,rotation.ravel(),root_position,root_delta,outward,phase,retreat_m]).all() or
        frame not in ('world','root') or not 0<=phase<=1 or not 0<=retreat_m<=.12 or (frame=='world' and retreat_m!=0) or abs(np.linalg.norm(outward)-1)>1e-6):
        raise ValueError('Explicit finite released-hand frame and bounded retreat required')
    if frame=='world':return position.copy(),rotation.copy()
    delta=Rotation.from_rotvec(root_delta[3:]).as_matrix()
    return root_position+root_delta[:3]+delta@(position-root_position)+phase*retreat_m*outward,delta@rotation
