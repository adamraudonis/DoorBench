"""Explicit float32 destination-pose admission for an unstepped planner.

This is a kinematic mapping check, not collision or physical qualification.
It never replaces the original native 1e-9 attained-state check, and never
writes a simulator. Both measured and reconstructed poses remain in the receipt.
"""
import hashlib
import json
import mujoco
import numpy as np


class DestinationKinematicsFailure(ValueError):
    def __init__(self, receipt):
        self.receipt=receipt
        super().__init__('Destination body measurements disagree with planner kinematics')


def admit_destination_return_kinematics(model, *, qpos, qvel, time_s,
        geometry_time_s, body_names, body_poses_xyz_wxyz):
    q=np.array(qpos,float,copy=True);v=np.array(qvel,float,copy=True)
    poses=np.array(body_poses_xyz_wxyz,float,copy=True)
    names=list(body_names)
    if (q.shape!=(model.nq,) or v.shape!=(model.nv,) or not names
            or len(names)!=len(set(names)) or poses.shape!=(len(names),7)
            or not np.isfinite(np.r_[q,v,poses.ravel(),time_s,geometry_time_s]).all()
            or time_s<0 or abs(time_s-geometry_time_s)>1e-10):
        raise ValueError('Require complete finite same-epoch state and unique measured bodies')
    required={'leaf_handle','robot/left_ankle_link','robot/right_ankle_link','robot/torso_link'}
    required.update(model.body(int(model.site_bodyid[model.site('robot/'+s+'_palm_touch').id])).name for s in ('rh','lh'))
    if not required.issubset(names):
        raise ValueError('Require actual handle, palm, foot and torso body poses')
    ids=np.array([model.body(n).id for n in names])
    free=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
    if len(free)!=1:raise ValueError('Require one free humanoid root')
    qa=int(model.jnt_qposadr[free[0]])
    norms=np.linalg.norm(poses[:,3:],axis=1);root_norm=np.linalg.norm(q[qa+3:qa+7])
    if max(abs(root_norm-1.),float(np.max(abs(norms-1.))))>2e-6:
        raise ValueError('Measured quaternion is not normalized within float32 admission bound')
    original_q=q.copy()
    q[qa+3:qa+7]/=root_norm
    rotations=[]
    for pose,norm in zip(poses,norms):
        matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,pose[3:]/norm)
        rotations.append(matrix.reshape(3,3))
    d=mujoco.MjData(model);d.qpos[:]=q;d.qvel[:]=v;d.time=float(time_s)
    mujoco.mj_kinematics(model,d)
    position=np.linalg.norm(d.xpos[ids]-poses[:,:3],axis=1)
    # Chord-to-angle avoids acos precision loss near identity.
    chord=np.linalg.norm(d.xmat[ids].reshape(-1,3,3)-rotations,axis=(1,2))
    angle=2*np.arcsin(np.clip(chord/(2*np.sqrt(2)),0,1))
    measured=dict(qpos=original_q.tolist(),qvel=v.tolist(),time_s=float(time_s),
        geometry_time_s=float(geometry_time_s),body_names=names,
        body_poses_xyz_wxyz=poses.tolist())
    receipt=dict(schema='doorbench.destination-return-kinematics.v1',
        profile='isaac-float32-2um-2urad-v1',
        passed=bool(np.max(position)<=2e-6 and np.max(angle)<=2e-6),
        maximum_position_error_m=float(np.max(position)),maximum_rotation_error_rad=float(np.max(angle)),
        position_limit_m=2e-6,rotation_limit_rad=2e-6,
        quaternion_normalization_limit=2e-6,
        maximum_root_quaternion_normalization_change=float(np.max(abs(q-original_q))),
        measured=measured,reconstructed_positions_m=d.xpos[ids].tolist(),
        reconstructed_rotations=d.xmat[ids].reshape(-1,3,3).tolist(),
        measured_sha256=hashlib.sha256(json.dumps(measured,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        simulation_steps=0,active_plant_writes=0,
        limitation='Kinematic mapping only; no collision-model, path or physical-motion qualification')
    if not receipt['passed']:raise DestinationKinematicsFailure(receipt)
    return d,receipt
