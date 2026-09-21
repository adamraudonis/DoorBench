"""Capture existing actual inputs for a later initialized-standing continuation.

This observer grants no stage permission and never changes motor commands.
Normal and friction patches retain their own independent slot inventories.
"""
import numpy as np

from .isaac_opening_measurements import contact_force_pairs,hand_contact_loads,pose_parts
from .isaac_post_opening_measurements import continuation_contact_summary

BODY_NAMES=('left_ankle_link','right_ankle_link','lh_palm','rh_palm','leaf','leaf_handle')
SCHEMA='doorbench.isaac-standing-continuation-observation.v1'


def sparse_contact_buffer(counts,starts,fields,*,capacity):
    counts,starts=np.asarray(counts),np.asarray(starts)
    if (counts.ndim!=2 or starts.shape!=counts.shape or counts.dtype.kind not in 'iu'
            or starts.dtype.kind not in 'iu' or np.any(counts<0) or np.any(starts<0)
            or int(counts.sum())>=capacity):
        raise ValueError('Complete nontruncated contact counts and starts required')
    pairs=[];slots=[];occupied=set()
    for i,j in zip(*np.nonzero(counts)):
        first,count=int(starts[i,j]),int(counts[i,j])
        selected=list(range(first,first+count))
        if first+count>capacity or occupied.intersection(selected):
            raise ValueError('Disjoint in-bounds contact slices required')
        occupied.update(selected);slots.extend(selected);pairs.append([int(i),int(j),first,count])
    result=dict(shape=list(counts.shape),pairs=pairs,slots=slots)
    for key,value in fields.items():
        array=np.asarray(value)
        if array.ndim!=2 or array.shape[0]!=capacity or not np.isfinite(array[slots]).all():
            raise ValueError('Finite occupied contact values required')
        result[key]=array[slots].tolist()
    return result


def pack_standing_continuation(*,time_s,pose_time_s,sensor_paths,filter_paths,
                               normal_matrix,normal_buffers,friction_buffers,
                               body_poses,capacity,physics_qualified):
    """Use one completed interval and copied same-epoch actual body poses."""
    if (not np.isfinite([time_s,pose_time_s]).all() or time_s<=0
            or abs(time_s-pose_time_s)>1e-10 or abs(time_s/.002-round(time_s/.002))>1e-7):
        raise ValueError('Same-epoch actual 500 Hz continuation observation required')
    af,ap,an,ad,ac,ast=[np.asarray(value) for value in normal_buffers]
    vectors,points,counts,starts=[np.asarray(value) for value in friction_buffers]
    vectors=vectors.reshape(capacity,3);points=points.reshape(capacity,3)
    summary=continuation_contact_summary(sensor_paths,filter_paths,af,an,ad,ac,ast,
        capacity=capacity,physics_qualified=physics_qualified)
    forces=contact_force_pairs(normal_matrix,vectors,counts,starts,capacity=capacity)
    if forces.shape[:2]!=ac.shape:raise ValueError('Normal and friction pair layouts differ')
    poses={}
    for name in BODY_NAMES:
        pose_parts(body_poses[name])
        poses[name]=np.asarray(body_poses[name]).tolist()
    normal=sparse_contact_buffer(ac,ast,dict(force_N=af,point_world=ap,normal_world=an,distance_m=ad),capacity=capacity)
    friction=sparse_contact_buffer(counts,starts,dict(force_N=vectors,point_world=points),capacity=capacity)
    hands=hand_contact_loads(sensor_paths,forces)
    matrix=np.asarray(normal_matrix)
    normal_pairs=[[int(i),int(j),*matrix[i,j].tolist()] for i,j in np.argwhere(np.any(matrix!=0,axis=-1))]
    return dict(schema=SCHEMA,time_s=float(time_s),pose_time_s=float(pose_time_s),
        contact_interval_s=[float(time_s-.002),float(time_s)],body_poses=poses,
        foot_loads_N=summary['foot_loads'].tolist(),hand_forces_world_N={name:value.tolist() for name,value in hands.items()},
        evidence=summary['evidence'],release_normal_world=None if summary['release_normal_world'] is None else summary['release_normal_world'].tolist(),
        raw=dict(capacity=capacity,pair_shape=list(matrix.shape[:2]),normal_force_pairs=normal_pairs,normal=normal,friction=friction),
        authorized_stages=0,scope='Measured continuation inputs only; no controller or physical stage qualification')
