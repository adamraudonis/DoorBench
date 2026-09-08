"""Evaluator-only reduction of synchronized actual straight-lever pad evidence."""
import numpy as np
from .isaac_pad_audit import shadow_physx_pad_grasp
from .sensor_balance_runtime import _number

RAW_SCHEMA='doorbench.shadow-raw-pad-evidence.v1'
RAW_FIELDS={'schema','interval_start_s','interval_end_s','geometry_time_s','clock','scope','contacts',
    'body_transforms_xyzw','handle_pair_forces_world_N','lever','contact_capacity','active_contact_count','normal_pair_force_consistency_error_N'}


def _physx_source_pair_error(contacts,pairs):
    """Replay v1 PhysX float32 product/axis-sum/difference/norm in source order.

    JSON preserves each source scalar but not NumPy's arithmetic dtype. The
    v1 producer reduces its copied PhysX float32 arrays *before* serialization.
    Casting scalars to float64 and then comparing that norm to the producer's
    recorded float32 norm is not a reproduction of that calculation.
    This is only the receipt comparison; a separate float64 force-balance
    calculation still has to meet the original 1e-3 N physical threshold.
    """
    groups={name:[] for name in pairs}
    for c in contacts:groups[c['body']].append(c)
    error=0.
    for name,patches in groups.items():
        force=np.asarray([c['normal_force_N'] for c in patches],float)
        normal=np.asarray([c['normal'] for c in patches],float).reshape(-1,3)
        pair=np.asarray(pairs[name],float)
        for value in (force,normal,pair):
            if not np.array_equal(value,value.astype(np.float32).astype(float)):
                raise ValueError('PhysX v1 force/normal/matrix scalars must preserve exact float32 source values')
        vectors=force.astype(np.float32)[:,None]*normal.astype(np.float32)
        difference=vectors.sum(axis=0)-pair.astype(np.float32)
        error=max(error,float(np.linalg.norm(difference)))
    return error


def evaluate_pad_evidence(evidence,*,time_s,physics_dt_s=.002):
    """Recompute distal surfaces, cylinder margins, load opposition and pair sums.

    Native contact geometry belongs to the interval start; PhysX supplies its
    synchronized end frames. The explicit clock prevents mixing those poses.
    World bodies/objects and identities remain evaluator data only.
    """
    e=evidence
    if type(e) is not dict or set(e)!=RAW_FIELDS or e['schema']!=RAW_SCHEMA:raise ValueError('Require complete raw pad evidence')
    if physics_dt_s!=.002 or not _number(time_s) or time_s<physics_dt_s:raise ValueError('Require an actual2ms interval endpoint')
    if not all(_number(e[k]) for k in ('interval_start_s','interval_end_s','geometry_time_s')):raise ValueError('Invalid raw contact clocks')
    if abs(e['interval_start_s']-(time_s-physics_dt_s))>1e-8 or abs(e['interval_end_s']-time_s)>1e-8:raise ValueError('Contact interval disagrees with actual step')
    if e['clock']=='physx-interval-end' and e['scope']=='complete-handle-body':geometry=time_s;physx=True
    elif e['clock']=='native-interval-start' and e['scope']=='native-lever-collider':geometry=time_s-physics_dt_s;physx=False
    else:raise ValueError('Unsupported or mixed engine contact epoch/scope')
    if abs(e['geometry_time_s']-geometry)>1e-8:raise ValueError('Contact geometry and body-frame epoch disagree')
    poses=e['body_transforms_xyzw'];contacts=e['contacts'];count=e['active_contact_count'];capacity=e['contact_capacity']
    if type(poses) is not dict or type(contacts) is not list or type(count) is not int or count<len(contacts):raise ValueError('Malformed actual body/contact count')
    if physx and (type(capacity) is not int or not count<capacity or capacity<=0):raise ValueError('Contact archive may be truncated')
    if not physx and capacity is not None:raise ValueError('Native dynamic contact list does not claim a fixed PhysX capacity')
    for name,pose in poses.items():
        value=np.asarray(pose,float)
        if type(name) is not str or not name.rsplit('/',1)[-1].startswith('rh_') or value.shape!=(7,) or not np.isfinite(value).all() or abs(np.linalg.norm(value[3:])-1)>1e-5:
            raise ValueError('Require exact right-hand synchronized body transforms')
    sums={name:np.zeros(3) for name in poses}
    for c in contacts:
        if type(c) is not dict or set(c)!={'body','position','normal','normal_force_N'} or c['body'] not in poses or not _number(c['normal_force_N']) or c['normal_force_N']<0:
            raise ValueError('Malformed raw hand contact; no precomputed pad labels accepted')
        point=np.asarray(c['position'],float);normal=np.asarray(c['normal'],float)
        if point.shape!=(3,) or normal.shape!=(3,) or not np.isfinite(np.r_[point,normal]).all() or abs(np.linalg.norm(normal)-1)>1e-5:raise ValueError('Invalid raw contact point/normal')
        sums[c['body']]+=c['normal_force_N']*normal
    pair_error=None;source_pair_error=None
    if physx:
        pairs=e['handle_pair_forces_world_N'];declared=e['normal_pair_force_consistency_error_N']
        if type(pairs) is not dict or set(pairs)!=set(poses) or not _number(declared) or not 0<=declared<=1e-3:raise ValueError('Missing independent PhysX pair-force evidence')
        pair_error=0.
        for name,value in pairs.items():
            v=np.asarray(value,float)
            if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError('Invalid actual pair-force vector')
            pair_error=max(pair_error,float(np.linalg.norm(sums[name]-v)))
        source_pair_error=_physx_source_pair_error(contacts,pairs)
        if pair_error>1e-3 or source_pair_error>1e-3 or abs(source_pair_error-declared)>1e-8:raise ValueError('Raw patches do not reproduce recorded pair forces')
    elif e['handle_pair_forces_world_N'] is not None or e['normal_pair_force_consistency_error_N'] is not None:
        raise ValueError('Native archive cannot invent independent PhysX pair measurements')
    lever=e['lever']
    if type(lever) is not dict or set(lever)!={'center','axis','half_length','radius'}:raise ValueError('Require measured straight-lever geometry')
    result=shadow_physx_pad_grasp(contacts,poses,**lever,profile='distal-pad-v1')
    result['invalid_loaded_patches']=sum(c['normal_force_N']>1e-6 and not c['pad_qualified'] for c in result['contacts'])
    result['raw_pair_force_reconstruction_error_N']=pair_error
    result['raw_source_pair_force_reconstruction_error_N']=source_pair_error
    result['raw_pair_force_source_arithmetic']='float32-product-axis-sum-difference-norm-v1' if physx else None
    result['raw_geometry_clock']=e['clock'];result['raw_contact_scope']=e['scope']
    return result
