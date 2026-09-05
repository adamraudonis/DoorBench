"""Optional local time parameterization of an unchanged MuJoCo pose sequence.

Only time knots change. Native coordinates, actor poses, frame correspondence,
contact labels and source ``native_time`` are preserved. This module enforces the
same sampled finite-difference derivative definitions as the independent planned
reference validator; it does not establish continuous-time acceleration bounds,
collision/contact feasibility, dynamic balance, or a time-optimal trajectory.

Adjacent interval stretch ratios are bounded to make the *discrete clock* gradual.
The caller must independently revalidate the saved result, including interpolation
and source/contact metadata. In particular, retiming cannot repair a bad pose or
make an oscillating pose sequence look natural. No files are written or adopted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class RetimeResult:
    time: np.ndarray
    native_time: np.ndarray | None
    interval_scale: np.ndarray
    metrics: dict
    success: bool


def _limits(value, default, name):
    result=np.asarray(default if value is None else value,dtype=float)
    if result.shape!=default.shape or not np.isfinite(result).all() or np.any(result<=0):
        raise ValueError(f'{name} must contain {len(default)} finite positive actor-DOF limits')
    return result.copy()


def _clock_envelope(scale,neighbor_ratio):
    """Smallest pointwise majorant with adjacent log-scale slope bounded."""
    slope=math.log(neighbor_ratio);offset=np.arange(len(scale))*slope
    logs=np.log(scale)
    forward=np.maximum.accumulate(logs+offset)-offset
    reverse=(np.maximum.accumulate((logs-offset)[::-1])[::-1]+offset)
    return np.exp(np.maximum(forward,reverse))


def _derivatives(local_displacement,world_displacement,intervals,velocity_limits,acceleration_limits):
    velocity=local_displacement/intervals[:,None]
    world_velocity=world_displacement/intervals[:,None]
    velocity_ratio=np.abs(velocity)/velocity_limits
    acceleration=np.diff(world_velocity,axis=0)/((intervals[1:]+intervals[:-1])*.5)[:,None]
    acceleration_ratio=np.abs(acceleration)/acceleration_limits
    return velocity_ratio,acceleration_ratio


def _summary(velocity_ratio,acceleration_ratio,labels):
    def worst(values,frame_offset):
        if not values.size:return None
        i,j=np.unravel_index(int(np.argmax(values)),values.shape)
        return {'frame':int(i+frame_offset),'dof':labels[j],'ratio':float(values[i,j])}
    return {'max_velocity_ratio':float(np.max(velocity_ratio)),
            'max_acceleration_ratio':float(np.max(acceleration_ratio)) if acceleration_ratio.size else 0.,
            'velocity_violations':int(np.sum(np.max(velocity_ratio,axis=1)>1.+1e-9)),
            'acceleration_violations':int(np.sum(np.max(acceleration_ratio,axis=1)>1.+1e-9)) if acceleration_ratio.size else 0,
            'worst_velocity':worst(velocity_ratio,0),'worst_acceleration':worst(acceleration_ratio,1)}


def retime_trajectory(model,qpos,time,*,actor_dof_indices=None,root_joint='actor_root',
                      native_time=None,velocity_limits=None,acceleration_limits=None,
                      max_iterations=200,max_interval_scale=20.,neighbor_ratio=1.15):
    """Return a locally stretched clock for full combined MuJoCo ``qpos``.

    Args:
        model: MuJoCo model containing the free actor root and actor joints.
        qpos: ``(N, model.nq)`` native + actor coordinates, never modified.
        time: Strictly increasing, nonnegative ``(N,)`` original actor clock.
        actor_dof_indices: Checked DOFs in limit-array order. Defaults to every
            joint whose name starts with ``actor_``; native DOFs are not limited.
        root_joint: Name of the free root, used to transport angular differences
            to world axes before acceleration differences, matching the validator.
        native_time: Optional nondecreasing source clock, returned as an exact copy.
        velocity_limits: Optional positive vector in actor-DOF order. Defaults:
            root translation .8 m/s, root rotation 1.5 rad/s, other DOFs 2.5 rad/s.
        acceleration_limits: Defaults: translation 3 m/s², world root rotation
            8 rad/s², other DOFs 15 rad/s².
        max_interval_scale: Hard cap relative to each original interval. An
            insufficient cap or iteration budget returns ``success=False``.
        neighbor_ratio: Maximum ratio of neighboring interval stretch factors.
            This regularizes discrete time knots, not a continuous clock spline.

    Each violated acceleration boundary stretches its two adjacent intervals.
    Simultaneously scaling them by s reduces that boundary's acceleration by s²;
    overlapping updates are recomputed until all bounds pass or the budget ends.
    A log-slope envelope spreads changes locally to avoid abrupt clock changes.
    This monotone feasible-clock search is deterministic, not globally time optimal.
    """
    import mujoco

    q=np.asarray(qpos,dtype=float);t=np.asarray(time,dtype=float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or t[0]<0 or np.any(np.diff(t)<=0):
        raise ValueError('time must contain at least two finite, strictly increasing nonnegative samples')
    if q.shape!=(len(t),model.nq) or not np.isfinite(q).all():
        raise ValueError(f'qpos must have shape ({len(t)}, {model.nq}) with finite coordinates')
    for j in range(model.njnt):
        kind=int(model.jnt_type[j]);address=int(model.jnt_qposadr[j])
        quaternion=q[:,address+3:address+7] if kind==int(mujoco.mjtJoint.mjJNT_FREE) else (
            q[:,address:address+4] if kind==int(mujoco.mjtJoint.mjJNT_BALL) else None)
        if quaternion is not None and np.any(np.abs(np.linalg.norm(quaternion,axis=1)-1)>1e-5):
            raise ValueError(f'qpos joint {model.joint(j).name!r} requires unit quaternions')
    source=None if native_time is None else np.asarray(native_time)
    if source is not None:
        if source.shape!=t.shape or source.dtype.kind not in 'iuf' or not np.isfinite(source).all() or source[0]<0 or np.any(source[1:]<source[:-1]):
            raise ValueError('native_time must be a finite, nondecreasing, nonnegative vector matching time')
        source=source.copy()
    if type(max_iterations) is not int or max_iterations<1:raise ValueError('max_iterations must be a positive integer')
    if not math.isfinite(max_interval_scale) or max_interval_scale<1:raise ValueError('max_interval_scale must be finite and at least 1')
    if not math.isfinite(neighbor_ratio) or neighbor_ratio<=1:raise ValueError('neighbor_ratio must be finite and greater than 1')
    try:root=model.joint(root_joint)
    except KeyError as exc:raise ValueError(f'Unknown free root joint {root_joint!r}') from exc
    if int(root.type[0])!=int(mujoco.mjtJoint.mjJNT_FREE):raise ValueError('root_joint must be a MuJoCo free joint')
    root_dof=int(root.dofadr[0]);root_address=int(root.qposadr[0])
    if actor_dof_indices is None:
        indices=np.flatnonzero([model.joint(int(model.dof_jntid[i])).name.startswith('actor_') for i in range(model.nv)])
    else:
        indices=np.asarray(actor_dof_indices)
        if indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer):raise ValueError('actor_dof_indices must be a vector of unique integer DOF indices')
    if not len(indices) or len(np.unique(indices))!=len(indices) or np.any(indices<0) or np.any(indices>=model.nv):
        raise ValueError('actor_dof_indices must be nonempty, unique and inside the model DOF range')
    indices=indices.astype(int,copy=True)
    vdefault=np.array([.8 if root_dof<=i<root_dof+3 else 1.5 if root_dof+3<=i<root_dof+6 else 2.5 for i in indices])
    adefault=np.array([3. if root_dof<=i<root_dof+3 else 8. if root_dof+3<=i<root_dof+6 else 15. for i in indices])
    vmax=_limits(velocity_limits,vdefault,'velocity_limits');amax=_limits(acceleration_limits,adefault,'acceleration_limits')
    labels=[]
    for i in indices:
        name=model.joint(int(model.dof_jntid[i])).name
        if root_dof<=i<root_dof+6:name+=':'+['x','y','z','rx','ry','rz'][i-root_dof]
        labels.append(name)
    displacement=np.empty((len(t)-1,model.nv))
    for i in range(len(t)-1):mujoco.mj_differentiatePos(model,displacement[i],1.,q[i],q[i+1])
    if not np.isfinite(displacement).all():raise ValueError('qpos differences exceed finite derivative range')
    world=displacement.copy();rotation=np.empty(9)
    for i in range(len(t)-1):
        mujoco.mju_quat2Mat(rotation,q[i,root_address+3:root_address+7])
        world[i,root_dof+3:root_dof+6]=rotation.reshape(3,3)@displacement[i,root_dof+3:root_dof+6]
    local=displacement[:,indices];world=world[:,indices];original=np.diff(t)
    before=_summary(*_derivatives(local,world,original,vmax,amax),labels)
    # Exact pass-through for an already valid clock; no cumulative-sum drift.
    intervals=original.copy();iterations=0;converged=False
    scale=np.maximum(1.,np.max(np.abs(local)/vmax,axis=1)/original)
    for iterations in range(1,max_iterations+1):
        scale=np.minimum(max_interval_scale,_clock_envelope(scale,neighbor_ratio))
        intervals=original*scale
        vr,ar=_derivatives(local,world,intervals,vmax,amax)
        if np.max(vr)<=1.+1e-9 and (not ar.size or np.max(ar)<=1.+1e-9):
            converged=True;break
        factors=np.ones(len(intervals))
        if ar.size:
            stretch=np.maximum(1.,np.sqrt(np.max(ar,axis=1))*1.002)
            # Do not stretch already feasible boundaries just for the safety margin.
            stretch[np.max(ar,axis=1)<=1.+1e-9]=1.
            factors[:-1]=np.maximum(factors[:-1],stretch)
            factors[1:]=np.maximum(factors[1:],stretch)
        next_scale=np.minimum(max_interval_scale,scale*factors)
        if np.max(np.abs(next_scale-scale))<1e-13:break
        if iterations<max_iterations:scale=next_scale
    changed=bool(np.any(scale>1.+1e-12))
    new_time=np.r_[t[0],t[0]+np.cumsum(intervals)] if changed else t.copy()
    # Recheck representable returned knots, not just pre-sum duration arithmetic.
    vr,ar=_derivatives(local,world,np.diff(new_time),vmax,amax)
    after=_summary(vr,ar,labels)
    achieved_neighbor=float(np.max(np.maximum(scale[1:]/scale[:-1],scale[:-1]/scale[1:]))) if len(scale)>1 else 1.
    success=converged and after['max_velocity_ratio']<=1.+1e-8 and after['max_acceleration_ratio']<=1.+1e-8
    metrics={'schema':'doorbench.retime.v1','frames':len(t),'iterations':iterations,
             'original_duration_s':float(t[-1]-t[0]),'duration_s':float(new_time[-1]-new_time[0]),
             'duration_scale':float((new_time[-1]-new_time[0])/(t[-1]-t[0])),
             'maximum_interval_scale':float(np.max(scale)),'changed_intervals':int(np.sum(scale>1.+1e-12)),
             'maximum_neighbor_scale_ratio':achieved_neighbor,'neighbor_ratio_limit':float(neighbor_ratio),
             'interval_scale_cap':float(max_interval_scale),'before':before,'after':after,
             'velocity_limits':vmax.tolist(),'acceleration_limits':amax.tolist(),'dof_labels':labels,
             'failure':None if success else 'Derivative bounds not met within the interval-scale cap or iteration budget.',
             'scope':'Sampled actor velocity and world-basis angular acceleration; unchanged poses and source clock. Independent geometry/contact/interpolation revalidation required.'}
    return RetimeResult(new_time,source,scale.copy(),metrics,bool(success))
