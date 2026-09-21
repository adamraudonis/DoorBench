"""Reconcile diagnosed numerical diagnostics after fresh actual-source checks.

Raw evidence, labels, coordinates and original gates remain exact. This does
not admit a source on its own: the caller first runs admit_isaac_release_context.
The archived immutable context is reused only after this bounded comparison;
the fresh diagnostic receipt stays separate from its canonical identity.
"""
import hashlib
import importlib.metadata
import json
import platform

import mujoco
import numpy as np

from .isaac_release_planning import IsaacReleasePlanningContext

SCHEMA = 'doorbench.isaac-release-numerical-reconciliation.v1'
DIAGNOSTIC_ABSOLUTE_TOLERANCE = 1e-15
_MEASURED = ('coordinate_admission', 'measured_body_admission', 'measured')
_MEASURED_HASH = ('coordinate_admission', 'measured_body_admission', 'measured_sha256')


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_encode(value).encode()).hexdigest()


def _fresh_gates(admission):
    if (admission.get('schema') != 'doorbench.isaac-release-source.v1'
            or admission.get('source_engine') != 'isaac-physx'
            or admission['source_qualification'].get('passed') is not True
            or admission['coordinate_admission'].get('passed') is not True
            or admission['coordinate_admission']['coordinate_admission']['state_admission'].get('passed') is not True
            or admission['measured_rest'].get('window_samples') != 251
            or any(type(admission.get(k)) is not int or admission[k] != 0
                   for k in ('physics_steps', 'source_sample_playback', 'authorized_stages'))):
        raise ValueError('Fresh original actual-source admission must pass before reconciliation')
    for field in (admission['material_frame_admission'],
                  admission['coordinate_admission']['measured_body_admission']):
        if (field.get('passed') is not True or field.get('position_limit_m') != 2e-6
                or field.get('rotation_limit_rad') != 2e-6):
            raise ValueError('Original fresh two-micrometer/two-microradian admission required')
    material = admission['material_frame_admission']
    body = admission['coordinate_admission']['measured_body_admission']
    if body.get('quaternion_normalization_limit') != 2e-6:
        raise ValueError('Original quaternion normalization bound required')
    for value, limit in ((material['maximum_position_error_m'], 2e-6),
                         (material['maximum_body_position_error_m'], 2e-6),
                         (material['maximum_body_rotation_error_rad'], 2e-6),
                         (body['maximum_position_error_m'], 2e-6),
                         (body['maximum_rotation_error_rad'], 2e-6)):
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= limit:
            raise ValueError('Fresh numerical admission did not satisfy original limits')


def reconcile_isaac_release_context(candidate, fresh_context):
    """Return (archived canonical context, separate fresh comparison receipt).

Only three diagnosed quantities may differ by <=1e-15: the transformed free
root angular velocity, material-frame maximum rotation error, and per-contact
radial alignment. The measured-payload checksum must match its own complete
payload on both sides. It is never ignored or replaced with a claimed hash.
"""
    if not isinstance(fresh_context, IsaacReleasePlanningContext) or type(candidate) is not dict:
        raise ValueError('Explicit candidate and freshly admitted Isaac planning context required')
    if candidate.get('schema') != 'doorbench.isaac-profiled-release-candidate.v1':
        raise ValueError('Original distinct Isaac release candidate required')
    fresh_context.verify_inputs()
    fresh = fresh_context.admission
    _fresh_gates(fresh)
    archived = candidate.get('source_admission')
    if type(archived) is not dict or _sha(archived) != candidate.get('source_context_sha256'):
        raise ValueError('Archived context must match its exact canonical SHA256')
    _fresh_gates(archived)
    if (not np.array_equal(candidate.get('initial_qpos'), fresh_context.qpos)
            or not np.array_equal(archived.get('initial_qpos'), fresh_context.qpos)
            or candidate.get('initial_time_s') != fresh_context.terminal_time_s):
        raise ValueError('Exact normalized source coordinates and epoch cannot be reconciled')
    for admission in (archived, fresh):
        body = admission['coordinate_admission']['measured_body_admission']
        if body.get('measured_sha256') != _sha(body['measured']):
            raise ValueError('Measured payload checksum differs from its own complete payload')
    scene = fresh_context.scene()
    model = scene.m
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError('Original model must contain exactly one free root')
    first = int(model.jnt_dofadr[int(free[0])]) + 3
    angular = set(range(first, first + 3))
    for admission in (archived, fresh):
        velocity = np.asarray(admission['coordinate_admission']['measured_body_admission']['measured']['qvel'])
        if velocity.shape != (model.nv,) or velocity.dtype.kind not in 'fiu' or not np.isfinite(velocity).all():
            raise ValueError('Complete finite original-model planning velocity required')
    differences = []

    def allowed(path):
        return ((len(path) == 5 and path[:4] == _MEASURED + ('qvel',) and path[4] in angular)
                or path == ('material_frame_admission', 'maximum_body_rotation_error_rad')
                or (len(path) == 4 and path[:2] == ('measured_rest', 'endpoint_contacts')
                    and type(path[2]) is int and path[3] == 'inward_radial_normal_alignment'))

    def compare(old, new, path=()):
        if type(old) is not type(new):
            raise ValueError('Source type differs at ' + repr(path))
        if type(old) is dict:
            if set(old) != set(new):
                raise ValueError('Source field inventory differs at ' + repr(path))
            for key in sorted(old):
                compare(old[key], new[key], path + (key,))
        elif type(old) is list:
            if len(old) != len(new):
                raise ValueError('Source list inventory differs at ' + repr(path))
            for index, (a, b) in enumerate(zip(old, new)):
                compare(a, b, path + (index,))
        else:
            if type(old) is float and (not np.isfinite(old) or not np.isfinite(new)):
                raise ValueError('Finite source scalars required')
            if old == new:
                return
            row = dict(path=list(path), archived=old, fresh=new)
            if path == _MEASURED_HASH:
                row['kind'] = 'verified-dependent-payload-sha256'
            elif allowed(path) and type(old) is float and abs(old - new) <= DIAGNOSTIC_ABSOLUTE_TOLERANCE:
                row.update(kind='diagnosed-derived-roundoff', absolute_difference=abs(old - new))
            else:
                raise ValueError('Unreconciled exact source field differs at ' + repr(path))
            differences.append(row)

    compare(archived, fresh)
    fresh_context.verify_inputs()
    bound = IsaacReleasePlanningContext(_encode(archived))
    bound.verify_inputs()
    receipt = dict(schema=SCHEMA, passed=True, fresh_original_admission_passed=True,
        archived_context_sha256=bound.sha256, fresh_context_sha256=fresh_context.sha256,
        exact_initial_qpos=True, transformed_root_angular_velocity_indices=sorted(angular),
        diagnostic_absolute_tolerance=DIAGNOSTIC_ABSOLUTE_TOLERANCE,
        differences=differences, difference_count=len(differences),
        fresh_python=platform.python_version(),
        fresh_packages={name:importlib.metadata.version(name) for name in ('numpy', 'scipy', 'mujoco')},
        source_state_sha256=fresh['source_qualification']['state_sha256'],
        input_sha256=fresh['input_sha256'].copy(),
        physics_steps=0, active_state_writes=0, authorized_stages=0,
        scope='Fresh original physical/contact/kinematic admission, then only diagnosed derived roundoff reconciliation; no raw measurement, model, threshold, label, coordinate or archived receipt is changed')
    return bound, receipt
