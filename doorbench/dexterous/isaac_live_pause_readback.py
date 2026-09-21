"""Read-only same-process plant/controller inventory for a planning suspension.

No application update, articulation update, controller call, FK, target write or
state restoration is performed. This is an equality guard, not phase admission.
"""
import hashlib
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np


RAW_ARTICULATION_GETTERS = (
    'get_root_transforms', 'get_root_velocities', 'get_link_transforms',
    'get_link_velocities', 'get_link_accelerations', 'get_dof_positions',
    'get_dof_velocities', 'get_dof_position_targets', 'get_dof_velocity_targets',
    'get_dof_actuation_forces', 'get_dof_projected_joint_forces',
)

ARTICULATION_PROPERTY_GETTERS = (
    'get_masses', 'get_inertias', 'get_coms', 'get_dof_limits',
    'get_dof_stiffnesses', 'get_dof_dampings', 'get_dof_armatures',
    'get_dof_friction_properties', 'get_dof_max_forces', 'get_dof_max_velocities',
    'get_material_properties', 'get_contact_offsets', 'get_rest_offsets',
)


def _copy(value):
    if type(value).__module__.split('.')[0] == 'torch':
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _bytes(value):
    return dict(byte_count=len(value), sha256=hashlib.sha256(value).hexdigest())


def controller_inventory(roots):
    """Digest actual object fields, preserving aliases and native MuJoCo state.

    Unknown object types reject instead of being silently omitted. Native state
    is serialized in this process only; these bytes are never deserialized or
    installed in the live plant. IDs also detect replacing an equal-valued
    controller/filter/array. No property discovery or arbitrary repr is used.
    """
    if type(roots) is not dict or not roots or any(type(k) is not str for k in roots):
        raise ValueError('Named retained controller roots required')
    seen = {}

    def encode(value, path):
        cls = type(value); name = cls.__module__ + '.' + cls.__qualname__
        if value is None or cls in (bool, int, str):
            return dict(type=name, value=value)
        if cls is float:
            return dict(type=name, bits=struct.pack('<d', value).hex())
        if isinstance(value, np.generic):
            a = np.asarray(value)
            return dict(type=name, dtype=a.dtype.str, **_bytes(a.tobytes()))
        if isinstance(value, Path):
            return dict(type=name, value=str(value))
        if cls is bytes:
            return dict(type=name, **_bytes(value))
        identity = id(value)
        if identity in seen:
            return dict(reference=seen[identity], python_id=identity)
        seen[identity] = path
        base = dict(type=name, python_id=identity)
        if isinstance(value, np.ndarray) or cls.__module__.split('.')[0] == 'torch':
            a = _copy(value)
            if a.dtype.hasobject:
                raise ValueError('Object arrays cannot be pause inventory: ' + path)
            return dict(base, dtype=a.dtype.str, shape=list(a.shape), **_bytes(a.tobytes(order='C')))
        if cls in (list, tuple):
            return dict(base, items=[encode(v, path + '/' + str(i)) for i, v in enumerate(value)])
        if cls is dict:
            if any(type(k) not in (str, int, float, bool) for k in value):
                raise ValueError('Unsupported controller mapping key: ' + path)
            keys = sorted(value, key=lambda k: (type(k).__name__, str(k)))
            return dict(base, items=[dict(key=encode(k, path + '/key'),
                value=encode(value[k], path + '/' + str(k))) for k in keys])
        if name in ('mujoco._structs.MjModel', 'mujoco._structs.MjData'):
            state = value.__getstate__()
            if type(state) is not bytes:
                raise ValueError('Expected native MuJoCo serialization bytes: ' + path)
            return dict(base, native_state=_bytes(state))
        if name == 'scipy.spatial.transform._rotation.Rotation':
            q = value.as_quat()
            return dict(base, quaternion_shape=list(q.shape), quaternion_bytes=_bytes(q.tobytes()))
        if (cls.__module__.startswith('doorbench.') or cls is SimpleNamespace or
                name == 'scipy.spatial.transform._rotation.Slerp'):
            return dict(base, fields=encode(vars(value), path + '/fields'))
        raise ValueError('Unsupported retained controller type at ' + path + ': ' + name)

    return {name:encode(value, name) for name, value in sorted(roots.items())}


def articulation_inventory(articulation):
    """Read raw backend tensors and already-existing cache/command buffers.

    Access cached fields via vars, not ArticulationData lazy state properties.
    All installed numeric data fields and TimestampedBuffers are included.
    """
    view = articulation.root_physx_view
    backend = {name:_copy(getattr(view, name)()) for name in
        (*RAW_ARTICULATION_GETTERS, *ARTICULATION_PROPERTY_GETTERS)}
    data = vars(articulation)['_data']
    cache = {}
    for name, value in sorted(vars(data).items()):
        if name in ('_root_physx_view', '_physics_sim_view'):
            continue
        cls = type(value)
        if cls.__name__ == 'TimestampedBuffer' and cls.__module__.startswith('isaaclab.utils.buffers'):
            state = vars(value)
            cache[name] = dict(timestamp=state['timestamp'],
                data=None if state['data'] is None else _copy(state['data']))
        elif isinstance(value, (np.ndarray, np.generic)) or cls.__module__.split('.')[0] == 'torch':
            cache[name] = _copy(value)
        elif value is None or cls in (bool, int, float, str):
            cache[name] = value
        elif cls in (list, tuple) and all(type(v) is str for v in value):
            cache[name] = list(value)
        else:
            raise ValueError('Unsupported articulation cache field: ' + name)
    targets = {name:_copy(value) for name, value in sorted(vars(articulation).items())
        if name in ('_joint_pos_target_sim', '_joint_vel_target_sim', '_joint_effort_target_sim')}
    if set(targets) != {'_joint_pos_target_sim', '_joint_vel_target_sim', '_joint_effort_target_sim'}:
        raise ValueError('All original articulation submission targets required')
    wrenches = {}
    for name in ('_instantaneous_wrench_composer', '_permanent_wrench_composer'):
        composer = vars(articulation)[name]
        fields = vars(composer)
        force = _copy(fields['_composed_force_b_torch'])
        torque = _copy(fields['_composed_torque_b_torch'])
        if fields['_active'] or fields['_link_poses_updated'] or np.any(force) or np.any(torque):
            raise ValueError('This motor-only pause requires inactive zero-wrench composers')
        wrenches[name] = dict(python_id=id(composer), active=bool(fields['_active']),
            link_poses_updated=bool(fields['_link_poses_updated']), force=force, torque=torque)
    return dict(backend=backend, cache=cache, submission_targets=targets, external_wrenches=wrenches)


def contact_inventory(view, dt):
    """Copy occupied normal/friction slots; preserve actual ordering and indices."""
    if dt != .002:
        raise ValueError('Original 500 Hz contact interval required')

    def occupied(buffers, names):
        arrays = [_copy(v) for v in buffers]
        counts, starts = arrays[-2:]
        if counts.shape != starts.shape or np.any(counts < 0):
            raise ValueError('Malformed contact slot accounting')
        slots = []
        for i in np.ndindex(counts.shape):
            first, count = int(starts[i]), int(counts[i])
            if first < 0 or first + count > len(arrays[0]):
                raise ValueError('Contact slot outside its actual buffer')
            slots.extend(range(first, first + count))
        selected = np.asarray(slots, dtype=np.int64)
        return dict(counts=counts, starts=starts, occupied_indices=selected,
            **{name:array[selected].copy() for name, array in zip(names, arrays[:-2])})

    return dict(matrix=_copy(view.get_contact_force_matrix(dt=dt)),
        normal=occupied(view.get_contact_data(dt), ('force', 'point', 'normal', 'distance')),
        friction=occupied(view.get_friction_data(dt), ('force', 'point')))


class PhysicsStepCounter:
    """Prospective callback independent of contact occurrence and cached clocks."""
    def __init__(self):
        self.count = 0
        self.elapsed = 0.
        self.invalid_interval = False

    def __call__(self, dt):
        self.count += 1
        self.elapsed += float(dt)
        self.invalid_interval |= not np.isfinite(dt) or abs(float(dt) - .002) > 1e-10

    def receipt(self):
        return dict(count=self.count, elapsed=self.elapsed, invalid_interval=self.invalid_interval)


def capture_native_pause_anchor(*, episode_id, step_index, epoch_s, sim, time_origin,
                                counter, robot, door, contacts, controllers,
                                invariant_getters, evidence_counts):
    """Compose explicit installed-backend inventory with the detached guard."""
    from .isaac_live_planning_pause import capture_pause_anchor
    count = counter.receipt()
    if count['count'] != step_index or count['invalid_interval']:
        raise ValueError('Actual physics callbacks must match completed recorded intervals')
    # Preserve the producer's existing clock-accounting tolerance: PhysX sends
    # float32 dt in callbacks, so cumulative time is not exactly N * float64 dt.
    # Callback count is exact, and the raw clock must be bit-identical across
    # the pause; this is not a tolerance on robot/controller state equality.
    if abs(float(sim.current_time) - time_origin - epoch_s) > .0001:
        raise ValueError('Actual backend clock differs from the completed evidence epoch')
    measured = dict(robot=articulation_inventory(robot), door=articulation_inventory(door),
        contacts=contact_inventory(contacts, .002),
        plant_invariants={name:_copy(getter()) for name, getter in sorted(invariant_getters.items())})
    return capture_pause_anchor(episode_id=episode_id, step_index=step_index, epoch_s=epoch_s,
        physics_clock=dict(sim_current_time=float(sim.current_time), time_origin=float(time_origin),
            sim_step_index=int(sim.current_time_step_index), callback=count, playing=bool(sim.is_playing())),
        measured=measured, controller=controller_inventory(controllers), retained_objects=controllers,
        evidence_counts=evidence_counts, pending_unaccepted_command=False)
