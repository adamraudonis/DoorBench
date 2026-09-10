"""Extract a measured Isaac archive epoch; no task or contact qualification."""
import numpy as np

from .destination_state_binding import freeze_destination_state, ROOT_CONVENTION

RECORDED_ROOT_CONVENTION = 'actor-origin pose and world actor-origin linear/angular velocity'


def extract_attained_state(*, configuration, motor_contract, provenance, physics, time_s):
    """Bind an exact recorded epoch without interpolation or state substitution.

    Array coordinate order comes from the same run's configuration, independently
    of motor-contract order. A valid binding is not permission to continue a
    failed physical episode; contact and geometry qualification remain separate.
    """
    if configuration.get('root_state_convention') != RECORDED_ROOT_CONVENTION:
        raise ValueError('Explicit actor-origin root convention required')
    names = configuration['robot_joint_names']
    door_names = configuration['door_joint_names']
    if (len(names) != 69 or len(set(names)) != 69
            or set(names) != set(motor_contract['joint_names'])):
        raise ValueError('Complete original robot coordinate inventory required')
    if not door_names or len(set(door_names)) != len(door_names):
        raise ValueError('Unique complete door coordinate inventory required')
    clock = np.asarray(physics['time_s'])
    if (clock.ndim != 1 or not len(clock) or not np.isfinite(clock).all()
            or clock[0] < 0 or np.any(np.diff(clock) <= 0)):
        raise ValueError('Finite strictly increasing archive epochs required')
    if isinstance(time_s, (bool, np.bool_)) or not np.isscalar(time_s) or not np.isfinite(time_s):
        raise ValueError('Finite requested epoch required')
    matches = np.flatnonzero(np.abs(clock - time_s) <= 1e-8)
    if len(matches) != 1:
        raise ValueError('One exact recorded epoch required; interpolation forbidden')
    index = int(matches[0])
    arrays = {}
    for key, width in [('root', 13), ('joints', 69), ('joint_velocity', 69),
                       ('door', len(door_names)), ('door_velocity', len(door_names))]:
        if key not in physics:
            raise ValueError('Missing measured state field: ' + key)
        array = np.asarray(physics[key])
        if array.shape != (len(clock), width) or not np.isfinite(array).all():
            raise ValueError('Incomplete or nonfinite measured state field: ' + key)
        arrays[key] = array[index]
    door_source = configuration['args']['door_usd']
    return freeze_destination_state(
        motor_contract=motor_contract, door_source_sha256=provenance['files'][door_source],
        time_s=float(clock[index]), measured_time_s=float(clock[index]),
        root_state_world=arrays['root'], root_state_convention=ROOT_CONVENTION,
        joint_position=dict(zip(names, arrays['joints'])),
        joint_velocity=dict(zip(names, arrays['joint_velocity'])),
        door_joint_order=door_names,
        door_position=dict(zip(door_names, arrays['door'])),
        door_velocity=dict(zip(door_names, arrays['door_velocity'])))
