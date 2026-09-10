"""Map a bound measured state into planner arrays without stepping a plant.

This proves coordinate coverage, not model geometry or contact equivalence.
Use the independent measured-body FK admission before planning a route.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .destination_state_binding import admit_destination_state


def planning_coordinates(model, binding, *, motor_contract, door_source_sha256):
    admission = admit_destination_state(binding, motor_contract=motor_contract,
        door_source_sha256=door_source_sha256,
        **{k: binding[k] for k in ('time_s', 'measured_time_s', 'root_state_world',
            'root_state_convention', 'joint_position', 'joint_velocity',
            'door_joint_order', 'door_position', 'door_velocity')})
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError('Planner requires exactly one free root')
    positions = {'robot/' + k: v for k, v in binding['joint_position'].items()}
    velocities = {'robot/' + k: v for k, v in binding['joint_velocity'].items()}
    if set(positions) & set(binding['door_position']):
        raise ValueError('Robot and door coordinate names overlap')
    positions.update(binding['door_position'])
    velocities.update(binding['door_velocity'])
    scalar = [j for j in range(model.njnt) if j != int(free[0])]
    if any(int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE)) for j in scalar):
        raise ValueError('Only declared scalar coordinates may accompany the root')
    if {model.joint(j).name for j in scalar} != set(positions):
        raise ValueError('Planner coordinate inventory differs from measured state')
    qpos = model.qpos0.copy()
    qvel = np.zeros(model.nv)
    qa = int(model.jnt_qposadr[free[0]])
    va = int(model.jnt_dofadr[free[0]])
    root = np.asarray(binding['root_state_world'])
    qpos[qa:qa+7] = root[:7]
    qvel[va:va+3] = root[7:10]
    # MuJoCo free-root translation rates are world-aligned; angular rates are
    # body-aligned. Isaac's bound root13 records both rates in world axes.
    rotation = Rotation.from_quat(root[[4, 5, 6, 3]])
    qvel[va+3:va+6] = rotation.inv().apply(root[10:13])
    for j in scalar:
        name = model.joint(j).name
        qpos[model.jnt_qposadr[j]] = positions[name]
        qvel[model.jnt_dofadr[j]] = velocities[name]
    return qpos, qvel, dict(state_admission=admission,
        scalar_coordinates=len(scalar), free_roots=1,
        simulation_steps=0, active_plant_writes=0,
        scope='Coordinate mapping only; original model provenance and measured-body FK must be independently checked')
