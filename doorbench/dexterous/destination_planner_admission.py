"""Require measured-state binding and full-body agreement before planning.

This is an unstepped coordinate/kinematic admission. Source geometry identity,
contact qualification, path clearance and physical execution remain separate.
"""
from .destination_planning_coordinates import planning_coordinates
from .destination_return_kinematics import admit_destination_return_kinematics


def admit_destination_planner(model, extracted, *, motor_contract, door_source_sha256):
    binding = extracted['binding']
    measured = extracted.get('measured_bodies')
    if measured is None:
        raise ValueError('Recorded feet, torso, palms and handle are required before planning')
    qpos, qvel, coordinate_receipt = planning_coordinates(model, binding,
        motor_contract=motor_contract, door_source_sha256=door_source_sha256)
    data, pose_receipt = admit_destination_return_kinematics(model,
        qpos=qpos, qvel=qvel, time_s=binding['time_s'],
        geometry_time_s=measured['geometry_time_s'],
        body_names=measured['body_names'], body_poses_xyz_wxyz=measured['body_poses_xyz_wxyz'])
    return data, dict(passed=True, coordinate_admission=coordinate_receipt,
        measured_body_admission=pose_receipt, state_sha256=binding['sha256'],
        scope=__doc__.strip())
