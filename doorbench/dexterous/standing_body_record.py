"""Compact measured body poses for subsequent unstepped planner admission."""
import numpy as np

ROBOT_BODIES = ('left_ankle_link', 'right_ankle_link', 'torso_link', 'rh_palm', 'lh_palm')
PLANNER_BODIES = tuple('robot/' + name for name in ROBOT_BODIES) + ('leaf_handle',)


def pack_standing_body_poses(robot_poses, handle_pose):
    robot = np.asarray(robot_poses)
    handle = np.asarray(handle_pose)
    if robot.shape != (5, 7) or handle.shape != (7,):
        raise ValueError('Require five robot body poses and one handle pose in XYZ/WXYZ order')
    poses = np.concatenate((robot, handle[None]), axis=0)
    if not np.isfinite(poses).all() or np.max(abs(np.linalg.norm(poses[:, 3:], axis=1)-1)) > 2e-6:
        raise ValueError('Require finite measured body poses with unit quaternions')
    return poses.copy()


POSE_CONVENTION = 'World body-origin XYZ/WXYZ at acquisition-physics time_s'


def extract_standing_body_poses(configuration, physics, *, time_s):
    """Read the exact measured epoch for independent planner admission."""
    if (configuration.get('standing_planner_body_names') != list(PLANNER_BODIES)
            or configuration.get('standing_planner_body_pose_convention') != POSE_CONVENTION):
        raise ValueError('Explicit ordered measured body inventory and convention required')
    clock = np.asarray(physics['time_s'])
    poses = np.asarray(physics['standing_body_poses'])
    if (clock.ndim != 1 or not len(clock) or not np.isfinite(clock).all()
            or clock[0] < 0 or np.any(np.diff(clock) <= 0)
            or poses.shape != (len(clock), 6, 7)):
        raise ValueError('Complete measured body archive with strictly increasing epochs required')
    if (isinstance(time_s, (bool, np.bool_)) or not np.isscalar(time_s)
            or not np.isfinite(time_s)):
        raise ValueError('Finite exact epoch required')
    matches = np.flatnonzero(abs(clock-time_s) <= 1e-10)
    if len(matches) != 1:
        raise ValueError('One exact measured body epoch required; interpolation forbidden')
    index = int(matches[0])
    measured = pack_standing_body_poses(poses[index, :5], poses[index, 5])
    return dict(geometry_time_s=float(clock[index]), body_names=list(PLANNER_BODIES),
                body_poses_xyz_wxyz=measured.tolist())
