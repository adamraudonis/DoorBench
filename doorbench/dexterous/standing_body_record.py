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
