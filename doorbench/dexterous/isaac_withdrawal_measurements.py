"""Unstepped complete hand/environment distances at measured PhysX state."""
import numpy as np

from .isaac_opening_measurements import OpeningGeometryMeasurements, pose_parts
from .standing_withdrawal_audit import clearance_pairs, environment_clearance


class IsaacWithdrawalMeasurements(OpeningGeometryMeasurements):
    def __init__(self, door_xml, robot_xml, joint_names):
        super().__init__(door_xml, robot_xml, joint_names)
        self.environment_pairs = clearance_pairs(self.m)
        hand_bodies = {int(self.m.geom_bodyid[g]) for g, _ in self.environment_pairs}
        environment_bodies = {int(self.m.geom_bodyid[g]) for _, g in self.environment_pairs}
        self.bodies = sorted(set(self.bodies) | hand_bodies)
        self.required_robot_body_names = tuple(sorted(
            self.m.body(b).name.removeprefix('robot/') for b in hand_bodies | {self.site_body}))
        # World-attached floor/frame colliders retain their authored fixed pose.
        # Every moving environmental collider must have an actual body witness.
        self.required_door_body_names = tuple(sorted(
            self.m.body(b).name for b in environment_bodies if b != 0))

    def read(self, *, time_s, pose_time_s, root, joints, angles, body_poses,
             handle_pose, leaf_pose):
        result = super().read(time_s=time_s, pose_time_s=pose_time_s, root=root,
            joints=joints, angles=angles, body_poses=body_poses,
            handle_pose=handle_pose, leaf_pose=leaf_pose)
        poses = {name: np.asarray(body_poses[name], float) for name in self.required_robot_body_names}
        door_poses = dict(body_poses, leaf=leaf_pose, leaf_handle=handle_pose)
        errors = []
        for name in self.required_door_body_names:
            pose = np.asarray(door_poses[name], float)
            p, rotation = pose_parts(pose)
            b = self.m.body(name).id
            errors.append((np.linalg.norm(self.d.xpos[b]-p),
                np.linalg.norm(self.d.xmat[b].reshape(3, 3)-rotation)))
            poses[name] = pose
        if errors:
            position, rotation = np.max(errors, axis=0)
            if position > .003 or rotation > .02:
                raise ValueError('Actual environmental body poses disagree with authored geometry')
            result['maximum_pose_position_error_m'] = max(result['maximum_pose_position_error_m'], float(position))
            result['maximum_pose_rotation_error'] = max(result['maximum_pose_rotation_error'], float(rotation))
        result.update(right_environment_clearance_m=environment_clearance(self.m, self.d, self.environment_pairs),
            geometry_time_s=float(pose_time_s), body_poses_xyz_wxyz={k: v.tolist() for k, v in poses.items()},
            environment_pair_count=len(self.environment_pairs), input_sha256=self.sources.copy(),
            scope='Original complete RH/environment authored geometry at measured PhysX state; no physics steps or direct PhysX distance-sensor claim')
        return result
