"""Privileged opening measurements; never a robot sensor observation.

The force adapter preserves counterpart identity for teacher qualification only.
The geometric calculator does not integrate or modify the active PhysX plant.
Its clearance is an authored-geometry estimate, checked against actual body poses;
the independent PhysX penetration/contact gates remain authoritative.
"""
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def pose_parts(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError('Expected finite measured xyz/wxyz body pose')
    if not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-5):
        raise ValueError('Measured quaternion must be normalized')
    return pose[:3], Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()


def contact_force_pairs(normal_matrix, friction_vectors, counts, starts, *, capacity):
    """Preserve actual surface pairs while summing distinct friction patches.

    Inputs must be copied before obtaining another PhysX contact buffer. Friction
    patches are not paired with individual normal contacts by index.
    """
    matrix = np.asarray(normal_matrix, dtype=float).copy()
    friction = np.asarray(friction_vectors, dtype=float)
    counts, starts = np.asarray(counts), np.asarray(starts)
    if (matrix.ndim != 3 or matrix.shape[-1] != 3 or counts.shape != matrix.shape[:2]
            or starts.shape != counts.shape or counts.dtype.kind not in 'iu'
            or starts.dtype.kind not in 'iu' or friction.shape != (capacity, 3)
            or np.any(counts < 0) or np.any(starts < 0)):
        raise ValueError('Malformed actual normal/friction contact buffers')
    if not np.isfinite(matrix).all() or counts.sum() >= capacity:
        raise ValueError('Nonfinite or potentially truncated contact evidence')
    occupied = set()
    for i, j in np.ndindex(counts.shape):
        count, start = int(counts[i, j]), int(starts[i, j])
        if not count:
            continue
        if start + count > capacity:
            raise ValueError('Friction contact slice exceeds capacity')
        slots = set(range(start, start + count))
        if occupied.intersection(slots):
            raise ValueError('Overlapping friction slices would double-count load')
        occupied.update(slots)
        vectors = friction[start:start + count]
        if not np.isfinite(vectors).all():
            raise ValueError('Nonfinite measured friction load')
        matrix[i, j] += vectors.sum(axis=0)
    return matrix


def panel_surface_loads(sensor_paths, filter_paths, pair_forces, leaf_pose, *,
                        leaf_path='/World/Door/Articulation/leaf', side='lh'):
    """Actual force projected against panel +Y, separately for palm and fingers.

    Forces act ON the robot. Net normal-plus-friction force is reduced per
    rigid body against this exact surface; unrelated contacts cannot qualify.
    This is a projected surface load, not a claim about tactile pad anatomy.
    """
    if side not in ('lh', 'rh'):
        raise ValueError('Expected left or right hand')
    paths = tuple(sensor_paths)
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate contact sensor rows')
    filters = np.asarray(filter_paths, dtype=str)
    forces = np.asarray(pair_forces, dtype=float)
    if filters.ndim != 2 or forces.shape != (*filters.shape, 3) or len(paths) != len(filters):
        raise ValueError('Contact row/filter order or shape mismatch')
    if not np.isfinite(forces).all():
        raise ValueError('Nonfinite actual pair force')
    _, rotation = pose_parts(leaf_pose)
    normal = rotation[:, 1]
    loads, vectors = {}, {}
    for i, path in enumerate(paths):
        name = path.rsplit('/', 1)[-1]
        if not name.startswith(side + '_'):
            continue
        matching = np.flatnonzero(filters[i] == leaf_path)
        if len(matching) != 1:
            raise ValueError('Every hand row must resolve the panel exactly once')
        vector = forces[i, matching[0]]
        loads[name] = max(0., float(-normal @ vector))
        vectors[name] = vector.tolist()
    if side + '_palm' not in loads:
        raise ValueError('Actual palm contact row is missing')
    return dict(total_normal_load_N=sum(loads.values()),
                palm_normal_load_N=loads[side + '_palm'], body_normal_loads_N=loads,
                body_panel_forces_world_N=vectors,
                scope='Privileged actual PhysX per-body projected normal-plus-friction surface load')


class OpeningGeometryMeasurements:
    """Unstepped geometry evaluated at synchronized actual robot/door state."""
    def __init__(self, door_xml, robot_xml, joint_names):
        import mujoco
        self.mujoco = mujoco
        robot_xml, door_xml = Path(robot_xml), Path(door_xml)
        if door_xml.is_dir():
            door_xml = door_xml / 'door.xml'
        scene = mujoco.MjSpec.from_file(str(door_xml))
        scene.memory = 128 * 1024 * 1024
        scene.attach(mujoco.MjSpec.from_file(str(robot_xml)), prefix='robot/',
                     frame=scene.worldbody.add_frame(pos=[0., -1.5, 0.]))
        self.m = scene.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.names = tuple(joint_names)
        self.root = int(m.jnt_qposadr[m.joint('robot/free_base').id])
        self.qa = np.array([m.jnt_qposadr[m.joint('robot/' + n).id] for n in self.names])
        self.door_qa = {role: m.jnt_qposadr[m.joint(name).id] for role, name in
                        [('operator', 'leaf_handle_hinge'), ('leaf', 'leaf_hinge'),
                         ('latch', 'leaf_latch_bolt_slide')]}
        self.geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and
                      m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
        if not self.geoms:
            raise ValueError('No authored right-hand collision shapes')
        self.bodies = sorted(set(int(m.geom_bodyid[g]) for g in self.geoms))
        self.lever = m.geom('leaf_handle_lever_col_n').id
        self.site = m.site('robot/rh_palm_touch').id
        self.site_body = int(m.site_bodyid[self.site])
        self.sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (door_xml, robot_xml)}

    def read(self, *, time_s, pose_time_s, root, joints, angles, body_poses,
             handle_pose, leaf_pose):
        if not np.isfinite([time_s, pose_time_s]).all() or abs(time_s - pose_time_s) > 1e-7:
            raise ValueError('Body poses and joint state must have the same actual clock')
        root = np.asarray(root, dtype=float)
        pose_parts(root[:7])
        q = np.asarray([joints[name] for name in self.names], dtype=float)
        if not np.isfinite(q).all():
            raise ValueError('Nonfinite measured robot joints')
        m, d, mj = self.m, self.d, self.mujoco
        d.qpos[self.root:self.root + 7] = root[:7]
        d.qpos[self.qa] = q
        for role, adr in self.door_qa.items():
            value = float(angles[role])
            if not np.isfinite(value):
                raise ValueError('Nonfinite measured mechanism position')
            d.qpos[adr] = value
        mj.mj_kinematics(m, d)
        errors = []
        actual = {int(b): body_poses[m.body(b).name.removeprefix('robot/')]
                  for b in set(self.bodies + [self.site_body])}
        actual.update({m.body('leaf_handle').id: handle_pose, m.body('leaf').id: leaf_pose})
        for body, pose in actual.items():
            pos, rotation = pose_parts(pose)
            errors.append((float(np.linalg.norm(d.xpos[body] - pos)),
                           float(np.linalg.norm(d.xmat[body].reshape(3, 3) - rotation))))
        position_error, rotation_error = np.max(errors, axis=0)
        if position_error > .003 or rotation_error > .02:
            raise ValueError('Actual body poses disagree with authored clearance geometry')
        palm_pos, palm_rot = pose_parts(actual[self.site_body])
        site_rot = np.empty(9)
        mj.mju_quat2Mat(site_rot, m.site_quat[self.site])
        quat = Rotation.from_matrix(palm_rot @ site_rot.reshape(3, 3)).as_quat()
        palm_pose = np.r_[palm_pos + palm_rot @ m.site_pos[self.site], quat[[3, 0, 1, 2]]]
        gap = min(float(mj.mj_geomDistance(m, d, g, self.lever, 1., None)) for g in self.geoms)
        return dict(time_s=float(time_s), right_palm_pose=palm_pose.tolist(),
                    right_lever_clearance_m=gap, maximum_pose_position_error_m=float(position_error),
                    maximum_pose_rotation_error=float(rotation_error), native_mirror_steps=0,
                    scope='Authored geometry at measured PhysX state; active PhysX contact audit remains separate')
