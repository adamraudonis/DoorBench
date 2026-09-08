"""Isaac 5.1/PhysX sensor emulation without semantic contact selection.

This adapter is CPU/NumPy first for correctness. Live GPU sign, static-collider
coverage and synchronization tests are still required; see DEXTEROUS_SENSORS.md.
Do not give this object, the stage, or tensor views to the student controller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sensor_contract import AngularTaxelGrid, finite_array, rotation_xyzw


def _host_copy(value):
    # PhysX normal/friction getters may reuse count/start buffers. Copy EVERY
    # buffer before calling the next getter (IsaacSim upstream issue #122).
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


@dataclass(frozen=True)
class TactileMount:
    body_path: str
    position_body: tuple[float, float, float]
    quaternion_xyzw_body: tuple[float, float, float, float]
    grid: AngularTaxelGrid


def all_scene_contact_paths(stage):
    """Exhaustive scene selection by physical API, never by object name or role.

    Dynamic/articulated collision shapes use their nearest rigid-body ancestor;
    static shapes use their own prim. Includes robot self contacts and floor.
    Run before the physics view is created; rebuilding topology requires rebuild.
    """
    from pxr import Usd, UsdPhysics
    paths = set()
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        candidate = prim
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                candidate = ancestor
                break
            ancestor = ancestor.GetParent()
        paths.add(str(candidate.GetPath()))
    if not paths:
        raise ValueError("Scene has no collision participants")
    return tuple(sorted(paths))


def _collapse_pairs(forces, points, counts, starts, *, capacity):
    """Erase the counterpart dimension before any local feature computation."""
    forces, points = np.asarray(forces, dtype=float), np.asarray(points, dtype=float)
    counts, starts = np.asarray(counts), np.asarray(starts)
    if (counts.ndim != 2 or starts.shape != counts.shape or counts.dtype.kind not in "iu" or
            starts.dtype.kind not in "iu" or np.any(counts < 0) or np.any(starts < 0) or
            forces.shape != (capacity, 3) or points.shape != (capacity, 3)):
        raise ValueError("Malformed PhysX contact buffers")
    used = int(counts.sum())
    if used >= capacity:
        raise RuntimeError("Contact buffer reached capacity; refuse possibly truncated touch")
    result = []
    for row_counts, row_starts in zip(counts, starts):
        indices = []
        for count, start in zip(row_counts, row_starts):
            if count and start + count > capacity:
                raise ValueError("Contact slice exceeds buffer capacity")
            indices.extend(range(int(start), int(start + count)))
        if len(indices) != len(set(indices)):
            raise ValueError("Overlapping contact buffers would double-count forces")
        result.append((finite_array(points[indices]), finite_array(forces[indices])))
    return result


class PhysXTaxelAdapter:
    def __init__(self, physics_view, stage, mounts, *, capacity=16384):
        self.mounts = tuple(mounts)
        if not self.mounts or type(capacity) is not int or capacity < 2:
            raise ValueError("Declare tactile mounts and a positive contact capacity")
        self.capacity = capacity
        # Exact one-to-many filters are required by the native API for detailed
        # points. Every physical counterpart is included identically; no leaf,
        # handle, robot, floor or object-class filter is accepted as an input.
        all_paths = all_scene_contact_paths(stage)
        self.body_paths = tuple(dict.fromkeys(mount.body_path for mount in self.mounts))
        if not set(self.body_paths).issubset(all_paths):
            raise ValueError("Each tactile mount must belong to a collidable robot body")
        self.bodies = physics_view.create_rigid_body_view(list(self.body_paths))
        returned_paths = tuple(self.bodies.prim_paths)
        if set(returned_paths) != set(self.body_paths) or len(returned_paths) != len(self.body_paths):
            raise ValueError("Rigid body view did not resolve the declared robot bodies")
        # Use actual backend row order for BOTH contact and transform views.
        self.body_paths = returned_paths
        self.contacts = physics_view.create_rigid_contact_view(list(self.body_paths),
            filter_patterns=[list(all_paths) for _ in self.body_paths], max_contact_data_count=capacity)
        if self.contacts.sensor_count != len(self.body_paths) or self.contacts.filter_count != len(all_paths):
            raise ValueError("Contact view does not cover every scene counterpart")
        if tuple(self.contacts.sensor_paths) != self.body_paths:
            raise ValueError("Contact and transform sensor row order disagree")
        if set(np.asarray(self.contacts.filter_paths, dtype=str).reshape(-1)) != set(all_paths):
            raise ValueError("Backend silently dropped or changed scene collision participants")
        self.body_index = {path: i for i, path in enumerate(self.body_paths)}
        self.dimension = sum(mount.grid.dimension for mount in self.mounts)

    def read(self, *, physics_dt):
        if not np.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("Use the actual positive physics timestep, not camera/control period")
        force, point, normal, separation, counts, starts = [_host_copy(x) for x in self.contacts.get_contact_data(physics_dt)]
        normal_vectors = np.asarray(force).reshape(self.capacity, 1) * np.asarray(normal).reshape(self.capacity, 3)
        normal_samples = _collapse_pairs(normal_vectors, np.asarray(point).reshape(self.capacity, 3), counts, starts, capacity=self.capacity)
        friction, friction_point, friction_counts, friction_starts = [_host_copy(x) for x in self.contacts.get_friction_data(physics_dt)]
        friction_samples = _collapse_pairs(np.asarray(friction).reshape(self.capacity, 3),
            np.asarray(friction_point).reshape(self.capacity, 3), friction_counts, friction_starts, capacity=self.capacity)
        # Friction patches need not coincide with the individual normal contact
        # points. Bin each at its reported position instead of pairing indices.
        poses = finite_array(_host_copy(self.bodies.get_transforms()), (len(self.body_paths), 7))
        arrays = []
        for mount in self.mounts:
            row = self.body_index[mount.body_path]
            rotation = rotation_xyzw(poses[row, 3:])
            origin = poses[row, :3] + rotation @ finite_array(mount.position_body, (3,))
            sensor_rotation = rotation @ rotation_xyzw(mount.quaternion_xyzw_body)
            points = np.concatenate((normal_samples[row][0], friction_samples[row][0]))
            forces = np.concatenate((normal_samples[row][1], friction_samples[row][1]))
            arrays.append(mount.grid.bin_forces(points, forces, origin, sensor_rotation))
        return np.concatenate(arrays)


def enqueue_robot_sensors(builder, articulation_data, imu_data, tactile, *, joint_indices, environment_index=0, capture_s):
    """Read only declared joint sensors and local IMU channels from Lab 2.3.2.

    Intentionally does not read root pose/velocity, projected_gravity_b, door
    state, goal or contact labels. No raw orientation oracle is called an IMU.
    """
    i = environment_index
    values = {"joint_position": articulation_data.joint_pos[i][joint_indices],
              "joint_velocity": articulation_data.joint_vel[i][joint_indices],
              "imu_gyro": imu_data.ang_vel_b[i], "imu_accelerometer": imu_data.lin_acc_b[i],
              "tactile": tactile}
    for key, value in values.items():
        builder.push(key, _host_copy(value), capture_s=capture_s)


def mounts_from_layout(layout, body_paths_by_name):
    """Bind a fixed robot layout, independent of door/task geometry or identity."""
    from .sensor_contract import INTERFACE_VERSION
    if layout.get("interface_version") != INTERFACE_VERSION:
        raise ValueError("Sensor layout version mismatch")
    mounts = []
    for row in layout["sensors"]:
        if row["gamma"] != 0:
            raise ValueError("Unsupported tactile foveation")
        grid = AngularTaxelGrid(row["width"], row["height"], tuple(row["fov_degrees"]))
        if grid.dimension != row["dimension"]:
            raise ValueError("Sensor grid dimension mismatch")
        mounts.append(TactileMount(body_paths_by_name[row["body_name"]],
            tuple(row["position_body_m"]), tuple(row["quaternion_xyzw_body"]), grid))
    if sum(mount.grid.dimension for mount in mounts) != layout["tactile_dimension"]:
        raise ValueError("Tactile layout dimension mismatch")
    return tuple(mounts)


def enable_tactile_reporting(stage, mounts):
    """Call before sim.reset(); does not change collision geometry/materials."""
    from pxr import PhysxSchema, UsdPhysics
    for path in {mount.body_path for mount in mounts}:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Tactile mount is missing its imported rigid body")
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)


def enqueue_camera(builder, key, camera_data, *, capture_s, available_s, environment_index=0):
    """Only RGB pixels are copied; camera pose/depth/segmentation stay private.

    Caller must timestamp the physics state rendered and when it becomes usable.
    Reusing an old image requires its old capture timestamp, not a fresh stamp.
    """
    if key not in ("rgb_left", "rgb_right"):
        raise ValueError("Unknown policy camera")
    rgb = _host_copy(camera_data.output["rgb"][environment_index])[..., :3]
    builder.push(key, rgb, capture_s=capture_s, available_s=available_s)
