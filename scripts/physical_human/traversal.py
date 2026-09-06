"""Native-step checks for release, unassisted walking, and portal passage."""

from collections import Counter

import mujoco
import numpy as np


def geom_bounds(model, data, geoms):
    """World AABBs of native collision shapes (conservative for rotated shapes)."""
    result = []
    for g in geoms:
        rotation = data.geom_xmat[g].reshape(3, 3)
        size = model.geom_size[g]
        kind = model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = np.full(3, size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = size[0] + abs(rotation[:, 2]) * size[1]
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            radius = np.sqrt((rotation**2) @ (size**2))
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radius = size[0] * np.sqrt((rotation[:, :2] ** 2).sum(axis=1)) + size[
                1
            ] * abs(rotation[:, 2])
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            radius = abs(rotation) @ size
        else:
            raise ValueError(f"Unsupported actor collision shape: {kind}")
        result.append([data.geom_xpos[g] - radius, data.geom_xpos[g] + radius])
    return np.asarray(result)


def portal_interval(model, data, geom, plane_y=0.0):
    """Exact X extent of a collision primitive intersecting the portal plane.

    Rotated limb/foot AABBs overestimate this slice and falsely reject limbs
    near a jamb even when the physical shape has clear space around it.
    """
    center = data.geom_xpos[geom]
    rotation = data.geom_xmat[geom].reshape(3, 3)
    size = model.geom_size[geom]
    kind = int(model.geom_type[geom])
    if kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        radius = np.full(3, size[0]) if kind == mujoco.mjtGeom.mjGEOM_SPHERE else size
        covariance = (rotation * radius**2) @ rotation.T
        dy = plane_y - center[1]
        fraction = 1 - dy * dy / covariance[1, 1]
        if fraction < 0:
            return None
        conditional_center = center[0] + covariance[0, 1] / covariance[1, 1] * dy
        extent = np.sqrt(
            max(
                0.0,
                (covariance[0, 0] - covariance[0, 1] ** 2 / covariance[1, 1])
                * fraction,
            )
        )
        return conditional_center - extent, conditional_center + extent
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius, half = size[:2]
        ax, ay = rotation[:2, 2]
        dy = plane_y - center[1]
        if abs(ay) < 1e-12:
            if abs(dy) > radius:
                return None
            extent = half * abs(ax) + np.sqrt(max(0.0, radius**2 - dy**2))
            return center[0] - extent, center[0] + extent
        lo, hi = sorted([(dy - radius) / ay, (dy + radius) / ay])
        lo, hi = max(-half, lo), min(half, hi)
        if lo > hi:
            return None
        ts = [lo, hi]
        radial = radius * ax * np.sign(ay) / np.hypot(ax, ay)
        for value in [-radial, radial]:
            t = (dy - value) / ay
            if lo <= t <= hi:
                ts.append(t)
        points = []
        for t in ts:
            extent = np.sqrt(max(0.0, radius**2 - (dy - ay * t) ** 2))
            points.extend([center[0] + ax * t - extent, center[0] + ax * t + extent])
        return min(points), max(points)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        corners = np.array(
            [[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]
        )
        vertices = center + (corners * size) @ rotation.T
        points = [v[0] for v in vertices if abs(v[1] - plane_y) < 1e-12]
        for i, a in enumerate(vertices):
            for b in vertices[i + 1 :]:
                if (a[1] - plane_y) * (b[1] - plane_y) < 0:
                    points.append(
                        a[0] + (plane_y - a[1]) / (b[1] - a[1]) * (b[0] - a[0])
                    )
        return (min(points), max(points)) if points else None
    raise ValueError(f"Unsupported portal collision primitive: {kind}")


class TraversalAudit:
    def __init__(self, model, walking_start):
        self.model = model
        self.start = walking_start
        self.body_names = [model.body(i).name or "" for i in range(model.nbody)]
        self.actor = np.array(
            [name.startswith(("actor_", "hand_")) for name in self.body_names]
        )
        self.hand = np.array(
            [name.startswith(("hand_", "actor_wrist_")) for name in self.body_names]
        )
        self.geoms = [
            g
            for g in range(model.ngeom)
            if self.actor[model.geom_bodyid[g]] and model.geom_contype[g]
        ]
        self.feet = [model.body("actor_ankle_" + s).id for s in "lr"]
        self.soles = [model.site("actor_site_sole_" + s).id for s in "lr"]
        self.root = model.body("actor_pelvis").id
        self.floor = model.geom("floor").id
        self.stance_origins = [None, None]
        self.stance_samples = [0, 0]
        self.max_slip = 0.0
        self.bad_ground_impulse = 0.0
        self.world_impulses = Counter()
        self.first_contact_times = {}
        self.release_hand_impulse = 0.0
        self.max_tilt = 0.0
        self.max_arm_angular = 0.0
        self.max_arm_linear = 0.0
        self.arm = [
            model.body("actor_" + n + "_l").id for n in ["shoulder", "elbow", "wrist"]
        ]
        self.portal_violations = Counter()
        self.portal_samples = 0
        self.crossed = set()
        self.min_portal_margin = float("inf")
        self.samples = 0
        self.final_bounds = None
        self.final_speed = None
        self.initial_side = None
        self.last_touch = None

    def observe(self, data, phase, walking_goal=None):
        m = self.model
        dt = m.opt.timestep
        self.samples += 1
        ground = np.zeros(2)
        for ci in range(data.ncon):
            c = data.contact[ci]
            bodies = m.geom_bodyid[[c.geom1, c.geom2]]
            actor = self.actor[bodies]
            if actor.sum() != 1:
                continue
            g = c.geom1 if actor[0] else c.geom2
            other = c.geom2 if actor[0] else c.geom1
            body = m.geom_bodyid[g]
            force = np.zeros(6)
            mujoco.mj_contactForce(m, data, ci, force)
            normal = max(0.0, float(force[0]))
            magnitude = float(np.linalg.norm(force[:3]))
            if other == self.floor:
                if body in self.feet:
                    ground[self.feet.index(body)] += normal
                else:
                    self.bad_ground_impulse += magnitude * dt
            else:
                if not self.hand[body] or data.time >= self.start:
                    pair = (
                        (m.geom(g).name or str(g))
                        + " / "
                        + (m.geom(other).name or str(other))
                    )
                    self.world_impulses[pair] += magnitude * dt
                    if magnitude > 0.001:
                        self.first_contact_times.setdefault(pair, float(data.time))
                if self.hand[body] and magnitude > 0.05:
                    self.last_touch = float(data.time)
                if self.hand[body] and data.time >= self.start:
                    self.release_hand_impulse += magnitude * dt
        for i in range(2):
            planted = (
                data.time > 0.5
                and ground[i] > 20
                and (walking_goal is None or walking_goal["contacts"][i])
            )
            if planted:
                point = data.site_xpos[self.soles[i], :2]
                if self.stance_origins[i] is None:
                    self.stance_origins[i] = point.copy()
                self.max_slip = max(
                    self.max_slip, float(np.linalg.norm(point - self.stance_origins[i]))
                )
                self.stance_samples[i] += 1
            else:
                self.stance_origins[i] = None
        if phase not in (
            "settle",
            "reach",
            "place around lever",
            "grasp",
            "settle grip",
        ):
            for body in self.arm:
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(
                    m, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
                )
                self.max_arm_angular = max(
                    self.max_arm_angular, float(np.linalg.norm(velocity[:3]))
                )
                self.max_arm_linear = max(
                    self.max_arm_linear, float(np.linalg.norm(velocity[3:]))
                )
        bounds = geom_bounds(m, data, self.geoms)
        if self.initial_side is None:
            self.initial_side = (bounds[:, 1, 1] < 0).tolist()
        at_portal = (bounds[:, 0, 1] <= 0) & (bounds[:, 1, 1] >= 0)
        if data.time >= self.start:
            self.max_tilt = max(
                self.max_tilt, float(np.arccos(np.clip(data.xmat[self.root, 8], -1, 1)))
            )
            for i in np.flatnonzero(at_portal):
                interval = portal_interval(m, data, self.geoms[i])
                if interval is None:
                    continue
                margin = min(
                    interval[0] + 0.005, 0.814 - interval[1], 2.085 - bounds[i, 1, 2]
                )
                self.min_portal_margin = min(self.min_portal_margin, float(margin))
                self.portal_samples += 1
                if margin < 0:
                    self.portal_violations[
                        m.geom(self.geoms[i]).name or str(self.geoms[i])
                    ] += 1
            self.crossed.update(np.flatnonzero(bounds[:, 0, 1] > 0.075).tolist())
        self.final_bounds = bounds
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            m, data, mujoco.mjtObj.mjOBJ_BODY, self.root, velocity, 0
        )
        self.final_speed = float(np.linalg.norm(velocity[3:]))

    def result(self, data, walker):
        final_clearance = float(self.final_bounds[:, 0, 1].min() - 0.075)
        impulses = {k: v for k, v in self.world_impulses.items() if v > 1e-9}
        result = {
            "native_samples": self.samples,
            "walking_started": walker is not None,
            "final_whole_body_clearance_beyond_frame_m": final_clearance,
            "final_root_speed_m_s": self.final_speed,
            "portal_samples": self.portal_samples,
            "portal_violations": dict(self.portal_violations),
            "minimum_portal_plane_margin_m": self.min_portal_margin
            if self.portal_samples
            else None,
            "max_planted_foot_slip_m": self.max_slip,
            "stance_samples": self.stance_samples,
            "nonfoot_ground_impulse_ns": self.bad_ground_impulse,
            "unintended_environment_impulses_ns": impulses,
            "first_unintended_contact_times_s": self.first_contact_times,
            "walking_hand_impulse_ns": self.release_hand_impulse,
            "last_hand_environment_contact_s": self.last_touch,
            "max_walking_root_tilt_deg": float(np.rad2deg(self.max_tilt)),
            "max_active_arm_angular_speed_rad_s": self.max_arm_angular,
            "max_active_arm_linear_speed_m_s": self.max_arm_linear,
            "qp_failures": [] if walker is None else walker.failures,
            "max_qp_primal_residual": None if walker is None else walker.max_residual,
        }
        result["checks"] = {
            "whole_body_through_portal": walker is not None
            and final_clearance > 0.20
            and self.portal_samples > 0
            and not self.portal_violations,
            "settled_after_traversal": self.final_speed < 0.03,
            "no_body_or_walking_hand_environment_contact": not impulses,
            "only_feet_touch_ground": self.bad_ground_impulse < 1e-9,
            "planted_foot_slip_under_10_mm": self.max_slip < 0.01,
            "walking_tilt_under_15_deg": self.max_tilt < np.deg2rad(15),
            "continuous_release_and_walk_arm": self.max_arm_angular < 4
            and self.max_arm_linear < 1.2,
            "all_walking_qps_solved": walker is not None and not walker.failures,
        }
        result["checks"] = {k: bool(v) for k, v in result["checks"].items()}
        result["passed"] = all(result["checks"].values())
        return result
