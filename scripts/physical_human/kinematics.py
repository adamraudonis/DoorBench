"""Check achieved native poses, including soft-limit violations under contact.

The bounds are prototype acceptance criteria, not population-wide clinical
limits. Joint ranges come from the scene; hand ranges retain MyoHand values.
"""

import mujoco
import numpy as np


def angle(a, b):
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.arccos(
        np.clip(np.sum(a * b, axis=-1) / np.maximum(denominator, 1e-12), -1, 1)
    )


class KinematicAudit:
    def __init__(self, model):
        self.model = model
        self.joints = np.array(
            [
                i
                for i in range(model.njnt)
                if model.jnt_limited[i]
                and model.joint(i).name.startswith(("actor_", "hand_"))
            ]
        )
        self.names = [model.joint(i).name for i in self.joints]
        self.qids = model.jnt_qposadr[self.joints]
        self.vids = model.jnt_dofadr[self.joints]
        self.ranges = model.jnt_range[self.joints]
        self.minimum = np.full(len(self.joints), np.inf)
        self.maximum = -self.minimum.copy()
        self.speed = np.zeros(len(self.joints))
        self.keypoints = np.array(
            [
                [model.site(f"hand_keypoint_{side}_{i:02}").id for i in range(21)]
                for side in ["l", "r"]
            ]
        )
        self.elbows = [model.body("actor_elbow_" + side).id for side in ["l", "r"]]
        self.wrist_bend = np.zeros(2)
        self.thumb_mcp_bend = np.zeros(2)
        self.thumb_ip_bend = np.zeros(2)
        self.frames = 0
        self.arm_names = ["actor_shoulder_l", "actor_elbow_l", "actor_wrist_l"]
        self.arm_bodies = [model.body(n).id for n in self.arm_names]
        self.max_work_arm_angular_speed = np.zeros(3)
        self.max_work_arm_linear_speed = np.zeros(3)
        self.current_arm_angular_speed = 0.0

    def observe(self, data, phase=None):
        q = data.qpos[self.qids]
        self.minimum = np.minimum(self.minimum, q)
        self.maximum = np.maximum(self.maximum, q)
        self.speed = np.maximum(self.speed, abs(data.qvel[self.vids]))
        pts = data.site_xpos[self.keypoints]
        self.wrist_bend = np.maximum(
            self.wrist_bend,
            angle(pts[:, 0] - data.xpos[self.elbows], pts[:, 9] - pts[:, 0]),
        )
        self.thumb_mcp_bend = np.maximum(
            self.thumb_mcp_bend, angle(pts[:, 2] - pts[:, 1], pts[:, 3] - pts[:, 2])
        )
        self.thumb_ip_bend = np.maximum(
            self.thumb_ip_bend, angle(pts[:, 3] - pts[:, 2], pts[:, 4] - pts[:, 3])
        )
        self.frames += 1
        self.current_arm_angular_speed = 0.0
        if phase in ("press lever", "pull", "hold open"):
            for i, body in enumerate(self.arm_bodies):
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(
                    self.model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
                )
                angular, linear = (
                    np.linalg.norm(velocity[:3]),
                    np.linalg.norm(velocity[3:]),
                )
                self.max_work_arm_angular_speed[i] = max(
                    self.max_work_arm_angular_speed[i], angular
                )
                self.max_work_arm_linear_speed[i] = max(
                    self.max_work_arm_linear_speed[i], linear
                )
                self.current_arm_angular_speed = max(
                    self.current_arm_angular_speed, angular
                )

    def result(self):
        excess = np.maximum(
            0,
            np.maximum(
                self.ranges[:, 0] - self.minimum, self.maximum - self.ranges[:, 1]
            ),
        )
        fingers = np.array([name.startswith("hand_") for name in self.names])
        violations = []
        for i in np.flatnonzero(excess > np.deg2rad(0.5)):
            violations.append(
                f"{self.names[i]} exceeded its joint range by {np.rad2deg(excess[i]):.3f} degrees"
            )
        for i, side in enumerate(["left", "right"]):
            if self.wrist_bend[i] > np.deg2rad(55):
                violations.append(f"{side} wrist bend exceeded 55 degrees")
            if self.thumb_mcp_bend[i] > np.deg2rad(85):
                violations.append(f"{side} thumb MCP bend exceeded 85 degrees")
            if self.thumb_ip_bend[i] > np.deg2rad(85):
                violations.append(f"{side} thumb IP bend exceeded 85 degrees")
        if np.max(self.speed[fingers]) > 15:
            violations.append("Digit angular speed exceeded 15 rad/s")
        if np.max(self.max_work_arm_angular_speed) > 4:
            violations.append(
                "Working arm angular speed exceeded 4 rad/s; inspect for an IK branch jump"
            )
        if np.max(self.max_work_arm_linear_speed) > 1:
            violations.append("Working arm linear speed exceeded 1 m/s")
        return {
            "passed": not violations,
            "violations": violations,
            "samples": self.frames,
            "max_work_arm_angular_speed_rad_s": dict(
                zip(self.arm_names, self.max_work_arm_angular_speed.tolist())
            ),
            "max_work_arm_linear_speed_m_s": dict(
                zip(self.arm_names, self.max_work_arm_linear_speed.tolist())
            ),
            "joint_limit_tolerance_deg": 0.5,
            "max_joint_limit_excess_deg": float(np.rad2deg(excess.max())),
            "max_digit_speed_rad_s": float(np.max(self.speed[fingers])),
            "max_wrist_bend_deg": dict(
                zip(["left", "right"], np.rad2deg(self.wrist_bend).tolist())
            ),
            "max_thumb_mcp_bend_deg": dict(
                zip(["left", "right"], np.rad2deg(self.thumb_mcp_bend).tolist())
            ),
            "max_thumb_ip_bend_deg": dict(
                zip(["left", "right"], np.rad2deg(self.thumb_ip_bend).tolist())
            ),
            "joint_angles_deg": {
                n: {
                    "limits": np.rad2deg(self.ranges[i]).tolist(),
                    "observed": np.rad2deg([self.minimum[i], self.maximum[i]]).tolist(),
                    "max_speed_rad_s": float(self.speed[i]),
                }
                for i, n in enumerate(self.names)
            },
        }
