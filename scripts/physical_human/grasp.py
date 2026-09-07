"""Task-specific grasp checks on achieved poses and native contact forces.

This prototype uses an overhand lever grasp: fingers above the grip and the
thumb phalanges below it. Anatomical joint ranges alone cannot establish this.
All positions are measured in the moving lever frame, never screen coordinates.
These thresholds are engineering acceptance criteria, not clinical norms.
"""

import math

import mujoco
import numpy as np

WORK_PHASES = ("press lever", "pull", "hold open")


class GripSqueeze:
    """Small joint torques pressing the five pads toward the grip centreline.

    Applied through the existing torque-limited digit actuators only. The lever
    remains passive. Pose servos supply posture; this bias supplies grip load
    that a zero-gap, purely positional pose cannot reliably maintain.
    """

    def __init__(self, model):
        self.model = model
        self.grip = model.geom("lever_grip").id
        self.lever = model.body("lever").id
        names = [model.joint(j).name for j in model.actuator_trnid[:, 0]]
        self.pads = []
        for digit in range(1, 6):
            geoms = (
                ["hand_l_proximal_thumb_contact0", "hand_l_distal_thumb_contact0"]
                if digit == 1
                else [
                    f"hand_l_{digit}proxph_contact0",
                    f"hand_l_midph{digit}_contact0",
                    f"hand_l_distph{digit}_contact0",
                ]
            )
            joints = (
                ("cmc_flexion", "cmc_abduction", "mp_flexion", "ip_flexion")
                if digit == 1
                else (
                    f"mcp{digit}_flexion",
                    f"mcp{digit}_abduction",
                    f"pm{digit}_flexion",
                    f"md{digit}_flexion",
                )
            )
            ids = np.array([names.index("hand_l_" + n) for n in joints])
            self.pads.append(
                ([model.geom(g).id for g in geoms], ids, 3.0 if digit == 1 else 2.0)
            )
        self.vids = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        self.jac = np.zeros((3, model.nv))
        self.closest = np.zeros(6)

    def torque(self, data):
        result = np.zeros(self.model.nu)
        rotation = data.xmat[self.lever].reshape(3, 3)
        origin = data.xpos[self.lever]
        for geoms, ids, force in self.pads:
            gap, geom, point = min(
                (
                    (
                        mujoco.mj_geomDistance(
                            self.model, data, self.grip, g, 0.03, self.closest
                        ),
                        g,
                        self.closest[3:].copy(),
                    )
                    for g in geoms
                ),
                key=lambda result: result[0],
            )
            if gap > 0.02:
                continue
            # Recover millimetre-scale pad separation under task load. A fixed
            # squeeze alone can be cancelled by the digit posture servo.
            pad_load = min(12.0, force + 1200 * max(gap, 0.0))
            local = (point - origin) @ rotation
            inward = rotation @ np.array([0, -0.066 - local[1], -local[2]])
            inward *= pad_load / max(np.linalg.norm(inward), 1e-8)
            mujoco.mj_jac(
                self.model, data, self.jac, None, point, self.model.geom_bodyid[geom]
            )
            result[ids] = np.clip(self.jac[:, self.vids[ids]].T @ inward, -0.35, 0.35)
        return result


class HandleWrench:
    """Drive the passive mechanism with bounded arm-motor torques.

    The geometric target stays attached to the *observed* handle. A fictitious
    handle pose several degrees ahead distorted the grasp during actuation.
    Jacobian-transpose torques instead supply the task load through contact.
    """

    def __init__(self, model):
        self.model = model
        self.lever = model.body("lever").id
        self.door = model.body("door").id
        self.wrist = model.body("actor_wrist_l").id
        names = [model.joint(j).name for j in model.actuator_trnid[:, 0]]
        self.ids = np.array(
            [
                i
                for i, n in enumerate(names)
                if n.startswith(
                    (
                        "actor_shoulder_l",
                        "actor_elbow_l",
                        "actor_pronation_l",
                        "actor_wrist_l",
                    )
                )
            ]
        )
        self.vids = model.jnt_dofadr[model.actuator_trnid[self.ids, 0]]
        self.jac = np.zeros((3, model.nv))

    def torque(self, data, lever_target, door_target=None, door_speed=0.0):
        model = self.model
        lever_joint = model.joint("lever_hinge")
        door_joint = model.joint("door_hinge")
        lever_angle = float(data.qpos[lever_joint.qposadr[0]])
        lever_speed = float(data.qvel[lever_joint.dofadr[0]])
        # Spring feedforward plus bounded position/velocity feedback.
        lever_torque = np.clip(
            1.8 * lever_target + 12 * (lever_target - lever_angle) - 0.6 * lever_speed,
            -1.8,
            1.8,
        )
        rotation = data.xmat[self.lever].reshape(3, 3)
        radius = 0.078
        point = data.xpos[self.lever] + rotation @ np.array([-radius, -0.066, 0])
        force = rotation @ np.array([0, 0, -lever_torque / radius])
        if door_target is not None:
            theta = float(data.qpos[door_joint.qposadr[0]])
            speed = float(data.qvel[door_joint.dofadr[0]])
            torque = np.clip(
                80 * (door_target - theta) + 30 * (door_speed - speed), -10, 10
            )
            arm = point - data.xpos[self.door]
            force += (
                torque
                * np.array([-arm[1], arm[0], 0])
                / max(arm[0] ** 2 + arm[1] ** 2, 0.1)
            )
        mujoco.mj_jac(model, data, self.jac, None, point, self.wrist)
        result = np.zeros(model.nu)
        result[self.ids] = self.jac[:, self.vids].T @ force
        return result


def digit_for_geom(name):
    if name.startswith(
        ("hand_l_proximal_thumb_contact", "hand_l_distal_thumb_contact")
    ):
        return "thumb"
    for n, label in zip(range(2, 6), ("index", "middle", "ring", "little")):
        if name.startswith(
            tuple(f"hand_l_{part}{n}_contact" for part in ("midph", "distph"))
        ):
            return label
        if name.startswith(f"hand_l_{n}proxph_contact"):
            return label
    return None


def opposed_contacts(contacts):
    """Return the largest thumb/finger radial separation with real normal load.

    Input is (digit, lever-local XYZ metres, normal force N). The grip axis is
    X and its centreline is Y=-.066, Z=0 in the lever body. A stem/rose/palm
    contact must never qualify as a finger holding the usable grip.
    """
    usable = []
    for digit, p, force in contacts:
        p = np.asarray(p)
        if digit and force >= 0.05 and -0.115 <= p[0] <= -0.015:
            radial = p[1:] - [-0.066, 0]
            norm = np.linalg.norm(radial)
            if norm > 0.001:
                usable.append((digit, radial / norm, p, force))
    thumbs = [
        c for c in usable if c[0] == "thumb" and c[2][2] < -0.003 and c[2][1] < -0.068
    ]
    fingers = [c for c in usable if c[0] != "thumb"]
    per_finger = {}
    for finger in fingers:
        degrees = max(
            (
                math.degrees(math.acos(float(np.clip(thumb[1] @ finger[1], -1, 1))))
                for thumb in thumbs
                if abs(thumb[2][0] - finger[2][0]) < 0.075
            ),
            default=0.0,
        )
        per_finger[finger[0]] = max(per_finger.get(finger[0], 0), degrees)
    return (
        max(per_finger.values(), default=0.0),
        sum(v >= 120 for v in per_finger.values()),
        sum(c[3] for c in thumbs),
    )


class GraspAudit:
    def __init__(self, model):
        self.model = model
        self.lever = model.body("lever").id
        self.grip = model.geom("lever_grip").id
        self.thumb_sites = [model.site(f"hand_keypoint_l_{i:02}").id for i in (3, 4)]
        self.finger_sites = np.array(
            [
                [
                    model.site(f"hand_keypoint_l_{i:02}").id
                    for i in (6 + 4 * n, 7 + 4 * n, 8 + 4 * n)
                ]
                for n in range(4)
            ]
        )
        self.phases = {}

    def measure(self, data):
        rotation = data.xmat[self.lever].reshape(3, 3)
        origin = data.xpos[self.lever]
        thumb = (data.site_xpos[self.thumb_sites] - origin) @ rotation
        fingers = (data.site_xpos[self.finger_sites] - origin) @ rotation
        finger_sides = np.all(fingers[:, :, 1] + 0.066 > 0.003, axis=1)
        thumb_opposite_side = bool(np.all(thumb[:, 1] + 0.066 < -0.003))
        contacts = []
        for i in range(data.ncon):
            c = data.contact[i]
            if self.grip not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.grip else c.geom1
            digit = digit_for_geom(self.model.geom(other).name or "")
            if digit is None:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, force)
            contacts.append((digit, (c.pos - origin) @ rotation, float(force[0])))
        separation, loaded_fingers, thumb_force = opposed_contacts(contacts)
        below = bool(thumb[:, 2].mean() < -0.006 and thumb[1, 2] < -0.006)
        return {
            "thumb_below_grip": below,
            "four_fingers_together": bool(np.all(finger_sides)),
            "finger_side_checks": dict(
                zip(("index", "middle", "ring", "little"), finger_sides.tolist())
            ),
            "thumb_on_opposite_side": thumb_opposite_side,
            "minimum_finger_side_clearance_mm": float(
                np.min(fingers[:, :, 1] + 0.066) * 1000
            ),
            "thumb_side_clearance_mm": float(-np.max(thumb[:, 1] + 0.066) * 1000),
            "thumb_tip_height_mm": float(thumb[1, 2] * 1000),
            "thumb_normal_force_n": thumb_force,
            "opposition_deg": separation,
            "opposed_loaded_fingers": loaded_fingers,
            "opposed_grasp": below
            and thumb_opposite_side
            and bool(np.all(finger_sides))
            and separation >= 120
            and loaded_fingers == 4,
        }

    def observe(self, data, phase):
        state = self.measure(data)
        if phase not in WORK_PHASES:
            return state
        stats = self.phases.setdefault(
            phase,
            {
                "samples": 0,
                "opposed": 0,
                "thumb_below": 0,
                "thumb_loaded": 0,
                "fingers_together": 0,
                "thumb_opposite": 0,
                "current_gap_s": 0.0,
                "longest_gap_s": 0.0,
                "min_opposition_deg": 180.0,
            },
        )
        stats["samples"] += 1
        stats["opposed"] += int(state["opposed_grasp"])
        stats["thumb_below"] += int(state["thumb_below_grip"])
        stats["thumb_loaded"] += int(state["thumb_normal_force_n"] >= 0.05)
        stats["fingers_together"] += int(state["four_fingers_together"])
        stats["thumb_opposite"] += int(state["thumb_on_opposite_side"])
        stats["current_gap_s"] = (
            0.0
            if state["opposed_grasp"]
            else stats["current_gap_s"] + self.model.opt.timestep
        )
        stats["longest_gap_s"] = max(stats["longest_gap_s"], stats["current_gap_s"])
        stats["min_opposition_deg"] = min(
            stats["min_opposition_deg"], state["opposition_deg"]
        )
        return state

    def result(self):
        phases = {}
        violations = []
        for phase in WORK_PHASES:
            s = self.phases.get(phase)
            if not s:
                violations.append(f"Missing {phase} observations")
                continue
            n = s["samples"]
            p = {
                "samples": n,
                "opposed_fraction": s["opposed"] / n,
                "thumb_below_fraction": s["thumb_below"] / n,
                "thumb_loaded_fraction": s["thumb_loaded"] / n,
                "four_fingers_together_fraction": s["fingers_together"] / n,
                "thumb_opposite_side_fraction": s["thumb_opposite"] / n,
                "longest_missing_opposition_s": s["longest_gap_s"],
            }
            phases[phase] = p
            if n * self.model.opt.timestep < 0.2:
                violations.append(f"{phase}: fewer than 0.2 seconds observed")
            if (
                s["fingers_together"] != n
                or s["thumb_opposite"] != n
                or s["thumb_below"] != n
            ):
                violations.append(f"{phase}: thumb/finger side separation was lost")
            if p["opposed_fraction"] < 0.90 or p["longest_missing_opposition_s"] > 0.12:
                violations.append(
                    f"{phase}: sustained opposing thumb/finger contact is missing"
                )
        return {
            "passed": not violations,
            "violations": violations,
            "grasp": "four fingers on the far side; thumb phalanges on the near side and underneath",
            "thumb_contact": "proximal or distal phalanx; metacarpal/palm never qualifies",
            "required_side_separation_fraction": 1.0,
            "minimum_radial_separation_deg": 120,
            "minimum_loaded_fingers": 4,
            "minimum_normal_force_n": 0.05,
            "required_fraction_per_phase": 0.90,
            "maximum_contact_gap_s": 0.12,
            "phases": phases,
        }
