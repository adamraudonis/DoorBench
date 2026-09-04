"""Per-step label tracking for door interaction episodes.

Labels (all bool unless noted) computed from MuJoCo state each step:
  touched_door              robot body contacted any door geom (leaf / operator / hardware)
  touched_operator          robot contacted the operator (handle / bar / pull ...)
  operator_actuated         operator joint travelled >= 70 % of its useful travel
  latch_released            latch bolt retracted >= 80 % of throw (or no latch)
  lock_released             engaged lock released by the robot (thumbturn / keypad / REX / slide bolt)
  door_opened               primary joint >= open threshold (10 deg / 0.1 m)
  door_open_clear           opening wide enough for the robot (>= clearance angle / travel)
  robot_passed_through      robot base crossed the door plane inside the opening
  door_closed_after         door back within closed threshold after passing (close tasks)
  door_slammed              closing velocity at contact with stop > threshold
  door_damaged              any damage event (see below)
  damage_events             list of {step, kind, part, value, threshold}
  robot_fell                robot base height < fall threshold (if a robot base body is given)
  hardware_misuse           excessive torque/force on operator (beyond yield) or wrong-direction actuation
  max_leaf_contact_force    N (peak normal force on the leaf from the robot)
  max_operator_torque       N*m or N (peak generalized force on the operator joint)
  time_to_touch / time_to_open / time_to_pass   seconds (None if never)
  energy_J                  sum |tau * dq| on door joints (work done on the door mechanism)
  success                   task-specific predicate (see env.TASK_SUCCESS)

Damage model thresholds come from spec.json -> physics.damage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class EpisodeLabels:
    touched_door: bool = False
    touched_operator: bool = False
    operator_actuated: bool = False
    latch_released: bool = False
    lock_released: bool = False
    door_opened: bool = False
    door_open_clear: bool = False
    robot_passed_through: bool = False
    door_closed_after: bool = False
    door_slammed: bool = False
    door_damaged: bool = False
    robot_fell: bool = False
    hardware_misuse: bool = False
    damage_events: list = field(default_factory=list)
    max_leaf_contact_force: float = 0.0
    max_operator_torque: float = 0.0
    max_door_angle: float = 0.0
    time_to_touch: float | None = None
    time_to_open: float | None = None
    time_to_pass: float | None = None
    energy_J: float = 0.0
    steps: int = 0
    sim_time: float = 0.0
    success: bool = False
    task: str = ""
    notes: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# lock-role joints that a robot withdraws to release a lock (0 = engaged, + = withdrawn); ALL of them must be
# withdrawn for `lock_released`.  Coupled parts (thumbturn + deadbolt, wheel + bolts, handle + hook) move together.
LOCK_RELEASE_PATTERNS = ("deadbolt_slide", "deadbolt_thumbturn_hinge", "aux_bolt_slide", "slide_bolt_slide", "slide_latch_slide",
                         "hook_hinge", "hook_thumbturn_hinge", "join_bolt_slide", "dog_", "bolt_", "garage_slide_lock_slide",
                         "hatch_bolt_slide", "pin_slide")
# momentary buttons: ANY press releases (REX button on a maglock, elevator call button)
RELEASE_BUTTON_PATTERNS = ("rex_button_slide", "call_button_slide")
# joints that are driven by the environment / by other lock parts and never count as a robot-side release
LOCK_IGNORE_PATTERNS = ("keypad_key_", "lock_bar_", "electric_bolt_slide")
LEGACY_LOCK_JOINTS = ("leaf_deadbolt_thumbturn_hinge", "rex_button_slide", "leaf_aux_bolt_slide", "slide_latch_slide", "leaf_slide_bolt_slide", "leaf_pin_slide", "leaf_deadbolt_slide")


class LabelTracker:
    """Stateful per-step labeller.  Construct once per episode.

    `operator_joints` / `lock_joints` are the joint names with role "operator" / "lock" in model.json (DoorEnv passes
    them); without them the tracker falls back to `meta["operator_joint"]` and a fixed list of lock joint names.
    """

    def __init__(self, model, spec: dict, meta: dict, robot_body_names: list[str] | None = None, robot_base_body: str | None = None, clearance_angle_deg: float = 60.0, clearance_travel_m: float = 0.55, robot_width_m: float = 0.5, operator_joints: list[str] | None = None, lock_joints: list[str] | None = None, latch_joints: list[str] | None = None):
        import mujoco
        self.mj = mujoco
        self.m = model
        self.spec = spec
        self.meta = meta
        self.phys = spec.get("physics", {})
        self.damage = self.phys.get("damage", {})
        self.L = EpisodeLabels(task=spec.get("task", ""))
        self.robot_bodies = set(robot_body_names or [])
        self.robot_base = robot_base_body
        self.clear_angle = math.radians(clearance_angle_deg)
        self.clear_travel = clearance_travel_m
        if spec.get("kinematics", {}).get("type") == "slide_vertical":
            self.clear_travel = max(self.clear_travel, 1.9)      # overhead / garage / roll-up: the opening must clear the robot's height
        self.robot_width = robot_width_m
        # joint ids
        self.pj = self._jid(meta.get("primary_joint"))
        self.oj = self._jid(meta.get("operator_joint")) if meta.get("operator_joint") else -1
        self.op_joints = [j for j in (self._jid(n) for n in (operator_joints or [])) if j >= 0] or ([self.oj] if self.oj >= 0 else [])
        self.bj = self._jid("leaf_latch_bolt_slide")
        # latch parts (0 = extended, + = retracted); only those carried by the primary leaf, so the second leaf of a
        # pair or the upper leaf of a dutch door does not hold `latch_released` back
        lj = [j for j in (self._jid(n) for n in (latch_joints or [])) if j >= 0 and model.jnt_range[j][1] - model.jnt_range[j][0] > 1e-6]
        if lj and self.pj >= 0:
            under = [j for j in lj if self._under(model.jnt_bodyid[j], model.jnt_bodyid[self.pj])]
            lj = under or lj
        self.latch_joints = lj or ([self.bj] if self.bj >= 0 else [])
        self.is_hinge = self.pj >= 0 and int(model.jnt_type[self.pj]) == int(mujoco.mjtJoint.mjJNT_HINGE)
        self.open_thr = math.radians(10) if self.is_hinge else 0.10
        self.closed_thr = math.radians(3) if self.is_hinge else 0.03
        # geom classification
        self.geom_sem = {}
        self.door_geoms, self.op_geoms, self.leaf_geoms, self.frame_geoms, self.glass_geoms = set(), set(), set(), set(), set()
        for g in range(model.ngeom):
            b = model.geom_bodyid[g]
            bname = mujoco.mj_id2name(mujoco.mjtObj.mjOBJ_BODY, b) if False else mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if bname in self.robot_bodies:
                continue
            if b == 0:
                self.frame_geoms.add(g)
                continue
            self.door_geoms.add(g)
            if any(k in gname for k in ("handle", "lever", "knob", "pad", "exit_device", "pull", "paddle", "wheel", "thumb", "pin_knob", "bolt_knob", "latch_knob", "grip", "crossbar", "push")):
                self.op_geoms.add(g)
            if "slab" in gname or "glass" in gname or "wing" in gname or "flap" in gname or "strip" in gname or "curtain" in gname:
                self.leaf_geoms.add(g)
            if "glass" in gname:
                self.glass_geoms.add(g)
        self.robot_geoms = {g for g in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) in self.robot_bodies)}
        self.pass_plane_site = self._sid("door_plane_center")
        self.goal_site = self._sid("goal_point")
        self.approach_site = self._sid("approach_point")
        self._prev_side = None
        self._prev_qd = None
        self._op_over = {}
        self._f = np.zeros(6)
        self.lock_engaged = bool(spec.get("lock", {}).get("engaged"))
        self.lock_releasable = bool(spec.get("lock", {}).get("robot_side_release", True))
        names = list(lock_joints) if lock_joints is not None else [n for n in LEGACY_LOCK_JOINTS if self._jid(n) >= 0]
        names = [n for n in names if self._jid(n) >= 0 and not any(p in n for p in LOCK_IGNORE_PATTERNS)]
        self.release_buttons = [self._jid(n) for n in names if any(p in n for p in RELEASE_BUTTON_PATTERNS)]
        self.lock_release_joints = [self._jid(n) for n in names if any(p in n for p in LOCK_RELEASE_PATTERNS) and not any(p in n for p in RELEASE_BUTTON_PATTERNS)]
        # a lift pin (gates, baby gates) is modelled as the operator but is what releases the lock
        self.lock_release_joints += [j for j in range(model.njnt) if "pin_slide" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "")]
        self.lock_release_joints = [j for j in dict.fromkeys(self.lock_release_joints) if model.jnt_range[j][1] - model.jnt_range[j][0] > 1e-6]
        self.keypad_joints = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(model.njnt) if "keypad_key_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "")}
        self.code = spec.get("lock", {}).get("code")
        self._key_seq = []
        self._key_down = set()

    def _jid(self, name):
        if not name:
            return -1
        return self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, name)

    def _sid(self, name):
        return self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_SITE, name)

    def _under(self, body: int, ancestor: int) -> bool:
        """True if `body` is `ancestor` or one of its descendants."""
        b = int(body)
        for _ in range(32):
            if b == ancestor:
                return True
            if b == 0:
                return False
            b = int(self.m.body_parentid[b])
        return False

    def q(self, d, j):
        return float(d.qpos[self.m.jnt_qposadr[j]]) if j >= 0 else 0.0

    def dq(self, d, j):
        return float(d.qvel[self.m.jnt_dofadr[j]]) if j >= 0 else 0.0

    # ------------------------------------------------------------------
    def step(self, d, robot_base_pos=None):
        m, L, mj = self.m, self.L, self.mj
        L.steps += 1
        L.sim_time = float(d.time)
        qd = self.q(d, self.pj)
        L.max_door_angle = max(L.max_door_angle, abs(qd))
        # ---- contacts
        max_leaf_f = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            r1, r2 = g1 in self.robot_geoms, g2 in self.robot_geoms
            if not (r1 or r2):
                continue
            other = g2 if r1 else g1
            mj.mj_contactForce(m, d, i, self._f)
            fn = abs(float(self._f[0]))
            if other in self.door_geoms:
                if not L.touched_door:
                    L.touched_door = True
                    L.time_to_touch = float(d.time)
                if other in self.op_geoms:
                    L.touched_operator = True
                if other in self.leaf_geoms:
                    max_leaf_f = max(max_leaf_f, fn)
                    thr = self.damage.get("leaf_dent_force_N") or 1e9
                    if other in self.glass_geoms:
                        thr = self.damage.get("glass_break_force_N") or thr
                    if fn > thr:
                        self._damage(d, "impact", mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, other), fn, thr)
        L.max_leaf_contact_force = max(L.max_leaf_contact_force, max_leaf_f)
        # ---- operator(s): any operator joint (both touch bars of a pair count) at >= 70 % of its travel
        for oj in self.op_joints:
            lo, hi = m.jnt_range[oj]
            qo = self.q(d, oj)
            if hi - lo > 1e-6 and (qo - lo) >= 0.7 * (hi - lo):
                L.operator_actuated = True
            tau = abs(float(d.qfrc_constraint[m.jnt_dofadr[oj]] + d.qfrc_applied[m.jnt_dofadr[oj]]))
            L.max_operator_torque = max(L.max_operator_torque, tau)
            ythr = self.damage.get("operator_yield_torque_Nm") or 1e9
            # yield is quasi-static: the load must be sustained (>= 10 ms), a single-step constraint impulse from the
            # operator hitting its stop is not an overload
            self._op_over[oj] = self._op_over.get(oj, 0) + 1 if tau > ythr else 0
            if self._op_over[oj] * m.opt.timestep >= 0.01 - 1e-9:
                self._damage(d, "operator_overload", mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, oj), tau, ythr)
                L.hardware_misuse = True
        # ---- latch / lock
        if self.latch_joints:
            if all((self.q(d, j) - m.jnt_range[j][0]) >= 0.8 * (m.jnt_range[j][1] - m.jnt_range[j][0]) for j in self.latch_joints):
                L.latch_released = True
        else:
            L.latch_released = True
        if self.lock_engaged and not L.lock_released and self.lock_releasable:
            if self.lock_release_joints and all((self.q(d, j) - m.jnt_range[j][0]) > 0.8 * (m.jnt_range[j][1] - m.jnt_range[j][0]) for j in self.lock_release_joints):
                L.lock_released = True
            for j in self.release_buttons:
                lo, hi = m.jnt_range[j]
                if hi - lo > 1e-6 and (self.q(d, j) - lo) > 0.8 * (hi - lo):
                    L.lock_released = True
            if self.code and self.keypad_joints:
                for name, j in self.keypad_joints.items():
                    key = name.split("keypad_key_")[1].split("_slide")[0]
                    pressed = self.q(d, j) > 0.7 * m.jnt_range[j][1]
                    if pressed and key not in self._key_down:
                        self._key_seq.append(key)
                        self._key_down.add(key)
                    if not pressed and key in self._key_down:
                        self._key_down.discard(key)
                if "".join(self._key_seq[-len(self.code):]) == self.code:
                    L.lock_released = True
                    L.notes.append(f"keypad code entered at t={d.time:.2f}")
        # ---- door state
        if abs(qd) >= self.open_thr and not L.door_opened:
            L.door_opened = True
            L.time_to_open = float(d.time)
        clear = (abs(qd) >= self.clear_angle) if self.is_hinge else (abs(qd) >= min(self.clear_travel, 0.95 * m.jnt_range[self.pj][1] if self.pj >= 0 and m.jnt_limited[self.pj] else self.clear_travel))
        if clear:
            L.door_open_clear = True
        # slam: closing speed when reaching closed
        dqd = self.dq(d, self.pj)
        if self._prev_qd is not None and abs(qd) < self.closed_thr and abs(self._prev_qd) >= self.closed_thr:
            vthr = self.damage.get("slam_velocity_rad_s", 4.0)
            if abs(dqd) > vthr:
                L.door_slammed = True
                self._damage(d, "slam", "leaf", abs(dqd), vthr)
        self._prev_qd = qd
        # ---- robot pass-through (base crosses the y=0 plane within the opening x-range)
        if robot_base_pos is not None:
            x, y, z = robot_base_pos
            side = 1 if y > 0 else -1
            Wo = self.spec["opening"]["width"]
            if self._prev_side is not None and side != self._prev_side and abs(x) < Wo / 2 + 0.3 and not L.robot_passed_through:
                L.robot_passed_through = True
                L.time_to_pass = float(d.time)
            self._prev_side = side
            if z < 0.35 and self.robot_base:
                L.robot_fell = True
            if L.robot_passed_through and abs(qd) < self.closed_thr:
                L.door_closed_after = True
        # ---- energy on door joints
        if self.pj >= 0:
            L.energy_J += abs(float(d.qfrc_applied[m.jnt_dofadr[self.pj]] + d.qfrc_constraint[m.jnt_dofadr[self.pj]]) * dqd) * m.opt.timestep
        return L

    def _damage(self, d, kind, part, value, thr):
        self.L.door_damaged = True
        if len(self.L.damage_events) < 50:
            self.L.damage_events.append({"step": self.L.steps, "t": float(d.time), "kind": kind, "part": part, "value": float(value), "threshold": float(thr)})

    def finalize(self):
        L = self.L
        t = L.task
        if t == "open_and_traverse":
            L.success = L.door_opened and L.robot_passed_through and not L.door_damaged
        elif t == "open_only":
            L.success = L.door_open_clear and not L.door_damaged
        elif t == "traverse_open":
            L.success = L.robot_passed_through and not L.touched_door and not L.door_damaged
        elif t == "close":
            L.success = abs(L.max_door_angle) > 0 and L.door_closed_after if False else (not L.door_damaged and L.notes.count("closed") >= 0 and L.door_closed_after) or (not L.door_damaged and (L.max_door_angle > 0) and self._closed_now)
        elif t == "unlock_open_traverse":
            L.success = L.lock_released and L.door_opened and L.robot_passed_through and not L.door_damaged
        elif t == "locked_recognize":
            L.success = (not L.door_opened) and (not L.door_damaged) and (not L.hardware_misuse)
        elif t == "push_through":
            L.success = L.robot_passed_through and not L.door_damaged
        elif t == "hold_and_pass":
            L.success = L.door_opened and L.robot_passed_through and not L.door_damaged and not L.door_slammed
        elif t == "peek":
            L.success = L.door_opened and (L.max_door_angle < self.clear_angle) and not L.robot_passed_through and not L.door_damaged
        else:
            L.success = L.door_opened and not L.door_damaged
        return L

    _closed_now = False

    def mark_closed(self, d):
        self._closed_now = abs(self.q(d, self.pj)) < self.closed_thr

    def mark_touch(self, d, operator: bool = False):
        """Record a touch from a programmatic hand (no robot geoms: DoorEnv.apply_* call this)."""
        L = self.L
        if not L.touched_door:
            L.touched_door = True
            L.time_to_touch = float(d.time)
        if operator:
            L.touched_operator = True
