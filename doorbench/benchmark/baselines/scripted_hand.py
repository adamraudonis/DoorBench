"""Scripted-hand baseline: the per-family heuristic of `scripts/demo_mujoco.py`, expressed through the policy interface.

It is an *oracle* heuristic - it reads the door's spec (joint names, lock parts, keypad code when the robot is
allowed to know it) and does what a competent person would do:

  1. release every robot-side lock part (thumbturn, slide bolts, dogs, hooks, lift pins, REX / call buttons),
     entering a keypad code key by key when there is one,
  2. actuate the operator(s) (lever / knob / thumb latch / touch bars of both leaves / handwheel one full turn),
  3. push, pull, slide or lift the leaf (both leaves of a pair) with the sign-off QA's calibrated push force,
  4. walk the base through once the opening is clear, hold the door until it is past, let go (closers return),
     or close it behind for close scenarios,
  5. `peek`: open to 30 deg and hold; `open_only`: open and hold; `close`: close at a controlled speed;
     `locked_recognize`: try gently for a few seconds, give up (`done`) if the leaf did not move.

It has no perception and no arm kinematics: it cannot fail for reasons a robot would (reach, grasp, balance), and it
cannot open what its hand cannot reach (far-side locks, unpowered / env-released access control).
"""
from __future__ import annotations

import math

import numpy as np

from ..policy import Policy
from ..scenarios import TRAVERSE_TASKS

DEG = math.radians
MOMENTARY = ("rex_button_slide", "call_button_slide")


class ScriptedHandPolicy(Policy):
    name = "scripted_hand"
    description = "Oracle heuristic hand from scripts/demo_mujoco.py: releases robot-side lock parts (keypad code, thumbturn, bolts, dogs), actuates the operator(s), pushes / slides / lifts the leaf with the QA push force, walks the base through when the opening is clear, lets go or closes behind."
    control_dt = 0.004

    # ------------------------------------------------------------------ setup
    def reset(self, info: dict, env=None) -> None:
        self.info = info
        self.task = info["task"]
        self.require_closed = bool(info["scenario"] in ("traverse_close",))
        self.joints = info["joints"]
        self.lim = info["torque_limits"]
        self.pj = info["primary_joint"]
        self.sj = info["secondary_joint"] if info.get("secondary_joint") in self.joints else None
        self.leaves = [j for j in (self.pj, self.sj) if j and self.lim.get(j, 0) > 0]
        self.push = float(self.lim.get(self.pj, 60.0)) or 60.0
        self.is_hinge = self.joints.get(self.pj, {}).get("type", "hinge") == "hinge"
        spec = info["spec"]
        kin = info["kinematics"].get("type", "hinge_vertical")
        self.kin = kin
        meta = info["meta"]
        dmg = spec.get("physics", {}).get("damage", {})
        self.yield_tau = float(dmg.get("operator_yield_torque_Nm") or 1e9)
        # operators: IR-role operators + meta.operator_joint (stalls use their slide latch as the operator)
        ops = list(info["operator_joints"])
        if meta.get("operator_joint") and meta["operator_joint"] in self.joints and meta["operator_joint"] not in ops:
            ops.append(meta["operator_joint"])
        # hand-lifted gravity latches (gate forks, latch bars) have no operator: lift them like one
        ops += [l for l in info["latch_joints"] if self.lim.get(l, 0) > 0 and l not in ops]
        self.ops = [o for o in ops if self.lim.get(o, 0) > 0]
        self.wheel = next((o for o in self.ops if "wheel" in o), None)
        # lock parts the hand may move (limit > 0), minus keys (entered in code order) and momentary buttons
        locks = [l for l in info["lock_joints"] if self.lim.get(l, 0) > 0 and l not in self.ops]
        self.keys = [l for l in locks if "keypad_key_" in l]
        self.buttons = [l for l in locks if any(p in l for p in MOMENTARY)]
        self.locks = [l for l in locks if l not in self.keys and l not in self.buttons]
        code = info["lock"].get("code") or ""
        self.code_keys = []
        for k in code:
            jn = f"leaf_keypad_key_{ {'*': 'star', '#': 'hash'}.get(k, k) }_slide"
            if jn in self.keys:
                self.code_keys.append(jn)
        # automatic door: powered operator opens for us (elevator: after the call button)
        act = info["kinematics"].get("actuator") or {}
        self.automatic = bool(meta.get("actuators")) and info["family"] in ("automatic_sliding", "automatic_swing", "elevator") and act.get("powered", True) is not False
        # an engaged lock the robot may release but with no part to move (card reader, key, privacy button, maglock
        # without a REX button, electric strike): present the credential / key
        lk = info["lock"]
        self.badge_needed = bool(lk.get("engaged")) and bool(lk.get("robot_side_release")) and not (self.locks or self.code_keys or self.buttons) and lk.get("kind") not in ("delayed_egress", "jam_stuck", "chain", "swing_bar_guard")
        self.badged = False
        # leaf targets
        rng = self.joints.get(self.pj, {}).get("range") or ([0.0, DEG(90)] if self.is_hinge else [0.0, 1.0])
        lo, hi = float(rng[0]), float(rng[1])
        both_ways = bool(meta.get("both_ways")) or (lo < -1e-6 < hi)
        self.hi = hi if hi > 1e-6 else (DEG(90) if self.is_hinge else 1.0)
        if kin == "rotor":
            self.mode = "rotor"
            self.target = None
        elif kin == "slide_vertical":
            self.mode = "vertical"
            self.target = self.hi + 0.1       # against the top stop (a PD without integral action sags under the uncounterbalanced weight)
        elif kin.startswith("slide"):
            self.mode = "slide"
            self.target = self.hi + 0.05      # against the end stop
        else:
            self.mode = "swing"
            self.target = min(DEG(80), max(0.85 * self.hi, min(0.97 * self.hi, DEG(62))))
        if self.task == "peek":
            self.target = DEG(30) if self.is_hinge else min(0.3, 0.6 * self.hi)
            if self.mode == "vertical":
                self.target = min(1.0, 0.5 * self.hi)
        if both_ways and self.mode == "swing":
            self.target = min(self.target, 0.97 * self.hi)
        # gains (as in the demo hands)
        p = self.push
        self.kp, self.kd = (3.0 * p, 0.35 * p) if self.mode == "swing" else ((4.0 * p, 1.0 * p) if self.mode == "slide" else (2.0 * p, 0.6 * p))
        # timeline
        self.t_press = 0.5
        self.t_lock = self.t_press
        n_keys = len(self.code_keys)
        self.key_times = [(self.t_lock + i * 0.5, self.t_lock + i * 0.5 + 0.3, jn) for i, jn in enumerate(self.code_keys)]
        t_keys_end = self.t_lock + n_keys * 0.5 + (0.4 if n_keys else 0.0)
        self.t_op = t_keys_end + (0.8 if (self.locks or self.buttons or self.badge_needed) else 0.0)
        self.t_push = self.t_op + (0.6 if (self.ops or self.wheel) else 0.0)
        if self.wheel:
            self.t_push = None      # after the wheel has turned
        self.t_wheel_done = None
        self.goal_y = float(info["goal_point"][1])
        self.t_pass = None
        self.released = False
        self.hold_after = 1.0
        self.give_up_at = self.t_push + 3.0 if self.t_push is not None else 8.0
        self.done = False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _q(obs, j):
        v = obs["joints"].get(j)
        return (v["q"], v["dq"]) if v else (0.0, 0.0)

    def servo(self, obs, j, target, kp, kd, limit, out):
        q, dq = self._q(obs, j)
        tau = float(np.clip(kp * (target - q) - kd * dq, -limit, limit))
        out[j] = out.get(j, 0.0) + tau
        return tau

    def velocity(self, obs, j, v_target, kv, limit, out, floor=0.0):
        q, dq = self._q(obs, j)
        tau = float(np.clip(kv * (v_target - dq), floor, limit))
        out[j] = out.get(j, 0.0) + tau
        return tau

    def operate(self, obs, j, effort, out, frac=1.0):
        """PD an operator / lock part to `frac` of its travel at a controlled rate."""
        rng = self.joints.get(j, {}).get("range") or [0.0, 1.0]
        lo, hi = float(rng[0]), float(rng[1])
        travel = max(hi - lo, 1e-3)
        effort = min(effort, self.lim.get(j, effort))
        return self.servo(obs, j, lo + frac * travel, effort / (0.15 * travel), effort / (6.0 * travel), effort, out)

    def op_effort(self, j):
        hinge = self.joints.get(j, {}).get("type") == "hinge"
        if j.startswith("dog_"):
            e = 14.0
        elif "exit_device" in j:
            e = 110.0
        elif hinge:
            e = 4.0
        else:
            e = 60.0
        return min(e, 0.8 * self.yield_tau, self.lim.get(j, e))

    def lock_effort(self, j):
        hinge = self.joints.get(j, {}).get("type") == "hinge"
        if j.startswith("dog_"):
            return 14.0
        return 3.0 if hinge else 60.0

    # ------------------------------------------------------------------ act
    def act(self, obs: dict) -> dict:
        t = obs["t"]
        out: dict[str, float] = {}
        bx, by = obs["base"]["pos"][0], obs["base"]["pos"][1]
        flags = obs["flags"]
        task = self.task
        base_v = [0.0, 0.0]
        walking = False
        # ---- when may the base walk
        if task == "traverse_open":
            walking = t > 0.3
        elif task in TRAVERSE_TASKS:
            walking = flags["door_open_clear"] or (self.automatic and t > self.t_press)
        if walking and self.t_pass is None:
            base_v = [float(np.clip(-2.0 * bx, -0.5, 0.5)), 1.0 if flags["door_open_clear"] or task == "traverse_open" else (0.6 if by < -0.9 else 0.0)]
        elif self.automatic and t > self.t_press and by < -0.9 and self.t_pass is None:
            base_v = [float(np.clip(-2.0 * bx, -0.5, 0.5)), 0.6]        # step into the sensor's range
        if task in TRAVERSE_TASKS and self.t_pass is None and by >= self.goal_y + 0.3:
            self.t_pass = t
        if task == "traverse_open":
            return {"base_velocity": base_v}
        if t < self.t_press:
            return {"base_velocity": base_v}
        badge = False
        if self.badge_needed and not self.badged and not flags["lock_released"]:
            badge, self.badged = True, True
        # ---- 1. lock parts (held retracted), keypad code, momentary buttons
        for jn in self.locks:
            if self.joints.get(jn, {}).get("type") == "hinge":
                self.operate(obs, jn, self.lock_effort(jn), out)       # thumbturns, dogs, hook levers: PD to the end of travel
            else:
                out[jn] = out.get(jn, 0.0) + self.lock_effort(jn)      # bolts: a steady pull (the QA's 60 N)
        for t0, t1, jn in self.key_times:
            if t0 <= t < t1:
                out[jn] = out.get(jn, 0.0) + 10.0                      # keypad key: steady 10 N press
        if self.buttons and t < self.t_op + 0.5 and not flags["lock_released"]:
            for jn in self.buttons:
                out[jn] = out.get(jn, 0.0) + 20.0                      # REX / call button: steady 20 N press
        # ---- close task: close at a controlled speed (no slam)
        if task == "close":
            self._close(obs, out)
            return {"torques": out, "badge": badge}
        # ---- let go / close behind after passing
        if self.t_pass is not None:
            if self.require_closed:
                self._close(obs, out)
                return {"torques": out}
            # let go of swing / sliding doors (closers return); keep holding hatches and overhead doors, which would fall
            if t > self.t_pass + self.hold_after and self.kin in ("hinge_vertical", "slide_horizontal", "rotor"):
                self.released = True
            if self.released:
                return {"torques": {jn: v for jn, v in out.items() if jn in self.locks}}
        # ---- locked_recognize: try gently, then give up
        gentle = task == "locked_recognize"
        if gentle and t > self.give_up_at:
            q, _ = self._q(obs, self.pj)
            if abs(q) < (DEG(5) if self.is_hinge else 0.03):
                return {"torques": {}, "done": True}
        # ---- 2. operator(s)
        q_leaf, _ = self._q(obs, self.pj)
        release_op = DEG(20) if self.is_hinge else 0.15
        if t >= self.t_op:
            if self.wheel:
                rng = self.joints[self.wheel].get("range") or [0.0, 2 * math.pi]
                end = float(rng[1])
                qw, _ = self._q(obs, self.wheel)
                if qw < end - 0.05 and self.t_wheel_done is None:
                    self.velocity(obs, self.wheel, 2.4, 25.0, min(60.0, self.lim.get(self.wheel, 60.0)), out)
                else:
                    if self.t_wheel_done is None:
                        self.t_wheel_done = t
                        self.t_push = t + 0.4
                        self.give_up_at = self.t_push + 3.0
                    self.servo(obs, self.wheel, end - 0.02, 40.0, 4.0, min(30.0, self.lim.get(self.wheel, 30.0)), out)
            for o in self.ops:
                if o == self.wheel:
                    continue
                if q_leaf < release_op or self.mode in ("slide", "vertical") or task in ("open_only", "peek"):
                    self.operate(obs, o, self.op_effort(o), out)
        # ---- 3. leaf
        if self.t_push is not None and t >= self.t_push and not self.automatic:
            if self.mode == "rotor":
                if self.t_pass is None:
                    self.velocity(obs, self.pj, 0.9, 120.0, self.push, out, floor=0.0)
            else:
                lim = min(self.push, 25.0 if self.is_hinge else 60.0) if gentle else self.push     # a normal push, not the QA's strong one
                for lf in self.leaves:
                    tgt = self.target if lf == self.pj else self._secondary_target(lf)
                    self.servo(obs, lf, tgt, self.kp, self.kd, lim, out)
        return {"torques": out, "base_velocity": base_v, "badge": badge}

    def _secondary_target(self, lf):
        rng = self.joints.get(lf, {}).get("range")
        if not rng:
            return self.target
        lo, hi = float(rng[0]), float(rng[1])
        if hi <= 1e-6 < -lo:            # mirrored leaf (opens negative)
            return -self.target if abs(self.target) <= -lo else 0.9 * lo
        return min(self.target, 0.97 * hi) if hi > 1e-6 else self.target

    def _close(self, obs, out):
        """Drive every leaf back to 0 at a controlled speed: fast far from the stop, slow near it (no slam)."""
        for lf in self.leaves:
            q, dq = self._q(obs, lf)
            near = abs(q) < (DEG(12) if self.is_hinge else 0.12)
            v = (0.35 if near else 1.0) * (1.0 if self.is_hinge else 0.5)
            v_t = -v if q > 0 else v
            if abs(q) < (DEG(1.0) if self.is_hinge else 0.01):
                # seat it: a modest constant push so the latch bolt rides over the strike lip
                out[lf] = out.get(lf, 0.0) + (-1.0 if q >= 0 else 1.0) * min(0.25 * self.push, 40.0 if self.is_hinge else 120.0)
                continue
            kv = 2.0 * self.push
            tau = float(np.clip(kv * (v_t - dq), -self.push, self.push))
            out[lf] = out.get(lf, 0.0) + tau
