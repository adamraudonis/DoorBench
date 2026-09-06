"""Scripted-hand baseline: the per-family heuristic of `scripts/demo_mujoco.py`, expressed through the policy interface.

It is an *oracle* heuristic - it reads the door's spec (joint names, lock parts, keypad code when the robot is
allowed to know it) and selects an authored mechanism sequence:

  1. release every robot-side lock part (thumbturn, slide bolts, dogs, hooks, lift pins, REX / call buttons),
     entering a keypad code key by key when there is one, presenting a badge where the lock needs one,
  2. actuate the operator(s) (lever / knob / thumb latch / touch bars of both leaves / handwheel one full turn),
  3. push, pull, slide or lift the leaf (both leaves of a pair) with the sign-off QA's calibrated push force,
  4. walk the base through once the opening is clear, hold the door until it is past, let go (closers return),
     or close it behind for `open_then_close`,
  5. `close_only`: close at a controlled speed and seat the latch; `locked_recognize`: try gently for a few seconds,
     declare the door locked (`declare_locked`) if the leaf did not move.

Human suite (advanced, opt in with `--suite human`): `knock_and_wait` knocks first and waits 3.3 s before the
sequence above; `wait_for_human` steps aside, waits until the person coming through has finished their path, then
opens; `hold_open_for_human` opens, steps beside the doorway away from the person's path, holds the leaf until the
person is through the plane, then walks through.

It has no perception, arm kinematics or balance model. Authored side permissions restrict hardware access,
but generalized forces do not establish human reach or a feasible grasp. Pocket edge-pull extraction uses
bounded world forces on actual articulated sites; that still does not simulate a complete human hand.
"""
from __future__ import annotations

import math

import numpy as np

from ..policy import Policy

DEG = math.radians
MOMENTARY = ("rex_button_slide", "call_button_slide")
TRAVERSE = ("open_and_traverse", "unlock_and_traverse", "open_then_close", "hold_open_for_human", "wait_for_human", "knock_and_wait")


class ScriptedHandPolicy(Policy):
    name = "scripted_hand"
    description = "Oracle heuristic hand from scripts/demo_mujoco.py: releases robot-side lock parts (keypad code, thumbturn, bolts, dogs, badge), actuates the operator(s), pushes / slides / lifts the leaf with the QA push force, walks the base through when the opening is clear, lets go or closes behind; knocks, waits for or holds the door for the simulated person of the human suite."
    control_dt = 0.004

    def control_period(self,env):
        # The chain's force-feedback handoff is resolved at its native step.
        # Holding this high-stiffness contact controller for8 steps changes
        # its stability. The actual rate is saved in every episode result.
        return float(env.m.opt.timestep) if env.meta.get('rollup_hoist') else self.control_dt

    # ------------------------------------------------------------------ setup
    def reset(self, info: dict, env=None) -> None:
        self.info = info
        self.env = env
        self.scenario = info["scenario"]
        self.traverse = self.scenario in TRAVERSE
        self.require_closed = self.scenario == "open_then_close"
        self.joints = info["joints"]
        self.lim = info["torque_limits"]
        self.pj = info["primary_joint"]
        self.sj = info["secondary_joint"] if info.get("secondary_joint") in self.joints else None
        self.leaves = [j for j in (self.pj, self.sj) if j and self.lim.get(j, 0) > 0]
        if info["family"] == "sliding_bypass":
            self.leaves = self.leaves[:1]  # independent panels: one selected leaf per attempt
        self.sequential = info["family"] == "bifold" and len(self.leaves) > 1
        self.active_leaf_index = 0
        self.push = float(self.lim.get(self.pj, 60.0)) or 60.0
        self.is_hinge = self.joints.get(self.pj, {}).get("type", "hinge") == "hinge"
        spec = info["spec"]
        kin = info["kinematics"].get("type", "hinge_vertical")
        self.kin = kin
        meta = info["meta"]
        self.rotary_locksets=meta.get('rotary_locksets',[])
        approach=1. if spec['robot'].get('approach_side')=='+y' else -1.
        self.rotary_free_egress=bool(self.rotary_locksets) and all(
            row['inside_face']==approach for row in self.rotary_locksets)
        self.vault_control=None
        if meta.get('vault_boltwork'):
            from ..vault_control import VaultControl
            self.vault_control=VaultControl(env)
        self.dutch=meta.get('dutch_joining_bolt')
        dutch_operation=meta.get('dutch_operation')
        if self.dutch:
            if dutch_operation=='upper_then_lower':
                self.leaves=[j for j in (self.dutch['upper_joint'],self.dutch['lower_joint']) if self.lim.get(j,0)>0]
                self.sequential=len(self.leaves)>1
            else:
                selected=self.dutch['upper_joint'] if dutch_operation=='upper_only' else self.dutch['lower_joint']
                self.leaves=[selected] if self.lim.get(selected,0)>0 else []
        self.sectional = meta.get('sectional_track')
        self.rollup = meta.get('rollup_curtain')
        self.lift = self.sectional or self.rollup
        self.hoist_rules = None
        self.hoist_keeper=None;self.hoist_transition=None;self.hoist_ready=False;self.hoist_held=False
        self.hoist_failed=None;self.hoist_positioned=False;self.hoist_departed=False
        if meta.get('rollup_hoist'):
            if env is None:raise ValueError('A material-chain controller requires the native environment')
            from ...rollup_hoist import compile_hoist
            self.hoist_rules=compile_hoist(env.m,meta)
            from ...hoist_keeper import compile_keeper
            self.hoist_keeper=compile_keeper(env.m,meta)
        self.knob_covers = meta.get('knob_covers', [])
        from ..security_release import SecurityRelease
        self.security_release=SecurityRelease(env) if env is not None else None
        self.lift_target = None
        self.lift_time = None
        dmg = spec.get("physics", {}).get("damage", {})
        self.yield_tau = float(dmg.get("operator_yield_torque_Nm") or 1e9)
        # operators: IR-role operators + meta.operator_joint (stalls use their slide latch as the operator)
        ops = list(info["operator_joints"])
        if meta.get("operator_joint") and meta["operator_joint"] in self.joints and meta["operator_joint"] not in ops:
            ops.append(meta["operator_joint"])
        # hand-lifted gravity latches (gate forks, latch bars) have no operator: lift them like one
        ops += [l for l in info["latch_joints"] if self.lim.get(l, 0) > 0 and l not in ops]
        self.pocket = meta.get('pocket_edge_pull')
        edge_joint = self.pocket['joint'] if self.pocket else None
        self.hatch_support = meta.get('hatch_support') or {}
        stay_release = self.hatch_support.get('support_release_joint')
        wall_inputs={b['joint'] for b in (meta.get('automatic_activation') or {}).get('buttons',[])}
        self.ops = [o for o in ops if self.lim.get(o, 0) > 0 and o not in (edge_joint, stay_release)
                    and o not in wall_inputs]
        self.stay_withdrawn = False
        self.closing_leaf_index = 0
        self.pocket_phase = None
        self.wheel = next((o for o in self.ops if "wheel" in o), None)
        # lock parts the hand may move (limit > 0), minus keys (entered in code order) and momentary buttons
        locks = [l for l in info["lock_joints"] if self.lim.get(l, 0) > 0 and l not in self.ops]
        self.keys = [l for l in locks if "keypad_key_" in l]
        self.buttons = [l for l in locks if any(p in l for p in MOMENTARY)]
        self.locks = [l for l in locks if l not in self.keys and l not in self.buttons
                      and not self._already_released(l)]
        if self.dutch and dutch_operation!='upper_only':
            # Leave a joined pair coupled. Withdrawing its bolt merely
            # because it is classified as a lock defeats its purpose.
            self.locks=[j for j in self.locks if j!=self.dutch['joint']]
        if self.dutch and dutch_operation=='upper_only':
            self.ops=[]  # The lower leaf's ordinary latch stays latched.
        code = (info["lock"].get("code") or "") if info["lock"].get("engaged") else ""
        if self.rotary_free_egress:
            # The inside trim always retracts its own latch cam. Exterior
            # credentials have no role in leaving through this face.
            code='';self.locks=[];self.buttons=[]
        self.code_keys = []
        for k in code:
            jn = f"leaf_keypad_key_{ {'*': 'star', '#': 'hash'}.get(k, k) }_slide"
            if jn in self.keys:
                self.code_keys.append(jn)
        # automatic door: powered operator opens for us (elevator: after the call button)
        act = info["kinematics"].get("actuator") or {}
        self.automatic = bool(meta.get("actuators")) and info["family"] in ("automatic_sliding", "automatic_swing", "elevator") and act.get("powered", True) is not False
        self.automatic = self.automatic or bool(self.sectional and self.sectional['drive']['mode']=='powered')
        activation = meta.get("automatic_activation") or {}
        self.activation_buttons = [b['joint'] for b in activation.get('buttons', []) if b.get('face') == -1 and self.lim.get(b['joint'], 0) > 0] if self.automatic else []
        self.wave_site = next((s for s in activation.get('wave_sites', []) if s.endswith('_n_zone')), None) if self.automatic else None
        if self.automatic:
            # Electric retraction belongs to the door controller. The hand
            # activates the wall station, not a simultaneous bar/lock gesture.
            self.ops = []
        # Only an actual credential-controlled class accepts a badge. A
        # missing/far-side thumbturn, key or slide bolt cannot be bypassed by
        # an invented credential when no manual input is available.
        lk = info["lock"]
        credential=lk.get('kind') in ('card_reader','mag_lock','electric_strike') or bool(meta.get('turnstile_locks'))
        self.badge_needed = credential and bool(lk.get("engaged")) and bool(lk.get("robot_side_release")) and not (self.locks or self.code_keys or self.buttons)
        self.badge_needed=self.badge_needed and not self.rotary_free_egress
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
        elif kin == "hinge_horizontal":
            self.mode = "swing"
            self.target = self.hi + 0.05      # a hatch lid / tilt-up panel rests against its stop or strut
        else:
            self.mode = "swing"
            self.target = min(DEG(80), max(0.85 * self.hi, min(0.97 * self.hi, DEG(62))))
        if both_ways and self.mode == "swing":
            self.target = min(self.target, 0.97 * self.hi)
        if info['family'] == 'bifold':
            self.target = self.hi + .05
        # gains (as in the demo hands)
        p = self.push
        self.kp, self.kd = (3.0 * p, 0.35 * p) if self.mode == "swing" else ((4.0 * p, 1.0 * p) if self.mode == "slide" else (2.0 * p, 0.6 * p))
        # timeline (hand time = sim time - delay)
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
        self.plane_y = float((info.get("pass_plane") or {}).get("center", [0, 0, 0])[1])
        self.travel_sign=float((info.get('pass_plane') or {}).get('traverse_direction',[0,1,0])[1]) or 1.
        self.t_pass = None
        self.released = False
        self.hold_after = 1.0
        self.give_up_at = self.t_push + 3.0 if self.t_push is not None else 8.0
        self.done = False
        # ---- human suite
        self.delay = 0.0            # hand time offset: knock-and-wait, waiting for the person to come through
        self.knocked = False
        self.knock_at = 0.3
        self.hold_for_human = self.scenario == "hold_open_for_human"
        self.human_through = False
        self.aside = None           # [x, y] the base steps to before doing anything else
        self.park_x = 0.0
        self._stall = {}            # leaf -> time the closing servo stalled (friction feed-forward ramp)
        human = info.get("human")
        path = (human or {}).get("path") or []
        if self.scenario == "knock_and_wait":
            self.delay = self.knock_at + 3.3       # `waited` needs the door opened >= 3 s after the knock
        elif self.scenario == "wait_for_human" and path:
            self.delay = float(path[-1][0]) + 0.3   # the person has finished their path (the env opens the door for them)
            # the person passes beside the start zone on the side away from the handle: step to the other side
            hx = float(path[2][1]) if len(path) > 2 else 0.0
            sx = float(info["base"]["start"][0])
            self.aside = [(-1.0 if hx > sx else 1.0) * 0.5 + 0.0, float(info["base"]["start"][1])]
        elif self.hold_for_human and path:
            # the person walks up behind the robot on the handle side and through the plane: hold the leaf from the
            # hinge side of the doorway, 0.8 m before the wall (outside the 0.52 m collision radius of their path)
            hx = float(path[1][1]) if len(path) > 1 else 0.25
            self.aside = [(-1.0 if hx >= 0 else 1.0) * 0.6, -0.8]
            self.park_x = self.aside[0] * 0.75          # 0.45 m beside the goal centre (inside the 0.5 m goal zone)

    # ------------------------------------------------------------------ helpers
    def _already_released(self, joint):
        """An initially withdrawn bolt/turned thumbturn is not another opening handle."""
        info = self.joints[joint]
        rng = info.get("range")
        return bool(rng and rng[1] > rng[0] and
                    info.get("initial", 0.) >= rng[0] + .9 * (rng[1] - rng[0]))

    def _moving_leaves(self, obs):
        if not self.sequential:
            return self.leaves
        current = self.leaves[self.active_leaf_index]
        q, dq = self._q(obs, current)
        target = self.target if current == self.pj else self._secondary_target(current)
        if abs(q) >= .92 * min(abs(target), max(abs(v) for v in self.joints[current]["range"])) and abs(dq) < .18:
            self.active_leaf_index = min(self.active_leaf_index + 1, len(self.leaves) - 1)
        # Previously opened banks remain supported while the next bank opens.
        return self.leaves[:self.active_leaf_index + 1]

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

    @staticmethod
    def _toward(bx, by, tx, ty, speed=1.0):
        dx, dy = tx - bx, ty - by
        n = math.hypot(dx, dy)
        if n < 0.03:
            return [0.0, 0.0], True
        s = min(speed, 3.0 * n)
        return [s * dx / n, s * dy / n], False

    # ------------------------------------------------------------------ act
    def act(self, obs: dict) -> dict:
        action=self._act(obs)
        if self.rotary_locksets:
            from ..rotary_control import surface_action
            action=surface_action(self.env,self.rotary_locksets,action)
        return action

    def _act(self, obs: dict) -> dict:
        t = obs["t"]
        tt = t - self.delay            # hand time
        if self.security_release and tt>=0:
            service=self.security_release.act()
            if service is not None:return service
        out: dict[str, float] = {}
        bx, by = obs["base"]["pos"][0], obs["base"]["pos"][1]
        flags = obs["flags"]
        base_v = [0.0, 0.0]
        walking = False
        human = obs.get("human_xy")
        if self.hoist_rules and tt>=0:
            if self.hoist_failed:return {'base_velocity':[0.,0.],'mechanism_failure':self.hoist_failed}
            h=self.info['meta']['rollup_hoist'];keeper=self.hoist_keeper
            stand_y=float(self.env.d.site_xpos[keeper.grip_site,1])+.40
            if not self.hoist_positioned:
                velocity,arrived=self._toward(bx,by,float(h['wheel_center_m'][0]),stand_y,speed=.6)
                if not arrived:return {'base_velocity':velocity}
                self.hoist_positioned=True
            if self.hoist_held and self.traverse and not self.hoist_departed:
                # After both hands leave the held mechanism, align with the
                # aperture before walking forward; do not cut through a jamb.
                intervals=obs.get('passage_intervals') or []
                if not intervals:return {'base_velocity':[0.,0.]}
                lo,hi=min(intervals,key=lambda interval:abs((interval[0]+interval[1])/2-bx))
                velocity,arrived=self._toward(bx,by,(lo+hi)/2,stand_y,speed=.6)
                if not arrived:return {'base_velocity':velocity}
                self.hoist_departed=True
        # ---- knock first (knock_and_wait), step aside (human scenarios)
        knock = False
        if self.scenario == "knock_and_wait" and not self.knocked and t >= self.knock_at:
            knock, self.knocked = True, True
        at_aside = True
        if self.aside is not None and self.t_pass is None:
            base_v, at_aside = self._toward(bx, by, self.aside[0], self.aside[1])
        # ---- hold-open: the person is through once they are past the plane by their radius (+ margin)
        if self.hold_for_human and human is not None and self.travel_sign*(human[1]-self.plane_y)>.5:
            self.human_through = True
        # ---- when may the base walk: once the hand is at work (never during the knock / wait delay) and the opening
        # is clear *now* (a door that closed again behind the person must be re-opened first)
        clear = flags.get("door_clear_now", flags["door_open_clear"])
        if self.traverse and self.t_pass is None and tt >= self.t_press:
            walking = (clear or self.automatic) and (not self.hoist_rules or self.hoist_held)
            if self.hold_for_human and not self.human_through:
                walking = False
        if walking:
            # through the centre of the opening; the person of hold_open_for_human parks 0.8 m past the goal, so stop
            # short of the goal centre and to the side of their path (outside the 0.52 m collision radius)
            x_t = self.park_x if (self.hold_for_human and self.travel_sign*(by-self.plane_y)>.3) else 0.0
            intervals = obs.get('passage_intervals')
            if intervals and abs(by-self.plane_y)<1.2:
                lo,hi=min(intervals,key=lambda interval:abs((interval[0]+interval[1])/2-bx))
                x_t=(lo+hi)/2
            base_v = [float(np.clip(-2.0 * (bx - x_t), -0.5, 0.5)), self.travel_sign*(1.0 if clear else (0.6 if self.travel_sign*(by-self.plane_y)<-.9 else 0.0))]
        elif self.automatic and self.traverse and tt > self.t_press and self.travel_sign*(by-self.plane_y)<-.9 and self.t_pass is None and at_aside:
            base_v = [float(np.clip(-2.0 * bx, -0.5, 0.5)), self.travel_sign*.6]        # step into the sensor's range
        if self.traverse and self.t_pass is None and self.travel_sign*(by-self.goal_y)>=(-0.1 if self.hold_for_human else 0.3):
            self.t_pass = t
        if self.wave_site and not clear and self.t_pass is None:
            station = obs.get('sites', {}).get(self.wave_site)
            if station is not None:
                base_v, _ = self._toward(bx,by,float(station[0]),float(station[1])-.35,speed=.6)
        if tt < self.t_press:
            return {"base_velocity": base_v, "knock": knock}
        if self.vault_control:
            closing=self.scenario=='close_only' or (self.require_closed and self.t_pass is not None)
            action=self.vault_control.act(closing=closing,goal=self.target,
                hands_off=self.t_pass is not None and not closing)
            return {**action,'base_velocity':base_v}
        badge = False
        for button in self.activation_buttons:
            if tt < self.t_press + .8:
                # A finite steady press seats the physical spring-return cap.
                # The generic 4 ms handle PD has excessive damping for this
                # millimetre stroke and can chatter below the switch point.
                out[button]=20.
        if self.badge_needed and not self.badged and not flags["lock_released"]:
            badge, self.badged = True, True
        # ---- 1. lock parts (held retracted), keypad code, momentary buttons
        for jn in self.locks:
            if self.joints.get(jn, {}).get("type") == "hinge":
                self.operate(obs, jn, self.lock_effort(jn), out)       # thumbturns, dogs, hook levers: PD to the end of travel
            else:
                out[jn] = out.get(jn, 0.0) + self.lock_effort(jn)      # bolts: a steady pull (the QA's 60 N)
        for t0, t1, jn in self.key_times:
            if t0 <= tt < t1:
                out[jn] = out.get(jn, 0.0) + 10.0                      # keypad key: steady 10 N press
        if self.buttons and tt < self.t_op + 0.5 and not flags["lock_released"]:
            for jn in self.buttons:
                out[jn] = out.get(jn, 0.0) + 20.0                      # REX / call button: steady 20 N press
        # ---- close_only: close at a controlled speed (no slam)
        if self.scenario == "close_only":
            if self.sectional and self.automatic:
                return {'torques':out,'badge':badge}
            contact = self._close(obs, out)
            return {"torques": out, "badge": badge, **contact}
        # ---- let go / close behind after passing
        if self.t_pass is not None:
            if self.require_closed:
                if self.sectional and self.automatic:
                    for button in self.activation_buttons:
                        if t < self.t_pass+.8:
                            out[button]=20.
                    return {'torques':out}
                contact = self._close(obs, out)
                return {"torques": out, **contact}
            # let go of swing / sliding doors (closers return); keep holding hatches and overhead doors, which would fall
            if t > self.t_pass + self.hold_after and self.kin in ("hinge_vertical", "slide_horizontal", "rotor"):
                self.released = True
            if self.released:
                return {"torques": {jn: v for jn, v in out.items() if jn in self.locks}}
        # ---- locked_recognize: try gently, then declare
        gentle = self.scenario == "locked_recognize"
        if gentle and tt > self.give_up_at:
            q, _ = self._q(obs, self.pj)
            if abs(q) < (DEG(5) if self.is_hinge else 0.03):
                return {"torques": {}, "declare_locked": True}
        # ---- 2. operator(s)
        q_leaf, _ = self._q(obs, self.pj)
        release_op = DEG(20) if self.is_hinge else 0.15
        if tt >= self.t_op:
            if self.wheel:
                rng = self.joints[self.wheel].get("range") or [0.0, 2 * math.pi]
                end = float(rng[1])
                qw, _ = self._q(obs, self.wheel)
                if qw < end - 0.05 and self.t_wheel_done is None:
                    self.velocity(obs, self.wheel, 2.4, 25.0, min(60.0, self.lim.get(self.wheel, 60.0)), out)
                else:
                    if self.t_wheel_done is None:
                        self.t_wheel_done = tt
                        self.t_push = tt + 0.4
                        self.give_up_at = self.t_push + 3.0
                    self.servo(obs, self.wheel, end - 0.02, 40.0, 4.0, min(30.0, self.lim.get(self.wheel, 30.0)), out)
            for o in self.ops:
                if o == self.wheel:
                    continue
                if q_leaf < release_op or self.mode in ("slide", "vertical"):
                    self.operate(obs, o, self.op_effort(o), out)
        # ---- 3. leaf
        if self.t_push is not None and tt >= self.t_push and not self.automatic:
            if self.lift:
                return {"torques": out, "base_velocity": base_v, "badge": badge,
                        **self._lift_action(obs, closing=False)}
            elif self.mode == "rotor":
                if self.t_pass is None:
                    self.velocity(obs, self.pj, 0.9, 120.0, self.push, out, floor=0.0)
            else:
                lim = min(self.push, 25.0 if self.is_hinge else 60.0) if gentle else self.push     # a normal push, not the QA's strong one
                for lf in self._moving_leaves(obs):
                    tgt = self.target if lf == self.pj else self._secondary_target(lf)
                    self.servo(obs, lf, tgt, self.kp, self.kd, lim, out)
                    if self.info['family'] == 'bifold':
                        # Overcome the measured folded-panel joint friction
                        # at low speed; reaching an angle is not clearance.
                        out[lf] = float(np.clip(out[lf] + math.copysign(.4*self.push,tgt), -lim, lim))
                if self.pocket:
                    p = self.pocket
                    q, _ = self._q(obs, p['leaf_joint'])
                    if q >= p['final_push_switch_q']:
                        # Transfer to the exposed leading edge before the cup
                        # enters the wall. An edge contact can push, not pull.
                        effort = float(np.clip(out.pop(p['leaf_joint'], 0.), 0., 120.))
                        site = p['final_push_site']
                        return {'torques': out, 'base_velocity': base_v, 'badge': badge,
                                'site_forces': {site: (np.asarray(p['final_push_direction'])*effort).tolist()},
                                'contact_site': site, 'contact_joint': p['leaf_joint']}
        contacts = self._covered_knob_contacts(out)
        return {"torques": out, "base_velocity": base_v, "badge": badge, "knock": knock, **contacts,
                "active_leaf": self.leaves[self.active_leaf_index] if self.leaves else self.pj,
                "activate_sensor": self.wave_site if tt < self.t_press + 8. and not clear else None}

    def _covered_knob_contacts(self, out):
        commands = {}
        primary = None
        for record in self.knob_covers:
            effort = out.pop(record['operator_joint'], 0.)
            if abs(effort)<1e-8:continue
            face = next((f for f in record['faces'] if f['face']==-1),None)
            if face is None:continue
            m,d=self.env.m,self.env.d
            joint=m.joint(record['operator_joint']).id
            force=float(np.clip(effort/(2*face['knob_radius_m']),-20.,20.))
            for site in face['grip_sites']:
                sid=m.site(site).id
                normal=d.site_xmat[sid].reshape(3,3)[:,2]
                commands[site]=(force*np.cross(d.xaxis[joint],normal)).tolist()
                if primary is None:primary=(record['operator_joint'],site)
        return {'site_forces':commands,'contact_joint':primary[0],'contact_site':primary[1]} if commands else {}

    def _secondary_target(self, lf):
        rng = self.joints.get(lf, {}).get("range")
        if not rng:
            return self.target
        lo, hi = float(rng[0]), float(rng[1])
        if self.info['family'] == 'bifold':
            return hi+.05 if hi>1e-6 else lo-.05
        if hi <= 1e-6 < -lo:            # mirrored leaf (opens negative)
            return -self.target if abs(self.target) <= -lo else 0.9 * lo
        return min(self.target, 0.97 * hi) if hi > 1e-6 else self.target

    def _close(self, obs, out):
        """Drive every leaf back to 0 at a controlled speed: fast far from the stop, slow near it (no slam).  A leaf
        that stalls against track / hinge friction (a dirty-track slider resists ~100 N) gets a feed-forward push on
        top of the velocity servo, which brakes it again once it moves."""
        t = obs["t"]
        if self.lift:
            return self._lift_action(obs, closing=True)
        support = self.hatch_support
        release = support.get('support_release_joint')
        if release:
            q, dq = self._q(obs, self.pj)
            pin, _ = self._q(obs, release)
            if q >= support['nominal_angle_rad'] - .06:
                # Lift/support the lid while withdrawing the load-bearing pin.
                # The stay catches by contact; no equality or pose reset releases it.
                self.operate(obs, release, min(60., self.lim.get(release, 60.)), out)
                if pin < .9 * support['release_position_m']:
                    self.servo(obs, self.pj, support['nominal_angle_rad']+.03,
                               self.kp, self.kd, self.push, out)
                    return {'contact_joint': release, 'contact_site': support['support_release_site']}
                self.stay_withdrawn = True
            elif self.stay_withdrawn:
                out.pop(release, None)
        if self.pocket:
            p = self.pocket
            q, dq = self._q(obs, p['leaf_joint'])
            extracted = p['recessed_leaf_q'] - q
            if extracted < p['face_grip_after_extract_m']:
                deployed, _ = self._q(obs, p['joint'])
                if self.pocket_phase is None:
                    self.pocket_phase = 'press'
                if deployed >= p['minimum_grasp_q']:
                    self.pocket_phase = 'extract'
                elif self.pocket_phase == 'extract' and deployed < .5:
                    self.pocket_phase = 'press'
                if self.pocket_phase == 'press':
                    site, joint = p['press_site'], p['joint']
                    force = np.asarray(p['press_direction']) * 12.
                else:
                    site, joint = p['grip_site'], p['leaf_joint']
                    # Pull the deployed rocker itself: its pivot and stop
                    # transmit the load into the leaf, with no leaf teleport
                    # or hidden torque holding the rocker open.
                    effort = float(np.clip(100.*(.25+dq), 0., min(self.push,120.)))
                    force = np.asarray(p['extract_direction']) * effort
                out.pop(p['joint'], None)
                out.pop(p['leaf_joint'], None)
                return {'site_forces': {site: force.tolist()},
                        'contact_site': site, 'contact_joint': joint}
            self.pocket_phase = 'face_pull'
        closing = self.leaves
        if self.sequential:
            closing = list(reversed(self.leaves))
            q, dq = self._q(obs, closing[self.closing_leaf_index])
            if abs(q) < DEG(1.) and abs(dq) < .1:
                self.closing_leaf_index = min(self.closing_leaf_index+1, len(closing)-1)
            # Support the other bank until the current one seats.
            for lf in closing[self.closing_leaf_index+1:]:
                target = self.target if lf == self.pj else self._secondary_target(lf)
                self.servo(obs, lf, target, self.kp, self.kd, self.push, out)
            closing = closing[:self.closing_leaf_index+1]
        for lf in closing:
            q, dq = self._q(obs, lf)
            near = abs(q) < (DEG(12) if self.is_hinge else 0.12)
            v = (0.35 if near else 1.0) * (1.0 if self.is_hinge else 0.5)
            v_t = -v if q > 0 else v
            sgn = -1.0 if q >= 0 else 1.0
            if abs(q) < (DEG(1.0) if self.is_hinge else 0.01):
                # seat it: a steady push so the latch bolt rides over the strike lip (a slider's needs to beat the
                # track friction; a hinged leaf or a vertical door gets a modest one so nothing slams)
                seat = min(0.25 * self.push, 40.0) if self.is_hinge else (min(0.25 * self.push, 120.0) if self.mode == "vertical" else min(0.5 * self.push, 200.0))
                out[lf] = out.get(lf, 0.0) + sgn * seat
                self._stall.pop(lf, None)
                continue
            kv = 2.0 * self.push
            tau = kv * (v_t - dq)
            if abs(dq) < 0.3 * v and self.mode != "vertical":
                # stalled against friction: feed-forward that builds up over 0.5 s (vertical doors close under gravity)
                t0 = self._stall.setdefault(lf, t)
                tau += sgn * min(0.5 * self.push, self.push * max(0.0, t - t0))
            else:
                self._stall.pop(lf, None)
            out[lf] = out.get(lf, 0.0) + float(np.clip(tau, -self.push, self.push))
        return {}

    def _lift_action(self, obs, closing):
        """Pull the actual bottom grip toward a moving point on its track.

        The moving target limits speed; its force is bounded independently of
        QA calibration. Roller contacts/cables determine every panel pose.
        """
        state = obs['lift_state']
        t = obs['t']
        if self.lift_target is None or getattr(self,'lift_closing',None)!=closing:
            self.lift_target = state['travel_m']
            self.lift_time = t
            self.lift_start = self.lift_target
            self.lift_closing = closing
        if self.hoist_rules:
            return self._hoist_action(obs,closing)
        goal = 0. if closing else state['span_m']
        u=min(1.,max(0.,t-self.lift_time)/12.)
        self.lift_target=self.lift_start+(goal-self.lift_start)*(10*u**3-15*u**4+6*u**5)
        cap = self.lift['drive']['manual_max_force_N']
        if self.sectional:
            from ...geometry.sectional import track_path
            target, _ = track_path(self.sectional['progress']['closed_s_m']+self.lift_target, self.sectional['path'])
            force = np.r_[0.,3500.*(target-np.asarray(state['point'])[1:])-200.*np.asarray(state['velocity'])[1:]]
        else:
            from ...rollup import rollup_handle_force
            floor=self.rollup['progress']['closed_z_m']
            command=rollup_handle_force(state['point'][2],state['grip_speed_m_s'],
                start_m=floor+self.lift_start,goal_m=floor+goal,elapsed_s=t-self.lift_time,
                mass_kg=state['grip_effective_mass_kg'],force_limit_N=cap)
            force=np.array([0.,0.,command['force_N']])
        force *= min(1.,cap/max(float(np.linalg.norm(force)),1e-12))
        site = self.lift['manual_grip_site']
        return {'site_forces': {site: force.tolist()},
                'contact_site': site, 'contact_joint': self.pj}

    def _hoist_action(self,obs,closing):
        from ...rollup_hoist import hoist_control
        from ...hoist_keeper import begin_keeper_transition,keeper_transition_action,keeper_open_force
        m,d=self.env.m,self.env.d;h,k=self.hoist_rules,self.hoist_keeper
        if self.hoist_failed or (self.hoist_held and not closing):return {}
        # Re-evaluate load-bearing contacts at current geometry with the
        # preceding held input. A position-only refresh leaves contact forces
        # stale; the runner clears applied inputs after every native step.
        pending=d.qfrc_applied.copy()
        try:
            d.qfrc_applied[:]=self.env.last_applied_qfrc
            self.env._with_passive(lambda:self.env.mj.mj_forward(m,d))
        finally:d.qfrc_applied[:]=pending
        if self.hoist_transition is None and not self.hoist_ready:
            self.hoist_transition=begin_keeper_transition(m,d,h,k,mode='release')
        if self.hoist_transition is not None:
            action=keeper_transition_action(m,d,h,k,self.hoist_transition)
            self.hoist_transition=action['next_state']
            if action['failed']:
                self.hoist_failed=action['reason'];return {'mechanism_failure':action['reason']}
            if action['done']:
                if self.hoist_transition['mode']=='engage':self.hoist_held=True
                else:self.hoist_ready=True;self.lift_time=obs['t']
                self.hoist_transition=None
            forces=action['site_forces']
            chain=next((site for site in forces if site!=k.grip_name),None)
            return {'site_forces':forces,**({'contact_site':chain,'contact_joint':self.pj} if chain else {})}
        height=float(d.site_xpos[h.bottom_site,2])
        if closing and height<=h.closed_z+.01:
            # The floor supports a shut curtain. Do not lift it to align a
            # keeper gap or leave an invisible hand supporting the chain.
            return {}
        if not closing and height>=h.open_z-.025:
            self.hoist_transition=begin_keeper_transition(m,d,h,k,mode='engage')
            return self._hoist_action(obs,closing)
        control=hoist_control(m,d,h,opening=not closing,elapsed_s=max(0.,obs['t']-self.lift_time))
        return {'site_forces':{control['site']:control['force_N'],**keeper_open_force(m,d,k)},
                'contact_site':control['site'],'contact_joint':self.pj}
