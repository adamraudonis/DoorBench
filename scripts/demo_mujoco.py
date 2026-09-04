#!/usr/bin/env python
"""Physics demo (MuJoCo): a programmatic hand opens DoorBench doors through DoorEnv and records a video.

The hand does what a robot policy does through `doorbench.benchmark.DoorEnv`: it applies generalized forces with
`env.apply_joint_torque` to the operator joint (press the lever / touch bars, turn the handwheel, lift the thumb
latch, press the keypad keys in code order), then pushes the leaf, holds it while a synthetic robot base walks through
(passed to `env.step(robot_base_pos=...)`), and lets go so closers can return.  Frames are rendered offscreen from two
of the cameras defined in every door.xml (`robot_view` + `iso` by default) with a HUD: door id, sim time, joint
states and the LabelTracker labels as they flip (touched / actuated / unlatched / unlocked / opened / clear /
traversed / closed after / damaged).  The orange sphere is the hand, the blue cylinder the robot base.

Usage
  python scripts/demo_mujoco.py                              # the 7 default doors -> docs/media/demo_<id>.mp4 + .gif
  python scripts/demo_mujoco.py --ids db0050_swing_single --out /tmp/demo
  python scripts/demo_mujoco.py --list

Requires mujoco, pillow, imageio + imageio-ffmpeg  (uv pip install imageio imageio-ffmpeg).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# id -> (subtitle, hand class name, options)
DEFAULT_DEMOS = {
    "db0050_swing_single": ("lever + LCN 4040 closer, grade-1 deadlatch, 112 kg", "SwingHand", {"seconds": 14.0}),
    "db0345_sliding_single": ("patio slider, thumb latch + hook lock (engaged)", "SlidingHand", {"seconds": 10.0}),
    "db0148_garage_sectional": ("garage sectional, 5 sections, counterbalanced", "GarageHand", {"seconds": 10.0, "push_scale": 0.45}),
    "db0066_revolving": ("revolving door, 3 wings, speed governor", "RevolvingHand", {"seconds": 12.0, "push_scale": 2.5}),
    "db0019_swing_double": ("panic pair, surface vertical rod exit devices + closers", "PairHand", {"seconds": 14.0}),
    "db0179_vault": ("vault door, handwheel drives 4 bolts, 1.08 t", "VaultHand", {"seconds": 13.0, "cams": ("detail_handle", "iso")}),
    "db0526_swing_single": ("keypad lever (4-digit code, engaged) + spring hinge", "KeypadHand", {"seconds": 13.0, "cams": ("detail_handle", "robot_view")}),
}

HAND_RGBA = (1.0, 0.55, 0.1, 0.95)
ROBOT_RGBA = (0.25, 0.55, 1.0, 0.35)
ARROW_RGBA = (1.0, 0.85, 0.2, 0.9)


# ----------------------------------------------------------------------------------------------- programmatic hands
class Hand:
    """Drives one door through DoorEnv.  `act(t)` applies this step's generalized forces and updates the HUD state."""

    def __init__(self, env, push: float, **opts):
        self.env, self.m, self.d = env, env.m, env.d
        self.push = push
        self.opts = opts
        self.action = "approach"
        self.markers = []          # world positions of the hand(s)
        self.arrow = None          # (from, to) push arrow
        self.released = False
        self.t_pass = None         # time the robot finished passing

    # --- helpers
    @property
    def L(self):
        return self.env.tracker.L

    def q(self, name):
        j = self.env._jid(name)
        return float(self.d.qpos[self.m.jnt_qposadr[j]]) if j >= 0 else 0.0

    def dq(self, name):
        j = self.env._jid(name)
        return float(self.d.qvel[self.m.jnt_dofadr[j]]) if j >= 0 else 0.0

    def site(self, name):
        import mujoco
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.d.site_xpos[sid].copy() if sid >= 0 else None

    def body(self, name):
        import mujoco
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.d.xpos[bid].copy() if bid >= 0 else None

    def first_site(self, *names):
        for n in names:
            p = self.site(n)
            if p is not None:
                return p
        return None

    def torque(self, joint, tau):
        self.env.apply_joint_torque(joint, tau)

    def servo(self, joint, target, kp, kd, limit):
        """PD toward `target`, saturated at +-limit: a steady push far from the target, a hold near it."""
        tau = float(np.clip(kp * (target - self.q(joint)) - kd * self.dq(joint), -limit, limit))
        self.torque(joint, tau)
        return tau

    def velocity(self, joint, v_target, kv, limit, floor=0.0):
        tau = float(np.clip(kv * (v_target - self.dq(joint)), floor, limit))
        self.torque(joint, tau)
        return tau

    def operate(self, joint, effort, frac=1.0):
        """Move an operator to `frac` of its travel at a controlled rate (a hand does not slam a lever into its stop)."""
        lo, hi = self.m.jnt_range[self.env._jid(joint)]
        target = lo + frac * (hi - lo)
        travel = max(hi - lo, 1e-3)
        return self.servo(joint, target, kp=effort / (0.15 * travel), kd=effort / (6.0 * travel), limit=effort)

    # --- protocol
    def can_pass(self):
        return self.L.door_open_clear

    def robot_path(self, proxy, t, dt):
        return None                 # None -> straight march (RobotProxy default)

    def act(self, t):
        raise NotImplementedError

    def hud_joints(self):
        """[(label, joint name, unit)] shown in the HUD."""
        out = []
        if self.env.meta.get("primary_joint"):
            out.append(("leaf", self.env.meta["primary_joint"]))
        if self.env.meta.get("operator_joint"):
            out.append(("operator", self.env.meta["operator_joint"]))
        if self.env._jid("leaf_latch_bolt_slide") >= 0:
            out.append(("latch bolt", "leaf_latch_bolt_slide"))
        return out


class SwingHand(Hand):
    """Lever / knob swing door: press the operator, push the leaf by the handle, hold for the robot, let go."""

    def __init__(self, env, push, t_press=0.7, t_push=1.3, target_deg=None, op_tau=4.0, hold_after_pass=1.0, release_op_deg=20.0, **opts):
        super().__init__(env, push, **opts)
        self.t_press, self.t_push, self.op_tau, self.hold_after, self.release_op = t_press, t_push, op_tau, hold_after_pass, math.radians(release_op_deg)
        j = env.pj
        hi = float(env.m.jnt_range[j][1]) if env.m.jnt_limited[j] else math.radians(90)
        self.target = math.radians(target_deg) if target_deg else min(math.radians(80), 0.85 * hi)
        self.op = env.meta.get("operator_joint")
        self.pj = env.meta["primary_joint"]
        self.kp, self.kd = 3.0 * push, 0.35 * push

    def act(self, t):
        self.markers = [self.first_site("leaf_handle_grip_n", "leaf_handle_grip_p", "leaf_edge_mid")]
        q = self.q(self.pj)
        if t < self.t_press:
            self.action = "approach"
            return
        if self.released:
            self.action = "let go" + (" (closer returns)" if self.env.spec["closer"]["model"] != "none" else "")
            return
        if self.op and q < self.release_op:
            self.operate(self.op, self.op_tau)
            self.action = "press " + self.env.spec["operator"]["model"].split("_")[0]
        if t >= self.t_push:
            tau = self.servo(self.pj, self.target, self.kp, self.kd, self.push)
            self.action = "push leaf" if q < self.target - 0.1 else "hold open"
            self.arrow = tau
        if self.t_pass is not None and t > self.t_pass + self.hold_after:
            self.released = True


class KeypadHand(SwingHand):
    """Keypad lever: press the code keys in order (each key is a real joint), then lever + push like SwingHand."""

    def __init__(self, env, push, t_start=0.6, press_s=0.3, gap_s=0.2, key_force=10.0, **opts):
        self.code = env.spec["lock"].get("code") or ""
        if any(env._jid(self.key_joint(k)[0]) < 0 for k in self.code):
            self.code = ""          # keypad without physical keys (e.g. lever_euro_backplate + keypad lock): nothing to press
        self.keys = [(t_start + i * (press_s + gap_s), t_start + i * (press_s + gap_s) + press_s, k) for i, k in enumerate(self.code)]
        t_end = self.keys[-1][1] + 0.5 if self.keys else t_start
        super().__init__(env, push, t_press=t_end, t_push=t_end + 0.6, **opts)
        self.key_force = key_force

    def key_joint(self, k):
        lab = {"*": "star", "#": "hash"}.get(k, k)
        return f"leaf_keypad_key_{lab}_slide", f"leaf_keypad_key_{lab}"

    def act(self, t):
        for t0, t1, k in self.keys:
            jn, bn = self.key_joint(k)
            if t0 <= t < t1 and self.env._jid(jn) >= 0:
                self.torque(jn, self.key_force)
                self.action = f"press key {k}  (code {'*' * len(self.code)})"
                self.markers = [self.body(bn)]
                return
            if t < t0:
                self.action = "move to next key"
                self.markers = [self.body(bn)]
                return
        super().act(t)

    def hud_joints(self):
        return super().hud_joints() + [(f"key {k}", self.key_joint(k)[0]) for k in self.code[:2]]


class PairHand(SwingHand):
    """Panic pair: two hands push both touch bars (exit devices), then both leaves; both closers return."""

    def __init__(self, env, push, **opts):
        super().__init__(env, push, op_tau=110.0, **opts)
        self.ops = [n for n in env.operator_joints]
        self.leaves = [env.meta["primary_joint"]] + ([env.meta["secondary_joint"]] if env.meta.get("secondary_joint") else [])

    def act(self, t):
        self.markers = [p for p in (self.site(f"{lf.rsplit('_hinge', 1)[0]}_exit_device_push") for lf in self.leaves) if p is not None] or [self.first_site("leaf_edge_mid")]
        q = self.q(self.pj)
        if t < self.t_press:
            self.action = "approach"
            return
        if self.released:
            self.action = "let go (closers return)"
            return
        if q < self.release_op:
            for o in self.ops:
                self.operate(o, self.op_tau)
            self.action = "push both touch bars"
        if t >= self.t_push:
            for lf in self.leaves:
                self.servo(lf, self.target, self.kp, self.kd, self.push)
            self.action = "push both leaves" if q < self.target - 0.1 else "hold open"
        if self.t_pass is not None and t > self.t_pass + self.hold_after:
            self.released = True

    def hud_joints(self):
        return [("leaf a", self.leaves[0])] + ([("leaf b", self.leaves[1])] if len(self.leaves) > 1 else []) + [("bar a", self.ops[0])] + ([("bar b", self.ops[1])] if len(self.ops) > 1 else [])


class SlidingHand(Hand):
    """Patio slider: lift the thumb latch (raises the hook out of the keeper), then slide the leaf by the handle."""

    def __init__(self, env, push, t_press=0.7, t_push=1.6, target_m=None, op_tau=3.0, **opts):
        super().__init__(env, push, **opts)
        self.t_press, self.t_push, self.op_tau = t_press, t_push, op_tau
        self.op, self.pj = env.meta.get("operator_joint"), env.meta["primary_joint"]
        hi = float(env.m.jnt_range[env.pj][1])
        self.target = target_m or min(1.0, 0.8 * hi)
        self.kp, self.kd = 4.0 * push, 1.0 * push
        # where the robot walks: the part of the opening the leaf vacates
        leaf_body = env.m.jnt_bodyid[env.pj]
        self.x_leaf0 = float(env.d.xpos[leaf_body][0])
        self.axis_x = float(np.sign(env.m.jnt_axis[env.pj][0]) or 1.0)
        self.W = env.spec["leaf"]["width"]

    def act(self, t):
        self.markers = [self.first_site("leaf_handle_grip_n", "leaf_handle_grip_p", "leaf_ext_pull_grip_p", "leaf_edge_mid")]
        if t < self.t_press:
            self.action = "approach"
            return
        if self.op:
            self.operate(self.op, self.op_tau)
            self.action = "lift thumb latch (hook)"
        if t >= self.t_push:
            self.servo(self.pj, self.target, self.kp, self.kd, self.push)
            self.action = "slide leaf" if self.q(self.pj) < self.target - 0.05 else "hold open"

    def hud_joints(self):
        out = super().hud_joints()
        if self.env._jid("leaf_hook_hinge") >= 0:
            out.append(("hook", "leaf_hook_hinge"))
        return out

    def robot_path(self, proxy, t, dt):
        x = self.x_leaf0 - self.axis_x * self.W / 2 + self.axis_x * self.target / 2
        proxy.x = x
        return None


class GarageHand(Hand):
    """Sectional garage door: lift by the handle until it is overhead (counterbalance carries most of the weight)."""

    def __init__(self, env, push, t_push=0.8, target_m=None, **opts):
        super().__init__(env, push, **opts)
        self.t_push, self.pj = t_push, env.meta["primary_joint"]
        hi = float(env.m.jnt_range[env.pj][1])
        self.target = target_m or 0.95 * hi
        self.kp, self.kd = 2.0 * push, 0.6 * push

    def act(self, t):
        self.markers = [self.first_site("lift_handle_grip_n", "lift_handle_grip_p", "leaf_edge_mid")]
        if t < self.t_push:
            self.action = "approach"
            return
        self.servo(self.pj, self.target, self.kp, self.kd, self.push)
        self.action = "lift door" if self.q(self.pj) < self.target - 0.1 else "hold overhead"


class RevolvingHand(Hand):
    """Revolving door: push a wing at walking pace; the robot enters a compartment and is carried around."""

    def __init__(self, env, push, t_push=0.6, omega=0.9, **opts):
        super().__init__(env, push, **opts)
        self.t_push, self.omega, self.pj = t_push, omega, env.meta["primary_joint"]
        self.r = 0.5 * env.spec["kinematics"].get("drum_diameter", env.meta.get("drum_diameter", 3.0)) * 0.62
        self.theta = None

    def act(self, t):
        self.markers = [self.first_site("wing_0_push", "wing_1_push", "wing_2_push")]
        if t < self.t_push:
            self.action = "approach"
            return
        self.velocity(self.pj, self.omega, 120.0, self.push, floor=0.0)
        self.action = f"push wing ({math.degrees(self.q(self.pj)):.0f} deg turned)"

    def can_pass(self):
        return self.q(self.pj) > math.radians(25)

    def robot_path(self, proxy, t, dt):
        q = self.q(self.pj)
        if self.theta is None:
            if q > math.radians(25):
                self.theta, self.q_enter = -math.pi / 2, q
            else:
                return np.array([0.0, -1.5, proxy.z])
        th = self.theta + (q - self.q_enter)
        if th < math.pi / 2:
            return np.array([self.r * math.cos(th), self.r * math.sin(th), proxy.z])
        proxy.done = True
        if proxy.t_done is None:
            proxy.t_done = t
        return np.array([0.0, min(1.6, self.r + (t - proxy.t_done) * proxy.speed), proxy.z])


class VaultHand(Hand):
    """Vault door: spin the handwheel a full turn (retracts the boltwork), then push the 1 t leaf open."""

    def __init__(self, env, push, t_wheel=0.6, omega=2.4, target_deg=75, **opts):
        super().__init__(env, push, **opts)
        self.t_wheel, self.omega = t_wheel, omega
        self.op, self.pj = env.meta["operator_joint"], env.meta["primary_joint"]
        self.wheel_end = float(env.m.jnt_range[env.oj][1])
        self.target = math.radians(target_deg)
        self.kp, self.kd = 3.0 * push, 0.5 * push
        self.t_open = None

    def act(self, t):
        self.markers = [self.first_site("wheel_grip_n", "wheel_grip_p")]
        if t < self.t_wheel:
            self.action = "approach"
            return
        qw = self.q(self.op)
        if qw < self.wheel_end - 0.05:
            self.velocity(self.op, self.omega, 25.0, 60.0, floor=0.0)
            self.action = f"turn handwheel ({qw / self.wheel_end * 100:.0f} % of a turn)"
            return
        self.servo(self.op, self.wheel_end - 0.02, 40.0, 4.0, 30.0)     # keep the boltwork retracted
        if self.t_open is None:
            self.t_open = t + 0.4
        if t >= self.t_open:
            self.markers = [self.first_site("leaf_edge_mid", "wheel_grip_n")]
            self.servo(self.pj, self.target, self.kp, self.kd, self.push)
            self.action = "push leaf (1.08 t)" if self.q(self.pj) < self.target - 0.1 else "hold open"
        else:
            self.action = "boltwork retracted"

    def hud_joints(self):
        return [("leaf", self.pj), ("wheel", self.op)] + [(f"bolt {i}", f"bolt_{i}_slide") for i in range(2) if self.env._jid(f"bolt_{i}_slide") >= 0]


HANDS = {c.__name__: c for c in (SwingHand, KeypadHand, PairHand, SlidingHand, GarageHand, RevolvingHand, VaultHand)}


class RobotProxy:
    """A synthetic robot base that walks from approach_point through the opening once the hand says it can pass."""

    def __init__(self, env, speed=1.0, z=0.5):
        self.env, self.speed, self.z = env, speed, z
        self.x, self.y = 0.0, -1.5
        self.moving = self.done = False
        self.t_done = None

    def step(self, hand, t, dt):
        p = hand.robot_path(self, t, dt)
        if p is not None:
            return p
        if not self.moving and hand.can_pass():
            self.moving = True
        if self.moving and not self.done:
            self.y += self.speed * dt
            if self.y >= 1.3:
                self.done, self.t_done = True, t
        return np.array([self.x, self.y, self.z])


# ----------------------------------------------------------------------------------------------- rendering
def _add_geom(scene, gtype, size, pos, rgba):
    import mujoco
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def _add_arrow(scene, p0, p1, width, rgba):
    import mujoco
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(p0, float), np.asarray(p1, float))
    scene.ngeom += 1


def render_views(renderer, env, cams, hand, robot_pos):
    import mujoco
    frames = []
    for cam in cams:
        renderer.update_scene(env.d, camera=cam)
        scn = renderer.scene
        for p in hand.markers:
            if p is not None:
                _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.045, 0, 0), p, HAND_RGBA)
        if robot_pos is not None:
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, (0.13, 0.45, 0), robot_pos, ROBOT_RGBA)
        frames.append(renderer.render().copy())
    return frames


_FONTS = {}


def _font(size):
    from PIL import ImageFont
    if size not in _FONTS:
        try:
            _FONTS[size] = ImageFont.load_default(size=size)
        except TypeError:
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def _fmt_joint(env, jn):
    import mujoco
    j = env._jid(jn)
    if j < 0:
        return "-"
    q = float(env.d.qpos[env.m.jnt_qposadr[j]])
    if int(env.m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
        return f"{math.degrees(q):5.1f} deg"
    return f"{q * 1000:6.1f} mm" if abs(q) < 0.1 else f"{q:5.2f} m"


def draw_hud(width, height, env, hand, title, subtitle, t, done_labels=None):
    from PIL import Image, ImageDraw
    L = env.tracker.L
    img = Image.new("RGB", (width, height), (20, 22, 26))
    dr = ImageDraw.Draw(img)
    f_big, f, f_small = _font(17), _font(14), _font(12)
    dr.text((10, 6), title, fill=(245, 245, 245), font=f_big)
    dr.text((10 + dr.textlength(title, font=f_big) + 14, 9), subtitle, fill=(170, 175, 185), font=f)
    dr.text((width - 10 - dr.textlength(f"t = {t:5.2f} s", font=f_big), 6), f"t = {t:5.2f} s", fill=(245, 245, 245), font=f_big)
    # joint states + hand action
    parts = [f"{lab}: {_fmt_joint(env, jn)}" for lab, jn in hand.hud_joints()]
    line = "   ".join(parts)
    dr.text((10, 32), line, fill=(200, 205, 215), font=f)
    act = f"hand: {hand.action}"
    dr.text((width - 10 - dr.textlength(act, font=f), 32), act, fill=(255, 170, 70), font=f)
    # label chips
    tr = env.tracker
    chips = [("touched", L.touched_door)]
    if tr.op_joints:
        chips.append(("actuated", L.operator_actuated))
    if tr.latch_joints:
        chips.append(("unlatched", L.latch_released))
    if env.spec["lock"].get("engaged"):
        chips.append(("unlocked", L.lock_released))
    chips += [("opened", L.door_opened), ("clear", L.door_open_clear), ("traversed", L.robot_passed_through)]
    if env.spec["closer"]["model"] != "none" or hand.released:
        chips.append(("closed after", L.door_closed_after))
    chips.append(("damaged", L.door_damaged))
    x, y = 10, 56
    for name, on in chips:
        w = dr.textlength(name, font=f_small) + 16
        bad = name in ("damaged", "slammed")
        fill = ((190, 50, 50) if on else (45, 48, 55)) if bad else ((40, 150, 80) if on else (45, 48, 55))
        dr.rounded_rectangle((x, y, x + w, y + 20), radius=6, fill=fill)
        dr.text((x + 8, y + 3), name, fill=(255, 255, 255) if on else (140, 145, 155), font=f_small)
        x += w + 6
    legend = "orange sphere = hand   blue cylinder = robot base (synthetic)   labels: doorbench.benchmark.LabelTracker"
    dr.text((10, y + 27), legend, fill=(120, 125, 135), font=f_small)
    if done_labels is not None:
        verdict = f"task {L.task}: {'SUCCESS' if done_labels.success else 'not successful'}"
        dr.text((width - 10 - dr.textlength(verdict, font=f), y + 25), verdict, fill=(120, 220, 140) if done_labels.success else (240, 120, 120), font=f)
    return np.asarray(img)


# ----------------------------------------------------------------------------------------------- driver
def qa_push(door_dir, env):
    """The calibrated 'strong push' the sign-off QA used for this door (falls back to the same formula)."""
    try:
        with open(os.path.join(door_dir, "qa.json")) as f:
            p = json.load(f)["metrics"].get("qa_push")
        if p:
            return float(p)
    except Exception:
        pass
    import mujoco
    m, d = env.m, env.d
    dof = m.jnt_dofadr[env.pj]
    mujoco.mj_forward(m, d)
    bias = abs(float(d.qfrc_bias[dof] - d.qfrc_passive[dof]))
    fl = float(m.dof_frictionloss[dof])
    is_hinge = int(m.jnt_type[env.pj]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    return min(2.0 * (bias + fl) + (60.0 if is_hinge else 80.0), 800.0 if is_hinge else 4000.0)


def run_demo(door_dir, out_dir, hand_name=None, subtitle="", seconds=12.0, fps=30, view=(448, 336), cams=("robot_view", "iso"), gif=True, gif_max_mb=4.0, snapshot_dir=None, push_scale=None, **hand_opts):
    import mujoco
    import imageio.v2 as iio
    from PIL import Image
    from doorbench.benchmark import DoorEnv

    door_id = os.path.basename(os.path.normpath(door_dir))
    env = DoorEnv(door_dir, tier="full")
    env.max_steps = 10 ** 9
    env.reset()
    push = qa_push(door_dir, env) * (push_scale or 1.0)
    fam = env.spec["family"]
    if hand_name is None:
        hand_name = "PairHand" if env.meta.get("pair") else ("VaultHand" if fam == "vault" else ("RevolvingHand" if fam == "revolving" else ("GarageHand" if env.spec["kinematics"]["type"] == "slide_vertical" else ("SlidingHand" if env.spec["kinematics"]["type"].startswith("slide") else ("KeypadHand" if env.spec["lock"].get("code") else "SwingHand")))))
    hand = HANDS[hand_name](env, push, **hand_opts)
    proxy = RobotProxy(env)
    cams = [c for c in cams if mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_CAMERA, c) >= 0] or ["iso"]
    W, H = view
    renderer = mujoco.Renderer(env.m, height=H, width=W)
    dt = env.m.opt.timestep
    steps_per_frame = max(1, int(round(1.0 / (fps * dt))))
    n_frames = int(seconds * fps)
    title = door_id
    os.makedirs(out_dir, exist_ok=True)
    mp4 = os.path.join(out_dir, f"demo_{door_id}.mp4")
    writer = iio.get_writer(mp4, fps=fps, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=16, ffmpeg_log_level="error")
    gif_frames = []
    t0 = time.time()
    hud_h = 112     # 336 + 112 = 448 = 28 * 16 (x264 macro blocks)
    robot_pos = np.array([0.0, -1.5, proxy.z])
    for k in range(n_frames):
        for _ in range(steps_per_frame):
            t = env.d.time
            hand.act(t)
            robot_pos = proxy.step(hand, t, dt)
            if proxy.done and hand.t_pass is None:
                hand.t_pass = t
            env.step(robot_base_pos=robot_pos)
        views = render_views(renderer, env, cams, hand, robot_pos)
        row = np.concatenate(views, axis=1)
        final = env.labels() if k >= n_frames - fps else None
        hud = draw_hud(row.shape[1], hud_h, env, hand, title, subtitle, env.d.time, final)
        frame = np.concatenate([row, hud], axis=0)
        writer.append_data(frame)
        if gif and k % 2 == 0:
            gif_frames.append(frame)
        if snapshot_dir and k in (0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1):
            os.makedirs(snapshot_dir, exist_ok=True)
            Image.fromarray(frame).save(os.path.join(snapshot_dir, f"{door_id}_{k:04d}.png"))
    writer.close()
    renderer.close()
    labels = env.labels().to_dict()
    env.close()
    out = {"id": door_id, "mp4": mp4, "mp4_mb": os.path.getsize(mp4) / 1e6, "labels": labels, "render_s": time.time() - t0, "hand": hand_name, "push": push}
    if gif:
        gif_path = os.path.join(out_dir, f"demo_{door_id}.gif")
        write_gif(gif_frames, gif_path, fps / 2, gif_max_mb)
        out["gif"], out["gif_mb"] = gif_path, os.path.getsize(gif_path) / 1e6
    return out


def write_gif(frames, path, fps, max_mb):
    from PIL import Image
    for width, colors, stride in ((480, 128, 1), (420, 96, 1), (360, 64, 1), (320, 48, 2)):
        ims = []
        for fr in frames[::stride]:
            im = Image.fromarray(fr)
            h = int(round(im.height * width / im.width))
            ims.append(im.resize((width, h), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=colors))
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(round(1000 * stride / fps)), loop=0, optimize=True)
        if os.path.getsize(path) <= max_mb * 1e6:
            return
    print(f"warning: {path} is {os.path.getsize(path) / 1e6:.1f} MB (> {max_mb} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--out", default="docs/media")
    ap.add_argument("--ids", default="", help="comma-separated door ids (default: the 7 demo doors)")
    ap.add_argument("--seconds", type=float, default=None, help="override video length")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=448, help="width of each camera view")
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--cams", default="", help="comma-separated camera names (default robot_view,iso)")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--snapshot-dir", default="", help="also save 5 PNG frames per door here")
    ap.add_argument("--push-scale", type=float, default=None, help="multiply the QA push force")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, (sub, hand, o) in DEFAULT_DEMOS.items():
            print(f"{k:28s} {hand:14s} {sub}")
        return
    ids = [s for s in a.ids.split(",") if s] or list(DEFAULT_DEMOS)
    results = []
    for did in ids:
        sub, hand, opts = DEFAULT_DEMOS.get(did, ("", None, {}))
        opts = dict(opts)
        if a.seconds:
            opts["seconds"] = a.seconds
        if a.cams:
            opts["cams"] = tuple(a.cams.split(","))
        if a.push_scale:
            opts["push_scale"] = a.push_scale
        opts.setdefault("seconds", 12.0)
        r = run_demo(os.path.join(a.assets, "doors", did), a.out, hand_name=hand, subtitle=sub, fps=a.fps, view=(a.width, a.height), gif=not a.no_gif, snapshot_dir=a.snapshot_dir or None, **opts)
        L = r["labels"]
        flags = " ".join(k for k in ("touched_door", "operator_actuated", "latch_released", "lock_released", "door_opened", "door_open_clear", "robot_passed_through", "door_closed_after", "door_damaged", "success") if L.get(k))
        print(f"{did:28s} {r['hand']:13s} push={r['push']:7.1f}  mp4 {r['mp4_mb']:.2f} MB" + (f"  gif {r['gif_mb']:.2f} MB" if "gif_mb" in r else "") + f"  ({r['render_s']:.1f} s)  labels: {flags}")
        results.append(r)
    return results


if __name__ == "__main__":
    main()
