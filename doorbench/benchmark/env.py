"""DoorEnv: a MuJoCo environment wrapping one DoorBench door (optionally with a robot model).

Features
  * loads door.xml / door_simple.xml / door_minimal.xml by tier
  * optional robot MJCF merged into the scene (e.g. MuJoCo Menagerie humanoids) via `attach_robot`
  * asymmetric closer damping + backcheck, ratchet (one-way) rotors, maglock breakaway, keypad/REX/badge
    release logic implemented in a passive-force callback and per-step hooks
  * task presets and label tracking (see labels.py), gymnasium-style API (reset/step/labels)
  * "programmatic hand": apply wrenches at grip sites to unit-test doors without a robot

Usage
  env = DoorEnv("assets/doors/db0001_rollup", tier="full")
  env.reset()
  for _ in range(1000):
      env.apply_site_force("leaf_handle_grip_n", [0, 0, -30])   # press lever down
      env.step()
  print(env.labels().to_dict())
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .labels import LabelTracker

TASK_DESCRIPTIONS = {
    "open_and_traverse": "Approach from -y, operate hardware, open, pass through to +y",
    "open_only": "Open the door past the clearance threshold",
    "traverse_open": "Door starts open; pass through without touching it",
    "close": "Door starts open; close and latch it",
    "unlock_open_traverse": "Release the lock from the robot side, then open and pass",
    "locked_recognize": "Try, recognise the door is locked, stop without damage",
    "push_through": "Free-swinging door: push through",
    "hold_and_pass": "Self-closing door: open, hold, pass before it closes",
    "peek": "Open partially (< clearance) and hold",
}


def load_manifest(assets_root: str) -> dict:
    with open(os.path.join(assets_root, "manifest.json")) as f:
        return json.load(f)


class DoorEnv:
    def __init__(self, door_dir: str, tier: str = "full", robot_xml: str | None = None, robot_body_prefix: str = "", robot_base_body: str | None = None, timestep: float | None = None, seed: int = 0):
        import mujoco
        self.mj = mujoco
        self.door_dir = door_dir
        self.tier = tier
        with open(os.path.join(door_dir, "spec.json")) as f:
            self.spec = json.load(f)
        with open(os.path.join(door_dir, "model.json")) as f:
            self.model_json = json.load(f)
        self.meta = self.model_json["meta"]
        xml = {"full": "door.xml", "simple": "door_simple.xml", "minimal": "door_minimal.xml"}[tier]
        self.xml_path = os.path.join(door_dir, xml)
        if robot_xml:
            self.m = self._merge_with_robot(self.xml_path, robot_xml)
        else:
            self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        if timestep:
            self.m.opt.timestep = timestep
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.robot_prefix = robot_body_prefix
        self.robot_base = robot_base_body
        robot_bodies = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.m.nbody) if robot_body_prefix and (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(robot_body_prefix)]
        self.robot_bodies = robot_bodies
        self.pj = self._jid(self.meta.get("primary_joint"))
        self.oj = self._jid(self.meta.get("operator_joint")) if self.meta.get("operator_joint") else -1
        # joint names by IR role (present in this tier): operators (both leaves of a pair) and lock parts
        self.operator_joints = [j["name"] for b in self.model_json["bodies"] if (j := b.get("joint")) and j.get("role") == "operator" and self._jid(j["name"]) >= 0]
        self.lock_joints = [j["name"] for b in self.model_json["bodies"] if (j := b.get("joint")) and j.get("role") == "lock" and self._jid(j["name"]) >= 0]
        self.latch_joints = [j["name"] for b in self.model_json["bodies"] if (j := b.get("joint")) and j.get("role") == "latch" and self._jid(j["name"]) >= 0]
        self._install_passive_callback()
        self.tracker = None
        self.task = self.spec.get("task", "open_and_traverse")
        self.max_steps = 4000
        self._breakable = {w["name"]: w for w in self.meta.get("breakable_welds", [])}
        self.unlocked_by_env = False

    # ------------------------------------------------------------------
    def _jid(self, name):
        if not name:
            return -1
        return self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, name)

    def _merge_with_robot(self, door_xml, robot_xml):
        """Attach a robot MJCF (its own file with meshes) to the door scene using MjSpec."""
        mujoco = self.mj
        spec = mujoco.MjSpec.from_file(door_xml)
        robot = mujoco.MjSpec.from_file(robot_xml)
        site = spec.worldbody.add_site(name="robot_attach", pos=[0.0, -1.5, 0.0])
        frame = spec.worldbody.add_frame(pos=[0.0, -1.5, 0.0])
        try:
            spec.attach(robot, prefix=self.robot_prefix or "robot/", frame=frame)
        except TypeError:
            frame.attach_body(robot.worldbody, self.robot_prefix or "robot/", "")
        return spec.compile()

    def _install_passive_callback(self):
        """Asymmetric closer damping, backcheck, ratchets, actuator-free hold-open."""
        mujoco = self.mj
        m = self.m
        rules = []
        for b in self.model_json["bodies"]:
            j = b.get("joint")
            if not j:
                continue
            jid = self._jid(j["name"])
            if jid < 0:
                continue
            dof = m.jnt_dofadr[jid]
            if j.get("damping_closing") is not None and j.get("damping_opening") is not None and j["damping_closing"] > 0:
                rules.append(("closer", dof, float(j["damping_closing"]), float(j["damping_opening"]), float(m.dof_damping[dof]), j.get("backcheck_angle"), j.get("backcheck_damping") or 0.0))
            if j.get("ratchet_one_way"):
                rules.append(("ratchet", dof, 0.0, 0.0, 0.0, None, 0.0))
        self._rules = rules
        pet_magnet = self.spec.get("kinematics", {}).get("magnet_force_N", 0.0)
        if pet_magnet and self.pj >= 0:
            rules.append(("magnet", m.jnt_dofadr[self.pj], float(pet_magnet), 0.0, 0.0, None, 0.0))

        def cb(model, data):
            for kind, dof, b_close, b_open, b_base, bc_ang, bc_damp in self._rules:
                v = data.qvel[dof]
                q = data.qpos[model.jnt_qposadr[model.dof_jntid[dof]]]
                if kind == "closer":
                    # model damping already applies b_base symmetric; add the difference for the active direction
                    b_target = b_close if v < 0 else b_open
                    if bc_ang is not None and v > 0 and q > bc_ang:
                        b_target += bc_damp
                    data.qfrc_passive[dof] += -(b_target - b_base) * v
                elif kind == "ratchet":
                    if v < 0:
                        data.qfrc_passive[dof] += -200.0 * v - 50.0 * min(0.0, q - math.floor(q / 1e-9 + 0) * 0) * 0
                elif kind == "magnet":
                    # detent near closed: pet flap magnet strip ~ F * arm within +-3 deg
                    arm = self.spec["leaf"]["height"]
                    if abs(q) < math.radians(3):
                        data.qfrc_passive[dof] += -math.copysign(b_close * arm, q) * (1 - abs(q) / math.radians(3))
        self._cb = cb
        mujoco.set_mjcb_passive(cb)

    # ------------------------------------------------------------------
    def reset(self, task: str | None = None, randomize: bool = False):
        mujoco = self.mj
        mujoco.mj_resetData(self.m, self.d)
        # initial joint values from the IR (e.g. retracted deadbolts, rest angles)
        for b in self.model_json["bodies"]:
            j = b.get("joint")
            if j and j.get("initial"):
                jid = self._jid(j["name"])
                if jid >= 0:
                    self.d.qpos[self.m.jnt_qposadr[jid]] = j["initial"]
        self.task = task or self.task
        if self.task in ("traverse_open", "close") and self.pj >= 0 and self.m.jnt_limited[self.pj]:
            lo, hi = self.m.jnt_range[self.pj]
            self.d.qpos[self.m.jnt_qposadr[self.pj]] = lo + 0.8 * (hi - lo)
            bj = self._jid("leaf_latch_bolt_slide")
            if bj >= 0:
                self.d.qpos[self.m.jnt_qposadr[bj]] = 0.0
        if randomize:
            self._domain_randomize()
        mujoco.mj_forward(self.m, self.d)
        self.tracker = LabelTracker(self.m, self.spec, self.meta, self.robot_bodies, self.robot_base, operator_joints=self.operator_joints, lock_joints=self.lock_joints, latch_joints=self.latch_joints)
        self.tracker.L.task = self.task
        self.unlocked_by_env = False
        self._t0 = self.d.time
        return self.observation()

    def _domain_randomize(self):
        m = self.m
        if self.pj >= 0:
            dof = m.jnt_dofadr[self.pj]
            m.dof_frictionloss[dof] *= self.rng.uniform(0.5, 1.8)
            m.dof_damping[dof] *= self.rng.uniform(0.7, 1.4)
            if m.jnt_stiffness[self.pj] > 0:
                m.jnt_stiffness[self.pj] *= self.rng.uniform(0.85, 1.2)
        for b in range(1, m.nbody):
            m.body_mass[b] *= self.rng.uniform(0.9, 1.1)

    def observation(self):
        d, m = self.d, self.m
        obs = {"time": float(d.time), "door_q": float(d.qpos[m.jnt_qposadr[self.pj]]) if self.pj >= 0 else 0.0,
               "door_dq": float(d.qvel[m.jnt_dofadr[self.pj]]) if self.pj >= 0 else 0.0,
               "operator_q": float(d.qpos[m.jnt_qposadr[self.oj]]) if self.oj >= 0 else None,
               "joint_q": {mujoco_name: float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt) if (mujoco_name := self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_JOINT, j))}}
        return obs

    def step(self, ctrl=None, robot_base_pos=None):
        if ctrl is not None and self.m.nu:
            self.d.ctrl[:] = ctrl
        self._lock_logic()
        self.mj.mj_step(self.m, self.d)
        self.d.qfrc_applied[:] = 0
        self.d.xfrc_applied[:] = 0
        if robot_base_pos is None and self.robot_base:
            bid = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, self.robot_base)
            robot_base_pos = self.d.xpos[bid].copy() if bid >= 0 else None
        self.tracker.step(self.d, robot_base_pos)
        done = self.tracker.L.steps >= self.max_steps
        return self.observation(), done

    # --- lock / access-control logic ------------------------------------
    def _lock_logic(self):
        """Maglock breakaway, delayed egress timer, REX/keypad/badge release (changes joint ranges & welds)."""
        m, d, mujoco = self.m, self.d, self.mj
        L = self.tracker.L if self.tracker else None
        lock = self.spec.get("lock", {})
        # maglock: release when REX pressed / code entered / badge(); break when constraint force exceeds holding force
        for name, w in self._breakable.items():
            eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, name)
            if eid < 0 or not d.eq_active[eid]:
                continue
            release = self.unlocked_by_env or (L is not None and L.lock_released)
            if lock.get("model") == "delayed_egress" and self.oj >= 0:
                # 3 s sustained push on the bar -> release 15 s later (IBC 1010.1.9.7 simplified: 15 s after initiation)
                q = d.qpos[m.jnt_qposadr[self.oj]]
                if q > 0.5 * m.jnt_range[self.oj][1]:
                    self._delay_t = getattr(self, "_delay_t", None) or d.time
                if getattr(self, "_delay_t", None) and d.time - self._delay_t > 15.0:
                    release = True
            if release:
                d.eq_active[eid] = 0
                continue
            # breakaway
            for i in range(d.nefc):
                if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[i] == eid:
                    if abs(d.efc_force[i]) > w["holding_force_N"]:
                        d.eq_active[eid] = 0
                        if L:
                            L.door_damaged = True
                            L.damage_events.append({"t": float(d.time), "kind": "maglock_forced", "part": name, "value": float(abs(d.efc_force[i])), "threshold": w["holding_force_N"]})
                        break
        # keypad / card / electric strike: restore operator range when released
        if L is not None and (L.lock_released or self.unlocked_by_env) and self.oj >= 0 and lock.get("engaged") and lock.get("model") in ("keypad_code_4", "keypad_code_6", "keypad_mechanical", "card_reader", "electric_strike", "electric_bolt", "privacy_button", "keyed_cylinder"):
            for b in self.model_json["bodies"]:
                j = b.get("joint")
                if j and j["name"] == self.meta.get("operator_joint"):
                    full = self._operator_full_travel()
                    if full and m.jnt_range[self.oj][1] < full - 1e-6:
                        m.jnt_range[self.oj][1] = full
        # turnstile credential release
        if self.meta.get("locked") and self.pj >= 0 and (self.unlocked_by_env or (L is not None and L.lock_released)):
            m.jnt_limited[self.pj] = 0

    def _operator_full_travel(self):
        from .. import hardware as H
        op = H.OPERATORS.get(self.spec["operator"]["model"])
        return op.travel if op else None

    def badge(self):
        """Present a valid credential (card reader / turnstile / maglock)."""
        self.unlocked_by_env = True
        if self.tracker:
            self.tracker.L.lock_released = True
            self.tracker.L.notes.append(f"badge presented at t={self.d.time:.2f}")

    # --- programmatic hand -------------------------------------------------
    # Without a robot model there are no robot geoms, so the tracker cannot see contacts: applying a wrench with
    # these helpers counts as the hand touching the door (grip / push sites and operator joints -> the operator).
    def apply_site_force(self, site_name: str, force_xyz, torque_xyz=(0, 0, 0)):
        sid = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            raise KeyError(site_name)
        bid = self.m.site_bodyid[sid]
        pos = self.d.site_xpos[sid]
        self.mj.mj_applyFT(self.m, self.d, np.asarray(force_xyz, float), np.asarray(torque_xyz, float), pos, bid, self.d.qfrc_applied)
        if self.tracker:
            self.tracker.mark_touch(self.d, operator=("grip" in site_name or "push" in site_name))

    def apply_joint_torque(self, joint_name: str, tau: float):
        j = self._jid(joint_name)
        if j < 0:
            raise KeyError(joint_name)
        self.d.qfrc_applied[self.m.jnt_dofadr[j]] += tau
        if self.tracker and tau:
            self.tracker.mark_touch(self.d, operator=joint_name in self.operator_joints)

    def close(self):
        """Remove the global passive-force callback (it references this env's model); call when done with the env."""
        if getattr(self, "_cb", None) is not None:
            self.mj.set_mjcb_passive(None)
            self._cb = None

    def grip_sites(self):
        return [self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_SITE, i) for i in range(self.m.nsite) if "grip" in (self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_SITE, i) or "") or "push" in (self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_SITE, i) or "")]

    def labels(self):
        self.tracker.mark_closed(self.d)
        return self.tracker.finalize()

    def render(self, camera="iso", width=640, height=480):
        r = self.mj.Renderer(self.m, height=height, width=width)
        r.update_scene(self.d, camera=camera)
        img = r.render()
        r.close()
        return img
