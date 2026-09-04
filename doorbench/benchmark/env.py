"""DoorEnv: a MuJoCo environment wrapping one DoorBench door (optionally with a robot model).

Features
  * loads door.xml / door_simple.xml / door_minimal.xml by tier (through MjSpec so bodies can be added)
  * optional robot MJCF merged into the scene (e.g. MuJoCo Menagerie humanoids) via `robot_xml`
  * asymmetric closer damping + backcheck, ratchet (one-way) rotors, maglock breakaway, keypad/REX/badge
    release logic implemented in a passive-force callback and per-step hooks
  * benchmark scenarios (spec.json["benchmark"], see scenarios.py): start-zone sampling, a kinematic simulated
    human for the `hold_open_for_human` / `wait_for_human` scenarios, per-step reward from the scenario's reward
    table, scenario success criteria, time budget
  * label tracking (see labels.py), gymnasium-style API (reset/step/labels)
  * "programmatic hand": apply wrenches at grip sites to unit-test doors without a robot

Usage
  env = DoorEnv("assets/doors/db0001_rollup", tier="full")
  env.reset(scenario="open_and_traverse", seed=3)
  for _ in range(1000):
      env.apply_site_force("leaf_handle_grip_n", [0, 0, -30])   # press lever down
      obs, done = env.step()
      r = env.reward()                                          # reward of the last step
  print(env.labels().to_dict(), env.episode_return, env.success)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .labels import LabelTracker
from .scenarios import SCENARIO_TYPES, build_benchmark, make_scenario, sample_start, human_pose

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

# legacy task -> scenario type whose reward table is used when reset(task=...) is called
LEGACY_TASK_SCENARIO = {"open_and_traverse": "open_and_traverse", "open_only": "open_and_traverse", "traverse_open": "open_and_traverse",
                        "close": "close_only", "unlock_open_traverse": "unlock_and_traverse", "locked_recognize": "locked_recognize",
                        "push_through": "open_and_traverse", "hold_and_pass": "open_and_traverse", "peek": "open_and_traverse"}

# success-criterion names that are labels rather than reward events
LABEL_ALIASES = {"damage": "door_damaged", "slam": "door_slammed", "traversed": "robot_passed_through", "hardware_misuse": "hardware_misuse",
                 "unlock": "lock_released", "unlatch": "latch_released", "touch_handle": "touched_operator"}


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
        self.robot_xml = robot_xml
        self.robot_prefix = robot_body_prefix or ("robot/" if robot_xml else "")
        self.robot_base = robot_base_body
        self.timestep = timestep
        self.rng = np.random.default_rng(seed)
        self.benchmark = self.spec.get("benchmark") or build_benchmark(self.spec, self.spec.get("physics", {}), self.model_json)
        self.scenario_names = [s["name"] for s in self.benchmark["scenarios"]]
        self._human_enabled = False
        self._human = None
        self.m, self.d = self._build(with_human=False)
        self._rebind()
        self.tracker = None
        self.task = self.spec.get("task", "open_and_traverse")
        self._scenario = None
        self._legacy_task = None
        self.max_steps = 4000
        self.unlocked_by_env = False
        self._fired = {}
        self._last_reward = 0.0
        self.episode_return = 0.0
        self.events = []
        self._done = False
        self.start_pose = None

    # ------------------------------------------------------------------
    def _build(self, with_human: bool):
        """Compile the door (+ robot, + human capsule) from MjSpec."""
        mujoco = self.mj
        spec = mujoco.MjSpec.from_file(self.xml_path)
        if self.robot_xml:
            robot = mujoco.MjSpec.from_file(self.robot_xml)
            spec.worldbody.add_site(name="robot_attach", pos=[0.0, -1.5, 0.0])
            frame = spec.worldbody.add_frame(pos=[0.0, -1.5, 0.0])
            try:
                spec.attach(robot, prefix=self.robot_prefix, frame=frame)
            except TypeError:
                frame.attach_body(robot.worldbody, self.robot_prefix, "")
        if with_human:
            hb = self.benchmark.get("human", {"radius_m": 0.22, "height_m": 1.75})
            r, h = float(hb["radius_m"]), float(hb["height_m"])
            body = spec.worldbody.add_body(name="human", mocap=True, pos=[0.0, -50.0, h / 2])
            body.add_geom(name="human_capsule", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[r, max(0.05, h / 2 - r), 0.0], rgba=[0.95, 0.55, 0.3, 0.9], group=0)
        m = spec.compile()
        if self.timestep:
            m.opt.timestep = self.timestep
        return m, mujoco.MjData(m)

    def _rebind(self):
        """Recompute ids and reinstall the passive callback after (re)compiling the model."""
        mujoco, m = self.mj, self.m
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
        self.robot_bodies = [n for n in names if self.robot_prefix and n.startswith(self.robot_prefix)]
        self.pj = self._jid(self.meta.get("primary_joint"))
        self.oj = self._jid(self.meta.get("operator_joint")) if self.meta.get("operator_joint") else -1
        self.sj = self._jid(self.meta.get("secondary_joint")) if self.meta.get("secondary_joint") else -1
        self.bj = self._jid("leaf_latch_bolt_slide")
        if self.bj < 0:
            for j in range(m.njnt):
                n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                if n.endswith("latch_bolt_slide"):
                    self.bj = j
                    break
        self._breakable = {w["name"]: w for w in self.meta.get("breakable_welds", [])}
        hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "human")
        self._human_mocap = int(m.body_mocapid[hid]) if hid >= 0 else -1
        self._human_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == hid} if hid >= 0 else set()
        self._robot_free_joint = -1
        for j in range(m.njnt):
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j]) or "") in self.robot_bodies:
                self._robot_free_joint = j
                break
        self._install_passive_callback()

    def _jid(self, name):
        if not name:
            return -1
        return self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, name)

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

        sig = (int(m.nv), int(m.nbody), int(m.ngeom), int(m.njnt))

        def cb(model, data):
            # the callback is global to the process: ignore models that are not ours (other envs, plain mujoco use)
            if (int(model.nv), int(model.nbody), int(model.ngeom), int(model.njnt)) != sig:
                return
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
                        data.qfrc_passive[dof] += -200.0 * v
                elif kind == "magnet":
                    # detent near closed: pet flap magnet strip ~ F * arm within +-3 deg
                    arm = self.spec["leaf"]["height"]
                    if abs(q) < math.radians(3):
                        data.qfrc_passive[dof] += -math.copysign(b_close * arm, q) * (1 - abs(q) / math.radians(3))
        self._cb = cb      # installed only around mj_step / mj_forward (see _with_passive): a permanently installed
                           # global callback breaks MjModel.from_xml_path / MjSpec.compile elsewhere in the process

    def _with_passive(self, fn):
        mujoco = self.mj
        mujoco.set_mjcb_passive(self._cb)
        try:
            return fn()
        finally:
            mujoco.set_mjcb_passive(None)

    # ------------------------------------------------------------------
    # scenarios
    # ------------------------------------------------------------------
    def scenario(self, name: str | None = None, build: bool = True) -> dict:
        """The door's scenario of that type (or the current / primary one).  With build=True a scenario type the door
        does not list is constructed on the fly, so any scenario can be run on any door."""
        if name is None:
            return self._scenario or self.benchmark["scenarios"][0]
        if isinstance(name, dict):
            return name
        for s in self.benchmark["scenarios"]:
            if s["name"] == name:
                return s
        if build and name in SCENARIO_TYPES:
            return make_scenario(name, self.spec, self.spec.get("physics", {}), self.model_json)
        raise KeyError(f"{name}: not one of this door's scenarios {self.scenario_names}")

    def reset(self, scenario: str | dict | None = None, seed: int | None = None, randomize: bool = False, task: str | None = None):
        mujoco = self.mj
        self._legacy_task = None
        if scenario is None and task is not None and task not in SCENARIO_TYPES:
            self._legacy_task = task
            scenario = LEGACY_TASK_SCENARIO.get(task, "open_and_traverse")
        elif scenario is None and task is not None:
            scenario = task
        sc = self.scenario(scenario)
        self._scenario = sc
        self.task = self._legacy_task or sc["name"]
        needs_human = bool(sc.get("human"))
        if needs_human and not self._human_enabled:
            self.m, self.d = self._build(with_human=True)
            self._human_enabled = True
            self._rebind()
        mujoco.mj_resetData(self.m, self.d)
        # initial joint values from the IR (e.g. retracted deadbolts, rest angles)
        for b in self.model_json["bodies"]:
            j = b.get("joint")
            if j and j.get("initial"):
                jid = self._jid(j["name"])
                if jid >= 0:
                    self.d.qpos[self.m.jnt_qposadr[jid]] = j["initial"]
        starts_open = sc["initial_state"].get("door") == "open" or self._legacy_task in ("traverse_open", "close")
        if starts_open and self.pj >= 0:
            if self.m.jnt_limited[self.pj]:
                lo, hi = self.m.jnt_range[self.pj]
                self.d.qpos[self.m.jnt_qposadr[self.pj]] = lo + 0.8 * (hi - lo)
            else:
                self.d.qpos[self.m.jnt_qposadr[self.pj]] = 1.2
            if self.bj >= 0:
                self.d.qpos[self.m.jnt_qposadr[self.bj]] = 0.0
        self._was_open = starts_open
        if randomize:
            self._domain_randomize()
        # robot start pose (sampled from the start zone; deterministic in seed)
        if seed is None:
            seed = int(self.rng.integers(0, 2 ** 31 - 1))
        self.start_pose = sample_start(sc, seed)
        self._place_robot(self.start_pose)
        # human
        self._human = None
        if needs_human and self._human_mocap >= 0:
            h = dict(sc["human"])
            h.update({"wait": 0.0, "prev_xy": None, "crossed_t": None, "collided": False, "done": False, "clear_at_crossing": None, "held_fired": False})
            self._human = h
            x, y = human_pose(h, 0.0)
            self.d.mocap_pos[self._human_mocap] = [x, y, h["height_m"] / 2]
        elif self._human_mocap >= 0:
            self.d.mocap_pos[self._human_mocap] = [0.0, -50.0, 1.0]
        self._with_passive(lambda: mujoco.mj_forward(self.m, self.d))
        self.tracker = LabelTracker(self.m, self.spec, self.meta, self.robot_bodies, self.robot_base)
        self.tracker.L.task = self.task
        self.unlocked_by_env = False
        self._t0 = self.d.time
        self._fired = {}
        self.events = []
        self._last_reward = 0.0
        self.episode_return = 0.0
        self._done = False
        self._knock_t = None
        self._delay_t = None
        self._robot_base_pos = None
        self._env_driving = False
        self.max_steps = int(math.ceil(sc["time_budget_s"] / self.m.opt.timestep))
        return self.observation()

    def _place_robot(self, start):
        """Put an attached robot with a free root joint at the sampled start pose (x, y, yaw)."""
        j = self._robot_free_joint
        if j < 0:
            return
        adr = self.m.jnt_qposadr[j]
        z = float(self.m.qpos0[adr + 2])
        self.d.qpos[adr:adr + 3] = [start["xy"][0], start["xy"][1], z]
        half = 0.5 * start["yaw"]
        self.d.qpos[adr + 3:adr + 7] = [math.cos(half), 0.0, 0.0, math.sin(half)]

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
               "joint_q": {mujoco_name: float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt) if (mujoco_name := self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_JOINT, j))},
               "scenario": self._scenario["name"] if self._scenario else None, "start": self.start_pose}
        if self._human is not None and self._human_mocap >= 0:
            p = d.mocap_pos[self._human_mocap]
            obs["human_xy"] = [float(p[0]), float(p[1])]
        return obs

    def step(self, ctrl=None, robot_base_pos=None):
        if ctrl is not None and self.m.nu:
            self.d.ctrl[:] = ctrl
        self._lock_logic()
        if self._human is not None:
            self._human_motion()
        self._with_passive(lambda: self.mj.mj_step(self.m, self.d))
        self.d.qfrc_applied[:] = 0
        self.d.xfrc_applied[:] = 0
        if robot_base_pos is None and self.robot_base:
            bid = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, self.robot_base)
            robot_base_pos = self.d.xpos[bid].copy() if bid >= 0 else None
        self._robot_base_pos = None if robot_base_pos is None else np.asarray(robot_base_pos, float)
        self.tracker.step(self.d, robot_base_pos)
        self._last_reward = 0.0
        if self._human is not None:
            self._human_events()
        self._update_rewards()
        sc = self._scenario
        done = self.tracker.L.steps >= self.max_steps or self._done or (sc is not None and self.d.time - self._t0 >= sc["time_budget_s"])
        return self.observation(), done

    # --- rewards ---------------------------------------------------------
    def _fire(self, name):
        sc = self._scenario
        if sc is None or name in self._fired or name not in sc["rewards"]:
            return
        t = float(self.d.time - self._t0)
        self._fired[name] = t
        v = float(sc["rewards"][name])
        self.events.append({"t": t, "event": name, "reward": v})
        self._last_reward += v

    def _door_q(self):
        return float(self.d.qpos[self.m.jnt_qposadr[self.pj]]) if self.pj >= 0 else 0.0

    def _door_clear_now(self):
        thr = self._scenario["thresholds"]
        q = abs(self._door_q())
        if thr.get("clear_rad") is not None:
            return q >= thr["clear_rad"] - 1e-6
        return q >= (thr.get("clear_m") or 0.5) - 1e-6

    def _door_opened_now(self):
        thr = self._scenario["thresholds"]
        q = abs(self._door_q())
        return q >= (thr["open_rad"] if thr.get("open_rad") is not None else (thr.get("open_m") or 0.3))

    def _bolt_extended(self):
        if self.bj < 0:
            return True
        throw = self.m.jnt_range[self.bj][1]
        return throw <= 0 or self.d.qpos[self.m.jnt_qposadr[self.bj]] < 0.2 * throw

    def _update_rewards(self):
        sc, L, m, d = self._scenario, self.tracker.L, self.m, self.d
        if sc is None:
            return
        dt = m.opt.timestep
        env_driving = getattr(self, "_env_driving", False)     # wait_for_human: the person is working the door, not the robot
        if L.touched_operator:
            self._fire("touch_handle")
        elif self.oj >= 0 and not env_driving:
            lo, hi = m.jnt_range[self.oj]
            if hi - lo > 1e-6 and (d.qpos[m.jnt_qposadr[self.oj]] - lo) > 0.1 * (hi - lo):
                self._fire("touch_handle")
        if self.bj >= 0 and not env_driving and "unlatch" not in self._fired:
            # bolt retracted by the operator (not pushed in by the strike lip while closing)
            throw = m.jnt_range[self.bj][1]
            op_ok = self.oj < 0 or (m.jnt_range[self.oj][1] - m.jnt_range[self.oj][0] < 1e-6) or (d.qpos[m.jnt_qposadr[self.oj]] - m.jnt_range[self.oj][0]) >= 0.5 * (m.jnt_range[self.oj][1] - m.jnt_range[self.oj][0])
            if throw > 0 and d.qpos[m.jnt_qposadr[self.bj]] >= 0.8 * throw and op_ok:
                self._fire("unlatch")
        if L.lock_released:
            self._fire("unlock")
        opened = self._door_opened_now()
        if opened and not self._was_open and not env_driving:
            self._fire("opened")
            if self._knock_t is not None and "waited" in sc["rewards"] and "waited" not in self._fired and (d.time - self._knock_t) >= 3.0:
                self._fire("waited")
        if L.robot_passed_through:
            self._fire("traversed")
        closed_now = abs(self._door_q()) < self.tracker.closed_thr
        shut = abs(self._door_q()) < (math.radians(1.0) if self.tracker.is_hinge else 0.01)   # bolt can only drop into the keeper when fully shut
        if L.door_closed_after:
            self._fire("closed_behind")
            if shut and self._bolt_extended():
                self._fire("latched_behind")
        if self._was_open and closed_now:
            self._fire("closed")
            if shut and self._bolt_extended():
                self._fire("latched")
        if L.door_damaged:
            self._fire("damage")
        if L.door_slammed:
            self._fire("slam")
        if L.hardware_misuse:
            self._fire("hardware_misuse")
        # knock: robot leaf contact in [5 N, dent threshold) while closed
        f = getattr(self.tracker, "step_leaf_force", 0.0)
        if "knocked" in sc["rewards"] and self._knock_t is None and closed_now and 5.0 <= f < (self.tracker.damage.get("leaf_dent_force_N") or 1e9):
            self.knock()
        pen = sc["rewards"].get("time_penalty_per_s", 0.0) * dt
        self._last_reward += pen
        self.episode_return += self._last_reward

    def reward(self) -> float:
        """Reward of the last step (event rewards that fired + the time penalty)."""
        return self._last_reward

    def _flag(self, crit: str) -> bool:
        neg = crit.startswith("!")
        name = crit.lstrip("!")
        v = name in self._fired
        if not v and self.tracker is not None:
            alias = LABEL_ALIASES.get(name, name)
            v = bool(getattr(self.tracker.L, alias, False))
            if name == "opened" and not v:
                v = self._door_opened_now() and not self._was_open
        return (not v) if neg else v

    @property
    def success(self) -> bool:
        if self._scenario is None or self.tracker is None:
            return False
        return all(self._flag(c) for c in self._scenario["success"])

    def declare_locked(self):
        """The policy declares the door locked (locked_recognize).  Ends the episode."""
        if self._scenario is not None and abs(self._door_q()) < self.tracker.closed_thr and not self.tracker.L.door_damaged:
            self._fire("recognized_locked")
        self.tracker.L.notes.append(f"declared locked at t={self.d.time:.2f}")
        self._done = True

    def knock(self):
        """Register a knock on the closed leaf (fires automatically from robot contacts; callable for programmatic hands)."""
        if self._knock_t is None and abs(self._door_q()) < self.tracker.closed_thr:
            self._knock_t = float(self.d.time)
            self._fire("knocked")
            self.tracker.L.notes.append(f"knocked at t={self.d.time:.2f}")

    # --- simulated human ---------------------------------------------------
    def _human_motion(self):
        """Advance the kinematic human along its path; pause before a closed door; open the door for a human coming
        from the far side (wait_for_human)."""
        h, m, d = self._human, self.m, self.d
        sc = self._scenario
        dt = m.opt.timestep
        t_path = (d.time - self._t0) - h["wait"]
        x, y = human_pose(h, t_path)
        plane = sc["pass_plane"]["center"]
        tdir = sc["pass_plane"]["traverse_direction"]
        # signed distance along the human's own travel direction (before the plane < 0)
        sgn = 1.0 if h["direction"] == "same_as_robot" else -1.0
        s_now = sgn * ((x - plane[0]) * tdir[0] + (y - plane[1]) * tdir[1])
        if h.get("waits_at_closed_door") and -0.7 <= s_now < 0.0 and not self._door_clear_now():
            h["wait"] += dt
            x, y = human_pose(h, t_path - dt)
        self._env_driving = False
        if h["direction"] == "opposite_to_robot" and self.pj >= 0:
            # the person opens the door (servo on the leaf + operator) from 1.2 m before to 0.8 m past the plane, then
            # closes it behind them (0.8 - 1.8 m past); door events fired while the environment drives the door are
            # not rewarded
            thr = sc["thresholds"]
            hinge = int(m.jnt_type[self.pj]) == int(self.mj.mjtJoint.mjJNT_HINGE)
            dof = m.jnt_dofadr[self.pj]
            q, v = d.qpos[m.jnt_qposadr[self.pj]], d.qvel[dof]
            target = None
            if -1.2 <= s_now <= 0.8:
                target = (thr["clear_rad"] + 0.2) if thr.get("clear_rad") is not None else (thr.get("clear_m") or 0.5) + 0.1
                if m.jnt_limited[self.pj]:
                    target = min(target, m.jnt_range[self.pj][1] - 0.02)
                kp, kv, fmax = (150.0, 25.0, 80.0) if hinge else (400.0, 60.0, 150.0)
            elif s_now > 0.8 and not h.get("closed_behind"):
                h.setdefault("close_t0", float(d.time))
                if abs(q) <= (0.5 * math.radians(1.0) if hinge else 0.005) or d.time - h["close_t0"] > 3.0:
                    h["closed_behind"] = True          # shut (or gave up after 3 s): let go of the door
                else:
                    target = 0.0
                    kp, kv, fmax = (30.0, 8.0, 20.0) if hinge else (150.0, 40.0, 60.0)
            if target is not None:
                self._env_driving = True
                d.qfrc_applied[dof] += float(np.clip(kp * (target - q) - kv * v, -fmax, fmax))
                if self.oj >= 0 and m.jnt_limited[self.oj] and -1.2 <= s_now <= 0.3:
                    # a soft hand on the operator (a stiff servo would slam it into its stop = operator overload)
                    odof = m.jnt_dofadr[self.oj]
                    oq, ov = d.qpos[m.jnt_qposadr[self.oj]], d.qvel[odof]
                    ohi = 0.9 * m.jnt_range[self.oj][1]
                    ohinge = int(m.jnt_type[self.oj]) == int(self.mj.mjtJoint.mjJNT_HINGE)
                    okp, okv, ofmax = (6.0, 0.2, 2.5) if ohinge else (300.0, 10.0, 60.0)
                    d.qfrc_applied[odof] += float(np.clip(okp * (ohi - oq) - okv * ov, -ofmax, ofmax))
        d.mocap_pos[self._human_mocap] = [x, y, h["height_m"] / 2]
        h["prev_s"] = h.get("s", None)
        h["s"] = s_now
        h["xy"] = (x, y)
        h["t_path"] = t_path

    def _human_events(self):
        h, m, d, L = self._human, self.m, self.d, self.tracker.L
        sc = self._scenario
        # collision: robot geoms vs the human capsule, or base within the two radii
        collided = False
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in self._human_geoms and c.geom2 in self.tracker.robot_geoms) or (c.geom2 in self._human_geoms and c.geom1 in self.tracker.robot_geoms):
                collided = True
                break
        if not collided and self._robot_base_pos is not None:
            rb = self.benchmark.get("robot", {}).get("body_radius_m", 0.3)
            if math.hypot(self._robot_base_pos[0] - h["xy"][0], self._robot_base_pos[1] - h["xy"][1]) < rb + h["radius_m"]:
                collided = True
        if collided and not h["collided"]:
            h["collided"] = True
            self._fire("collision_with_human")
        # plane crossing
        if h.get("prev_s") is not None and h["prev_s"] < 0.0 <= h["s"] and h["crossed_t"] is None:
            h["crossed_t"] = float(d.time - self._t0)
            h["clear_at_crossing"] = self._door_clear_now()
            h["robot_crossed_before"] = bool(L.robot_passed_through)
        if h["crossed_t"] is not None and h["direction"] == "same_as_robot" and not h["held_fired"] and h["clear_at_crossing"] and not h["collided"]:
            # held until the person is fully through: body centre one radius past the plane
            if h["s"] >= h["radius_m"] + 0.05 and self._door_clear_now():
                h["held_fired"] = True
                self._fire("held_for_human")
        if not h["done"] and h["t_path"] >= h["path"][-1][0]:
            h["done"] = True
            if h["direction"] == "opposite_to_robot" and not h["collided"] and not h.get("robot_crossed_before", False):
                self._fire("yielded_to_human")

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
    def apply_site_force(self, site_name: str, force_xyz, torque_xyz=(0, 0, 0)):
        sid = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            raise KeyError(site_name)
        bid = self.m.site_bodyid[sid]
        pos = self.d.site_xpos[sid]
        self.mj.mj_applyFT(self.m, self.d, np.asarray(force_xyz, float), np.asarray(torque_xyz, float), pos, bid, self.d.qfrc_applied)

    def apply_joint_torque(self, joint_name: str, tau: float):
        j = self._jid(joint_name)
        self.d.qfrc_applied[self.m.jnt_dofadr[j]] += tau

    def grip_sites(self):
        return [self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_SITE, i) for i in range(self.m.nsite) if "grip" in (self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_SITE, i) or "") or "push" in (self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_SITE, i) or "")]

    def labels(self):
        self.tracker.mark_closed(self.d)
        L = self.tracker.finalize()
        if self._scenario is not None and self._legacy_task is None:
            L.success = self.success
        L.notes = list(L.notes)
        L.reward_events = list(self.events)
        L.episode_return = float(self.episode_return)
        return L

    def close(self):
        """Nothing to release: the passive callback is only installed around the env's own mj_step / mj_forward calls."""
        self.mj.set_mjcb_passive(None)

    def render(self, camera="iso", width=640, height=480):
        r = self.mj.Renderer(self.m, height=height, width=width)
        r.update_scene(self.d, camera=camera)
        img = r.render()
        r.close()
        return img
