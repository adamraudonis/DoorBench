"""Benchmark runner: evaluate a policy over doors x scenarios x seeds in MuJoCo, in parallel, and write a result JSON.

    doorbench benchmark run --policy scripted_hand --doors all --seeds 3 --scenarios default --workers 8 --out results/scripted_hand.json
    doorbench benchmark run --policy my_pkg.policies:MyPolicy --doors family:swing_single --dry-run

Reference embodiment: DoorEnv's programmatic hand (generalized forces on named door joints, clamped) plus a synthetic
robot base that walks with the commanded planar velocity and can only cross the wall plane while the opening is
clear (see `doorbench.benchmark.policy` for the interface).  Robot embodiments (`Policy.embodiment == "robot"`) build
their own DoorEnv with a robot attached and drive its actuators.

Determinism: MuJoCo is deterministic for a given model and input sequence; seed 0 evaluates the nominal door, seeds
>= 1 apply DoorEnv's domain randomisation (`reset(randomize=True)` with `DoorEnv(seed=seed)`: hinge friction,
damping, closer stiffness, masses) and jitter the base start by up to +-0.2 m in x.  Policies receive the seed in
`door_info`.

Output: a JSON document validated by `results/schema.json` (`scripts/validate_result.py`) with one entry per episode
(outcome, timestamped events, time-to-traverse, damage, peak forces, energy) and an aggregate (success rate overall
/ per family / per difficulty / per task / per scenario / per lock state, doors solved on every seed, mean
time-to-traverse, damage rate).
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from .policy import Policy, load_policy_class, policy_meta, resolve_policy_spec
from .scenarios import SCENARIOS, TRAVERSE_TASKS, Scenario, door_is_closed, parse_scenarios, success_of

SCHEMA_VERSION = "1.0"
FLAG_KEYS = ("touched_door", "touched_operator", "operator_actuated", "latch_released", "lock_released", "door_opened", "door_open_clear", "robot_passed_through", "door_closed_after", "door_slammed", "door_damaged", "robot_fell", "hardware_misuse")
EVENT_FLAGS = ("touched_door", "operator_actuated", "latch_released", "lock_released", "door_opened", "door_open_clear", "robot_passed_through", "door_closed_after", "door_slammed", "door_damaged", "robot_fell")
LABEL_NUMBERS = ("max_leaf_contact_force", "max_operator_torque", "max_door_angle", "energy_J", "time_to_touch", "time_to_open", "time_to_pass")
ENV_DRIVEN_LOCK_PARTS = ("lock_bar_", "electric_bolt_slide")
BASE_MAX_SPEED = 1.5      # m/s
BASE_RADIUS = 0.30        # m: half-depth of the wall band the base may only enter while the opening is clear
BASE_MIN_OPENING = 0.45   # m: narrower openings (pet doors) cannot be passed by the reference base
BASE_Z = 0.5
OUTCOMES = ("success", "fail", "damaged", "fell", "timeout", "error")


# ----------------------------------------------------------------------------------------------- door selection
def load_manifest(assets: str) -> dict:
    with open(os.path.join(assets, "manifest.json")) as f:
        return json.load(f)


def select_doors(manifest: dict, arg: str) -> list[dict]:
    """all | family:<f>[,<f>] | difficulty:<n>[,<n>] | task:<t> | first:<n> | sample:<n>[:<seed>] | ids:<a,b> | <a,b> | @file"""
    doors = [d for d in manifest["doors"] if not d.get("error")]
    arg = (arg or "all").strip()
    if arg == "all":
        return doors
    if arg.startswith("@"):
        with open(arg[1:]) as f:
            ids = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        return [d for d in doors if d["id"] in set(ids)]
    if ":" in arg:
        kind, _, val = arg.partition(":")
        if kind == "family":
            fams = set(val.split(","))
            return [d for d in doors if d["family"] in fams]
        if kind == "difficulty":
            lv = {int(x) for x in val.split(",")}
            return [d for d in doors if int(d.get("difficulty", 0)) in lv]
        if kind == "task":
            ts = set(val.split(","))
            return [d for d in doors if d.get("task") in ts]
        if kind == "lock":
            want = val
            return [d for d in doors if (want == "locked" and d.get("lock_engaged")) or (want == "unlocked" and not d.get("lock_engaged"))]
        if kind == "first":
            return doors[: int(val)]
        if kind == "sample":
            n, _, sd = val.partition(":")
            rng = np.random.default_rng(int(sd) if sd else 0)
            idx = sorted(rng.choice(len(doors), size=min(int(n), len(doors)), replace=False).tolist())
            return [doors[i] for i in idx]
        if kind == "ids":
            arg = val
        else:
            raise ValueError(f"unknown door selector {arg!r}")
    ids = [x.strip() for x in arg.split(",") if x.strip()]
    by_id = {d["id"]: d for d in doors}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown door ids: {missing[:5]}")
    return [by_id[i] for i in ids]


# ----------------------------------------------------------------------------------------------- per-episode pieces
def qa_push_for(door_dir: str, env) -> float:
    """The calibrated strong push of the sign-off QA (qa.json metrics.qa_push).  Free-swinging families (saloon,
    revolving, turnstiles, bifold, accordion, bypass, strips, pet flaps) are not calibrated by the QA: the same
    formula applies with a floor of 150 N*m / 300 N (a strong human push on a wing or a folding panel)."""
    try:
        with open(os.path.join(door_dir, "qa.json")) as f:
            p = json.load(f)["metrics"].get("qa_push")
        if p:
            return float(p)
    except Exception:
        pass
    m, d = env.m, env.d
    if env.pj < 0:
        return 100.0
    dof = m.jnt_dofadr[env.pj]
    env.mj.mj_forward(m, d)
    bias = abs(float(d.qfrc_bias[dof] - d.qfrc_passive[dof]))
    fl = float(m.dof_frictionloss[dof])
    is_hinge = int(m.jnt_type[env.pj]) == int(env.mj.mjtJoint.mjJNT_HINGE)
    push = min(2.0 * (bias + fl) + (60.0 if is_hinge else 80.0), 800.0 if is_hinge else 4000.0)
    return max(push, 150.0 if is_hinge else 300.0)


HAND_LIFTED_LATCHES = ("fork_hinge", "latch_bar_hinge")   # gravity latches a hand lifts directly (no operator drives them)


def operator_reachable(spec: dict, joint_name: str) -> bool:
    """Is this operator joint on the robot's face of the door?  Mirrors `geometry.common.operator_faces`: the robot
    stands at -y; `sides == "push_side"` puts the exit device on the face the door swings away from, so a robot on
    the pull side only has the far-side pull / lever (`*_far_*`); a handleset's interior knob sits on the far face
    (the robot has the thumb piece); `sides == "far"` is never reachable."""
    sides = spec.get("operator", {}).get("sides", "both")
    is_push = bool(spec.get("robot", {}).get("is_push", True))
    if sides == "far":
        return False
    if sides == "push_side":
        return (not is_push) if "_far_" in joint_name else is_push
    try:
        from .. import hardware as H
        kind = H.OPERATORS[spec["operator"]["model"]].kind
    except Exception:
        kind = ""
    if kind == "handleset" and joint_name.endswith("_handle_hinge"):
        return False
    return True


def torque_limits(env, door_dir: str) -> dict[str, float]:
    """Hand strength per robot-interactive joint (N*m for hinges, N for slides); 0 = not reachable."""
    mj, m = env.mj, env.m
    HINGE = int(mj.mjtJoint.mjJNT_HINGE)
    push = qa_push_for(door_dir, env)
    lock = env.spec.get("lock", {})
    far_side_lock = bool(lock.get("engaged")) and not bool(lock.get("robot_side_release", True))
    out = {}
    for b in env.model_json["bodies"]:
        j = b.get("joint")
        if not j:
            continue
        jid = env._jid(j["name"])
        if jid < 0:
            continue
        role, name = j.get("role"), j["name"]
        hinge = int(m.jnt_type[jid]) == HINGE
        if role in ("primary", "secondary"):
            lim = push
        elif role == "operator":
            lim = ((60.0 if "wheel" in name else 30.0) if hinge else 300.0) if operator_reachable(env.spec, name) else 0.0
        elif role == "lock":
            if any(p in name for p in ENV_DRIVEN_LOCK_PARTS) or far_side_lock:
                lim = 0.0
            elif "keypad_key_" in name:
                lim = 30.0
            else:
                lim = 30.0 if hinge else 200.0
        elif role == "latch" and any(p in name for p in HAND_LIFTED_LATCHES):
            lim = 10.0 if hinge else 60.0
        else:                      # latch bolts are driven through the operator, mechanism joints are not robot-interactive
            lim = 0.0
        out[name] = float(lim)
    return out


class SyntheticBase:
    """A point robot base: walks with the commanded planar velocity, blocked by the wall plane unless the opening is clear."""

    def __init__(self, start, half_opening: float, max_speed=BASE_MAX_SPEED, radius=BASE_RADIUS):
        self.pos = np.array([start[0], start[1], BASE_Z], float)
        self.half = float(half_opening)
        self.vmax, self.r = float(max_speed), float(radius)

    def step(self, v, dt: float, clear: bool):
        v = np.asarray(v, float).ravel()[:2] if v is not None else np.zeros(2)
        if not np.all(np.isfinite(v)):
            v = np.zeros(2)
        s = float(np.hypot(*v))
        if s > self.vmax:
            v = v * (self.vmax / s)
        x0, y0 = float(self.pos[0]), float(self.pos[1])
        x, y = x0 + float(v[0]) * dt, y0 + float(v[1]) * dt
        entering = abs(y) < self.r or (y0 < 0) != (y < 0)
        allowed = clear and 2 * self.half >= BASE_MIN_OPENING and abs(x) < self.half - 0.05
        if entering and not allowed:
            if y0 <= -self.r:
                y = min(y, -self.r)
            elif y0 >= self.r:
                y = max(y, self.r)
            else:
                y = y0
        if abs(y) < self.r:
            x = float(np.clip(x, -(self.half - 0.05), self.half - 0.05)) if self.half > 0.1 else x
        self.pos[0], self.pos[1] = float(np.clip(x, -3.0, 3.0)), float(np.clip(y, -3.0, 3.0))
        return self.pos


class AutoDoorSensor:
    """Environment-side presence sensor / door operator for automatic doors and elevator landing doors: drives the
    door's position actuators (model.json meta.actuators) open while the robot base is within the sensor range,
    holds `hold_open_s`, ramps at the spec's open / close speed.  Elevator doors open once the call button was
    pressed (lock released); an engaged, unreleased lock keeps the operator off ("night mode")."""

    def __init__(self, env):
        mj, m = env.mj, env.m
        acts = env.meta.get("actuators") or []
        kin = env.spec.get("kinematics", {})
        act = kin.get("actuator") or {}
        fam = env.spec.get("family", "")
        self.enabled = bool(acts) and fam in ("automatic_sliding", "automatic_swing", "elevator") and act.get("powered", True) is not False
        if not self.enabled:
            return
        self.env = env
        self.ids, self.hi = [], []
        for a in acts:
            aid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, a["name"])
            if aid >= 0:
                self.ids.append(aid)
                self.hi.append(float((a.get("ctrlrange") or [0, 1])[1]))
        self.range = float(act.get("sensor_range_m", 1.5))
        self.hold = float(act.get("hold_open_s", 2.0))
        span = max(self.hi) if self.hi else 1.0
        self.v_open = float(act.get("open_speed_m_s") or (span / float(act.get("open_time_s") or 3.0)))
        self.v_close = float(act.get("close_speed_m_s") or 0.8 * self.v_open)
        sid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "door_plane_center")
        self.center = m.site_pos[sid][:2].copy() if sid >= 0 else np.zeros(2)
        self.elevator = fam == "elevator"
        self.locked = bool(env.spec.get("lock", {}).get("engaged"))
        self.target = [0.0] * len(self.ids)
        self.last_seen = -1e9
        self.dt = float(m.opt.timestep)

    def step(self, base_xy, t: float):
        if not self.enabled:
            return
        env = self.env
        L = env.tracker.L
        released = bool(L.lock_released or env.unlocked_by_env)
        present = float(np.hypot(*(np.asarray(base_xy, float) - self.center))) < self.range
        if self.elevator:
            present = present and released
        elif self.locked and not released:
            present = False
        if present:
            self.last_seen = t
        want = (t - self.last_seen) < self.hold
        for k, aid in enumerate(self.ids):
            goal = self.hi[k] if want else 0.0
            self.target[k] += float(np.clip(goal - self.target[k], -self.v_close * self.dt, self.v_open * self.dt))
            env.d.ctrl[aid] = self.target[k]


# ----------------------------------------------------------------------------------------------- episode
@dataclass
class Job:
    door: dict                # manifest entry
    door_dir: str
    scenario: str
    seed: int
    tier: str
    policy_spec: str
    time_budget_s: float | None = None
    wall_timeout_s: float = 120.0
    randomize: bool = True
    control_dt: float | None = None
    extra: dict = field(default_factory=dict)


_POLICY: Policy | None = None
_POLICY_SPEC: str | None = None
_WARN = {"n": 0}


def _quiet_mujoco():
    try:
        import mujoco

        def _h(msg):
            _WARN["n"] += 1
        mujoco.set_mju_user_warning(_h)
    except Exception:
        pass


def _policy_for(spec: str) -> Policy:
    global _POLICY, _POLICY_SPEC
    if _POLICY is None or _POLICY_SPEC != spec:
        cls = load_policy_class(spec)
        _POLICY = cls()
        _POLICY_SPEC = spec
    return _POLICY


def _init_worker(policy_spec: str):
    _quiet_mujoco()
    try:
        _policy_for(policy_spec)
    except Exception:
        traceback.print_exc()


def _r(x, nd=4):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), nd)
    return x


def build_door_info(env, job: Job, scenario: Scenario, task: str, budget: float, control_dt: float, limits: dict, base_start) -> dict:
    mj, m = env.mj, env.m
    HINGE = int(mj.mjtJoint.mjJNT_HINGE)
    joints = {}
    for b in env.model_json["bodies"]:
        j = b.get("joint")
        if not j:
            continue
        jid = env._jid(j["name"])
        if jid < 0 or j.get("role") not in ("primary", "secondary", "operator", "latch", "lock"):
            continue
        rng = [float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])] if m.jnt_limited[jid] else None
        joints[j["name"]] = {"role": j.get("role"), "type": "hinge" if int(m.jnt_type[jid]) == HINGE else "slide", "range": rng, "label": j.get("label", "")}
    spec = env.spec
    lock = spec.get("lock", {})
    from .. import hardware as H
    lk = H.LOCKS.get(lock.get("model", "none"))
    sites = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
    gp = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "goal_point")
    ap = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "approach_point")
    return {
        "id": spec["id"], "family": spec["family"], "task": task, "scenario": scenario.name, "tier": job.tier, "seed": job.seed,
        "difficulty": job.door.get("difficulty"), "time_budget_s": budget, "control_dt": control_dt, "dt": float(m.opt.timestep),
        "spec": spec, "meta": env.meta, "joints": joints,
        "primary_joint": env.meta.get("primary_joint"), "secondary_joint": env.meta.get("secondary_joint"),
        "operator_joints": list(env.operator_joints), "lock_joints": list(env.lock_joints), "latch_joints": list(env.latch_joints),
        "lock": {"model": lock.get("model", "none"), "kind": getattr(lk, "kind", "none"), "engaged": bool(lock.get("engaged")), "robot_side_release": bool(lock.get("robot_side_release", True)),
                 "code": lock.get("code") if lock.get("robot_side_release", True) else None},
        "closer": spec.get("closer", {}).get("model", "none"), "operator": spec.get("operator", {}).get("model", "none"),
        "kinematics": spec.get("kinematics", {}), "opening_width": float(spec["opening"]["width"]), "leaf_width": float(spec["leaf"]["width"]),
        "approach_point": m.site_pos[ap].tolist() if ap >= 0 else [0.0, -1.5, 0.0], "goal_point": m.site_pos[gp].tolist() if gp >= 0 else [0.0, 1.5, 0.0],
        "sites": [s for s in sites if s], "torque_limits": limits,
        "base": {"max_speed": BASE_MAX_SPEED, "radius": BASE_RADIUS, "start": [float(v) for v in base_start]},
    }


def run_episode(job: Job) -> dict:
    """One door x scenario x seed.  Returns the episode dict (never raises: errors become outcome 'error')."""
    t_wall = time.time()
    _WARN["n"] = 0
    scenario = SCENARIOS[job.scenario]
    ep = {"door_id": job.door["id"], "family": job.door["family"], "difficulty": job.door.get("difficulty"), "scenario": job.scenario, "seed": job.seed, "tier": job.tier,
          "task": None, "success": False, "outcome": "error", "sim_time": 0.0, "steps": 0, "wall_s": 0.0, "events": [], "labels": {}, "error": None}
    env = None
    try:
        policy = _policy_for(job.policy_spec)
        pcls = type(policy)
        from .env import DoorEnv
        if getattr(policy, "embodiment", "hand_base") == "robot":
            env = pcls.make_env(job.door_dir, job.tier, job.seed)
        else:
            env = DoorEnv(job.door_dir, tier=job.tier, seed=job.seed)
        env.max_steps = 10 ** 9
        spec = env.spec
        task = scenario.task_for(spec)
        ep["task"] = task
        budget = float(job.time_budget_s or scenario.budget_for(spec))
        env.reset(task=task, randomize=bool(job.randomize and job.seed > 0))
        mj, m, d = env.mj, env.m, env.d
        dt = float(m.opt.timestep)
        control_dt = float(job.control_dt or getattr(policy, "control_dt", 0.01) or dt)
        decim = max(1, int(round(control_dt / dt)))
        control_dt = decim * dt
        rng = np.random.default_rng([job.seed, int(job.door.get("index", 0))])
        jitter = float(rng.uniform(-0.2, 0.2)) if job.seed > 0 else 0.0
        ap = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "approach_point")
        start = (m.site_pos[ap].copy() if ap >= 0 else np.array([0.0, -1.5, 0.0]))
        start[0] += jitter
        robot = bool(env.robot_base)
        limits = torque_limits(env, job.door_dir)
        info = build_door_info(env, job, scenario, task, budget, control_dt, limits, start)
        half = 0.5 * float(spec["opening"]["width"])
        base = SyntheticBase(start, half)
        sensor = AutoDoorSensor(env)
        policy.reset(info, env=env)
        # ---- fast accessors
        HINGE = int(mj.mjtJoint.mjJNT_HINGE)
        is_hinge = env.pj >= 0 and int(m.jnt_type[env.pj]) == HINGE
        jl = [(n, int(m.jnt_qposadr[env._jid(n)]), int(m.jnt_dofadr[env._jid(n)])) for n in info["joints"]]
        jdof = {n: dof for n, _, dof in jl}
        sl = [(n, mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, n)) for n in info["sites"]]
        pq = int(m.jnt_qposadr[env.pj]) if env.pj >= 0 else -1
        pd_ = int(m.jnt_dofadr[env.pj]) if env.pj >= 0 else -1
        sj = env._jid(env.meta.get("secondary_joint")) if env.meta.get("secondary_joint") else -1
        sq = int(m.jnt_qposadr[sj]) if sj >= 0 else -1
        sd_ = int(m.jnt_dofadr[sj]) if sj >= 0 else -1
        base_bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, env.robot_base) if robot else -1
        goal_y = float(info["goal_point"][1])
        prange = (float(m.jnt_range[env.pj][0]), float(m.jnt_range[env.pj][1])) if env.pj >= 0 and m.jnt_limited[env.pj] else None
        L = env.tracker.L
        engaged = bool(spec.get("lock", {}).get("engaged"))
        tail_after_goal = 6.0 if spec.get("closer", {}).get("model", "none") != "none" else 2.0   # let closers return (slam / closed-after labels)

        def base_pos():
            return d.xpos[base_bid] if robot else base.pos

        def clear():
            if L.door_open_clear:
                return True
            if task == "traverse_open" and prange is not None and pq >= 0:
                return (abs(float(d.qpos[pq])) - prange[0]) >= 0.75 * (prange[1] - prange[0])
            return False

        lean = robot     # robot embodiments read their own state from the env handle; skip the per-joint / per-site dicts

        def obs():
            bp = base_pos()
            return {"t": float(d.time), "door_q": float(d.qpos[pq]) if pq >= 0 else 0.0, "door_dq": float(d.qvel[pd_]) if pd_ >= 0 else 0.0,
                    "secondary_q": float(d.qpos[sq]) if sq >= 0 else None, "secondary_dq": float(d.qvel[sd_]) if sd_ >= 0 else None,
                    "joints": {} if lean else {n: {"q": float(d.qpos[qa]), "dq": float(d.qvel[da])} for n, qa, da in jl},
                    "sites": {} if lean else {n: d.site_xpos[sid].tolist() for n, sid in sl},
                    "base": {"pos": [float(bp[0]), float(bp[1]), float(bp[2])]},
                    "flags": {"touched": L.touched_door, "operator_actuated": L.operator_actuated, "latch_released": L.latch_released, "lock_released": L.lock_released,
                              "door_opened": L.door_opened, "door_open_clear": L.door_open_clear, "passed_through": L.robot_passed_through, "damaged": L.door_damaged},
                    "locked": engaged and not (L.lock_released or env.unlocked_by_env)}

        n_steps = int(round(budget / dt))
        prev = {k: False for k in EVENT_FLAGS}
        events = []
        t_goal = t_ok = t_closed = t_damage = None
        outcome = "fail"
        action = {}
        tq = {}
        torques = []
        base_v = np.zeros(2)
        step = 0
        while step < n_steps:
            t = float(d.time)
            if step % decim == 0:
                action = policy.act(obs()) or {}
                tq = action.get("torques") or {}
                torques = [(jdof[n], max(-limits.get(n, 0.0), min(limits.get(n, 0.0), float(v)))) for n, v in tq.items() if n in jdof and limits.get(n, 0.0) > 0 and v]
                bv = action.get("base_velocity")
                base_v = np.asarray(bv, float).ravel()[:2] if bv is not None else np.zeros(2)
                if action.get("badge") and not env.unlocked_by_env and spec.get("lock", {}).get("robot_side_release", True):
                    env.badge()
                    events.append(["badge", round(t, 3)])
                if time.time() - t_wall > job.wall_timeout_s:
                    outcome = "timeout"
                    break
                # label transitions -> events
                for k in EVENT_FLAGS:
                    v = bool(getattr(L, k))
                    if v and not prev[k]:
                        events.append([k, round(t, 3)])
                        prev[k] = True
            # ---- apply the action
            if torques:
                for dof, tau in torques:
                    d.qfrc_applied[dof] += tau
                if env.tracker and not L.touched_door:
                    env.tracker.mark_touch(d, operator=False)
                if not L.touched_operator and any(n in env.operator_joints for n in tq):
                    env.tracker.mark_touch(d, operator=True)
            ctrl = action.get("ctrl")
            if ctrl is not None and m.nu:
                d.ctrl[:] = np.asarray(ctrl, float).ravel()[: m.nu]
            bp = base_pos()
            sensor.step(bp[:2], t)
            if not robot:
                base.step(base_v, dt, clear())
            env.step(robot_base_pos=None if robot else base.pos)
            step += 1
            t = float(d.time)
            # ---- termination
            if L.door_damaged and L.touched_door:
                t_damage = t_damage if t_damage is not None else t
                if t - t_damage > 0.5:
                    outcome = "damaged"
                    break
            if L.robot_fell:
                outcome = "fell"
                break
            if action.get("done"):
                break
            by = float(base_pos()[1])
            if task in TRAVERSE_TASKS:
                if by >= goal_y and t_goal is None:
                    t_goal = t
                    events.append(["goal_reached", round(t, 3)])
                if t_goal is not None:
                    q_now = float(d.qpos[pq]) if pq >= 0 else 0.0
                    if scenario.require_closed:
                        if door_is_closed(q_now, is_hinge) and t > t_goal + 0.5:
                            t_closed = t_closed if t_closed is not None else t
                            if t - t_closed > 0.5:
                                break
                        else:
                            t_closed = None
                    elif t > t_goal + tail_after_goal or (t > t_goal + 2.0 and door_is_closed(q_now, is_hinge)):
                        break
            elif task == "open_only":
                if L.door_open_clear:
                    t_ok = t_ok if t_ok is not None else t
                    if t - t_ok > 1.0:
                        break
            elif task == "close":
                q_now = float(d.qpos[pq]) if pq >= 0 else 0.0
                if door_is_closed(q_now, is_hinge) and t > 0.5:
                    t_closed = t_closed if t_closed is not None else t
                    if t - t_closed > 1.0:
                        break
                else:
                    t_closed = None
        # ---- finalize
        labels = env.labels().to_dict()
        for k in EVENT_FLAGS:
            if labels.get(k) and not prev[k]:
                events.append([k, round(float(d.time), 3)])
        q_end = float(d.qpos[pq]) if pq >= 0 else 0.0
        # damage that happened before the policy ever touched the door (a flap or hatch reset in an unstable open
        # pose slamming shut on its own) is the environment's doing, not the policy's: recorded, not counted
        t_touch = labels.get("time_to_touch")
        dmg_events = labels.get("damage_events") or []
        env_damage = bool(labels.get("door_damaged")) and not any((e.get("t") or 0.0) >= (t_touch if t_touch is not None else float("inf")) - 1e-9 for e in dmg_events)
        if env_damage:
            labels["door_damaged"] = False
            labels["door_slammed"] = False
        ok = success_of(task, labels, scenario, q_end, is_hinge, goal_reached=t_goal is not None)
        if outcome in ("fail", "damaged") and labels.get("door_damaged"):
            outcome = "damaged"
        if ok and outcome not in ("timeout", "error"):
            outcome = "success"
        elif outcome == "fail" and labels.get("robot_fell"):
            outcome = "fell"
        elif outcome == "fail" and step >= n_steps and not ok:
            outcome = "fail"
        ep.update({
            "success": bool(ok), "outcome": outcome, "sim_time": _r(d.time, 3), "steps": int(step), "wall_s": _r(time.time() - t_wall, 3),
            "time_budget_s": budget, "control_dt": control_dt, "randomized": bool(job.randomize and job.seed > 0),
            "time_to_touch": _r(labels.get("time_to_touch"), 3), "time_to_open": _r(labels.get("time_to_open"), 3), "time_to_pass": _r(labels.get("time_to_pass"), 3),
            "time_to_goal": _r(t_goal, 3), "time_to_close": _r(t_closed, 3),
            "damage": bool(labels.get("door_damaged")), "env_damage": env_damage, "damage_events": [{"t": _r(e.get("t"), 3), "kind": e.get("kind"), "part": e.get("part"), "value": _r(e.get("value"), 1), "threshold": _r(e.get("threshold"), 1)} for e in dmg_events[:5]],
            "max_leaf_force_N": _r(labels.get("max_leaf_contact_force"), 1), "max_operator_torque": _r(labels.get("max_operator_torque"), 2),
            "max_door_angle": _r(labels.get("max_door_angle"), 4), "door_q_end": _r(q_end, 4), "energy_J": _r(labels.get("energy_J"), 2),
            "labels": {k: bool(labels.get(k)) for k in FLAG_KEYS}, "tracker_success": bool(labels.get("success")),
            "events": events, "mujoco_warnings": int(_WARN["n"]), "base_end": [_r(v, 3) for v in base_pos()],
        })
    except Exception as e:  # noqa: BLE001
        ep["error"] = f"{type(e).__name__}: {e}"[:400]
        ep["traceback"] = traceback.format_exc()[-1500:]
        ep["outcome"] = "error"
        ep["wall_s"] = _r(time.time() - t_wall, 3)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return ep


# ----------------------------------------------------------------------------------------------- aggregation
def _mean(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return round(sum(xs) / len(xs), 3) if xs else None


def _median(xs):
    xs = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not xs:
        return None
    n = len(xs)
    return round(xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]), 3)


def _group_stats(eps: list[dict]) -> dict:
    by_door = {}
    for e in eps:
        by_door.setdefault(e["door_id"], []).append(bool(e["success"]))
    n = len(eps)
    ns = sum(1 for e in eps if e["success"])
    pass_t = [e.get("time_to_pass") for e in eps if e["success"] and e.get("time_to_pass") is not None]
    return {
        "n_doors": len(by_door), "n_episodes": n, "n_success": ns, "success_rate": round(ns / n, 4) if n else 0.0,
        "doors_solved": sum(1 for v in by_door.values() if all(v)), "doors_solved_any": sum(1 for v in by_door.values() if any(v)),
        "damage_rate": round(sum(1 for e in eps if e.get("damage")) / n, 4) if n else 0.0,
        "mean_time_to_pass_s": _mean(pass_t), "median_time_to_pass_s": _median(pass_t),
        "mean_time_to_open_s": _mean([e.get("time_to_open") for e in eps if e.get("time_to_open") is not None]),
        "mean_max_leaf_force_N": _mean([e.get("max_leaf_force_N") for e in eps]),
        "mean_energy_J": _mean([e.get("energy_J") for e in eps]),
        "outcomes": {k: sum(1 for e in eps if e.get("outcome") == k) for k in OUTCOMES if any(e.get("outcome") == k for e in eps)},
    }


def lock_state_of(door: dict) -> str:
    if not door.get("lock_engaged"):
        return "unlocked"
    return "locked_releasable" if door.get("robot_side_release", True) else "locked_no_release"


def aggregate(episodes: list[dict], doors_by_id: dict | None = None) -> dict:
    eps = [e for e in episodes if e.get("outcome") != "error"] or episodes
    agg = _group_stats(eps)
    agg["n_errors"] = sum(1 for e in episodes if e.get("outcome") == "error")
    agg["mean_wall_s"] = _mean([e.get("wall_s") for e in episodes])
    agg["mean_sim_time_s"] = _mean([e.get("sim_time") for e in eps])
    agg["timeouts"] = sum(1 for e in episodes if e.get("outcome") == "timeout")

    def group(key):
        g = {}
        for e in eps:
            g.setdefault(str(key(e)), []).append(e)
        return {k: _group_stats(v) for k, v in sorted(g.items())}
    agg["by_family"] = group(lambda e: e["family"])
    agg["by_difficulty"] = group(lambda e: e.get("difficulty"))
    agg["by_task"] = group(lambda e: e.get("task"))
    agg["by_scenario"] = group(lambda e: e.get("scenario"))
    if doors_by_id:
        agg["by_lock_state"] = group(lambda e: lock_state_of(doors_by_id.get(e["door_id"], {})))
    return agg


# ----------------------------------------------------------------------------------------------- run
def git_commit(root: str) -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _error_episode(job: Job, err: str) -> dict:
    return {"door_id": job.door["id"], "family": job.door["family"], "difficulty": job.door.get("difficulty"), "scenario": job.scenario, "seed": job.seed, "tier": job.tier,
            "task": None, "success": False, "outcome": "error", "sim_time": 0.0, "steps": 0, "wall_s": 0.0, "events": [], "labels": {}, "error": err[:400]}


def _run_pool(jobs: list[Job], workers: int, policy_spec: str, collect, progress, max_attempts: int = 3):
    """Run the jobs on a process pool.  A worker that dies (segfault, OOM kill) breaks the pool: the unfinished jobs
    are resubmitted to a fresh pool (up to `max_attempts` times each; a job that keeps killing workers becomes an
    'error' episode) instead of hanging the run."""
    import concurrent.futures as cf
    import multiprocessing as mp
    from concurrent.futures.process import BrokenProcessPool
    ctx = mp.get_context("spawn") if sys.platform == "darwin" else mp.get_context()
    pending = list(range(len(jobs)))
    attempts = [0] * len(jobs)
    while pending:
        ex = cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(policy_spec,))
        futs = {}
        for i in pending:
            attempts[i] += 1
            futs[ex.submit(run_episode, jobs[i])] = i
        pending = []
        broken = False
        try:
            for fut in cf.as_completed(futs):
                i = futs[fut]
                try:
                    collect(fut.result())
                except BrokenProcessPool:
                    broken = True
                    if attempts[i] < max_attempts:
                        pending.append(i)
                    else:
                        collect(_error_episode(jobs[i], "worker process died repeatedly on this episode"))
                except Exception as e:  # noqa: BLE001
                    collect(_error_episode(jobs[i], f"{type(e).__name__}: {e}"))
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        if broken:
            progress(f"  a worker process died; restarting the pool for {len(pending)} unfinished episode(s)")
            pending.sort()


def make_jobs(doors: list[dict], assets: str, scenarios: list[Scenario], seeds: list[int], tier: str, policy_spec: str, **kw) -> list[Job]:
    return [Job(door=d, door_dir=os.path.join(assets, "doors", d["id"]), scenario=s.name, seed=seed, tier=tier, policy_spec=policy_spec, **kw)
            for d in doors for s in scenarios for seed in seeds]


def run_benchmark(policy_spec: str, doors: str = "all", seeds: int | list[int] = 3, scenarios: str = "default", workers: int = 8, tier: str = "full",
                  assets: str = "assets", time_budget_s: float | None = None, wall_timeout_s: float = 120.0, randomize: bool = True,
                  control_dt: float | None = None, label: str = "", out: str | None = None, progress=print) -> dict:
    """Evaluate `policy_spec` and return the result document (also written to `out` when given)."""
    import mujoco
    t0 = time.time()
    policy_spec = resolve_policy_spec(policy_spec)
    pcls = load_policy_class(policy_spec)
    pmeta = policy_meta(pcls)
    if hasattr(pcls, "check"):
        pcls.check()
    if getattr(pcls, "requires_tier", None) and tier != pcls.requires_tier:
        progress(f"note: {pmeta['name']} requires tier {pcls.requires_tier}; overriding --tier {tier}")
        tier = pcls.requires_tier
    manifest = load_manifest(assets)
    door_list = select_doors(manifest, doors)
    seed_list = list(range(int(seeds))) if isinstance(seeds, int) else [int(s) for s in seeds]
    scen = parse_scenarios(scenarios)
    jobs = make_jobs(door_list, assets, scen, seed_list, tier, policy_spec, time_budget_s=time_budget_s, wall_timeout_s=wall_timeout_s, randomize=randomize, control_dt=control_dt)
    progress(f"{pmeta['name']}: {len(door_list)} doors x {len(scen)} scenario(s) x {len(seed_list)} seed(s) = {len(jobs)} episodes, tier {tier}, {workers} worker(s)")
    episodes = []
    state = {"n_done": 0, "n_ok": 0, "t_last": time.time()}

    def collect(ep):
        episodes.append(ep)
        state["n_done"] += 1
        state["n_ok"] += bool(ep["success"])
        if ep.get("outcome") == "error" and state["n_done"] <= 20:
            progress(f"  error on {ep['door_id']} seed {ep['seed']}: {ep.get('error')}")
        if time.time() - state["t_last"] > 10 or state["n_done"] == len(jobs):
            state["t_last"] = time.time()
            el = state["t_last"] - t0
            n = state["n_done"]
            progress(f"  {n}/{len(jobs)} episodes, {state['n_ok']} successes ({100 * state['n_ok'] / n:.1f} %), {el:.0f} s elapsed, ETA {el / n * (len(jobs) - n):.0f} s")

    if workers <= 1:
        _init_worker(policy_spec)
        for job in jobs:
            collect(run_episode(job))
    else:
        _run_pool(jobs, workers, policy_spec, collect, progress)
    order = {d["id"]: i for i, d in enumerate(door_list)}
    sc_order = {s.name: i for i, s in enumerate(scen)}
    episodes.sort(key=lambda e: (order.get(e["door_id"], 0), sc_order.get(e["scenario"], 0), e["seed"]))
    wall = time.time() - t0
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    doors_by_id = {d["id"]: d for d in door_list}
    extra = {}
    try:
        extra = pcls.info() if hasattr(pcls, "info") else {}
    except Exception:
        pass
    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": {"name": "DoorBench", "dataset_version": manifest.get("version"), "dataset_generated": manifest.get("generated"), "n_doors_total": manifest.get("n_doors", len(manifest["doors"])), **git_commit(root)},
        "policy": {**pmeta, "spec": policy_spec, "extra": extra},
        "run": {"date": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "label": label, "simulator": "mujoco", "simulator_version": mujoco.__version__, "tier": tier,
                "scenarios": [s.to_dict() for s in scen], "seeds": seed_list, "n_doors": len(door_list), "door_selection": doors, "time_budget_s": time_budget_s or {s.name: s.time_budget_s for s in scen},
                "randomize": bool(randomize), "control_dt": control_dt or pmeta["control_dt"], "workers": workers, "wall_time_s": round(wall, 1),
                "host": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "cpu_count": os.cpu_count()},
                "command": " ".join(sys.argv)},
        "aggregate": aggregate(episodes, doors_by_id),
        "episodes": episodes,
    }
    if out:
        write_result(result, out)
        progress(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
    a = result["aggregate"]
    progress(f"{pmeta['name']}: {a['doors_solved']} / {a['n_doors']} doors solved on every seed ({a['doors_solved_any']} on at least one), episode success {a['success_rate'] * 100:.1f} %, damage {a['damage_rate'] * 100:.1f} %, {wall:.0f} s wall ({a['mean_wall_s']} s / episode)")
    return result


def write_result(result: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")


def dry_run(doors: str, scenarios: str, seeds: int, assets: str = "assets", out=print):
    manifest = load_manifest(assets)
    door_list = select_doors(manifest, doors)
    scen = parse_scenarios(scenarios)
    out(f"{len(door_list)} doors x {len(scen)} scenario(s) x {seeds} seed(s) = {len(door_list) * len(scen) * seeds} episodes")
    for d in door_list:
        tasks = ", ".join(s.task_for({"task": d.get("task")}) for s in scen)
        out(f"{d['id']:28s} {d['family']:22s} L{d.get('difficulty', '?')}  {d.get('lock', 'none'):18s} {'locked' if d.get('lock_engaged') else '      '}  {tasks}")
