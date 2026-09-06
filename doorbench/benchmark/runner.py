"""Benchmark runner: evaluate a policy over doors x scenarios x seeds in MuJoCo, in parallel, and write a result JSON.

    doorbench benchmark run --policy scripted_hand --doors all --seeds 3 --workers 8 --out results/scripted_hand.json
    doorbench benchmark run --policy scripted_hand --suite human --out results/scripted_hand_human.json     # advanced, opt-in
    doorbench benchmark run --policy my_pkg.policies:MyPolicy --doors family:swing_single --dry-run

Scenarios and suites
--------------------
Every door lists its scenarios in `spec.json["benchmark"]` (`doorbench.benchmark.scenarios`; summarised per door in
`manifest.json`).  They come in two *suites*:

* `core` (the default): `open_and_traverse`, `open_then_close`, `close_only`, `unlock_and_traverse`,
  `locked_recognize` - the door and the robot only, no simulated person anywhere.  Every door's primary scenario is
  a core scenario, so `--suite core` covers the 985 standard doors; the headline "N / 985 doors" number is core-only.
* `human` (advanced, opt in with `--suite human` or `--suite all`): `hold_open_for_human`, `wait_for_human`,
  `knock_and_wait` - a kinematic person is simulated by DoorEnv (79 doors list one of these).

The runner evaluates each door on the scenarios it lists in the chosen suite (`--scenarios` narrows that list).
Core and human episodes are never mixed: every episode carries its `suite`, the aggregate holds one table per
suite (`aggregate["core"]`, `aggregate["human"]`) and `scripts/validate_result.py` rejects mixed tables.

Success is the scenario's own criterion (`DoorEnv.success`: the `success` list of the scenario block, e.g.
`opened & traversed & !damage`, `unlock & opened & traversed & !damage`, `closed & latched & !damage & !slam`,
`!opened & !damage & !hardware_misuse`, `held_for_human & traversed & !collision_with_human & !damage`).  A door is
**solved** when every episode of it (every listed scenario x every seed) succeeded.

Reference embodiment: DoorEnv's programmatic hand (generalized forces on named door joints, clamped) plus a synthetic
robot base that starts at the scenario's seeded start pose, walks with the commanded planar velocity and can only
cross the wall plane while the opening is clear (see `doorbench.benchmark.policy`).  Robot embodiments
(`Policy.embodiment == "robot"`) build their own DoorEnv with a robot attached and drive its actuators.

Determinism: MuJoCo is deterministic for a given model and input sequence; seed 0 evaluates the nominal door, seeds
>= 1 apply DoorEnv's domain randomisation (`reset(randomize=True)` with `DoorEnv(seed=seed)`: hinge friction,
damping, closer stiffness, masses); the start pose is drawn from the scenario's start zone with the same seed.

Output: a JSON document validated by `results/schema.json` (`scripts/validate_result.py`) with one entry per episode
(outcome, timestamped events, reward events and return, time-to-traverse, damage, peak forces, energy) and, per
suite, an aggregate (success rate overall / per family / per difficulty / per scenario / per lock state, doors solved
on every episode, mean time-to-traverse, damage rate, human-collision rate).
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

from ..benchmark_eligibility import is_benchmark_eligible, require_benchmark_eligible, collection_counts
from .policy import Policy, load_policy_class, policy_meta, resolve_policy_spec
from .scenarios import CORE_SCENARIOS, HUMAN_SCENARIOS, SCENARIO_DESCRIPTIONS, SCENARIO_SUITE, SCENARIO_TYPES, SUITES, scenarios_in_suite

SCHEMA_VERSION = "1.1"
FLAG_KEYS = ("touched_door", "touched_operator", "operator_actuated", "latch_released", "lock_released", "door_opened", "door_open_clear", "robot_passed_through", "door_closed_after", "door_slammed", "door_damaged", "robot_fell", "hardware_misuse")
EVENT_FLAGS = ("touched_door", "operator_actuated", "latch_released", "lock_released", "door_opened", "door_open_clear", "robot_passed_through", "door_closed_after", "door_slammed", "door_damaged", "robot_fell")
ENV_DRIVEN_LOCK_PARTS = ("lock_bar_", "electric_bolt_slide")
BASE_MAX_SPEED = 1.5      # m/s
BASE_RADIUS = 0.30        # m: half-depth of the wall band the base may only enter while the opening is clear
BASE_MIN_OPENING = 0.45   # m: narrower openings (pet doors) cannot be passed by the reference base
BASE_Z = 0.5
OUTCOMES = ("success", "fail", "damaged", "fell", "timeout", "error", "native_failure", "mechanism_failure")
SUITE_CHOICES = ("core", "human", "all")


# ----------------------------------------------------------------------------------------------- door selection
def load_manifest(assets: str) -> dict:
    with open(os.path.join(assets, "manifest.json")) as f:
        return json.load(f)


def select_doors(manifest: dict, arg: str) -> list[dict]:
    """all | family:<f>[,<f>] | difficulty:<n>[,<n>] | scenario:<s>[,<s>] | lock:locked|unlocked | first:<n> | every:<n>[:<offset>] | sample:<n>[:<seed>] | ids:<a,b> | <a,b> | @file"""
    inventory = [d for d in manifest["doors"] if not d.get("error")]
    doors = [d for d in inventory if is_benchmark_eligible(d)]
    arg = (arg or "all").strip()
    if arg == "all":
        return doors
    if arg.startswith("@"):
        with open(arg[1:]) as f:
            ids = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        return select_doors(manifest, "ids:" + ",".join(ids))
    if ":" in arg:
        kind, _, val = arg.partition(":")
        if kind == "family":
            fams = set(val.split(","))
            for family in fams:
                require_benchmark_eligible(family, operation="benchmark selection")
            return [d for d in doors if d["family"] in fams]
        if kind == "difficulty":
            lv = {int(x) for x in val.split(",")}
            return [d for d in doors if int(d.get("difficulty", 0)) in lv]
        if kind == "scenario":
            want = set(val.split(","))
            return [d for d in doors if want & set(door_scenarios(d, "all"))]
        if kind == "task":
            ts = set(val.split(","))
            return [d for d in doors if d.get("task") in ts]
        if kind == "lock":
            want = val
            return [d for d in doors if (want == "locked" and d.get("lock_engaged")) or (want == "unlocked" and not d.get("lock_engaged"))]
        if kind == "first":
            return doors[: int(val)]
        if kind == "every":
            n, _, off = val.partition(":")
            return doors[int(off or 0):: int(n)]
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
    by_id = {d["id"]: d for d in inventory}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown door ids: {missing[:5]}")
    for i in ids:
        require_benchmark_eligible(by_id[i], operation="benchmark selection")
    return [by_id[i] for i in ids]


def door_scenarios(door: dict, suite: str) -> list[str]:
    """The scenario names a manifest entry lists in `suite` ('core' | 'human' | 'all')."""
    if not is_benchmark_eligible(door):
        return []
    b = door.get("benchmark") or {}
    names = list(b.get("scenarios") or [b.get("primary") or "open_and_traverse"])
    return scenarios_in_suite(names, suite)


def parse_scenarios(arg: str | None, suite: str) -> list[str] | None:
    """`--scenarios`: None / '' / 'all' = every scenario the door lists in the suite; 'primary' = the door's primary
    scenario only (always core); a comma list narrows to those types (they must belong to the suite)."""
    if not arg or arg in ("all", "suite"):
        return None
    if arg == "primary":
        return ["primary"]
    out = []
    for n in (x.strip() for x in arg.split(",") if x.strip()):
        if n not in SCENARIO_TYPES:
            raise KeyError(f"unknown scenario {n!r}; known: {', '.join(SCENARIO_TYPES)}")
        if suite != "all" and SCENARIO_SUITE[n] != suite:
            raise ValueError(f"scenario {n!r} belongs to the {SCENARIO_SUITE[n]!r} suite, not {suite!r} (use --suite {SCENARIO_SUITE[n]} or --suite all)")
        if n not in out:
            out.append(n)
    return out


def scenarios_for(door: dict, suite: str, only: list[str] | None) -> list[str]:
    if not is_benchmark_eligible(door):
        return []
    if only == ["primary"]:
        b = door.get("benchmark") or {}
        return [b.get("primary") or door_scenarios(door, "core")[0]]
    names = door_scenarios(door, suite)
    return [n for n in names if n in only] if only else names


# ----------------------------------------------------------------------------------------------- per-episode pieces
def qa_push_for(door_dir: str, env) -> float:
    """Use a source-matching QA effort, or calculate it on private native data.

    A stale sidecar must never set the force budget of rebuilt mechanics.
    The result is an oracle test effort, not a human strength certificate.
    """
    try:
        import hashlib
        from pathlib import Path
        with open(os.path.join(door_dir, "qa.json")) as f:
            report = json.load(f)
        source = report.get('source_sha256', {})
        matching = all(source.get(name) == hashlib.sha256(Path(door_dir,name).read_bytes()).hexdigest()
                       for name in ('spec.json','model.json','door.xml'))
        p = report['metrics'].get('qa_push')
        if matching and p:
            return float(p)
    except Exception:
        pass
    m = env.m
    if env.pj < 0:
        return 100.0
    from ..qa import qa_push
    pose = env.mj.MjData(m)
    phys = env.spec.get('physics', {})
    report = env._with_passive(lambda: qa_push(m, pose, env.pj, phys.get('mass', {}).get('dynamics_mass_kg', phys.get('mass', {}).get('total_kg')),
                                                env.spec['leaf']['width'], env.meta))
    return float(report['push'])


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
    far_side_lock = not bool(lock.get("robot_side_release", True))
    from .interactions import ContactSites
    contacts=ContactSites(env)
    paired_holds={r['joint']:r for r in env.meta.get('paired_leaf_holds',[])}
    out = {}
    for b in env.model_json["bodies"]:
        j = b.get("joint")
        if not j or j.get('type')=='free':
            continue
        jid = env._jid(j["name"])
        if jid < 0:
            continue
        role, name = j.get("role"), j["name"]
        hinge = int(m.jnt_type[jid]) == HINGE
        if not j.get("robot_interactive", True):
            out[name] = 0.0
            continue
        if role in ("primary", "secondary"):
            lim = push
        elif role == "operator":
            lim = ((60.0 if "wheel" in name else 30.0) if hinge else 300.0) if operator_reachable(env.spec, name) else 0.0
        elif role == "lock":
            if name in paired_holds:
                lim=paired_holds[name]['force_cap_N'] if paired_holds[name]['accessible_from_robot']else 0.
            elif any(p in name for p in ENV_DRIVEN_LOCK_PARTS) or far_side_lock:
                lim = 0.0
            elif "keypad_key_" in name:
                lim = 30.0
            else:
                lim = 30.0 if hinge else 200.0
        elif role == "latch" and any(p in name for p in HAND_LIFTED_LATCHES):
            lim = 10.0 if hinge else 60.0
        else:                      # latch bolts are driven through the operator, mechanism joints are not robot-interactive
            lim = 0.0
        # Generalized-force convenience cannot bypass an absent or far-face
        # input. Powered door actuators are driven separately by their actual
        # activation logic; passive transmitted loads remain in SiteForces.
        has_input=contacts.potential(name) if name in paired_holds else contacts.select(name)is not None
        out[name] = float(lim) if has_input else 0.0
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
        self.pos[0], self.pos[1] = float(np.clip(x, -4.0, 4.0)), float(np.clip(y, -4.0, 4.0))
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
        sectional = env.meta.get('sectional_track')
        self.sectional = sectional if sectional and sectional['drive']['mode'] == 'powered' else None
        self.enabled = bool(self.sectional) or (bool(acts) and fam in ("automatic_sliding", "automatic_swing", "elevator") and act.get("powered", True) is not False)
        if not self.enabled:
            return
        self.env = env
        self.elevator_control=None
        if fam=='elevator' and env.meta.get('elevator_interlocks'):
            from .elevator_control import ElevatorControl
            self.elevator_control=ElevatorControl(env)
            return
        self.ids, self.hi = [], []
        for a in acts:
            if a.get("role") in ("latch_retraction",'sectional_drive'):
                continue
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
        self.activation = env.meta.get("automatic_activation") or {"kind": 'push_button_wall' if self.sectional else act.get("sensor", "motion")}
        self.buttons = [(env._jid(b["joint"]), float(b["threshold_m"])) for b in self.activation.get("buttons", [])]
        self.retractors = []
        if self.sectional:
            from ..geometry.sectional import inspection_pose
            link=self.sectional['drive']['linkage']
            jid=env._jid(link['trolley_joint'])
            self.lift_qadr=int(m.jnt_qposadr[jid]);self.lift_dof=int(m.jnt_dofadr[jid])
            self.lift_end=inspection_pose(1.,self.sectional)['trolley_q']
            self.lift_target=float(env.d.qpos[self.lift_qadr])
            self.lift_open=bool(env._was_open);self.lift_pressed=False
            self.lift_started=False
            self.lift_site=m.site(self.sectional['powered_drive_site']).id
            self.lift_actuator=m.actuator(self.sectional['drive']['actuator']).id
        for row in env.meta.get("powered_latch_retraction", []):
            aid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, row["actuator"])
            jid = env._jid(row["joint"])
            if aid >= 0 and jid >= 0:
                self.retractors.append((aid, int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid]), row["travel"], row["max_force"]))

    def step(self, base_xy, t: float, action=None):
        if not self.enabled:
            return
        if self.elevator_control is not None:
            self.elevator_control.step(base_xy,t)
            return
        env = self.env
        L = env.tracker.L
        released = bool(L.lock_released or env.unlocked_by_env)
        present = float(np.hypot(*(np.asarray(base_xy, float) - self.center))) < self.range
        kind = self.activation.get("kind")
        if kind in ("push_button", "push_button_wall"):
            present = any(j >= 0 and float(env.d.qpos[env.m.jnt_qposadr[j]]) >= threshold for j, threshold in self.buttons)
        elif kind == "wave_to_open":
            site = (action or {}).get("activate_sensor")
            sid = env.mj.mj_name2id(env.m, env.mj.mjtObj.mjOBJ_SITE, site) if site in self.activation.get("wave_sites", []) else -1
            present = sid >= 0 and float(np.linalg.norm(env.d.site_xpos[sid][:2] - np.asarray(base_xy))) <= .75
        if self.env.spec.get("kinematics", {}).get("actuator", {}).get("push_and_go"):
            present = present or (self.env.pj >= 0 and abs(float(env.d.qpos[env.m.jnt_qposadr[self.env.pj]])) > .07)
        if self.elevator:
            present = present and released
        elif self.locked and not released:
            present = False
        if present:
            self.last_seen = t
        if self.sectional:
            # A physical garage wall button toggles the commanded state.
            # Presence cannot activate it, and releasing the pad is not a
            # command to close while the doorway is being traversed.
            if present and not self.lift_pressed:
                self.lift_open=not self.lift_open
                self.lift_started=True
            self.lift_pressed=present
            if not self.lift_started:
                env.d.ctrl[self.lift_actuator]=0.
                return
            goal=self.lift_end if self.lift_open else 0.
            self.lift_target += float(np.clip(goal-self.lift_target,-.30*self.dt,.30*self.dt))
            cap=self.sectional['drive']['max_force_N']
            effort=float(np.clip(3500.*(self.lift_target-env.d.qpos[self.lift_qadr])-200.*env.d.qvel[self.lift_dof],-cap,cap))
            env.d.ctrl[self.lift_actuator]=effort
            return
        want = (t - self.last_seen) < self.hold
        ready = True
        for aid, qa, va, travel, force in self.retractors:
            q, dq = float(env.d.qpos[qa]), float(env.d.qvel[va])
            env.d.ctrl[aid] = float(np.clip(force/(.2*travel)*(travel-q) - force/(6*travel)*dq, 0, force)) if want else 0.
            ready = ready and q >= .8*travel
        for k, aid in enumerate(self.ids):
            goal = self.hi[k] if want and ready else 0.0
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
    wall_timeout_s: float = 120.0  # episode execution after native initialization
    randomize: bool = True
    control_dt: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def suite(self) -> str:
        return SCENARIO_SUITE[self.scenario]


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


def build_door_info(env, job: Job, sc: dict, budget: float, control_dt: float, limits: dict, start: dict) -> dict:
    mj, m = env.mj, env.m
    HINGE = int(mj.mjtJoint.mjJNT_HINGE)
    joints = {}
    for b in env.model_json["bodies"]:
        j = b.get("joint")
        if not j or j.get('type')=='free':
            continue
        jid = env._jid(j["name"])
        if jid < 0 or j.get("role") not in ("primary", "secondary", "operator", "latch", "lock"):
            continue
        rng = [float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])] if m.jnt_limited[jid] else None
        joints[j["name"]] = {"role": j.get("role"), "type": "hinge" if int(m.jnt_type[jid]) == HINGE else "slide", "range": rng, "label": j.get("label", ""), "initial": float(env.d.qpos[m.jnt_qposadr[jid]]), "body": b["name"]}
    spec = env.spec
    lock = spec.get("lock", {})
    from .. import hardware as H
    lk = H.LOCKS.get(lock.get("model", "none"))
    sites = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
    gp = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "goal_point")
    ap = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "approach_point")
    goal = sc.get("goal")
    return {
        "id": spec["id"], "family": spec["family"], "scenario": sc["name"], "suite": SCENARIO_SUITE[sc["name"]], "tier": job.tier, "seed": job.seed,
        "difficulty": job.door.get("difficulty"), "time_budget_s": budget, "control_dt": control_dt, "dt": float(m.opt.timestep),
        "scenario_spec": sc, "start": start, "human": sc.get("human"),
        "spec": spec, "meta": env.meta, "joints": joints,
        "primary_joint": env.meta.get("primary_joint"), "secondary_joint": env.meta.get("secondary_joint"),
        "operator_joints": list(env.operator_joints), "lock_joints": list(env.lock_joints), "latch_joints": list(env.latch_joints),
        "lock": {"model": lock.get("model", "none"), "kind": getattr(lk, "kind", "none"), "engaged": bool(lock.get("engaged")), "robot_side_release": bool(lock.get("robot_side_release", True)),
                 "code": lock.get("code") if lock.get("robot_side_release", True) else None},
        "closer": spec.get("closer", {}).get("model", "none"), "operator": spec.get("operator", {}).get("model", "none"),
        "kinematics": spec.get("kinematics", {}), "opening_width": float(spec["opening"]["width"]), "leaf_width": float(spec["leaf"]["width"]),
        "approach_point": m.site_pos[ap].tolist() if ap >= 0 else [0.0, -1.5, 0.0],
        "goal_point": (list(goal["center"]) if goal else (m.site_pos[gp].tolist() if gp >= 0 else [0.0, 1.5, 0.0])),
        "pass_plane": sc.get("pass_plane"), "handle_targets": list(sc.get("handle_targets") or []),
        "sites": [s for s in sites if s], "torque_limits": limits,
        "base": {"max_speed": BASE_MAX_SPEED, "radius": BASE_RADIUS, "start": [float(start["xy"][0]), float(start["xy"][1]), BASE_Z], "yaw": float(start["yaw"])},
    }


def run_episode(job: Job, observer=None) -> dict:
    """One episode. Optional observer(event, env, base, action) records read-only snapshots.

    Events are reset, step (after physics), and final. Recording does not change the policy or termination.
    Exceptions, including observer failures, become explicit error episodes.
    """
    from ..native_warnings import capture_native_warnings
    with capture_native_warnings() as messages:
        result=_run_episode(job,observer,messages)
    result['mujoco_warning_messages']=messages
    result['mujoco_warnings']=len(messages)
    if messages:
        result['success']=False
        result['outcome']='native_failure'
        result.setdefault('native_failure',{})['messages']=messages
    return result


def _run_episode(job: Job, observer, native_messages) -> dict:
    require_benchmark_eligible(job.door, operation="episode evaluation")
    with open(os.path.join(job.door_dir, "spec.json")) as f:
        require_benchmark_eligible(json.load(f), operation="episode evaluation")
    t_wall = time.time()
    _WARN["n"] = 0
    ep = {"door_id": job.door["id"], "family": job.door["family"], "difficulty": job.door.get("difficulty"), "scenario": job.scenario, "suite": job.suite, "seed": job.seed, "tier": job.tier,
          "success": False, "outcome": "error", "sim_time": 0.0, "steps": 0, "wall_s": 0.0, "events": [], "reward_events": [], "episode_return": 0.0, "labels": {}, "error": None}
    env = None
    try:
        policy = _policy_for(job.policy_spec)
        pcls = type(policy)
        from .env import DoorEnv
        if getattr(policy, "embodiment", "hand_base") == "robot":
            env = pcls.make_env(job.door_dir, job.tier, job.seed)
        else:
            env = DoorEnv(job.door_dir, tier=job.tier, seed=job.seed)
        spec = env.spec
        sc = env.scenario(job.scenario)
        env.reset(scenario=sc, seed=job.seed, randomize=bool(job.randomize and job.seed > 0))
        ep['initialization_wall_s']=_r(time.time()-t_wall,3)
        if env.initialization_evidence is not None:ep['initialization_evidence']=env.initialization_evidence
        budget = float(job.time_budget_s or sc["time_budget_s"])
        mj, m, d = env.mj, env.m, env.d
        dt = float(m.opt.timestep)
        env.max_steps = int(math.ceil(budget / dt))
        period=getattr(policy,'control_period',None)
        preferred_dt=period(env) if callable(period) else getattr(policy,'control_dt',.01)
        control_dt = float(job.control_dt or preferred_dt or dt)
        decim = max(1, int(round(control_dt / dt)))
        control_dt = decim * dt
        start = env.start_pose
        robot = bool(env.robot_base)
        limits = torque_limits(env, job.door_dir)
        info = build_door_info(env, job, sc, budget, control_dt, limits, start)
        half = 0.5 * float(spec["opening"]["width"])
        base = SyntheticBase(start["xy"], half)
        sensor = AutoDoorSensor(env)
        policy.reset(info, env=env)
        from .site_forces import SiteForces
        site_forces = SiteForces(env, limits)
        # ---- fast accessors
        HINGE = int(mj.mjtJoint.mjJNT_HINGE)
        is_hinge = env.pj >= 0 and int(m.jnt_type[env.pj]) == HINGE
        jl = [(n, int(m.jnt_qposadr[env._jid(n)]), int(m.jnt_dofadr[env._jid(n)])) for n in info["joints"]]
        jdof = {n: dof for n, _, dof in jl}
        conditional_dofs={jdof[r['joint']]:r['joint']for r in env.meta.get('paired_leaf_holds',[])if r['joint']in jdof}
        sl = [(n, mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, n)) for n in info["sites"]]
        pq = int(m.jnt_qposadr[env.pj]) if env.pj >= 0 else -1
        pd_ = int(m.jnt_dofadr[env.pj]) if env.pj >= 0 else -1
        sj = env._jid(env.meta.get("secondary_joint")) if env.meta.get("secondary_joint") else -1
        sq = int(m.jnt_qposadr[sj]) if sj >= 0 else -1
        sd_ = int(m.jnt_dofadr[sj]) if sj >= 0 else -1
        base_bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, env.robot_base) if robot else -1
        goal = sc.get("goal")
        goal_xy = (float(goal["center"][0]), float(goal["center"][1])) if goal else None
        goal_r = float(goal["radius"]) if goal else 0.5
        L = env.tracker.L
        engaged = bool(spec.get("lock", {}).get("engaged"))
        has_closer = spec.get("closer", {}).get("model", "none") != "none" or bool(spec.get("kinematics", {}).get("self_closing"))
        scen = sc["name"]
        # after success, keep simulating a tail so closers return (slam / closed-after labels) and the base reaches the
        # goal zone before the episode ends (locked_recognize ends on the declaration / budget only)
        def tail_after_ok():
            if scen == "locked_recognize":
                return 0.0
            if goal:
                return 6.0 if (has_closer or t_goal is None) else 2.0
            return 1.0
        hm = env._human_mocap if getattr(env, "_human", None) is not None else -1

        def base_pos():
            return d.xpos[base_bid] if robot else base.pos

        tr = env.tracker
        prange_hi = float(m.jnt_range[env.pj][1]) if env.pj >= 0 and m.jnt_limited[env.pj] else None

        def clear_now():
            """Is the opening clear *right now*?  The reference rule of the labels (`LabelTracker.door_open_clear`:
            hinge >= 60 deg, slide >= 0.55 m or 95 % of the travel, overhead >= 1.9 m) evaluated instantaneously - the
            label itself is sticky and a door that closed again behind a person or under its closer must be
            re-opened - or the scenario's own clearance threshold."""
            if tr.passage.enabled:
                return env._door_clear_now()
            if env._door_clear_now():
                return True
            q = abs(float(d.qpos[pq])) if pq >= 0 else 0.0
            if is_hinge:
                return q >= tr.clear_angle
            lim = min(tr.clear_travel, 0.95 * prange_hi) if prange_hi is not None else tr.clear_travel
            return q >= max(lim, tr.open_thr)      # a lock-limited 2 mm range is not an opening

        def damaged_by_policy():
            """Damage after the policy first touched the door (a flap that slams on its own at reset, or the person of
            wait_for_human working the door, is the environment's doing)."""
            if not L.door_damaged:
                return False
            t_touch = L.time_to_touch if L.touched_door else None
            return t_touch is not None and any((e.get("t") or 0.0) >= t_touch - 1e-9 for e in L.damage_events)

        lean = robot     # robot embodiments read their own state from the env handle; skip the per-joint / per-site dicts
        t_episode=time.time()

        def obs():
            bp = base_pos()
            o = {"t": float(d.time), "door_q": float(d.qpos[pq]) if pq >= 0 else 0.0, "door_dq": float(d.qvel[pd_]) if pd_ >= 0 else 0.0,
                 "secondary_q": float(d.qpos[sq]) if sq >= 0 else None, "secondary_dq": float(d.qvel[sd_]) if sd_ >= 0 else None,
                 "joints": {} if lean else {n: {"q": float(d.qpos[qa]), "dq": float(d.qvel[da])} for n, qa, da in jl},
                 "sites": {} if lean else {n: d.site_xpos[sid].tolist() for n, sid in sl},
                 "base": {"pos": [float(bp[0]), float(bp[1]), float(bp[2])]},
                 "flags": {"touched": L.touched_door, "operator_actuated": L.operator_actuated, "latch_released": L.latch_released, "lock_released": L.lock_released,
                           "door_opened": L.door_opened, "door_open_clear": L.door_open_clear, "door_clear_now": clear_now(), "passed_through": L.robot_passed_through,
                           "closed_after": L.door_closed_after, "slammed": L.door_slammed, "damaged": L.door_damaged},
                 "locked": engaged and not (L.lock_released or env.unlocked_by_env),
                 "fired": list(env._fired), "return": float(env.episode_return), "success": bool(env.success),
                 "human_xy": [float(d.mocap_pos[hm][0]), float(d.mocap_pos[hm][1])] if hm >= 0 else None}
            o['passage_intervals'] = tr.passage.intervals(d)
            if env.meta.get('sectional_track') or env.meta.get('rollup_curtain'):
                from .lift_state import lift_state
                o['lift_state'] = lift_state(m, d, env.meta, with_velocity=True)
            return o

        prev = {k: False for k in EVENT_FLAGS}
        events = []
        t_goal = t_ok = t_damage = None
        outcome = "fail"
        action = {}
        tq = {}
        torques = []
        base_v = np.zeros(2)
        step = 0
        done = False
        stopped_by_policy = False
        if observer is not None:
            observer("reset", env, base_pos().copy(), {"torque_limits": dict(limits)})
        while not done:
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
                if action.get("knock"):
                    env.knock()
                    if env._knock_t is not None and not any(e[0] == "knock" for e in events):
                        events.append(["knock", round(t, 3)])
                if action.get("declare_locked"):
                    env.declare_locked()
                    events.append(["declare_locked", round(t, 3)])
                    stopped_by_policy = True
                if time.time() - t_episode > job.wall_timeout_s:
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
                applied_torque=False
                for dof, tau in torques:
                    if dof in conditional_dofs and site_forces.contacts.select(conditional_dofs[dof],d)is None:
                        continue
                    d.qfrc_applied[dof] += tau
                    applied_torque=True
                if applied_torque and env.tracker and not L.touched_door:
                    env.tracker.mark_touch(d, operator=False)
                if applied_torque and not L.touched_operator and any(n in env.operator_joints for n in tq):
                    env.tracker.mark_touch(d, operator=True)
            if not robot and (action.get("site_forces") or action.get('site_torques')):
                force_tau = site_forces.generalized(d, action.get("site_forces"),action.get('site_torques'))
                d.qfrc_applied[:] += force_tau
                if np.any(force_tau):
                    env.tracker.mark_touch(d, operator=True)
                for dof, limit in site_forces.limits.items():
                    d.qfrc_applied[dof] = np.clip(d.qfrc_applied[dof], -limit, limit)
            ctrl = action.get("ctrl")
            if ctrl is not None and m.nu:
                d.ctrl[:] = np.asarray(ctrl, float).ravel()[: m.nu]
            bp = base_pos()
            sensor.step(bp[:2], t, action)
            if not robot:
                intervals = tr.passage.intervals(d)
                clear_at_base = clear_now() if intervals is None else any(lo <= base.pos[0] <= hi for lo,hi in intervals)
                base.step(base_v, dt, clear_at_base)
            _, done = env.step(robot_base_pos=None if robot else base.pos)
            step += 1
            if observer is not None:
                observer("step", env, base_pos().copy(), action)
            t = float(d.time)
            # ---- termination
            warnings=[mj.mjtWarning(i).name for i,w in enumerate(d.warning) if w.number]
            if warnings or native_messages or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                ep['native_failure']={'warnings':warnings,'messages':list(native_messages),'time_s':t}
                outcome='native_failure'
                break
            if action.get('mechanism_failure'):
                ep['mechanism_failure']=str(action['mechanism_failure'])
                outcome='mechanism_failure'
                break
            if damaged_by_policy():
                t_damage = t_damage if t_damage is not None else t
                if t - t_damage > 0.5:
                    outcome = "damaged"
                    break
            if L.robot_fell:
                outcome = "fell"
                break
            if stopped_by_policy or action.get("done"):
                break
            bp = base_pos()
            if goal_xy is not None and t_goal is None and math.hypot(float(bp[0]) - goal_xy[0], float(bp[1]) - goal_xy[1]) <= goal_r:
                t_goal = t
                events.append(["goal_reached", round(t, 3)])
            tail = tail_after_ok()
            if tail > 0:
                if env.success:
                    t_ok = t_ok if t_ok is not None else t
                    if t - t_ok > tail:
                        break
                else:
                    t_ok = None
        if observer is not None:
            observer("final", env, base_pos().copy(), action)
        # ---- finalize
        labels = env.labels().to_dict()
        for k in EVENT_FLAGS:
            if labels.get(k) and not prev[k]:
                events.append([k, round(float(d.time), 3)])
        q_end = env._door_q()
        # damage that happened before the policy ever touched the door (a flap or hatch reset in an unstable open
        # pose slamming shut on its own) is the environment's doing, not the policy's: recorded, not counted
        t_touch = labels.get("time_to_touch")
        dmg_events = labels.get("damage_events") or []
        env_damage = bool(labels.get("door_damaged")) and not any((e.get("t") or 0.0) >= (t_touch if t_touch is not None else float("inf")) - 1e-9 for e in dmg_events)
        ok = bool(env.success)
        if env_damage and not ok:
            # re-evaluate the scenario criteria without the environment's own damage
            labels["door_damaged"] = False
            labels["door_slammed"] = False
            L.door_damaged = False
            L.door_slammed = False
            env._fired.pop("damage", None)
            env._fired.pop("slam", None)
            ok = bool(env.success)
        if outcome in ("fail", "damaged") and labels.get("door_damaged"):
            outcome = "damaged"
        if outcome in ('native_failure','mechanism_failure'):ok=False
        if ok and outcome not in ("timeout", "error"):
            outcome = "success"
        elif outcome == "fail" and labels.get("robot_fell"):
            outcome = "fell"
        fired = dict(env._fired)
        hum = getattr(env, "_human", None)
        ep.update({
            "success": ok, "outcome": outcome, "sim_time": _r(d.time, 3), "steps": int(step), "wall_s": _r(time.time() - t_wall, 3),
            "time_budget_s": budget, "control_dt": control_dt, "randomized": bool(job.randomize and job.seed > 0), "start": {"xy": [_r(v, 3) for v in start["xy"]], "yaw": _r(start["yaw"], 3)},
            "time_to_touch": _r(labels.get("time_to_touch"), 3), "time_to_open": _r(labels.get("time_to_open"), 3), "time_to_pass": _r(labels.get("time_to_pass"), 3),
            "time_to_goal": _r(t_goal, 3), "time_to_close": _r(fired.get("closed_behind", fired.get("closed")), 3),
            "expected_transit_s": sc.get("expected_transit_s"),
            "damage": bool(labels.get("door_damaged")), "env_damage": env_damage, "damage_events": [{"t": _r(e.get("t"), 3), "kind": e.get("kind"), "part": e.get("part"), "value": _r(e.get("value"), 1), "threshold": _r(e.get("threshold"), 1)} for e in dmg_events[:5]],
            "max_leaf_force_N": _r(labels.get("max_leaf_contact_force"), 1), "max_operator_torque": _r(labels.get("max_operator_torque"), 2),
            "max_door_angle": _r(labels.get("max_door_angle"), 4), "door_q_end": _r(q_end, 4), "energy_J": _r(labels.get("energy_J"), 2),
            "labels": {k: bool(labels.get(k)) for k in FLAG_KEYS}, "criteria": {c: bool(env._flag(c)) for c in sc.get("success", [])},
            "events": events, "reward_events": [[e["event"], _r(e["t"], 3), _r(e["reward"], 3)] for e in env.events], "episode_return": _r(env.episode_return, 3),
            "human_collision": bool(hum and hum.get("collided")) if hum is not None else None,
            "mujoco_warnings": int(_WARN["n"]), "base_end": [_r(v, 3) for v in base_pos()],
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
from ..result_aggregation import _mean, _median, _group_stats, lock_state_of, aggregate_suite, aggregate


# ----------------------------------------------------------------------------------------------- run
def git_commit(root: str) -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _error_episode(job: Job, err: str) -> dict:
    return {"door_id": job.door["id"], "family": job.door["family"], "difficulty": job.door.get("difficulty"), "scenario": job.scenario, "suite": job.suite, "seed": job.seed, "tier": job.tier,
            "success": False, "outcome": "error", "sim_time": 0.0, "steps": 0, "wall_s": 0.0, "events": [], "reward_events": [], "episode_return": 0.0, "labels": {}, "error": err[:400]}


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


def make_jobs(doors: list[dict], assets: str, suite: str, only: list[str] | None, seeds: list[int], tier: str, policy_spec: str, **kw) -> list[Job]:
    return [Job(door=d, door_dir=os.path.join(assets, "doors", d["id"]), scenario=s, seed=seed, tier=tier, policy_spec=policy_spec, **kw)
            for d in doors for s in scenarios_for(d, suite, only) for seed in seeds]


def run_benchmark(policy_spec: str, doors: str = "all", seeds: int | list[int] = 3, suite: str = "core", scenarios: str | None = None, workers: int = 8, tier: str = "full",
                  assets: str = "assets", time_budget_s: float | None = None, wall_timeout_s: float = 120.0, randomize: bool = True,
                  control_dt: float | None = None, label: str = "", out: str | None = None, progress=print) -> dict:
    """Evaluate `policy_spec` and return the result document (also written to `out` when given)."""
    import mujoco
    t0 = time.time()
    if suite not in SUITE_CHOICES:
        raise ValueError(f"suite must be one of {SUITE_CHOICES}, got {suite!r}")
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
    only = parse_scenarios(scenarios, suite)
    jobs = make_jobs(door_list, assets, suite, only, seed_list, tier, policy_spec, time_budget_s=time_budget_s, wall_timeout_s=wall_timeout_s, randomize=randomize, control_dt=control_dt)
    door_list = [d for d in door_list if scenarios_for(d, suite, only)]        # doors without a scenario in the suite are not evaluated
    scen_names = sorted({j.scenario for j in jobs}, key=SCENARIO_TYPES.index)
    progress(f"{pmeta['name']}: suite {suite}, {len(door_list)} doors, {len(jobs)} episodes ({', '.join(scen_names)} x {len(seed_list)} seed(s)), tier {tier}, {workers} worker(s)")
    if not jobs:
        raise ValueError(f"no episodes to run: none of the selected doors lists a scenario of the {suite!r} suite" + (f" among {only}" if only else ""))
    episodes = []
    state = {"n_done": 0, "n_ok": 0, "t_last": time.time()}

    def collect(ep):
        episodes.append(ep)
        state["n_done"] += 1
        state["n_ok"] += bool(ep["success"])
        if ep.get("outcome") == "error" and state["n_done"] <= 20:
            progress(f"  error on {ep['door_id']} {ep['scenario']} seed {ep['seed']}: {ep.get('error')}")
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
    episodes.sort(key=lambda e: (order.get(e["door_id"], 0), SCENARIO_TYPES.index(e["scenario"]), e["seed"]))
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
        "benchmark": {"name": "DoorBench", "dataset_version": manifest.get("version"), "dataset_generated": manifest.get("generated"), "n_doors_total": collection_counts(manifest["doors"])["n_doors_eligible"], **collection_counts(manifest["doors"]), **git_commit(root)},
        "policy": {**pmeta, "spec": policy_spec, "extra": extra},
        "run": {"date": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "label": label, "simulator": "mujoco", "simulator_version": mujoco.__version__, "tier": tier,
                "suite": suite, "scenarios": [{"name": s, "suite": SCENARIO_SUITE[s], "description": SCENARIO_DESCRIPTIONS[s]} for s in scen_names], "scenario_filter": scenarios or "all",
                "seeds": seed_list, "n_doors": len(door_list), "door_selection": doors, "time_budget_s": time_budget_s or "per scenario (spec.json benchmark.scenarios[].time_budget_s)",
                "randomize": bool(randomize), "control_dt": control_dt or pmeta["control_dt"], "workers": workers, "wall_time_s": round(wall, 1),
                "host": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "cpu_count": os.cpu_count()},
                "command": " ".join(sys.argv)},
        "aggregate": aggregate(episodes, doors_by_id),
        "episodes": episodes,
    }
    if out:
        write_result(result, out)
        progress(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
    for s, a in result["aggregate"].items():
        progress(f"{pmeta['name']} [{s} suite]: {a['doors_solved']} / {a['n_doors']} doors solved on every episode ({a['doors_solved_any']} on at least one), "
                 f"episode success {a['success_rate'] * 100:.1f} %, damage {a['damage_rate'] * 100:.1f} %"
                 + (f", human collisions {a['human_collision_rate'] * 100:.1f} %" if s == "human" else "") + f"; {a['mean_wall_s']} s wall / episode")
    progress(f"{wall:.0f} s wall in total")
    return result


def write_result(result: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")


def dry_run(doors: str, suite: str, scenarios: str | None, seeds: int, assets: str = "assets", out=print):
    manifest = load_manifest(assets)
    door_list = select_doors(manifest, doors)
    only = parse_scenarios(scenarios, suite)
    rows = [(d, scenarios_for(d, suite, only)) for d in door_list]
    n_eps = sum(len(s) for _, s in rows) * seeds
    n_doors = sum(1 for _, s in rows if s)
    out(f"suite {suite}: {n_doors} doors ({len(door_list)} selected) x their scenarios x {seeds} seed(s) = {n_eps} episodes")
    for d, s in rows:
        out(f"{d['id']:28s} {d['family']:22s} L{d.get('difficulty', '?')}  {d.get('lock', 'none'):18s} {'locked' if d.get('lock_engaged') else '      '}  {', '.join(s) if s else '(no scenario in this suite)'}")
