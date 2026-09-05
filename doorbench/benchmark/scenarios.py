"""Benchmark scenarios and reward criteria, emitted per door into ``spec.json["benchmark"]``.

Every door lists one or more *scenarios*.  A scenario fixes the initial door state, where the robot starts (a
randomisable start zone), what it must touch (handle targets = grip / push sites of ``model.json``), the plane it
must cross, the goal zone, an optional simulated human, a reward table (event -> value), a time budget and an
expected transit time.  ``docs/BENCHMARK.md`` documents the schema and every formula used below.

Public API
  build_benchmark(spec, phys, model_dict)      -> the whole ``benchmark`` block (seeded scenario assignment)
  make_scenario(name, spec, phys, model_dict)  -> one scenario of any type for any door (used by DoorEnv / tests)
  sample_start(scenario, seed)                 -> {"xy": [x, y], "z": z, "yaw": yaw} drawn from the start zone
  expected_transit_time(...)                   -> the per-scenario time estimate (documented formula)
"""
from __future__ import annotations

import math
import random

from ..benchmark_eligibility import benchmark_eligibility, is_benchmark_eligible, require_benchmark_eligible

SCENARIO_TYPES = ("open_and_traverse", "open_then_close", "close_only", "unlock_and_traverse", "locked_recognize",
                  "hold_open_for_human", "wait_for_human", "knock_and_wait")

# ---- suites.  Human-interaction scenarios are segregated: the `core` suite needs nothing but the door and the robot
# and is the default everywhere (runner, DoorEnv, viewer, result tables).  The `human` suite adds a kinematic simulated
# person (hold_open / wait) or social etiquette that presumes one (knock) and is an advanced, opt-in tier.
CORE_SCENARIOS = ("open_and_traverse", "open_then_close", "close_only", "unlock_and_traverse", "locked_recognize")
HUMAN_SCENARIOS = ("hold_open_for_human", "wait_for_human", "knock_and_wait")
SUITES = ("core", "human")
SCENARIO_SUITE = {**{n: "core" for n in CORE_SCENARIOS}, **{n: "human" for n in HUMAN_SCENARIOS}}
assert set(SCENARIO_SUITE) == set(SCENARIO_TYPES)


def suite_of(name: str) -> str:
    """'core' (no human involved, default suite) or 'human' (advanced, opt-in)."""
    return SCENARIO_SUITE[name]


def scenarios_in_suite(names, suite: str) -> list:
    """Filter scenario names: suite in {'core', 'human', 'all'}."""
    return list(names) if suite == "all" else [n for n in names if SCENARIO_SUITE[n] == suite]

SCENARIO_DESCRIPTIONS = {
    "open_and_traverse": "Start in the start zone, reach the handle, unlatch, open and walk through the door plane to the goal zone.",
    "open_then_close": "Open, walk through, then close the door behind you (latched if the door has a latch).",
    "close_only": "The door starts open; close it (and latch it) without slamming.",
    "unlock_and_traverse": "The door is locked but releasable from the robot's side (thumbturn, keypad code, slide bolt, badge, REX ...): release the lock, open and walk through.",
    "locked_recognize": "The door is locked with no release on the robot's side: probe, recognise the lock, declare it (env.declare_locked()) and stop without damage.",
    "hold_open_for_human": "A person walks up behind the robot: open the door, hold it open until the person has walked through, release it, then walk through yourself.",
    "wait_for_human": "A person comes through the door from the other side first: yield (no contact, stay out of the doorway), then open and walk through.",
    "knock_and_wait": "Knock on the closed leaf (5-200 N), wait at least 3 s, then open and walk through.",
}

# ---- reward values (event -> reward).  Positive events fire once per episode; time penalty is per second.
R = {
    "touch_handle": 1.0, "unlatch": 2.0, "unlock": 3.0, "opened": 3.0, "traversed": 10.0, "closed_behind": 3.0, "latched_behind": 1.0,
    "closed": 5.0, "latched": 2.0, "held_for_human": 5.0, "yielded_to_human": 5.0, "recognized_locked": 5.0, "knocked": 2.0, "waited": 3.0,
    "collision_with_human": -20.0, "damage": -10.0, "slam": -2.0, "hardware_misuse": -5.0, "time_penalty_per_s": -0.05,
}

EVENT_DESCRIPTIONS = {
    "touch_handle": "robot contacts the operator (handle / bar / pull) or moves the operator joint by >10 % of its travel",
    "unlatch": "latch bolt retracted >= 80 % of its throw (label latch_released)",
    "unlock": "engaged lock released from the robot side (label lock_released)",
    "opened": "primary joint past the open threshold (30 deg hinged / rotor, min(0.3 m, half travel) sliding)",
    "traversed": "robot base crossed the pass plane inside the opening (label robot_passed_through)",
    "closed_behind": "door back within the closed threshold after the robot passed",
    "latched_behind": "latch bolt re-extended (< 20 % throw) with the door closed after the robot passed",
    "closed": "door (which started open) brought within the closed threshold",
    "latched": "latch bolt extended with the door closed",
    "held_for_human": "the human crossed the pass plane while the opening was clear (door_open_clear) and without contact",
    "yielded_to_human": "the human finished its path with no contact and the robot did not cross the plane before the human",
    "recognized_locked": "env.declare_locked() called while the door is closed and undamaged",
    "knocked": "robot leaf contact between 5 N and the dent threshold while the door is closed",
    "waited": "door not opened for at least 3 s after the knock",
    "collision_with_human": "contact between a robot geom and the human capsule, or base within (r_robot + r_human) of the human",
    "damage": "any damage event (dent, puncture, glass, operator yield, latch shear, hinge tear-out, forced maglock)",
    "slam": "closing speed at the stop above the slam threshold",
    "hardware_misuse": "operator torque / force beyond its yield (label hardware_misuse)",
    "time_penalty_per_s": "applied every step as value * dt",
}

ROBOT = {"walk_speed_m_s": 0.7, "body_radius_m": 0.30, "height_m": 1.7, "push_force_N": 40.0, "lift_force_N": 120.0,
         "note": "nominal humanoid used for the expected-transit-time formula and the start-zone clearance"}
HUMAN = {"radius_m": 0.22, "height_m": 1.75, "speed_m_s": 1.1}

FREE_SWING = ("saloon", "strip_curtain", "pet_door", "revolving", "turnstile_tripod", "turnstile_fullheight")
SLIDING_LIKE = ("sliding_single", "sliding_bypass", "gate_sliding", "garage_sectional", "garage_tiltup", "rollup", "bifold", "accordion", "hatch_floor", "hatch_ceiling")
HUMAN_ELIGIBLE = ("swing_single", "swing_double", "pivot", "gate_swing", "cold_storage")
KNOCK_CONTEXTS = ("residential_interior", "commercial_office", "institutional")


# ---------------------------------------------------------------------------
# geometry helpers (model.json dict)
# ---------------------------------------------------------------------------
def _qrot(q, v):
    w, x, y, z = q
    cx, cy, cz = x, y, z
    # t = 2 * cross(q_xyz, v)
    tx, ty, tz = 2 * (cy * v[2] - cz * v[1]), 2 * (cz * v[0] - cx * v[2]), 2 * (cx * v[1] - cy * v[0])
    return (v[0] + w * tx + (cy * tz - cz * ty), v[1] + w * ty + (cz * tx - cx * tz), v[2] + w * tz + (cx * ty - cy * tx))


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def body_world_frames(model: dict) -> dict:
    """{body_name: (pos, quat)} in world coordinates at the authored (rest) configuration."""
    frames = {}
    by_name = {b["name"]: b for b in model["bodies"]}
    for b in model["bodies"]:
        chain = []
        cur = b
        while cur is not None:
            chain.append(cur)
            cur = by_name.get(cur["parent"]) if cur.get("parent") else None
        p, q = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
        for node in reversed(chain):
            off = _qrot(q, tuple(node["pos"]))
            p = (p[0] + off[0], p[1] + off[1], p[2] + off[2])
            q = _qmul(q, tuple(node["quat"]))
        frames[b["name"]] = (p, q)
    return frames


def site_world_positions(model: dict) -> dict:
    """{site_name: {"pos": [x, y, z], "body": body, "role": role}} for every site of the model."""
    frames = body_world_frames(model)
    out = {}
    for b in model["bodies"]:
        p, q = frames[b["name"]]
        for s in b.get("sites", []):
            off = _qrot(q, tuple(s["pos"]))
            out[s["name"]] = {"pos": [p[0] + off[0], p[1] + off[1], p[2] + off[2]], "body": b["name"], "role": s.get("role", "")}
    return out


def _ancestors(model: dict, body: str) -> list:
    by_name = {b["name"]: b for b in model["bodies"]}
    out = []
    cur = by_name.get(body)
    while cur is not None:
        out.append(cur["name"])
        cur = by_name.get(cur["parent"]) if cur.get("parent") else None
    return out


def _joint_body(model: dict, joint: str | None) -> str | None:
    for b in model["bodies"]:
        if b.get("joint") and b["joint"]["name"] == joint:
            return b["name"]
    return None


def active_leaf_body(model: dict) -> str | None:
    """Body of the leaf the robot operates: the operator joint's leaf for pairs, else the primary joint's body."""
    meta = model.get("meta", {})
    leaf_joints = {j for j in (meta.get("primary_joint"), meta.get("secondary_joint")) if j}
    leaf_bodies = {_joint_body(model, j) for j in leaf_joints}
    op_body = _joint_body(model, meta.get("operator_joint"))
    if op_body:
        for anc in _ancestors(model, op_body):
            if anc in leaf_bodies:
                return anc
    return _joint_body(model, meta.get("primary_joint"))


def handle_targets(model: dict) -> list:
    """Grip / push site names the robot should reach, preferring those on the active leaf."""
    sites = site_world_positions(model)
    leaf = active_leaf_body(model)
    on_leaf, others = [], []
    for name, s in sites.items():
        if s["role"] not in ("grip", "push"):
            continue
        if leaf and leaf in _ancestors(model, s["body"]):
            on_leaf.append(name)
        else:
            others.append(name)
    return on_leaf or others


# ---------------------------------------------------------------------------
# expected transit time
# ---------------------------------------------------------------------------
def _dist_xy(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def t_operate(spec: dict, phys: dict, unlock: bool, model: dict | None = None) -> float:
    """Seconds to work the hardware: 0 without an operator; 1.0 s lever / paddle / push / pull; 1.5 s knob, thumb latch,
    T-handle, cremone, handleset, lift latch, hook; 3 s wheel; 1.5 s per dog; + 2 s thumbturn / slide bolt / key /
    badge, + 1 s per keypad digit + 1 s."""
    from .. import hardware as H
    op = H.OPERATORS.get(spec["operator"]["model"])
    kind = op.kind if op else "none"
    t = 0.0
    if kind == "none":
        t = 0.0
    elif kind in ("lever", "paddle", "push_plate", "pull", "flush_pull", "ring_pull", "panic_touchbar", "panic_crossbar", "card_lever", "push_button_screen", "keypad_lever"):
        t = 1.0
    elif kind == "wheel":
        t = 3.0
    else:
        t = 1.5
    n_dogs = sum(1 for b in (model or {}).get("bodies", []) if b.get("joint") and b["joint"]["name"].startswith("dog_")) if model else 0
    if n_dogs:
        t += 1.5 * n_dogs                                    # every dog is thrown individually
    if unlock:
        code = spec["lock"].get("code")
        t += (1.0 + 1.0 * len(code)) if code else 2.0
    return t


def t_open_dynamics(spec: dict, phys: dict, clear: dict) -> float:
    """Time to open to the clearance threshold under the nominal push / lift force (see docs/BENCHMARK.md)."""
    kin = spec["kinematics"]["type"]
    fam = spec["family"]
    F = ROBOT["push_force_N"]
    m = phys["mass"]["total_kg"]
    W = spec["leaf"]["width"]
    if fam in FREE_SWING:
        if kin == "rotor":
            theta = clear["angle_rad"]
            return max(0.8, theta / 0.9)             # rotor turned at ~0.9 rad/s (revolving door / turnstile)
        return 0.5
    if kin in ("hinge_vertical", "hinge_horizontal"):
        I = phys.get("inertia_about_hinge_kg_m2") or (m * W * W / 3)
        r = max(0.3, W - 0.08)
        tau_res = phys.get("closer", {}).get("spring_preload_Nm", 0.0) + phys.get("hinge", {}).get("coulomb_torque_Nm", 0.0)
        if kin == "hinge_horizontal":
            cb = float(spec["kinematics"].get("counterbalance_fraction", 0.0) or 0.0)
            L = spec["leaf"]["height"]
            tau_res += m * 9.81 * (L / 2) * (1.0 - cb) * 0.5     # mean gravity moment over the lift
            F = ROBOT["lift_force_N"]
            r = L
        tau_net = max(0.1 * F * r, F * r - tau_res)
        b_open = phys.get("closer", {}).get("damping_opening", 0.0) + phys.get("hinge", {}).get("total_damping_symmetric", 0.0)
        theta = clear["angle_rad"]
        t = math.sqrt(2 * theta * I / tau_net) + theta * b_open / tau_net
        return min(12.0, max(0.6, t))
    if kin == "slide_horizontal":
        Fr = phys.get("roller", {}).get("coulomb_force_N", 5.0) or 5.0
        b = phys.get("roller", {}).get("viscous_damping_N_s_per_m", 0.0) or 0.0
        F_net = max(0.1 * F, F - Fr)
        d = clear["travel_m"]
        m_leaf = m / max(1, spec["leaf"].get("count", 1))
        t = math.sqrt(2 * d * m_leaf / F_net) + d * b / F_net
        return min(12.0, max(0.6, t))
    if kin == "slide_vertical":
        cb = float(spec["kinematics"].get("counterbalance_fraction", 0.0) or 0.0)
        Fr = phys.get("roller", {}).get("coulomb_force_N", 10.0) or 10.0
        lift = m * 9.81 * (1 - cb) + Fr
        v = 0.4 if lift <= ROBOT["lift_force_N"] else 0.4 * ROBOT["lift_force_N"] / lift
        return min(20.0, max(1.0, clear["travel_m"] / max(v, 0.05)))
    return 1.5


def clearance_thresholds(spec: dict, model: dict) -> dict:
    """Opening the robot needs: 60 deg (or max_open if smaller) for hinged leaves, min(0.55 m, travel) for sliders,
    1.9 m (or travel) for vertical doors, one sector for rotors."""
    kin = spec["kinematics"]["type"]
    meta = model.get("meta", {})
    prim = None
    for b in model["bodies"]:
        if b.get("joint") and b["joint"]["name"] == meta.get("primary_joint"):
            prim = b["joint"]
    rng = prim["range"] if prim and prim.get("range") else None
    if kin in ("hinge_vertical", "hinge_horizontal"):
        mo = math.radians(spec["kinematics"].get("max_open_deg") or 90)
        ang = min(math.radians(60), mo) if kin == "hinge_vertical" else min(math.radians(75), mo)
        return {"angle_rad": ang, "travel_m": None, "open_rad": math.radians(30), "open_m": None}
    if kin == "rotor":
        sector = math.radians(float(meta.get("ratchet_deg") or 90.0))
        return {"angle_rad": sector, "travel_m": None, "open_rad": math.radians(30), "open_m": None}
    travel = float(spec["kinematics"].get("travel_m") or (rng[1] - rng[0] if rng else 0.6) or 0.6)
    if kin == "slide_vertical":
        d = min(1.9, travel)
    else:
        d = min(0.55, travel)
    return {"angle_rad": None, "travel_m": d, "open_rad": None, "open_m": min(0.3, 0.5 * travel)}


def expected_transit_time(name: str, spec: dict, phys: dict, start_center, targets_xy, plane_xy, goal_xy, clear: dict, human: dict | None, model: dict | None = None) -> dict:
    """expected = t_approach + t_operate + t_open + t_pass + t_scenario (all terms returned for the docs / viewer)."""
    v = ROBOT["walk_speed_m_s"]
    reach = targets_xy or plane_xy
    d_approach = max(0.0, _dist_xy(start_center, reach) - 0.6)     # stop 0.6 m short of the handle (arm reach)
    t_appr = d_approach / v
    unlock = name == "unlock_and_traverse"
    t_op = t_operate(spec, phys, unlock, model) if name != "close_only" else 0.5
    t_opn = t_open_dynamics(spec, phys, clear) if name != "close_only" else 0.0
    d_pass = _dist_xy(plane_xy, goal_xy) + 0.6
    t_pass = d_pass / v if name not in ("close_only", "locked_recognize") else 0.0
    extra = 0.0
    closer_t = phys.get("closer", {}).get("closing_time_est_s")
    if closer_t and name not in ("close_only", "locked_recognize") and closer_t < t_pass:
        extra += 1.0                                              # self-closing faster than the walk: hold / re-open once
    W = spec["leaf"]["width"]
    if name in ("open_then_close", "close_only"):
        extra += 2.0 + W / v                                      # walk back around the leaf and pull / slide it shut
    if name == "knock_and_wait":
        extra += 1.0 + 3.0
    if name == "locked_recognize":
        extra += 4.0                                              # probe the hardware, then declare
    if human and name == "hold_open_for_human":
        extra += human["path"][-1][0] - (t_appr + t_op + t_opn) + 0.5 if human["path"] else 3.0
        extra = max(extra, 2.0)
    if human and name == "wait_for_human":
        extra += human["path"][-1][0] if human["path"] else 4.0    # wait for the person to clear the doorway
    total = t_appr + t_op + t_opn + t_pass + extra
    return {"approach_s": round(t_appr, 2), "operate_s": round(t_op, 2), "open_s": round(t_opn, 2), "pass_s": round(t_pass, 2), "scenario_extra_s": round(extra, 2), "total_s": round(total, 2)}


# ---------------------------------------------------------------------------
# scenario construction
# ---------------------------------------------------------------------------
def _path_from_points(points, speed, t0):
    """(t, x, y) waypoints walking the polyline at constant speed starting at t0."""
    out = [[round(t0, 3), round(points[0][0], 4), round(points[0][1], 4)]]
    t = t0
    for a, b in zip(points[:-1], points[1:]):
        t += _dist_xy(a, b) / speed
        out.append([round(t, 3), round(b[0], 4), round(b[1], 4)])
    return out


def make_scenario(name: str, spec: dict, phys: dict, model: dict) -> dict:
    require_benchmark_eligible(spec, operation="scenario generation")
    if name not in SCENARIO_TYPES:
        raise KeyError(name)
    meta = model.get("meta", {})
    sites = site_world_positions(model)
    appr = sites.get("approach_point", {}).get("pos", [0.0, -1.5, 0.0])
    goal = sites.get("goal_point", {}).get("pos", [0.0, 1.5, 0.0])
    plane = sites.get("door_plane_center", {}).get("pos", [0.0, 0.0, 1.0])
    horizontal = bool(meta.get("horizontal"))
    op = spec["opening"]
    normal = [0.0, 0.0, 1.0] if horizontal else [0.0, 1.0, 0.0]
    targets = handle_targets(model)
    targets_xy = None
    if targets:
        pts = [sites[t]["pos"] for t in targets]
        targets_xy = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    # ---- start zone: outside the swing arc of a leaf that opens toward the robot, at the spec'd start distance
    W = spec["leaf"]["width"]
    opens_toward_robot = spec["kinematics"]["type"] == "hinge_vertical" and not spec["robot"].get("is_push", True) and spec["family"] not in FREE_SWING
    arc = (W + 0.45) if opens_toward_robot else 0.0
    dist = max(float(spec["robot"].get("start_distance_m", 1.5)), abs(appr[1]), arc, 1.2)
    start_c = [round(appr[0], 3), round(-dist if appr[1] <= 0 else dist, 3), round(appr[2], 3)]
    yaw0 = math.atan2(plane[1] - start_c[1], plane[0] - start_c[0])
    radius = 0.30
    start = {"center": start_c, "radius": radius, "yaw": round(yaw0, 4), "yaw_range": [round(yaw0 - 0.35, 4), round(yaw0 + 0.35, 4)],
             "randomize": {"position": "uniform_disc", "radius": radius, "yaw_jitter_rad": 0.35, "seed_base": int(spec["seed"] % 100000),
                           "formula": "r = R*sqrt(u1), phi = 2*pi*u2, yaw = yaw0 + (2*u3 - 1)*yaw_jitter; u ~ U(0,1) from random.Random(seed_base + seed)"}}
    clear = clearance_thresholds(spec, model)
    plane_d = {"center": [round(float(c), 4) for c in plane], "normal": normal, "width": op["width"], "height": op["height"], "traverse_direction": [0.0, 0.0, (1.0 if goal[2] > plane[2] else -1.0)] if horizontal else [0.0, 1.0 if goal[1] > appr[1] else -1.0, 0.0]}
    goal_d = {"center": [round(float(c), 4) for c in goal], "radius": 0.5}
    side = 1.0 if float(meta.get("handle_cam_x", 0.3) or 0.3) >= 0 else -1.0     # latch-edge side of the opening (x sign)
    human = None
    initial = {"door": "closed", "lock_engaged": bool(spec["lock"].get("engaged")), "latched": spec["latch"]["model"] != "none"}
    has_operator = bool(meta.get("operator_joint")) or bool(targets)
    has_latch = spec["latch"]["model"] != "none" and any(b.get("joint") and b["joint"]["role"] == "latch" for b in model["bodies"])
    rewards = {}

    def add(*events):
        for e in events:
            rewards[e] = R[e]
    if name in ("open_and_traverse", "open_then_close", "unlock_and_traverse", "hold_open_for_human", "wait_for_human", "knock_and_wait"):
        if has_operator:
            add("touch_handle")
        if has_latch:
            add("unlatch")
        add("opened", "traversed", "damage", "slam", "time_penalty_per_s")
        success = ["opened", "traversed", "!damage"]
    if name == "open_then_close":
        add("closed_behind")
        if has_latch:
            add("latched_behind")
        success = ["opened", "traversed", "closed_behind", "!damage", "!slam"] + (["latched_behind"] if has_latch else [])
    if name == "close_only":
        initial["door"] = "open"
        if has_operator:
            add("touch_handle")
        add("closed", "damage", "slam", "time_penalty_per_s")
        if has_latch:
            add("latched")
        success = ["closed", "!damage", "!slam"] + (["latched"] if has_latch else [])
    if name == "unlock_and_traverse":
        add("unlock")
        success = ["unlock", "opened", "traversed", "!damage"]
    if name == "locked_recognize":
        if has_operator:
            add("touch_handle")
        add("recognized_locked", "damage", "hardware_misuse", "time_penalty_per_s")
        success = ["!opened", "!damage", "!hardware_misuse"]
    if name in ("hold_open_for_human", "wait_for_human"):
        add("collision_with_human")
        sp = HUMAN["speed_m_s"]
        if name == "hold_open_for_human":
            add("held_for_human")
            pre = expected_transit_time("open_and_traverse", spec, phys, start_c, targets_xy, (plane[0], plane[1]), (goal[0], goal[1]), clear, None, model)
            t0 = pre["approach_s"] + pre["operate_s"] + pre["open_s"] + 0.5
            pts = [(start_c[0] + side * 0.8, start_c[1] - 1.6), (plane[0] + side * 0.25, plane[1] - 0.9), (plane[0], plane[1]), (goal[0], goal[1] + 0.8)]
            human = {"radius_m": HUMAN["radius_m"], "height_m": HUMAN["height_m"], "speed_m_s": sp, "start_t_s": round(t0, 2), "direction": "same_as_robot",
                     "path": _path_from_points(pts, sp, t0), "waits_at_closed_door": True,
                     "note": "kinematic capsule; pauses 0.7 m before the plane while the opening is not clear"}
            success = ["held_for_human", "traversed", "!collision_with_human", "!damage"]
        else:
            add("yielded_to_human")
            pts = [(plane[0], goal[1] + 1.6), (plane[0], plane[1]), (plane[0] - side * 0.5, plane[1] - 1.0), (start_c[0] - side * 1.1, start_c[1] - 0.9)]
            human = {"radius_m": HUMAN["radius_m"], "height_m": HUMAN["height_m"], "speed_m_s": sp, "start_t_s": 0.5, "direction": "opposite_to_robot",
                     "path": _path_from_points(pts, sp, 0.5), "waits_at_closed_door": False,
                     "note": "kinematic capsule; the environment opens the door for the person (servo on the leaf + operator) while they are within 1.2 m of the plane"}
            success = ["yielded_to_human", "opened", "traversed", "!collision_with_human", "!damage"]
    if name == "knock_and_wait":
        add("knocked", "waited")
        success = ["knocked", "waited", "opened", "traversed", "!damage"]
    thr = {"open_rad": clear["open_rad"], "open_m": clear["open_m"], "clear_rad": clear["angle_rad"], "clear_m": clear["travel_m"]}
    tt = expected_transit_time(name, spec, phys, start_c, targets_xy, (plane[0], plane[1]), (goal[0], goal[1]), clear, human, model)
    budget = 5 * math.ceil((3.0 * tt["total_s"] + 10.0) / 5.0)
    return {
        "name": name, "suite": SCENARIO_SUITE[name], "requires_human": human is not None,
        "description": SCENARIO_DESCRIPTIONS[name], "initial_state": initial, "start": start,
        "approach_point": [round(float(c), 4) for c in appr], "handle_targets": targets, "pass_plane": plane_d,
        "goal": goal_d if name not in ("close_only", "locked_recognize") else None, "human": human,
        "thresholds": thr, "rewards": rewards, "success": success,
        "time_budget_s": float(budget), "expected_transit_s": tt["total_s"], "expected_transit_terms": tt,
    }


def assign_scenarios(spec: dict) -> list:
    """Seeded per-door scenario list (see docs/BENCHMARK.md, 'Scenario assignment')."""
    if not is_benchmark_eligible(spec):
        return []
    rng = random.Random(int(spec["seed"]) * 1000003 + 17)
    lock = spec["lock"]
    locked, releasable = bool(lock.get("engaged")), bool(lock.get("robot_side_release", True))
    fam = spec["family"]
    kin = spec["kinematics"]["type"]
    if locked and releasable:
        out = ["unlock_and_traverse"]
    elif locked:
        out = ["locked_recognize"]
    else:
        out = ["open_and_traverse"]
    self_closing = (spec["closer"]["model"] not in ("none", "gas_strut")) or bool(spec["kinematics"].get("self_closing")) or fam in ("automatic_swing", "automatic_sliding", "elevator")
    if fam in SLIDING_LIKE and not (locked and not releasable):
        out.append("open_then_close")
        if rng.random() < 0.35:
            out.append("close_only")
    elif kin == "hinge_vertical" and not locked and not self_closing and fam not in FREE_SWING and fam not in ("baby_gate", "stall"):
        if rng.random() < 0.30:
            out.append("open_then_close")
        if rng.random() < 0.15:
            out.append("close_only")
    if fam in HUMAN_ELIGIBLE and not locked and rng.random() < 0.20:
        out.append(rng.choice(["hold_open_for_human", "wait_for_human"]) if not self_closing else ("hold_open_for_human" if rng.random() < 0.7 else "wait_for_human"))
    if fam == "swing_single" and spec.get("context") in KNOCK_CONTEXTS and not locked and rng.random() < 0.08:
        out.append("knock_and_wait")
    return out


def build_benchmark(spec: dict, phys: dict, model: dict) -> dict:
    names = assign_scenarios(spec)
    assert not names or SCENARIO_SUITE[names[0]] == "core", names          # the primary (default) scenario never needs a person
    scen = [make_scenario(n, spec, phys, model) for n in names]
    return {"schema_version": "1.1", "robot": ROBOT, "human": HUMAN, "primary_scenario": names[0] if names else None,
            "benchmark_eligibility": benchmark_eligibility(spec),
            "suites": {s: scenarios_in_suite(names, s) for s in SUITES}, "scenarios": scen,
            "reward_values": R, "event_descriptions": EVENT_DESCRIPTIONS}


def benchmark_summary(bench: dict) -> dict:
    """Compact form for manifest.json."""
    p = bench["scenarios"][0] if bench["scenarios"] else {}
    names = [s["name"] for s in bench["scenarios"]]
    return {"scenarios": names, "primary": bench["primary_scenario"], "core": scenarios_in_suite(names, "core"),
            "human": scenarios_in_suite(names, "human"), "time_budget_s": p.get("time_budget_s"),
            "benchmark_eligibility": bench.get("benchmark_eligibility"),
            "expected_transit_s": p.get("expected_transit_s"), "has_human": any(s.get("human") for s in bench["scenarios"])}


def sample_start(scenario: dict, seed: int = 0) -> dict:
    """Draw a robot start pose from the scenario's start zone (documented formula; deterministic in seed)."""
    st = scenario["start"]
    rz = st.get("randomize", {})
    rng = random.Random(int(rz.get("seed_base", 0)) + int(seed))
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    R_ = float(rz.get("radius", st.get("radius", 0.3)))
    r = R_ * math.sqrt(u1)
    phi = 2 * math.pi * u2
    cx, cy, cz = st["center"]
    yaw = st["yaw"] + (2 * u3 - 1) * float(rz.get("yaw_jitter_rad", 0.35))
    return {"xy": [cx + r * math.cos(phi), cy + r * math.sin(phi)], "z": cz, "yaw": yaw, "seed": int(seed)}


def human_pose(human: dict, t: float):
    """Interpolated (x, y) of the human at path time t (clamped to the path ends)."""
    path = human["path"]
    if t <= path[0][0]:
        return path[0][1], path[0][2]
    for a, b in zip(path[:-1], path[1:]):
        if t <= b[0]:
            s = (t - a[0]) / max(1e-6, b[0] - a[0])
            return a[1] + s * (b[1] - a[1]), a[2] + s * (b[2] - a[2])
    return path[-1][1], path[-1][2]
