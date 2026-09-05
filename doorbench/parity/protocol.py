"""The Isaac parity protocol: data + pure functions shared by the MuJoCo and the Isaac Lab runner.

Nothing in this module imports a simulator.  A runner
  1. builds the per-door inputs once (``door_inputs``; the MuJoCo runner adds the gravity bias it measures),
  2. for every phase in ``PHASES`` resets (or continues), and every physics step asks ``phase_efforts`` which
     generalized forces to apply (MJCF joint names; the Isaac runner maps them to USD / RL-slot joints, drops the
     ones that do not exist in that file) and ``tendon_min_positions`` for the one-sided latch coupling,
  3. records curves at ``SAMPLE_HZ`` (DoorBench joint coordinates) and hands them to ``phase_metrics`` +
     ``phase_status`` so both simulators are judged by the same code,
  4. ``compare_door`` turns a MuJoCo record and a PhysX record into a per-phase verdict, discrepancy codes and a grade.

The schedule, efforts and thresholds are those of ``doorbench.qa.run_qa`` (the sign-off QA), expressed in simulated
time so that a 500 Hz MuJoCo run and a 120 Hz PhysX run follow the same protocol:

  settle   1.0 s free                                   (qa: settle)
  hold     push on the primary joint, 1 s (holding doors) or up to 6 s (free doors)   (qa: hold / free_opens)
  operate  thumbturn 2 N*m (t<1.2), aux bolts 3 N*m / 60 N, dogs 14 N*m, operator effort from 0.6 s, push from 1.2 s
           while q < 50 deg; 6.4 s                      (qa: actuate_opens)
  release  0.8 s, primary joint pinned, no efforts       (qa: latch_returns)
  relatch  close drive 6 s, re-push 1 s                  (qa: relatch)
  closer   12 s free from min(60 deg, 0.8 max_open)      (qa: closer_returns)
  locked   operator 6 N*m / 150 N + push, 2 s            (qa: locked_holds)

Joint values are DoorBench coordinates everywhere (MuJoCo q; USD q + ``doorbench:zero_offset``).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Iterable

from .. import hardware as H
from ..qa import door_flags, push_base, CLOSE_RATE_RAD_S, FREE_SWING_FAMILIES, PUSH_BASE_MAX, PUSH_BASE_MIN, PUSH_CAP

PROTOCOL_VERSION = "1.1"
# 1.1: the relatch closing drive is rate limited (qa.CLOSE_RATE_RAD_S).  Driven flat out it swung a 120 deg leaf
#      shut at 5.9 rad/s - 12 mm of leaf-edge travel per 2 ms step - and the slab tunnelled through the frame stop,
#      leaving the latch bolt wedged outside its strike.  The hand now stops pushing once the leaf is already
#      closing at a human ~1.3 m/s, which is what qa.run_qa does.  Records made at 1.0 used the un-limited drive.
# Version of the *metric definitions* (``phase_metrics``).  The protocol (schedule, efforts, expectations) is unchanged
# at 1.0, so records of both versions describe the same experiment - but a metric whose formula changed cannot be
# compared across the two, and ``compare_door`` refuses to instead of reporting a meaningless delta.
METRICS_VERSION = "1.1"
METRIC_DEF_CHANGED_IN = {
    # 1.1: |v| at the single 30 Hz sample nearest the crossing -> peak |v| over the 100 ms of approach before it
    "arrival_speed": "1.1",
    "speed_at_latch": "1.1",
}
ARRIVAL_WINDOW_S = 0.1      # s of approach the arrival / latch speed is measured over
SAMPLE_HZ = 30
PHASES = ("settle", "hold", "operate", "release", "relatch", "closer", "locked")
# phases that start from the reset state; the others continue from the previous phase
PHASE_RESETS = {"settle": True, "hold": True, "operate": True, "release": False, "relatch": False, "closer": True, "locked": True}
DURATIONS = {"settle": 1.0, "hold": 1.0, "hold_free": 6.0, "operate": 6.4, "release": 0.8, "relatch_close": 6.0, "relatch_push": 1.0, "closer": 12.0, "locked": 2.0}
# qa.py effort tables
OPERATOR_EFFORT = {"hinge": 4.0, "exit_device": 8.0, "wheel": 10.0, "dog": 14.0, "slide": 120.0}
LOCKED_EFFORT = {"hinge": 6.0, "slide": 150.0}
THUMBTURN_EFFORT, THUMBTURN_UNTIL_S = 2.0, 1.2
AUX_EFFORT = {"hinge": 3.0, "slide": 60.0}
DOG_EFFORT = 14.0
OPERATOR_FROM_S, PUSH_FROM_S, PUSH_STOP_DEG = 0.6, 1.2, 50.0
# the adaptive push (base sized by the leaf's own weight moment, capped) lives in doorbench.qa: one definition, used
# by the sign-off QA and by both parity runners
PUSH_BASE_CAP, PUSH_BASE_FLOOR = PUSH_BASE_MAX, PUSH_BASE_MIN
THUMBTURN_JOINTS = ("leaf_deadbolt_thumbturn_hinge", "leaf_a_deadbolt_thumbturn_hinge")
AUX_JOINTS = ("leaf_aux_bolt_slide", "slide_latch_slide", "leaf_slide_bolt_slide", "leaf_pin_slide", "leaf_thumb_hinge", "hatch_bolt_slide",
              "join_bolt_slide", "garage_slide_lock_slide", "leaf_hook_thumbturn_hinge", "leaf_a_hook_thumbturn_hinge")
LATCH_BOLT_JOINT = "leaf_latch_bolt_slide"
NO_RETURN_LATCH_KINDS = ("roller", "ball_catch", "magnetic")
KINDS = ("mjcf", "usd_full", "usd_rl")

# discrepancy / outcome codes (compare_door)
CODES = ("OK", "MUJOCO_FAIL", "PHYSX_NO_OPEN", "PHYSX_HOLD_FAIL", "EXPORT_WELD_MISSING", "SETTLE_DRIFT", "LIMIT_VIOLATION", "NAN",
         "CLOSER_NO_RETURN", "LATCH_NO_RETURN", "RELATCH_FAIL", "STRUCTURE_FAIL", "LOAD_FAIL", "METRIC_DELTA", "INFO_DISAGREE", "RL_CANON", "MISSING",
         "STALE_INPUTS", "METRICS_VERSION_SKEW")
# codes that make a door grade C (behavioural disagreement) vs B (quantitative)
STATUS_CODES = {"MUJOCO_FAIL", "PHYSX_NO_OPEN", "PHYSX_HOLD_FAIL", "EXPORT_WELD_MISSING", "LIMIT_VIOLATION", "NAN", "CLOSER_NO_RETURN", "LATCH_NO_RETURN", "RELATCH_FAIL"}
QUANT_CODES = {"SETTLE_DRIFT", "METRIC_DELTA", "INFO_DISAGREE"}
# codes that say the two records do not describe the same experiment: not a discrepancy, a bookkeeping fact
PROVENANCE_CODES = {"STALE_INPUTS", "METRICS_VERSION_SKEW"}

DEG = math.pi / 180.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _joint_table(model_json: dict) -> dict:
    out = {}
    for b in model_json["bodies"]:
        j = b.get("joint")
        if not j:
            continue
        out[j["name"]] = {
            "type": j["type"], "range": j.get("range"), "initial": float(j.get("initial") or 0.0), "modeled_at": float(j.get("modeled_at") or 0.0),
            "stiffness": float(j.get("stiffness") or 0.0), "damping": float(j.get("damping") or 0.0), "frictionloss": float(j.get("frictionloss") or 0.0),
            "armature": float(j.get("armature") or 0.0), "springref": float(j.get("springref") or 0.0), "role": j.get("role", ""), "body": b["name"], "axis": j.get("axis"),
            "damping_closing": j.get("damping_closing"), "damping_opening": j.get("damping_opening"), "backcheck_angle": j.get("backcheck_angle"),
            "backcheck_damping": j.get("backcheck_damping"), "ratchet_one_way": bool(j.get("ratchet_one_way")), "limit_solref": j.get("limit_solref"),
        }
        # USD drive target (springref - modeled_at) in DoorBench coordinates is simply springref (0 when no spring)
        out[j["name"]]["spring_target"] = float(j.get("springref") or 0.0) if out[j["name"]]["stiffness"] > 0 else out[j["name"]]["modeled_at"]
    return out


def _operator_effort(name: str, jtype: str) -> float:
    if jtype != "hinge":
        return OPERATOR_EFFORT["slide"]
    if name.startswith("dog_"):
        return OPERATOR_EFFORT["dog"]
    if "wheel" in name:
        return OPERATOR_EFFORT["wheel"]
    if "exit_device" in name:
        return OPERATOR_EFFORT["exit_device"]
    return OPERATOR_EFFORT["hinge"]


def read_rl_meta_text(usda_text: str) -> dict | None:
    """The ``doorbench:rl`` JSON attribute of a door_rl.usda, parsed from the .usda text (no pxr needed).

    The usda writer picks the quoting per value: ``'...'`` (most doors), ``"..."`` with ``\\"`` escapes, or a
    triple-quoted literal; backslash escapes are those of Python string literals."""
    m = re.search(r'doorbench:rl\s*=\s*', usda_text)
    if not m:
        return None
    i = m.end()
    delim = None
    for d in ('"""', "'''", '"', "'"):
        if usda_text.startswith(d, i):
            delim = d
            break
    if delim is None:
        return None
    i += len(delim)
    out = []
    n = len(usda_text)
    while i < n:
        c = usda_text[i]
        if c == "\\" and i + 1 < n:
            nxt = usda_text[i + 1]
            out.append({"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'"}.get(nxt, "\\" + nxt))
            i += 2
            continue
        if usda_text.startswith(delim, i):
            break
        out.append(c)
        i += 1
    return json.loads("".join(out))


def read_rl_meta(door_dir: str) -> dict | None:
    import os
    p = os.path.join(door_dir, "door_rl.usda")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        txt = f.read()
    try:
        return read_rl_meta_text(txt)
    except Exception:
        try:
            from pxr import Usd
            st = Usd.Stage.Open(p)
            return json.loads(st.GetDefaultPrim().GetAttribute("doorbench:rl").Get())
        except Exception:
            return None


# what ``inputs_hash`` covers: everything a runner *acts on*.  Diagnostics derived from these (how the push was sized,
# wall times, notes) belong outside them - adding a derived field to a hashed dict would mark every existing record of
# the dataset stale for no behavioural reason.
HASHED_INPUT_KEYS = ("door_id", "joints", "forces", "thresholds", "coupling", "schedule", "flags")


def inputs_hash(inputs: dict) -> str:
    keep = {k: inputs[k] for k in HASHED_INPUT_KEYS if k in inputs}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# per-door inputs
# ---------------------------------------------------------------------------
def door_inputs(spec: dict, model_json: dict, forces: dict | None = None, qa: dict | None = None, rl_meta: dict | None = None) -> dict:
    """Everything a runner needs for one door, derived from spec.json + model.json (+ optional qa.json / rl meta).

    ``forces``: {"bias": gravity bias at the primary DOF, "frictionloss", "preload"} measured in MuJoCo at qpos0
    (``scripts/parity_reference_mujoco.py``); without it the push is taken from qa.json's ``qa_push`` or estimated
    from model.json (no gravity term) - ``forces.source`` says which.
    """
    meta = model_json["meta"]
    joints = _joint_table(model_json)
    phys = spec.get("physics", {})
    kin = spec.get("kinematics", {})
    fam = spec["family"]
    pj = meta.get("primary_joint")
    if pj not in joints:
        raise ValueError(f"{spec['id']}: primary joint {pj!r} not in model.json")
    P = joints[pj]
    is_hinge = P["type"] == "hinge"
    unit = "hinge" if is_hinge else "slide"
    oj = meta.get("operator_joint") if meta.get("operator_joint") in joints else None
    sj = meta.get("secondary_joint") if meta.get("secondary_joint") in joints else None
    bolt = LATCH_BOLT_JOINT if LATCH_BOLT_JOINT in joints else None
    latch_joints = [n for n, j in joints.items() if j["role"] == "latch"]
    thumbturn = next((n for n in THUMBTURN_JOINTS if n in joints), None)
    aux = [{"joint": n, "effort": AUX_EFFORT["hinge" if joints[n]["type"] == "hinge" else "slide"]} for n in AUX_JOINTS if n in joints]
    dogs = [n for n in joints if n.startswith("dog_") and "hinge" in n]
    unlimited = [n for n, j in joints.items() if j["range"] is None]
    lk = H.LOCKS[spec["lock"]["model"]]
    lt = H.LATCHES[spec["latch"]["model"]]
    flags = door_flags(spec)
    flags.update({
        "free_swing_family": fam in FREE_SWING_FAMILIES, "locked_rotor": bool(meta.get("locked")), "rest_angle_deg": kin.get("rest_angle_deg"),
        "both_ways": bool(kin.get("both_ways")), "automatic": bool(meta.get("actuators")), "has_weld": any(e.get("kind") == "weld" and e.get("active", True) for e in model_json.get("equalities", [])),
        "breakable_welds": meta.get("breakable_welds", []), "chain_slack": float(lk.chain_slack or 0.0), "lock_kind": lk.kind, "latch_kind": lt.kind, "latch_id": lt.id,
        "lock_engaged_spec": bool(spec["lock"]["engaged"]), "robot_side_release": bool(spec["lock"].get("robot_side_release", True)), "condition": spec.get("condition"),
        "closer_model": spec["closer"]["model"], "closer_preload_Nm": float(phys.get("closer", {}).get("spring_preload_Nm", 0.0) or 0.0),
    })
    # ---- forces (qa.py adaptive push)
    W = float(spec["leaf"]["width"])
    mass = float(phys.get("mass", {}).get("total_kg", 0.0) or 0.0)
    fl = P["frictionloss"]
    preload = abs(P["stiffness"] * P["springref"]) if P["stiffness"] > 0 else 0.0
    base, cap = push_base(unit, mass, W), PUSH_CAP[unit]
    if forces is not None and forces.get("bias") is not None:
        # values from the compiled MJCF win (the XML rounds to 6 decimals; qa.py reads m.dof_frictionloss / m.qpos_spring)
        bias = float(forces["bias"])
        fl = float(forces.get("frictionloss", fl))
        preload = float(forces.get("preload", preload))
        push = min(2.0 * (bias + fl + preload) + base, cap)
        src = forces.get("source", "mujoco")
    elif qa and qa.get("metrics", {}).get("qa_push") is not None:
        push = float(qa["metrics"]["qa_push"])
        bias = max(0.0, (push - base) / 2.0 - fl - preload) if push < cap else None
        src = "qa.json"
    else:
        bias = None
        push = min(2.0 * (fl + preload) + base, cap)
        src = "model.json (no gravity bias)"
    static = (bias or 0.0) + fl + preload
    max_open_deg = float(kin.get("max_open_deg") or 90)
    travel = float(P["range"][1] - P["range"][0]) if P["range"] else None
    chain_limit = math.asin(min(0.99, lk.chain_slack / max(W - 0.1, 0.2))) if (lk.chain_slack and is_hinge) else 0.0
    thr = 2.0 * DEG if is_hinge else 0.015
    if (flags["locked_rotor"] or (flags["free_swing"] and flags["has_holding"])) and P["range"] is not None:
        thr = max(thr, float(P["range"][1]) + 1.0 * DEG)     # locked turnstiles / bolted flaps: the leaf may move within its locked play
    thresholds = {
        "thr": thr, "thr_free": 10.0 * DEG if is_hinge else 0.05, "target": (min(20.0, 0.5 * max_open_deg) * DEG) if is_hinge else 0.05,
        "thr_locked": thr + chain_limit, "chain_limit_rad": chain_limit, "chain_engaged": bool(spec["lock"]["engaged"] and lk.kind in ("chain", "swing_bar_guard") and is_hinge),
        "closer_start": min(60.0, 0.8 * max_open_deg) * DEG, "closer_pass": 6.0 * DEG, "closed_thr": 3.0 * DEG if is_hinge else 0.03,
        "open_thr_bench": 30.0 * DEG if is_hinge else min(0.3, 0.5 * (travel or 0.6)), "relatch_closed": 2.0 * DEG, "relatch_repush": 2.5 * DEG, "relatch_min_open": 5.0 * DEG,
        "release_min_open": 3.0 * DEG, "bolt_return_m": 0.006, "settle_primary": (0.05 if (is_hinge or kin.get("type") == "rotor") else 0.01), "settle_other": 0.02 if is_hinge else 0.002,
        "pen0_min_m": -0.012, "limit_tol": {"hinge": 2.0 * DEG, "slide": 0.01}, "v_cap_primary": 15.0 if is_hinge else 6.0, "v_cap_any": 50.0,
        "operator_travel": float(H.OPERATORS[spec["operator"]["model"]].travel), "operator_dead_travel": float(H.OPERATORS[spec["operator"]["model"]].dead_travel),
        "operator_yield": float(phys.get("damage", {}).get("operator_yield_torque_Nm") or 0.0), "slam_velocity": float(phys.get("damage", {}).get("slam_velocity_rad_s") or 0.0),
        "latch_throw_m": float(joints[bolt]["range"][1] - joints[bolt]["range"][0]) if bolt and joints[bolt]["range"] else float(lt.throw),
    }
    # ---- couplings: one-sided tendons (bolt >= scale * operator), bilateral equalities, welds, loop closures
    tendons = []
    for t in model_json.get("tendons", []):
        terms = [(str(a), float(c)) for a, c in t["sites"]]
        tendons.append({"name": t["name"], "terms": terms, "lo": float(t["range"][0]), "hi": float(t["range"][1])})
    latch_coupling = None
    if bolt and oj:
        for t in tendons:
            names = [a for a, _ in t["terms"]]
            if bolt in names and oj in names:
                c = dict(t["terms"])
                latch_coupling = {"scale": -c[oj] / c[bolt], "operator_joint": oj, "latch_joint": bolt, "tendon": t["name"]}
                break
        if latch_coupling is None:
            for e in model_json.get("equalities", []):
                if e.get("kind") == "joint" and e.get("a") == bolt and e.get("b") == oj:
                    latch_coupling = {"scale": float(e["polycoeff"][1]), "offset": float(e["polycoeff"][0]), "operator_joint": oj, "latch_joint": bolt, "tendon": None}
    coupling = {
        "latch": latch_coupling, "tendons": tendons,
        "mimics": [{"driven": e["a"], "driver": e["b"], "coeff": list(e["polycoeff"][:2])} for e in model_json.get("equalities", []) if e.get("kind") == "joint"],
        "welds": [{"body1": e["a"], "body2": e["b"], "active": e.get("active", True)} for e in model_json.get("equalities", []) if e.get("kind") == "weld"],
        "loop_closures": [{"body1": e["a"], "body2": e["b"]} for e in model_json.get("equalities", []) if e.get("kind") == "connect"],
        "actuators": meta.get("actuators", []),
    }
    # ---- RL canonical file: slot mapping + welded parts
    rl = None
    if rl_meta:
        slot_of = {}
        for slot, info in rl_meta.get("joints", {}).items():
            if info.get("active") and info.get("source"):
                slot_of[info["source"]] = slot
        rl = {"slot_of": slot_of, "door_joint": rl_meta.get("door_joint"), "operator_slot_joint": rl_meta.get("operator_slot_joint"), "latch_present": rl_meta.get("slots", {}).get("latch", "none") != "none",
              "operator_present": rl_meta.get("operator_slot_joint") is not None, "secondary_slot_joint": rl_meta.get("secondary_slot_joint"), "lock_engaged": bool(rl_meta.get("lock", {}).get("engaged")),
              "welded_static": rl_meta.get("welded_static", []), "omitted": rl_meta.get("omitted", []), "notes": rl_meta.get("notes", []),
              "latch_coupling": rl_meta.get("latch_coupling"), "targets": {slot: info.get("target", 0.0) for slot, info in rl_meta.get("joints", {}).items() if info.get("active")},
              # ground truth from the exporter (usd.py write_usd_rl): which mechanism parts door_rl.usda welded and in
              # which state.  Before this existed the schedule GUESSED "an engaged lock plus a thumbturn / aux bolt /
              # dog joint in the MJCF means the canonical file is welded shut", which was wrong for every door whose
              # release part is coupled to the operator (hook sliders, cremone bolts, wheel-driven dogs).
              "welded_engaged": [w["joint"] for w in rl_meta.get("welded_engaged", [])],
              "released_holding": [w["joint"] for w in rl_meta.get("released_holding", [])],
              "released_parts": [w["joint"] for w in rl_meta.get("released_parts", [])],
              "weld_ground_truth": "welded" in rl_meta,
              "env_release": [e.get("name") for e in rl_meta.get("env_release", [])]}
    inputs = {
        "protocol_version": PROTOCOL_VERSION, "door_id": spec["id"], "family": fam, "kinematics_type": kin.get("type"), "is_hinge": is_hinge, "unit": unit,
        "max_open_deg": max_open_deg, "travel_m": travel, "leaf_width_m": W, "mass_kg": mass, "task": spec.get("task"),
        "joints": joints, "primary_joint": pj, "operator_joint": oj, "secondary_joint": sj, "latch_bolt_joint": bolt, "latch_joints": latch_joints,
        "thumbturn_joint": thumbturn, "aux_joints": aux, "dog_joints": dogs, "unlimited_joints": unlimited,
        "flags": flags,
        "forces": {"bias": bias, "frictionloss": fl, "preload": preload, "static": static, "push": push, "close_drive": min(0.5 * push, 1.5 * static + 40.0),
                   "operator_effort": _operator_effort(oj, joints[oj]["type"]) if oj else None, "locked_effort": (LOCKED_EFFORT["hinge"] if joints[oj]["type"] == "hinge" else LOCKED_EFFORT["slide"]) if oj else None,
                   "thumbturn_effort": THUMBTURN_EFFORT, "dog_effort": DOG_EFFORT, "source": src},
        "push_base": base,      # derived from mass / width (doorbench.qa.push_base); reported, not hashed - see HASHED_INPUT_KEYS
        "thresholds": thresholds, "coupling": coupling, "rl": rl,
        "reference_qa": {k: qa["metrics"].get(k) for k in ("qa_push", "hold_displacement", "actuate_displacement", "operator_travel_reached", "bolt_after_release_m",
                                                            "relatch_closed_angle", "relatch_repush_angle", "closer_final_angle", "locked_displacement", "settle_drift")} if qa else None,
        "reference_checks": (qa or {}).get("checks"),
    }
    inputs["schedule"] = {k: expected_outcomes(inputs, k) for k in KINDS}
    inputs["inputs_hash"] = inputs_hash(inputs)
    return inputs


# ---------------------------------------------------------------------------
# what the door is expected to do, per phase and file kind
# ---------------------------------------------------------------------------
def _rl_blocking(inputs: dict, rl: dict) -> list:
    """Joints the protocol works to release this door that ``door_rl.usda`` welded in their ENGAGED state.

    ``rl["welded_engaged"]`` is the exporter's ground truth (every engaged latch / lock part with no canonical slot);
    intersecting it with the joints the protocol actually actuates (``thumbturn_joint``, ``aux_joints``,
    ``dog_joints``) is what decides whether the canonical door can open: a welded keypad key or thumbturn spindle
    does not hold the leaf, a welded slide bolt or an extra dog does."""
    release_set = set(inputs["dog_joints"]) | {a["joint"] for a in inputs["aux_joints"]}
    if inputs["thumbturn_joint"]:
        release_set.add(inputs["thumbturn_joint"])
    return [j for j in (rl.get("welded_engaged") or []) if j in release_set]


def expected_outcomes(inputs: dict, kind: str = "mjcf") -> dict:
    """{phase: expectation}.  Expectations: 'settle', 'hold', 'free_opens', 'opens', 'stays_closed',
    'bolt_returns', 'bolt_returns_info', 'relatches', 'relatches_info', 'closes', 'locked_holds', or 'na:<reason>'."""
    f, th = inputs["flags"], inputs["thresholds"]
    oj, bolt, is_hinge = inputs["operator_joint"], inputs["latch_bolt_joint"], inputs["is_hinge"]
    rl = inputs.get("rl") if kind == "usd_rl" else None
    exp = {"settle": "settle"}
    free_swing = f["free_swing"]
    # ---- hold / free_opens (qa.py: every family is pushed; a leaf nothing holds must move, a locked rotor / bolted
    # flap must not.  Free-swing doors used to be informational here - that is how 12 locked accordion folds and 10
    # revolving doors jammed on the header shipped signed off.)
    if free_swing:
        exp["hold"] = "hold" if (f["locked_rotor"] or f["has_holding"]) else "free_opens"
    elif f["has_holding"]:
        exp["hold"] = "hold"
    else:
        exp["hold"] = "free_opens"
    if rl is not None and exp["hold"] == "hold" and f["spring_latch"] and not f["lock_engaged"] and not rl["latch_present"]:
        exp["hold"] = "na:rl latch not in the canonical articulation (welded released)"
    if rl is not None and exp["hold"] == "hold" and rl.get("released_holding") and not _rl_blocking(inputs, rl):
        # door_rl.usda welded every part that holds this leaf in its RELEASED state (a hook / cremone bolt / dog that
        # the operator retracts and that has no canonical slot): nothing can hold the canonical leaf, by construction
        exp["hold"] = f"na:rl holding part welded released ({', '.join(rl['released_holding'][:3])})"
    # ---- operate
    if free_swing:
        exp["operate"] = "na:free-swing family"
    elif oj is None:
        exp["operate"] = "na:no operator joint" + (" (panic device on the far side)" if f["latch_kind"] in ("rim_latch", "mortise_latch", "vertical_rods") else "")
    elif f["env_release_only"]:
        exp["operate"] = "na:lock released by environment logic"
    elif not f["can_release"]:
        exp["operate"] = "na:no robot-side release (see locked)"
    else:
        exp["operate"] = "opens"
        if rl is not None:
            if not rl["operator_present"]:
                exp["operate"] = "na:rl operator slot empty (world-mounted operator)"
            elif rl.get("weld_ground_truth"):
                # the exporter records what door_rl.usda did with every mechanism part; the protocol contributes which
                # parts IT works to release the door (thumbturn, aux bolts, dogs).  A release part welded ENGAGED is
                # what the canonical file cannot undo; keypad keys and thumbturn spindles are welded too but do not
                # hold the leaf, and their release parts (or the latch) still open it.
                if f["lock_engaged"] and _rl_blocking(inputs, rl):
                    exp["operate"] = "stays_closed"
            else:
                welded_release = [n for n in ([inputs["thumbturn_joint"]] if inputs["thumbturn_joint"] else []) + [a["joint"] for a in inputs["aux_joints"]] + inputs["dog_joints"] if n and n != oj]
                if f["lock_engaged"] and welded_release:
                    exp["operate"] = "stays_closed"      # engaged lock whose release parts are welded engaged in door_rl.usda
    # ---- release / relatch (continuations of operate)
    if exp["operate"] == "opens" and bolt:
        rl_bolt_ok = rl is None or rl["latch_present"]
        if not rl_bolt_ok:
            exp["release"] = "na:rl latch slot empty"
            exp["relatch"] = "na:rl latch slot empty"
        else:
            exp["release"] = "bolt_returns" if f["latch_kind"] not in NO_RETURN_LATCH_KINDS else "bolt_returns_info"
            if is_hinge:
                exp["relatch"] = "relatches" if (f["latch_kind"] not in NO_RETURN_LATCH_KINDS and f["lock_kind"] != "jam_stuck") else "relatches_info"
            else:
                exp["relatch"] = "na:sliding door"
    else:
        exp["release"] = "na:no operate phase" if not bolt else "na:operate not expected to open"
        exp["relatch"] = exp["release"]
    # ---- closer
    if f["latch_id"] == "fork_gravity":
        exp["closer"] = "na:gravity fork latch"
    elif free_swing:
        exp["closer"] = "na:free-swing family"
    elif is_hinge and f["closer_model"] not in ("none", "gas_strut") and f["closer_preload_Nm"] > 0 and not f["both_ways"] and not f["env_release_only"] \
            and not (f["lock_engaged_spec"] and f["lock_kind"] in ("chain", "swing_bar_guard", "padlock")):
        exp["closer"] = "closes"
    else:
        exp["closer"] = "na:no closer test"
    # ---- locked
    if not free_swing and oj is not None and not f["can_release"] and not f["env_release_only"]:
        exp["locked"] = "locked_holds" if (rl is None or rl["operator_present"]) else "na:rl operator slot empty"
    else:
        exp["locked"] = "na:not a locked-no-release door"
    return exp


def phase_applies(inputs: dict, phase: str, kind: str = "mjcf") -> bool:
    return not inputs["schedule"][kind][phase].startswith("na:")


def phase_duration(inputs: dict, phase: str, kind: str = "mjcf") -> float:
    if phase == "hold":
        return DURATIONS["hold"] if inputs["schedule"][kind]["hold"] == "hold" else DURATIONS["hold_free"]
    if phase == "relatch":
        return DURATIONS["relatch_close"] + DURATIONS["relatch_push"]
    return DURATIONS[phase]


def phase_initial_state(inputs: dict, phase: str) -> dict:
    """Joint overrides (DoorBench coordinates) applied after the reset at the start of a phase."""
    if phase == "closer":
        st = {inputs["primary_joint"]: inputs["thresholds"]["closer_start"]}
        if inputs["latch_bolt_joint"]:
            st[inputs["latch_bolt_joint"]] = 0.0
        return st
    return {}


# ---------------------------------------------------------------------------
# per-step drive (pure): efforts by MJCF joint name
# ---------------------------------------------------------------------------
def phase_efforts(inputs: dict, phase: str, t: float, q: dict, kind: str = "mjcf", qd: dict | None = None) -> dict:
    """Generalized forces (N*m on hinges, N on slides) to apply at simulated time ``t`` (s since the phase start).

    ``q``: current joint values by MJCF name (at least the primary joint).  Time comparisons carry a 1e-9 slack so
    that 300 * 0.002 s counts as 0.6 s.
    ``qd``: current joint RATES by MJCF name.  Only the relatch close needs them - the hand stops pushing once the
    leaf is already swinging shut at CLOSE_RATE_RAD_S (see PROTOCOL_VERSION 1.1).  A runner that cannot supply
    rates drives flat out, which is the 1.0 behaviour.
    """
    eps = 1e-9
    F = inputs["forces"]
    pj, oj = inputs["primary_joint"], inputs["operator_joint"]
    eff = {}
    if phase == "hold":
        dur = phase_duration(inputs, "hold", kind)
        qp = q.get(pj, 0.0)
        # qa.py stops the free push once the door is past thr_free after 1 s
        if t < dur - eps and not (t >= DURATIONS["hold"] - eps and qp > inputs["thresholds"]["thr_free"] and inputs["schedule"][kind]["hold"] != "hold"):
            eff[pj] = F["push"]
    elif phase == "operate":
        if t >= DURATIONS["operate"] - eps:
            return eff
        if inputs["thumbturn_joint"] and t < THUMBTURN_UNTIL_S - eps:
            eff[inputs["thumbturn_joint"]] = F["thumbturn_effort"]
        for a in inputs["aux_joints"]:
            eff[a["joint"]] = a["effort"]
        for d in inputs["dog_joints"]:
            eff[d] = F["dog_effort"]
        if oj and t >= OPERATOR_FROM_S - eps:
            eff[oj] = F["operator_effort"]
        if t >= PUSH_FROM_S - eps and (not inputs["is_hinge"] or q.get(pj, 0.0) < PUSH_STOP_DEG * DEG):
            eff[pj] = F["push"]
    elif phase == "relatch":
        if t < DURATIONS["relatch_close"] - eps:
            # rate-limited close: a person stops shoving once the door is already swinging shut, and driving it flat
            # out tunnels the slab through the frame stop in one step (PROTOCOL_VERSION 1.1)
            if qd is None or not inputs["is_hinge"] or qd.get(pj, 0.0) > -CLOSE_RATE_RAD_S:
                eff[pj] = -F["close_drive"]
        elif t < DURATIONS["relatch_close"] + DURATIONS["relatch_push"] - eps:
            eff[pj] = F["push"]
    elif phase == "locked":
        if t < DURATIONS["locked"] - eps:
            if oj:
                eff[oj] = F["locked_effort"]
            eff[pj] = F["push"]
    return eff


def tendon_min_positions(inputs: dict, q: dict) -> dict:
    """One-sided MJCF fixed tendons (c_a q_a + sum c_i q_i >= lo, c_a > 0 for the first term): the minimum value of the
    first joint given the others.  {joint: q_min} clipped to the joint's upper limit.  PhysX has no tendon, so the
    Isaac runner clamps the joint state to this every step (MuJoCo enforces it natively)."""
    out = {}
    for t in inputs["coupling"]["tendons"]:
        (a, ca), *rest = t["terms"]
        if ca <= 0 or any(r[0] not in q for r in rest) or a not in inputs["joints"]:
            continue
        qmin = (t["lo"] - sum(c * q[n] for n, c in rest)) / ca
        rng = inputs["joints"][a]["range"]
        if rng is not None:
            qmin = min(qmin, rng[1])
        out[a] = max(qmin, out.get(a, -math.inf))
    return out


def servo_effort(inputs: dict, q: dict, v: dict) -> dict:
    """MJCF position actuators of automatic doors (ctrl = 0: servo toward closed) as feed-forward efforts, for a
    simulator that has no actuator: force = clip(kp * (0 - q) - kv * v, forcerange)."""
    out = {}
    for a in inputs["coupling"]["actuators"]:
        j = a.get("joint")
        if j not in q:
            continue
        lo, hi = a.get("forcerange", (-1e9, 1e9))
        f = float(a.get("kp", 0.0)) * (0.0 - q[j]) - float(a.get("kv", 0.0)) * v.get(j, 0.0)
        out[j] = float(min(max(f, lo), hi))
    return out


# ---------------------------------------------------------------------------
# metrics from recorded curves (pure)
# ---------------------------------------------------------------------------
def _at(curve: dict, joint: str, t_query: float):
    """Value of a joint curve at the sample nearest to t_query (None when the joint is not recorded)."""
    ts, qs = curve.get("t") or [], (curve.get("q") or {}).get(joint)
    if not ts or qs is None:
        return None
    k = min(range(len(ts)), key=lambda i: abs(ts[i] - t_query))
    return float(qs[k])


def _first_time(curve: dict, joint: str, pred, t_min: float = 0.0):
    ts, qs = curve.get("t") or [], (curve.get("q") or {}).get(joint)
    if not ts or qs is None:
        return None
    for tt, qq in zip(ts, qs):
        if tt >= t_min - 1e-9 and pred(qq):
            return float(tt)
    return None


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def approach_speed(curve: dict, joint: str, t_cross: float | None, closed_thr: float, window: float = ARRIVAL_WINDOW_S):
    """Peak |v| of ``joint`` over the ``window`` seconds of approach *before* the crossing at ``t_cross``.

    ``|v|`` at the single sample nearest the crossing (metrics 1.0) is a sampling artefact, not a measurement: the
    crossing sample is the first one already inside the closed band, so it reads the post-impact velocity - either
    ~0 (the leaf has stopped against the strike) or a rebound - and which of the two it lands on depends on where the
    30 Hz grid happens to fall relative to an impact that lasts a few milliseconds.  The peak over the last 100 ms of
    the approach is the speed the leaf actually arrives with and does not move when the grid does.  Both runners call
    this, so MuJoCo (500 Hz stepping) and PhysX (120 Hz) are reduced by the same formula from the same 30 Hz curves.
    """
    ts = curve.get("t") or []
    vs = (curve.get("v") or {}).get(joint)
    qs = (curve.get("q") or {}).get(joint)
    if not ts or not vs or t_cross is None:
        return None
    n = min(len(ts), len(vs))
    idx = [i for i in range(n) if t_cross - window - 1e-9 <= ts[i] < t_cross - 1e-9]
    if not idx:
        # the crossing happens inside the first sample interval: fall back to the last sample still outside the
        # closed band (or, failing that, the last sample before the crossing)
        idx = [i for i in range(n) if ts[i] < t_cross - 1e-9 and (qs is None or i >= len(qs) or abs(qs[i]) >= closed_thr)][-1:]
        if not idx:
            idx = [i for i in range(n) if ts[i] < t_cross - 1e-9][-1:]
    if not idx:
        return None
    return max(abs(float(vs[i])) for i in idx)


def primary_at_limit(inputs: dict, q_max) -> bool | None:
    """Did the leaf reach its opening stop (within the limit tolerance)?  None when the joint has no range."""
    rng = (inputs["joints"].get(inputs["primary_joint"]) or {}).get("range")
    if not rng:
        return None
    tol = inputs["thresholds"]["limit_tol"]["hinge" if inputs["is_hinge"] else "slide"]
    return bool(_finite(q_max) and abs(float(q_max) - float(rng[1])) <= tol)


def phase_metrics(inputs: dict, phase: str, curve: dict, ctx: dict | None = None) -> dict:
    """Metrics of one phase from its recorded curve.

    curve = {"t": [...], "q": {joint: [...]}, "v": {joint: [...]}, "minmax": {joint: [lo, hi]}, "finite": bool,
             "warnings": [...], "pen0_m": float (settle, MuJoCo), "vmax": {joint: float}}  (DoorBench coordinates)
    ctx: values carried between phases, e.g. {"opened": q at the end of operate}.
    """
    ctx = ctx or {}
    th, pj, oj, bolt, sj = inputs["thresholds"], inputs["primary_joint"], inputs["operator_joint"], inputs["latch_bolt_joint"], inputs["secondary_joint"]
    ts = curve.get("t") or []
    m = {"finite": bool(curve.get("finite", True)) and bool(ts), "warnings": list(curve.get("warnings") or []), "t_end": float(ts[-1]) if ts else None, "n_samples": len(ts)}
    qs = curve.get("q") or {}
    if phase == "settle":
        drift = {}
        for j, arr in qs.items():
            if len(arr) >= 2 and _finite(arr[0]) and _finite(arr[-1]):
                drift[j] = float(arr[-1] - arr[0])
        m["settle_drift_signed"] = drift
        m["settle_drift"] = abs(drift.get(pj, 0.0))
        others = {j: d for j, d in drift.items() if j != pj and j not in inputs["unlimited_joints"]}
        m["settle_drift_other_max"] = max((abs(d) for d in others.values()), default=0.0)
        m["settle_drift_other_joint"] = max(others, key=lambda j: abs(others[j])) if others else None
        m["pen0_m"] = curve.get("pen0_m")
        m["max_v_primary"] = (curve.get("vmax") or {}).get(pj)
    elif phase == "hold":
        q1 = _at(curve, pj, DURATIONS["hold"])
        t_free = _first_time(curve, pj, lambda x: x > th["thr_free"])
        m["q_at_1s"] = q1
        m["t_free"] = t_free
        # qa.py: q at exit = at 1.0 s (holding doors, or free doors already open by then) else at the first crossing after 1 s
        if t_free is not None and t_free > DURATIONS["hold"]:
            m["hold_displacement"] = _at(curve, pj, t_free)
        elif inputs["schedule"]["mjcf"]["hold"] != "hold" and t_free is None and ts:
            m["hold_displacement"] = float(qs[pj][-1]) if pj in qs else None
        else:
            m["hold_displacement"] = q1
        m["secondary_drift"] = (float(qs[sj][-1] - qs[sj][0]) if sj in qs and len(qs[sj]) > 1 else None)
    elif phase == "operate":
        opened = float(qs[pj][-1]) if pj in qs and qs[pj] else None
        m["opened"] = opened
        m["operator_travel_reached"] = float(qs[oj][-1]) if oj in qs and qs[oj] else None
        m["operator_travel_frac"] = (m["operator_travel_reached"] / th["operator_travel"]) if (m["operator_travel_reached"] is not None and th["operator_travel"]) else None
        if bolt in qs and qs[bolt]:
            throw = th["latch_throw_m"] or 1.0
            m["bolt_retract_max_m"] = float(max(qs[bolt]))
            m["bolt_retract_max_frac"] = m["bolt_retract_max_m"] / throw
            m["t_unlatch"] = _first_time(curve, bolt, lambda x: x >= 0.8 * throw)
        m["t_open"] = _first_time(curve, pj, lambda x: x > th["target"])
        m["t_open_bench"] = _first_time(curve, pj, lambda x: x > th["open_thr_bench"])
        m["q_primary_max"] = float(max(qs[pj])) if pj in qs and qs[pj] else None
        m["primary_at_limit"] = primary_at_limit(inputs, m["q_primary_max"])
    elif phase == "release":
        m["bolt_after_release_m"] = float(qs[bolt][-1]) if bolt in qs and qs[bolt] else None
        m["t_bolt_return"] = _first_time(curve, bolt, lambda x: x < th["bolt_return_m"]) if bolt in qs else None
        if oj in qs and qs[oj] and th["operator_travel"]:
            m["operator_after_release_frac"] = float(qs[oj][-1]) / th["operator_travel"]
        m["opened_before"] = ctx.get("opened")
    elif phase == "relatch":
        tc = DURATIONS["relatch_close"]
        m["relatch_closed_angle"] = _at(curve, pj, tc)
        m["relatch_repush_angle"] = float(qs[pj][-1]) if pj in qs and qs[pj] else None
        m["t_close"] = _first_time(curve, pj, lambda x: abs(x) < th["closed_thr"])
        m["arrival_speed"] = approach_speed(curve, pj, m["t_close"], th["closed_thr"])
        if bolt in qs and qs[bolt]:
            close_idx = [i for i, tt in enumerate(ts) if tt <= tc + 1e-9]
            m["bolt_min_during_close"] = float(min(qs[bolt][i] for i in close_idx)) if close_idx else None
            m["bolt_max_during_close"] = float(max(qs[bolt][i] for i in close_idx)) if close_idx else None
        m["opened_before"] = ctx.get("opened")
    elif phase == "closer":
        m["closer_final_angle"] = float(qs[pj][-1]) if pj in qs and qs[pj] else None
        m["closer_t_close"] = _first_time(curve, pj, lambda x: abs(x) < th["closed_thr"])
        vs = (curve.get("v") or {}).get(pj)
        if vs:
            m["peak_closing_speed"] = float(max(-x for x in vs))
            if m["closer_t_close"] is not None:
                # same approach-speed definition as relatch: the sample at the crossing is post-impact
                m["speed_at_latch"] = approach_speed(curve, pj, m["closer_t_close"], th["closed_thr"])
                m["slam"] = bool(th["slam_velocity"] and (m["speed_at_latch"] or 0.0) > th["slam_velocity"])
            # rebounds: velocity sign changes from closing to opening after the first closing
            reb, prev = 0, None
            for x in vs:
                s = 1 if x > 1e-3 else (-1 if x < -1e-3 else 0)
                if prev == -1 and s == 1:
                    reb += 1
                if s:
                    prev = s
            m["rebounds"] = reb
    elif phase == "locked":
        m["locked_displacement"] = float(qs[pj][-1]) if pj in qs and qs[pj] else None
        m["operator_travel_reached"] = float(qs[oj][-1]) if oj in qs and qs[oj] else None
    # ---- limits and sanity over the phase (P9 / P10)
    viol = []
    for j, mm in (curve.get("minmax") or {}).items():
        jt = inputs["joints"].get(j)
        if jt is None or jt["range"] is None:
            continue
        tol = th["limit_tol"]["hinge" if jt["type"] == "hinge" else "slide"]
        lo, hi = jt["range"]
        over = max(lo - mm[0], mm[1] - hi, 0.0)
        if over > tol:
            viol.append({"joint": j, "over": float(over), "min": float(mm[0]), "max": float(mm[1]), "range": [lo, hi]})
    m["limit_violations"] = viol
    vmax = curve.get("vmax") or {}
    m["max_v_primary"] = vmax.get(pj)
    m["max_v_any"] = max(vmax.values()) if vmax else None
    m["velocity_cap_hit"] = bool((m["max_v_primary"] or 0.0) > th["v_cap_primary"] or (m["max_v_any"] or 0.0) > th["v_cap_any"])
    return m


# ---------------------------------------------------------------------------
# pass / fail against the expectation (pure)
# ---------------------------------------------------------------------------
def phase_status(inputs: dict, phase: str, expected: str, m: dict) -> str:
    """'pass' | 'fail' | 'skip' | 'na' for one phase.  '*_info' expectations are judged like their base expectation
    (the caller decides how much weight an informational phase carries)."""
    if expected.startswith("na:"):
        return "na"
    if not m or not m.get("finite", False):
        return "fail"
    th = inputs["thresholds"]
    base = expected[:-5] if expected.endswith("_info") else expected
    if phase == "settle":
        drift_ok = m["settle_drift"] < th["settle_primary"] or bool(inputs["flags"]["rest_angle_deg"])
        pen_ok = m.get("pen0_m") is None or m["pen0_m"] > th["pen0_min_m"]
        return "pass" if (drift_ok and pen_ok and not m.get("warnings")) else "fail"
    if phase == "hold":
        if base == "hold":
            return "pass" if (m.get("q_at_1s") is not None and m["q_at_1s"] < th["thr"]) else "fail"
        return "pass" if m.get("t_free") is not None else "fail"
    if phase == "operate":
        opened = m.get("opened")
        if opened is None:
            return "fail"
        if base == "stays_closed":
            return "pass" if opened < th["thr_locked"] else "fail"
        if th["chain_engaged"]:
            return "pass" if (1.5 * DEG < opened < th["chain_limit_rad"] + 4.0 * DEG) else "fail"
        return "pass" if opened > th["target"] else "fail"
    if phase == "release":
        if m.get("opened_before") is not None and m["opened_before"] < th["release_min_open"]:
            return "skip"
        b = m.get("bolt_after_release_m")
        return "pass" if (b is not None and b < th["bolt_return_m"]) else "fail"
    if phase == "relatch":
        if m.get("opened_before") is None or m["opened_before"] <= th["relatch_min_open"]:
            return "skip"
        c, r = m.get("relatch_closed_angle"), m.get("relatch_repush_angle")
        return "pass" if (c is not None and r is not None and abs(c) < th["relatch_closed"] and r < th["relatch_repush"]) else "fail"
    if phase == "closer":
        f = m.get("closer_final_angle")
        return "pass" if (f is not None and abs(f) < th["closer_pass"]) else "fail"
    if phase == "locked":
        d = m.get("locked_displacement")
        return "pass" if (d is not None and d < th["thr_locked"]) else "fail"
    return "na"


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def _within(a, b, abs_tol: float, rel_tol: float = 0.0) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    d = abs(a - b)
    return d <= abs_tol or (rel_tol > 0 and d <= rel_tol * max(abs(a), abs(b)))


def operator_span(inputs: dict) -> float:
    """Operator travel used for tolerances: the hardware table's travel or, when the MJCF operator joint is authored
    with a larger range (fork / lift latches modelled as 1.2 rad hinges, baby-gate pin slides with a 50 mm range), the
    joint's range span.  ``thresholds.operator_travel`` alone would make the tolerance 0.006 rad on a fork lever."""
    oj = inputs.get("operator_joint")
    span = float(inputs["thresholds"].get("operator_travel") or 0.0)
    rng = (inputs["joints"].get(oj) or {}).get("range") if oj else None
    if rng:
        span = max(span, float(rng[1] - rng[0]))
    return span or 1.0


def locked_play(inputs: dict) -> float:
    """How far the leaf of a *holding* door may legitimately travel: the play its own hardware leaves it.

    ``thresholds.thr`` is 2 deg / 15 mm for a latched leaf, but a locked turnstile rotor or a bolted pet flap is
    authored with a range the leaf may move inside while "held" (``door_inputs`` raises ``thr`` to range[1] + 1 deg).
    Inside that window the resting point is not a behavioural fact - MuJoCo's soft limit parks the leaf against one
    end of it, PhysX's rigid limit against the other - so the comparison tolerance has to cover it."""
    base = 2.0 * DEG if inputs["is_hinge"] else 0.015
    return max(0.0, float(inputs["thresholds"]["thr"]) - base)


def metric_tolerances(inputs: dict, phase: str, expected: str | None = None) -> dict:
    """{metric: (abs_tol, rel_tol)} for the quantitative comparison of a phase (hinge / slide units).

    ``expected`` is the phase expectation (``inputs["schedule"]["mjcf"][phase]`` when omitted): the hold phase measures
    a different quantity for a door that must stay shut (latch slop, sub-degree) than for one that must swing open
    (tenths of a rad), so it cannot use one tolerance for both."""
    h = inputs["is_hinge"]
    ang, lin = (0.1, 0.05)
    if phase == "settle":
        return {"settle_drift": (0.02 if h else 0.005, 0.0)}
    if phase == "hold":
        exp = expected or inputs["schedule"]["mjcf"]["hold"]
        exp = exp[:-5] if exp.endswith("_info") else exp
        play = locked_play(inputs)
        free = (ang if h else lin, 0.2)
        # a free door's hold_displacement is q at the free-swing crossing (tenths of a rad / m): the latched-door
        # tolerance (0.01 rad / 3 mm) would call MuJoCo's soft-limit overshoot a discrepancy
        hold_tol = free if exp != "hold" else ((0.01 if h else 0.003) + play, 0.0)
        return {"hold_displacement": hold_tol, "t_free": (0.25, 0.3), "q_at_1s": (free[0] + play, free[1])}
    if phase == "operate":
        return {"opened": (ang if h else lin, 0.2), "q_primary_max": (ang if h else lin, 0.2), "t_open": (0.3, 0.3),
                "operator_travel_reached": (0.1 * operator_span(inputs), 0.0),
                "bolt_retract_max_m": (0.15 * (inputs["thresholds"]["latch_throw_m"] or 0.01), 0.0), "t_unlatch": (0.2, 0.0)}
    if phase == "release":
        return {"bolt_after_release_m": (0.002, 0.0), "t_bolt_return": (0.2, 0.0)}
    if phase == "relatch":
        return {"relatch_closed_angle": (1.0 * DEG, 0.0), "relatch_repush_angle": (1.0 * DEG, 0.0), "t_close": (0.3, 0.3), "arrival_speed": (0.1, 0.3)}
    if phase == "closer":
        return {"closer_final_angle": (2.0 * DEG, 0.0), "closer_t_close": (0.5, 0.3), "peak_closing_speed": (0.1, 0.3)}
    if phase == "locked":
        return {"locked_displacement": (0.01 if h else 0.003, 0.0)}
    return {}


def _phase_codes(inputs: dict, phase: str, expected: str, s_mj: str, s_px: str, m_mj: dict, m_px: dict) -> list:
    codes = []
    if not m_px.get("finite", True):
        codes.append("NAN")
    # joint-limit violations count only when PhysX exceeds a limit that MuJoCo (soft limits, solreflimit 0.005) respects,
    # or by more than twice MuJoCo's own overshoot on that joint
    mj_over = {v["joint"]: v["over"] for v in (m_mj.get("limit_violations") or [])}
    if any(v["over"] > 2.0 * mj_over.get(v["joint"], 0.0) for v in (m_px.get("limit_violations") or [])):
        codes.append("LIMIT_VIOLATION")
    info = expected.endswith("_info")
    base = expected[:-5] if info else expected
    if s_mj == "fail" and not info:          # an informational phase has no reference verdict (qa.py never tested it)
        codes.append("MUJOCO_FAIL")
    if s_mj == s_px:
        return codes
    if info:
        codes.append("INFO_DISAGREE")
        return codes
    if s_px == "fail" and s_mj == "pass":
        if phase == "settle":
            codes.append("SETTLE_DRIFT")
        elif base in ("hold", "stays_closed", "locked_holds"):
            codes.append("EXPORT_WELD_MISSING" if (inputs["flags"]["env_release_only"] and inputs["flags"]["has_weld"]) else "PHYSX_HOLD_FAIL")
        elif base in ("free_opens", "opens"):
            codes.append("PHYSX_NO_OPEN")
        elif base == "bolt_returns":
            codes.append("LATCH_NO_RETURN")
        elif base == "relatches":
            codes.append("RELATCH_FAIL")
        elif base == "closes":
            codes.append("CLOSER_NO_RETURN")
        else:
            codes.append("METRIC_DELTA")
    elif s_px == "pass" and s_mj == "fail":
        pass  # already MUJOCO_FAIL
    elif "skip" in (s_mj, s_px):
        codes.append("METRIC_DELTA")
    return codes


def _version_tuple(v) -> tuple:
    try:
        return tuple(int(x) for x in str(v or "1.0").split("."))
    except ValueError:
        return (1, 0)


def skewed_metrics(mj: dict | None, px: dict | None) -> list:
    """Metrics whose definition changed between the two records' ``metrics_version`` - not comparable, in either
    direction, until the older side is re-run.  A record without the field predates the field: metrics 1.0."""
    a, b = _version_tuple((mj or {}).get("metrics_version")), _version_tuple((px or {}).get("metrics_version"))
    if a == b:
        return []
    lo = min(a, b)
    return sorted(m for m, v in METRIC_DEF_CHANGED_IN.items() if _version_tuple(v) > lo)


def stale_reason(inputs: dict, mj: dict | None, px: dict | None) -> str | None:
    """Why these two records do not describe the same door, or None when they do.

    ``inputs_hash`` covers the joints, forces (the adaptive push!), thresholds, couplings, schedule and flags a runner
    was given.  Joining a PhysX run of one dataset revision with a MuJoCo reference of another silently attributes
    export / physics differences to doors that were simply not the same door in the two runs, so a mismatch is a
    hard X, never a discrepancy class."""
    h_ref = inputs.get("inputs_hash")
    h_mj, h_px = (mj or {}).get("inputs_hash"), (px or {}).get("inputs_hash")
    if h_mj and h_px and h_mj != h_px:
        return f"inputs_hash mujoco {h_mj} != physx {h_px}"
    if h_ref and h_px and h_px != h_ref:
        return f"inputs_hash physx {h_px} != current protocol inputs {h_ref}"
    if h_ref and h_mj and h_mj != h_ref:
        return f"inputs_hash mujoco {h_mj} != current protocol inputs {h_ref}"
    return None


def compare_door(inputs: dict, mj: dict | None, px: dict | None, kind: str = "usd_full") -> dict:
    """Verdict for one door and one USD kind.  ``mj`` / ``px`` are runner records: {"phases": {phase: {"status", "metrics",
    "expected"}}, "structure": {"status", ...}, "load_error": str | None}.  Returns per-phase agreement, codes, grade."""
    out = {"door_id": inputs["door_id"], "kind": kind, "phases": {}, "codes": [], "grade": "A"}
    if mj is None or px is None:
        out["codes"] = ["MISSING"]
        out["grade"] = "X"
        out["note"] = "missing " + ("mujoco" if mj is None else "physx") + " record"
        return out
    stale = stale_reason(inputs, mj, px)
    if stale:
        out["codes"], out["grade"], out["note"], out["stale"] = ["STALE_INPUTS"], "X", stale, True
        return out
    if px.get("load_error"):
        out["codes"], out["grade"], out["note"] = ["LOAD_FAIL"], "X", px["load_error"]
        return out
    skew = skewed_metrics(mj, px)
    if skew:
        out["metrics_version"] = {"mujoco": mj.get("metrics_version") or "1.0", "physx": px.get("metrics_version") or "1.0", "not_comparable": skew}
    if (px.get("structure") or {}).get("status") == "fail":
        out["codes"].append("STRUCTURE_FAIL")
    exp_mj, exp_px = inputs["schedule"]["mjcf"], inputs["schedule"][kind]
    grade_c, grade_b = False, bool(out["codes"])
    for phase in PHASES:
        e_mj, e_px = exp_mj[phase], exp_px[phase]
        r_mj, r_px = mj.get("phases", {}).get(phase, {}), px.get("phases", {}).get(phase, {})
        row = {"expected_mujoco": e_mj, "expected_physx": e_px, "mujoco": r_mj.get("status", "na"), "physx": r_px.get("status", "na"), "codes": [], "deltas": {}}
        if e_px.startswith("na:") and not e_mj.startswith("na:"):
            row["codes"] = ["RL_CANON"] if kind == "usd_rl" else []
            row["agree"] = None
            out["phases"][phase] = row
            continue
        if e_mj.startswith("na:"):
            row["agree"] = None
            out["phases"][phase] = row
            continue
        m_mj, m_px = r_mj.get("metrics", {}) or {}, r_px.get("metrics", {}) or {}
        # the RL expectation can differ (welded lock => stays_closed while MuJoCo opens): judged against its own expectation
        codes = _phase_codes(inputs, phase, e_px if e_px == e_mj else e_mj, row["mujoco"], row["physx"], m_mj, m_px)
        if e_px != e_mj:
            codes = [c for c in codes if c not in ("PHYSX_NO_OPEN", "PHYSX_HOLD_FAIL", "EXPORT_WELD_MISSING")] + (["RL_CANON"] if row["physx"] == "pass" else ["PHYSX_HOLD_FAIL" if e_px == "stays_closed" else "PHYSX_NO_OPEN"])
        agree = row["mujoco"] == row["physx"]
        if agree and row["mujoco"] == "pass" and e_px == e_mj:      # different expectations (RL welds) -> different numbers by construction
            # the leaf coasts into its stop in both runs: the value at the end of the phase differs only by how each
            # solver bounces off the limit (MuJoCo's soft limit returns ~17 % of the impact velocity, PhysX's
            # articulation limit is inelastic), so the peak - which both reach - is the comparable quantity
            rebound = phase == "operate" and bool(m_mj.get("primary_at_limit", primary_at_limit(inputs, m_mj.get("q_primary_max")))) \
                and bool(m_px.get("primary_at_limit", primary_at_limit(inputs, m_px.get("q_primary_max"))))
            # relatch continues from operate: when the two runs enter it from different angles (that same rebound off
            # the stop), its *timing* metrics measure two different experiments.  The verdict metrics of the phase
            # (relatch_closed_angle / relatch_repush_angle, both end states) stay graded.
            not_like_for_like = set()
            if phase == "relatch" and not _within(m_mj.get("opened_before"), m_px.get("opened_before"), 0.1 if inputs["is_hinge"] else 0.05, 0.2):
                not_like_for_like = {"t_close", "arrival_speed"}
            for name, (atol, rtol) in metric_tolerances(inputs, phase, e_mj).items():
                a, b = m_mj.get(name), m_px.get(name)
                if a is None and b is None:
                    continue
                if name in skew or name in not_like_for_like:
                    why = ("metric definition changed; re-run the older side" if name in skew else
                           f"phase entered at a different angle (mujoco {m_mj.get('opened_before')}, physx {m_px.get('opened_before')})")
                    row["deltas"][name] = {"mujoco": a, "physx": b, "delta": (None if (a is None or b is None) else float(b - a)),
                                           "abs_tol": atol, "rel_tol": rtol, "ok": None, "not_comparable": why}
                    if name in skew and "METRICS_VERSION_SKEW" not in codes:
                        codes.append("METRICS_VERSION_SKEW")
                    continue
                ok = _within(a, b, atol, rtol)
                waived = None
                if not ok and name == "opened" and rebound:
                    ok, waived = True, "both runs reached the joint limit; graded on q_primary_max"
                row["deltas"][name] = {"mujoco": a, "physx": b, "delta": (None if (a is None or b is None) else float(b - a)), "abs_tol": atol, "rel_tol": rtol, "ok": ok}
                if waived:
                    row["deltas"][name]["waived"] = waived
                if not ok and "METRIC_DELTA" not in codes:
                    codes.append("METRIC_DELTA")
            if phase == "settle":
                d_mj, d_px = m_mj.get("settle_drift_signed", {}) or {}, m_px.get("settle_drift_signed", {}) or {}
                worst = None
                for j in d_px:                      # MuJoCo records every joint; PhysX only the ones the file has
                    if j in inputs["unlimited_joints"]:
                        continue
                    tol = 0.02 if inputs["joints"].get(j, {}).get("type", "hinge") == "hinge" else 0.005    # per joint type (rad / m)
                    dd = abs(d_mj.get(j, 0.0) - d_px[j])
                    if dd > tol and (worst is None or dd > worst[1]):
                        worst = (j, dd, tol)
                if worst is not None:
                    row["deltas"]["settle_drift_joint"] = {"joint": worst[0], "mujoco": d_mj.get(worst[0], 0.0), "physx": d_px[worst[0]], "delta": worst[1], "abs_tol": worst[2], "ok": False}
                    if "SETTLE_DRIFT" not in codes:
                        codes.append("SETTLE_DRIFT")
        row["codes"] = codes
        row["agree"] = agree
        out["phases"][phase] = row
        for c in codes:
            if c not in out["codes"]:
                out["codes"].append(c)
        if any(c in STATUS_CODES for c in codes):
            grade_c = True
        elif any(c in QUANT_CODES for c in codes):
            grade_b = True
    if skew:
        out["codes"] = [c for c in out["codes"] if c != "METRICS_VERSION_SKEW"] + ["METRICS_VERSION_SKEW"]
    out["grade"] = "C" if grade_c else ("B" if grade_b else "A")
    # "OK" means every *comparable* thing agrees; a provenance code (a metric that could not be compared at all) is
    # recorded alongside it rather than instead of it, so the summary still counts the doors that are at parity
    if not [c for c in out["codes"] if c not in PROVENANCE_CODES]:
        out["codes"] = ["OK"] + out["codes"]
    return out


def summarize(compares: Iterable[dict]) -> dict:
    """Dataset-level counts from compare_door outputs: grades, codes, per phase agreement."""
    comps = list(compares)
    s = {"n": len(comps), "grades": {}, "codes": {}, "phases": {p: {"agree": 0, "disagree": 0, "na": 0} for p in PHASES}, "worst": [],
         "stale": {"n": 0, "doors": []}, "metrics_version_skew": {"n": 0, "metrics": [], "doors": []}}
    for c in comps:
        if "STALE_INPUTS" in c["codes"]:
            s["stale"]["n"] += 1
            if len(s["stale"]["doors"]) < 200:
                s["stale"]["doors"].append(c["door_id"])
        if "METRICS_VERSION_SKEW" in c["codes"]:
            s["metrics_version_skew"]["n"] += 1
            for m in (c.get("metrics_version") or {}).get("not_comparable", []):
                if m not in s["metrics_version_skew"]["metrics"]:
                    s["metrics_version_skew"]["metrics"].append(m)
            if len(s["metrics_version_skew"]["doors"]) < 200:
                s["metrics_version_skew"]["doors"].append(c["door_id"])
        s["grades"][c["grade"]] = s["grades"].get(c["grade"], 0) + 1
        for code in c["codes"]:
            s["codes"].setdefault(code, {"count": 0, "examples": []})
            s["codes"][code]["count"] += 1
            if len(s["codes"][code]["examples"]) < 8:
                s["codes"][code]["examples"].append(c["door_id"])
        for p, row in c.get("phases", {}).items():
            if row.get("agree") is None:
                s["phases"][p]["na"] += 1
            elif row["agree"]:
                s["phases"][p]["agree"] += 1
            else:
                s["phases"][p]["disagree"] += 1
    rank = {"X": 3, "C": 2, "B": 1, "A": 0}
    s["worst"] = [{"door_id": c["door_id"], "kind": c.get("kind"), "grade": c["grade"], "codes": c["codes"]} for c in sorted(comps, key=lambda c: -rank.get(c["grade"], 0))[:20] if c["grade"] != "A"]
    return s
