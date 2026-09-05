"""Isaac parity protocol (doorbench/parity): per-door inputs, verdict classification, and the MuJoCo reference runner.

  * door_inputs / expected_outcomes for representative doors (spring latch + knob, env-released weld, slide-bolt
    operator, vault dogs, far-side panic bar, saloon, revolving, maglock, locked turnstile)
  * phase_efforts follows the qa.py schedule; tendon clamp and servo helpers
  * compare_door classifies synthetic MuJoCo / PhysX records (OK, PHYSX_NO_OPEN, PHYSX_HOLD_FAIL, EXPORT_WELD_MISSING,
    SETTLE_DRIFT, LIMIT_VIOLATION, NAN, CLOSER_NO_RETURN, LATCH_NO_RETURN, MUJOCO_FAIL, RL_CANON, METRIC_DELTA)
  * MuJoCo runner on 3 doors: every applicable phase passes and the qa.json metrics are reproduced

Run:  pytest -q tests/test_parity_protocol.py       (~15 s; skipped when assets/ has not been generated)
"""
from __future__ import annotations

import copy
import json
import math
import os

import pytest

from doorbench.parity import protocol as P

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.environ.get("DOORBENCH_ASSETS", os.path.join(ROOT, "assets"))
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ASSETS, "manifest.json")), reason=f"generated dataset not found at {ASSETS}")


def _load(door_id):
    dd = os.path.join(ASSETS, "doors", door_id)
    with open(os.path.join(dd, "spec.json")) as f:
        spec = json.load(f)
    with open(os.path.join(dd, "model.json")) as f:
        mj = json.load(f)
    qa = None
    if os.path.isfile(os.path.join(dd, "qa.json")):
        with open(os.path.join(dd, "qa.json")) as f:
            qa = json.load(f)
    return spec, mj, qa, P.read_rl_meta(dd)


def _inputs(door_id):
    spec, mj, qa, rl = _load(door_id)
    return P.door_inputs(spec, mj, qa=qa, rl_meta=rl)


# ---------------------------------------------------------------------------
# inputs / expectations
# ---------------------------------------------------------------------------
def test_inputs_spring_latch_knob_door():
    inp = _inputs("db0002_swing_single")
    assert inp["is_hinge"] and inp["primary_joint"] == "leaf_hinge" and inp["operator_joint"] == "leaf_handle_hinge" and inp["latch_bolt_joint"] == "leaf_latch_bolt_slide"
    assert inp["flags"]["spring_latch"] and inp["flags"]["has_holding"] and not inp["flags"]["lock_engaged"]
    assert inp["forces"]["operator_effort"] == 4.0 and inp["forces"]["source"] == "qa.json"
    assert inp["forces"]["push"] == pytest.approx(60.360874, abs=1e-5)       # qa_push of the door
    assert inp["coupling"]["latch"]["scale"] == pytest.approx(0.015679, abs=1e-6) and inp["coupling"]["latch"]["operator_joint"] == "leaf_handle_hinge"
    sched = inp["schedule"]
    for kind in ("mjcf", "usd_full", "usd_rl"):
        assert sched[kind]["hold"] == "hold" and sched[kind]["operate"] == "opens" and sched[kind]["release"] == "bolt_returns" and sched[kind]["relatch"] == "relatches"
        assert sched[kind]["closer"].startswith("na:") and sched[kind]["locked"].startswith("na:")
    assert inp["rl"]["slot_of"] == {"leaf_hinge": "door_hinge", "leaf_handle_hinge": "operator_hinge", "leaf_latch_bolt_slide": "latch_slide"}
    th = inp["thresholds"]
    assert th["thr"] == pytest.approx(math.radians(2)) and th["target"] == pytest.approx(math.radians(20)) and th["latch_throw_m"] == pytest.approx(0.0127)


def test_inputs_env_release_weld_door():
    inp = _inputs("db0021_swing_single")          # delayed egress: weld leaf -> world, released by env logic
    assert inp["flags"]["env_release_only"] and inp["flags"]["has_weld"]
    assert inp["schedule"]["mjcf"] == {"settle": "settle", "hold": "hold", "operate": "na:lock released by environment logic", "release": "na:operate not expected to open",
                                       "relatch": "na:operate not expected to open", "closer": "na:no closer test", "locked": "na:not a locked-no-release door"}
    assert inp["coupling"]["welds"] == [{"body1": "leaf", "body2": "world", "active": True}]


def test_inputs_slide_bolt_gate_and_vault_dogs():
    gate = _inputs("db0033_gate_sliding")        # slide bolt is latch, lock and operator at once
    assert not gate["is_hinge"] and gate["forces"]["operator_effort"] == 120.0 and gate["thresholds"]["thr"] == 0.015
    assert gate["schedule"]["mjcf"]["operate"] == "opens" and gate["schedule"]["usd_rl"]["operate"] == "opens"   # the operator slot IS the release
    assert [a["joint"] for a in gate["aux_joints"]] == ["leaf_slide_bolt_slide"]
    vault = _inputs("db0124_vault")              # two dogs: dog_0 is the operator, dog_1 is welded in door_rl.usda
    assert vault["forces"]["operator_effort"] == 14.0 and vault["dog_joints"] == ["dog_0_hinge", "dog_1_hinge"]
    assert vault["schedule"]["mjcf"]["operate"] == "opens" and vault["schedule"]["usd_full"]["operate"] == "opens"
    assert vault["schedule"]["usd_rl"]["operate"] == "stays_closed"
    assert vault["coupling"]["mimics"][0]["driven"] == "bolt_0_slide"


def test_inputs_panic_far_side_saloon_revolving_maglock_turnstile():
    panic = _inputs("db0011_automatic_swing")    # touchbar on the push side, robot outside: no operator joint
    assert panic["operator_joint"] is None and panic["schedule"]["mjcf"]["operate"].startswith("na:no operator joint") and panic["schedule"]["mjcf"]["closer"] == "closes"
    assert panic["flags"]["automatic"] and panic["coupling"]["actuators"][0]["kp"] == 150.0
    saloon = _inputs("db0031_saloon")
    assert saloon["flags"]["free_swing"] and saloon["schedule"]["mjcf"]["hold"] == "free_opens_info" and saloon["schedule"]["mjcf"]["closer"].startswith("na:")
    rev = _inputs("db0066_revolving")
    assert rev["unlimited_joints"] == ["rotor_hinge"] and rev["schedule"]["mjcf"]["hold"] == "free_opens_info"
    mag = _inputs("db0026_swing_single")          # maglock engaged: MuJoCo weld, nothing in the USD
    assert mag["flags"]["env_release_only"] and mag["flags"]["has_weld"] and mag["schedule"]["usd_full"]["hold"] == "hold"
    turn = _inputs("db0187_turnstile_fullheight")  # locked rotor: must hold within its locked play
    assert turn["flags"]["locked_rotor"] and turn["schedule"]["mjcf"]["hold"] == "hold" and turn["thresholds"]["thr"] > 0.05


def test_inputs_hash_stable_and_forces_override():
    spec, mj, qa, rl = _load("db0002_swing_single")
    a = P.door_inputs(spec, mj, qa=qa, rl_meta=rl)
    b = P.door_inputs(spec, mj, qa=qa, rl_meta=rl)
    assert a["inputs_hash"] == b["inputs_hash"]
    c = P.door_inputs(spec, mj, forces={"bias": 0.0, "frictionloss": 0.180437, "preload": 0.0}, qa=qa, rl_meta=rl)
    assert c["forces"]["source"] == "mujoco" and c["forces"]["push"] == pytest.approx(2 * 0.180437 + 60)
    d = P.door_inputs(spec, mj, rl_meta=rl)                     # no qa.json, no MuJoCo: model.json estimate
    assert d["forces"]["source"].startswith("model.json") and d["forces"]["bias"] is None


# ---------------------------------------------------------------------------
# drive schedule and helpers
# ---------------------------------------------------------------------------
def test_phase_efforts_follow_qa_schedule():
    inp = _inputs("db0002_swing_single")
    pj, oj, push = inp["primary_joint"], inp["operator_joint"], inp["forces"]["push"]
    q0 = {pj: 0.0, oj: 0.0}
    assert P.phase_efforts(inp, "settle", 0.5, q0) == {}
    assert P.phase_efforts(inp, "hold", 0.0, q0) == {pj: push}
    assert P.phase_efforts(inp, "hold", 1.5, q0) == {}                          # holding door: 1 s only
    assert P.phase_efforts(inp, "operate", 0.598, q0) == {}
    assert P.phase_efforts(inp, "operate", 300 * 0.002, q0) == {oj: 4.0}      # operator from 0.6 s (float slack)
    assert P.phase_efforts(inp, "operate", 600 * 0.002, q0) == {oj: 4.0, pj: push}
    assert P.phase_efforts(inp, "operate", 2.0, {pj: math.radians(55), oj: 0.5}) == {oj: 4.0}   # push stops past 50 deg
    assert P.phase_efforts(inp, "operate", 6.4, q0) == {}
    assert P.phase_efforts(inp, "release", 0.1, q0) == {}
    assert P.phase_efforts(inp, "relatch", 1.0, q0) == {pj: -inp["forces"]["close_drive"]}
    assert P.phase_efforts(inp, "relatch", 6.5, q0) == {pj: push}
    assert P.phase_efforts(inp, "closer", 3.0, q0) == {}
    vault = _inputs("db0124_vault")
    e = P.phase_efforts(vault, "operate", 1.5, {vault["primary_joint"]: 0.0})
    assert e["dog_0_hinge"] == 14.0 and e["dog_1_hinge"] == 14.0 and e[vault["primary_joint"]] == vault["forces"]["push"]
    auto = _inputs("db0011_automatic_swing")
    tt = auto["thumbturn_joint"]
    assert tt == "leaf_deadbolt_thumbturn_hinge"
    # no operator joint: the operate phase is n/a, but the schedule function still drives the thumbturn until 1.2 s
    assert P.phase_efforts(auto, "operate", 0.5, {auto["primary_joint"]: 0.0}) == {tt: 2.0}
    assert tt not in P.phase_efforts(auto, "operate", 1.3, {auto["primary_joint"]: 0.0})


def test_tendon_clamp_and_servo():
    inp = _inputs("db0002_swing_single")
    bolt, oj = inp["latch_bolt_joint"], inp["operator_joint"]
    mins = P.tendon_min_positions(inp, {oj: 0.5, bolt: 0.0})
    assert mins == {bolt: pytest.approx(0.5 * 0.015679, abs=1e-6)}
    mins = P.tendon_min_positions(inp, {oj: 5.0, bolt: 0.0})
    assert mins[bolt] == pytest.approx(0.0127)                                  # clipped to the bolt's upper limit
    assert P.tendon_min_positions(inp, {oj: -0.3}) == {bolt: pytest.approx(-0.3 * 0.015679)}
    auto = _inputs("db0011_automatic_swing")
    pj = auto["primary_joint"]
    assert P.servo_effort(auto, {pj: 0.1}, {pj: 0.0}) == {pj: pytest.approx(-15.0)}
    assert P.servo_effort(auto, {pj: 1.0}, {pj: 1.0}) == {pj: -60.0}           # forcerange clip
    assert P.servo_effort(inp, {inp["primary_joint"]: 1.0}, {}) == {}


def test_phase_duration_and_initial_state():
    inp = _inputs("db0002_swing_single")
    assert P.phase_duration(inp, "hold") == 1.0 and P.phase_duration(inp, "operate") == 6.4 and P.phase_duration(inp, "relatch") == 7.0
    free = _inputs("db0003_cold_storage")
    assert P.phase_duration(free, "hold") == 6.0
    auto = _inputs("db0011_automatic_swing")
    st = P.phase_initial_state(auto, "closer")
    assert st[auto["primary_joint"]] == pytest.approx(math.radians(60)) and st["leaf_latch_bolt_slide"] == 0.0


# ---------------------------------------------------------------------------
# verdict classification on synthetic records
# ---------------------------------------------------------------------------
def _curve(inp, t_end, q_primary, q_bolt=None, q_op=None, hz=P.SAMPLE_HZ, extra=None):
    n = int(round(t_end * hz)) + 1
    ts = [k / hz for k in range(n)]
    q = {inp["primary_joint"]: [q_primary(t) for t in ts]}
    if q_bolt is not None:
        q[inp["latch_bolt_joint"]] = [q_bolt(t) for t in ts]
    if q_op is not None:
        q[inp["operator_joint"]] = [q_op(t) for t in ts]
    v = {inp["primary_joint"]: [0.0] + [(q[inp["primary_joint"]][k] - q[inp["primary_joint"]][k - 1]) * hz for k in range(1, n)]}
    minmax = {j: [min(a), max(a)] for j, a in q.items()}
    c = {"t": ts, "q": q, "v": v, "minmax": minmax, "vmax": {j: max(abs(x) for x in v[j]) for j in v}, "finite": True, "warnings": []}
    c.update(extra or {})
    return c


def _record(inp, kind, behaviour):
    """Synthetic runner record: behaviour = {phase: callable(t)->q_primary (+ optional bolt)}."""
    sched = inp["schedule"][kind]
    phases, ctx = {}, {}
    for phase in P.PHASES:
        expected = sched[phase]
        if expected.startswith("na:"):
            phases[phase] = {"expected": expected, "status": "na", "metrics": {}}
            continue
        spec = behaviour.get(phase, {})
        dur = P.phase_duration(inp, phase, kind)
        curve = _curve(inp, dur, spec.get("q", lambda t: 0.0), spec.get("bolt"), spec.get("op"), extra=spec.get("extra"))
        m = P.phase_metrics(inp, phase, curve, ctx)
        if phase == "operate":
            ctx["opened"] = m["opened"]
        phases[phase] = {"expected": expected, "status": P.phase_status(inp, phase, expected, m), "metrics": m}
    return {"phases": phases, "structure": {"status": "pass"}}


def _good_behaviour(inp):
    return {"settle": {"q": lambda t: 0.0}, "hold": {"q": lambda t: 0.002},
            "operate": {"q": lambda t: 0.0 if t < 1.2 else min(1.6, 0.6 * (t - 1.2)), "bolt": lambda t: 0.0 if t < 0.6 else min(0.0127, 0.03 * (t - 0.6)), "op": lambda t: 0.0 if t < 0.6 else min(0.87, 2.0 * (t - 0.6))},
            "release": {"q": lambda t: 1.6, "bolt": lambda t: max(0.0, 0.0127 - 0.05 * t), "op": lambda t: max(0.0, 0.87 - 3 * t)},
            "relatch": {"q": lambda t: max(0.0, 1.6 - 0.5 * t) if t < 6.0 else 0.002}}


def test_verdict_ok_and_metric_delta():
    inp = _inputs("db0002_swing_single")
    mj = _record(inp, "mjcf", _good_behaviour(inp))
    assert all(r["status"] == "pass" for p, r in mj["phases"].items() if r["status"] != "na")
    px = _record(inp, "usd_full", _good_behaviour(inp))
    v = P.compare_door(inp, mj, px, kind="usd_full")
    assert v["grade"] == "A" and v["codes"] == ["OK"]
    slow = _good_behaviour(inp)
    slow["operate"] = {"q": lambda t: 0.0 if t < 3.0 else min(1.6, 0.6 * (t - 3.0)), "bolt": slow["operate"]["bolt"], "op": slow["operate"]["op"]}   # opens 1.8 s later
    v = P.compare_door(inp, mj, _record(inp, "usd_full", slow), kind="usd_full")
    assert v["grade"] == "B" and "METRIC_DELTA" in v["codes"] and not v["phases"]["operate"]["deltas"]["t_open"]["ok"]


def test_verdict_physx_no_open_and_latch_no_return():
    inp = _inputs("db0002_swing_single")
    mj = _record(inp, "mjcf", _good_behaviour(inp))
    stuck = _good_behaviour(inp)
    stuck["operate"] = {"q": lambda t: 0.002, "bolt": lambda t: 0.0, "op": stuck["operate"]["op"]}   # handle turns, bolt never retracts
    v = P.compare_door(inp, mj, _record(inp, "usd_full", stuck), kind="usd_full")
    assert v["grade"] == "C" and "PHYSX_NO_OPEN" in v["codes"] and v["phases"]["operate"]["physx"] == "fail" and v["phases"]["operate"]["mujoco"] == "pass"
    assert v["phases"]["release"]["physx"] == "skip"           # the door never opened: release is vacuous in PhysX
    sticky = _good_behaviour(inp)
    sticky["release"] = {"q": lambda t: 1.6, "bolt": lambda t: 0.0127, "op": lambda t: 0.0}
    v = P.compare_door(inp, mj, _record(inp, "usd_full", sticky), kind="usd_full")
    assert "LATCH_NO_RETURN" in v["codes"] and v["grade"] == "C"


def test_verdict_hold_fail_weld_missing_and_rl_canon():
    latch = _inputs("db0002_swing_single")
    mj = _record(latch, "mjcf", _good_behaviour(latch))
    weak = _good_behaviour(latch)
    weak["hold"] = {"q": lambda t: 1.5 * t}
    v = P.compare_door(latch, mj, _record(latch, "usd_full", weak), kind="usd_full")
    assert "PHYSX_HOLD_FAIL" in v["codes"] and v["grade"] == "C"
    mag = _inputs("db0026_swing_single")     # maglock weld: env logic, not exported
    mj = _record(mag, "mjcf", {"hold": {"q": lambda t: 0.0}})
    px = _record(mag, "usd_full", {"hold": {"q": lambda t: min(1.57, 1.5 * t)}})
    v = P.compare_door(mag, mj, px, kind="usd_full")
    assert v["codes"] == ["EXPORT_WELD_MISSING"] and v["grade"] == "C"
    vault = _inputs("db0124_vault")          # rl: dog_1 welded engaged -> stays_closed is the RL expectation
    mj = _record(vault, "mjcf", {"hold": {"q": lambda t: 0.002}, "operate": {"q": lambda t: 0.0 if t < 1.2 else min(1.7, 0.7 * (t - 1.2))}})
    px = _record(vault, "usd_rl", {"hold": {"q": lambda t: 0.002}, "operate": {"q": lambda t: 0.001}})
    v = P.compare_door(vault, mj, px, kind="usd_rl")
    assert v["phases"]["operate"]["physx"] == "pass" and "RL_CANON" in v["phases"]["operate"]["codes"] and v["grade"] == "A"
    px_open = _record(vault, "usd_rl", {"hold": {"q": lambda t: 0.002}, "operate": {"q": lambda t: min(1.0, t)}})
    v = P.compare_door(vault, mj, px_open, kind="usd_rl")
    assert "PHYSX_HOLD_FAIL" in v["phases"]["operate"]["codes"] and v["grade"] == "C"


def test_verdict_settle_drift_limits_nan_closer_mujoco_fail_missing():
    auto = _inputs("db0011_automatic_swing")    # hold + closer phases
    good = {"settle": {"q": lambda t: 0.0}, "hold": {"q": lambda t: 0.002}, "closer": {"q": lambda t: max(0.0, math.radians(60) - 0.2 * t)}}
    mj = _record(auto, "mjcf", good)
    assert mj["phases"]["closer"]["status"] == "pass"
    # settle drift on the latch bolt in PhysX only (spring target lost)
    drift = copy.deepcopy(good)
    drift["settle"] = {"q": lambda t: 0.0, "bolt": lambda t: min(0.008, 0.02 * t)}
    v = P.compare_door(auto, mj, _record(auto, "usd_full", drift), kind="usd_full")
    assert "SETTLE_DRIFT" in v["codes"] and v["phases"]["settle"]["deltas"]["settle_drift_joint"]["joint"] == "leaf_latch_bolt_slide" and v["grade"] == "B"
    # closer never returns
    stuck = copy.deepcopy(good)
    stuck["closer"] = {"q": lambda t: math.radians(60)}
    v = P.compare_door(auto, mj, _record(auto, "usd_full", stuck), kind="usd_full")
    assert "CLOSER_NO_RETURN" in v["codes"] and v["grade"] == "C"
    # limit violation only in PhysX (door beyond max_open by 5 deg)
    over = copy.deepcopy(good)
    over["hold"] = {"q": lambda t: 0.002, "extra": {"minmax": {auto["primary_joint"]: [0.0, math.radians(105)]}}}
    v = P.compare_door(auto, mj, _record(auto, "usd_full", over), kind="usd_full")
    assert "LIMIT_VIOLATION" in v["codes"] and v["grade"] == "C"
    # NaN
    nan = copy.deepcopy(good)
    nan["hold"] = {"q": lambda t: 0.002, "extra": {"finite": False}}
    v = P.compare_door(auto, mj, _record(auto, "usd_full", nan), kind="usd_full")
    assert "NAN" in v["codes"] and v["grade"] == "C"
    # MuJoCo itself fails (reference problem, not a PhysX bug)
    bad_mj = _record(auto, "mjcf", dict(good, closer={"q": lambda t: math.radians(60)}))
    v = P.compare_door(auto, bad_mj, _record(auto, "usd_full", good), kind="usd_full")
    assert v["codes"] == ["MUJOCO_FAIL"] and v["grade"] == "C"
    # missing / load failure
    assert P.compare_door(auto, mj, None)["codes"] == ["MISSING"]
    v = P.compare_door(auto, mj, {"load_error": "spawn: boom", "phases": {}})
    assert v["codes"] == ["LOAD_FAIL"] and v["grade"] == "X"
    s = P.summarize([v, P.compare_door(auto, mj, _record(auto, "usd_full", good))])
    assert s["grades"] == {"X": 1, "A": 1} and s["codes"]["LOAD_FAIL"]["count"] == 1


# ---------------------------------------------------------------------------
# MuJoCo reference runner
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("door_id", ["db0002_swing_single", "db0033_gate_sliding", "db0011_automatic_swing"])
def test_mujoco_runner_reproduces_qa(door_id, tmp_path):
    pytest.importorskip("mujoco")
    from doorbench.parity.mujoco_runner import run_door, compact_record

    rec = run_door(os.path.join(ASSETS, "doors", door_id), cache_dir=str(tmp_path))
    assert rec["ok"], {p: r["status"] for p, r in rec["phases"].items()}
    assert rec["inputs"]["forces"]["source"] == "mujoco"
    assert rec["qa_reproduction"]["ok"] and rec["qa_reproduction"]["compared"] >= 2, rec["qa_reproduction"]
    for p, r in rec["phases"].items():
        if r["status"] != "na":
            assert r["curve"]["t"][0] == 0.0 and abs(r["curve"]["t"][-1] - P.phase_duration(rec["inputs"], p)) < 1e-6 or r["curve"].get("early_exit")
            assert set(r["curve"]["q"]) == set(rec["inputs"]["joints"])
    assert rec["sanity"]["finite"] and not rec["sanity"]["warnings"]
    # cached second run
    rec2 = run_door(os.path.join(ASSETS, "doors", door_id), cache_dir=str(tmp_path))
    assert rec2.get("cached") and rec2["inputs_hash"] == rec["inputs_hash"]
    lite = compact_record(rec)
    assert "curve" in lite["phases"]["hold"] and lite["phases"]["hold"]["curve"]["hz"] == 5
    assert json.dumps(lite)   # serialisable


def test_mujoco_runner_phase_behaviour():
    pytest.importorskip("mujoco")
    from doorbench.parity.mujoco_runner import MujocoDoor

    door = MujocoDoor(os.path.join(ASSETS, "doors", "db0002_swing_single"))
    rec = door.run()
    ph = rec["phases"]
    assert ph["hold"]["metrics"]["hold_displacement"] < math.radians(2)
    assert ph["operate"]["metrics"]["opened"] > math.radians(20) and ph["operate"]["metrics"]["bolt_retract_max_frac"] > 0.8 and ph["operate"]["metrics"]["t_unlatch"] < 1.2
    assert ph["release"]["metrics"]["bolt_after_release_m"] < 0.006 and ph["relatch"]["metrics"]["relatch_repush_angle"] < math.radians(2.5)
    assert ph["settle"]["metrics"]["settle_drift_other_max"] < 1e-3          # spring preloads hold the knob and the bolt in MuJoCo
    auto = MujocoDoor(os.path.join(ASSETS, "doors", "db0011_automatic_swing")).run()
    assert auto["phases"]["closer"]["status"] == "pass" and abs(auto["phases"]["closer"]["metrics"]["closer_final_angle"]) < math.radians(6)
    assert auto["emulations_used"] == ["servo_native"]
