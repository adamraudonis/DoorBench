"""Parity gate: the metric definitions, the tolerances that depend on the door, and the provenance guards.

These are the protocol / metric / report bugs that inflated the disagreement count of rounds 1 and 2:

* two records that describe different doors (``inputs_hash``) were joined and their differences attributed to PhysX,
* a metric whose definition changed was differenced across the two definitions,
* the operate phase was graded on the value at the end of the phase, which is a rebound off the joint stop,
* a door expected to swing open was judged with the latched-door hold tolerance,
* the settle classifier tagged every disagreement as a lost spring preload, spring or no spring,
* the sign-off push was a flat 60 N*m, ~100x the scale of a 0.14 kg pet flap.

Everything here is synthetic (no simulator, no dataset) except the push-scaling test, which only needs the formula.
"""
from __future__ import annotations

import json
import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "isaaclab"))

from doorbench import qa as QA  # noqa: E402
from doorbench.parity import protocol as P  # noqa: E402
from doorbench.parity import results as R  # noqa: E402
from doorbench.parity.report import build_report, plot_histogram  # noqa: E402
import merge_isaac_results as MERGE  # noqa: E402

DEG = math.pi / 180.0


# ---------------------------------------------------------------------------------------------------------------------
# a minimal protocol-inputs dict: enough for compare_door / phase_metrics without a dataset

def _inputs(hinge: bool = True, rng=(0.0, 2.0944), thr: float = 2.0 * DEG, hold: str = "hold", stiffness: float = 0.0, springref: float = 0.0) -> dict:
    joints = {"leaf_hinge": {"type": "hinge" if hinge else "slide", "range": list(rng), "stiffness": stiffness, "springref": springref,
                             "frictionloss": 0.0, "damping": 0.0, "armature": 0.0, "role": "primary", "initial": 0.0, "modeled_at": 0.0}}
    inp = {
        "door_id": "db0001_test", "family": "swing_single", "is_hinge": hinge, "unit": "hinge" if hinge else "slide",
        "primary_joint": "leaf_hinge", "operator_joint": None, "secondary_joint": None, "latch_bolt_joint": None,
        "joints": joints, "unlimited_joints": [], "thumbturn_joint": None, "aux_joints": [], "dog_joints": [], "latch_joints": [],
        "flags": {"rest_angle_deg": None, "env_release_only": False, "has_weld": False},
        "thresholds": {"thr": thr, "thr_free": 10.0 * DEG, "target": 20.0 * DEG, "thr_locked": thr, "chain_limit_rad": 0.0, "chain_engaged": False,
                       "closer_start": 60.0 * DEG, "closer_pass": 6.0 * DEG, "closed_thr": 3.0 * DEG, "open_thr_bench": 30.0 * DEG,
                       "relatch_closed": 2.0 * DEG, "relatch_repush": 2.5 * DEG, "relatch_min_open": 5.0 * DEG, "release_min_open": 3.0 * DEG,
                       "bolt_return_m": 0.006, "settle_primary": 0.05, "settle_other": 0.02, "pen0_min_m": -0.012,
                       "limit_tol": {"hinge": 2.0 * DEG, "slide": 0.01}, "v_cap_primary": 15.0, "v_cap_any": 50.0,
                       "operator_travel": 0.87, "operator_dead_travel": 0.0, "operator_yield": 0.0, "slam_velocity": 0.0, "latch_throw_m": 0.0127},
        "schedule": {k: {"settle": "settle", "hold": hold, "operate": "opens", "release": "na:x", "relatch": "na:x", "closer": "na:x", "locked": "na:x"} for k in P.KINDS},
        "inputs_hash": "aaaaaaaaaaaa",
    }
    return inp


def _rec(phases: dict, inputs_hash: str = "aaaaaaaaaaaa", metrics_version: str | None = None) -> dict:
    rec = {"phases": phases, "structure": {"status": "pass"}, "inputs_hash": inputs_hash}
    if metrics_version:
        rec["metrics_version"] = metrics_version
    return rec


def _ph(status: str, metrics: dict, expected: str = "opens") -> dict:
    base = {"finite": True, "warnings": [], "limit_violations": []}
    return {"expected": expected, "status": status, "metrics": base | metrics}


# ---------------------------------------------------------------------------------------------------------------------
# 1. staleness: two records must describe the same door

def test_compare_door_flags_stale_inputs():
    inp = _inputs()
    good = _rec({"operate": _ph("pass", {"opened": 1.0, "q_primary_max": 1.0, "t_open": 1.4})})
    stale = _rec({"operate": _ph("pass", {"opened": 1.0, "q_primary_max": 1.0, "t_open": 1.4})}, inputs_hash="bbbbbbbbbbbb")
    v = P.compare_door(inp, good, stale, kind="usd_full")
    assert v["grade"] == "X" and v["codes"] == ["STALE_INPUTS"] and "aaaaaaaaaaaa" in v["note"] and "bbbbbbbbbbbb" in v["note"]
    assert P.compare_door(inp, good, good, kind="usd_full")["grade"] == "A"
    # a reference that is itself older than the current protocol inputs is caught too
    old_ref = _rec({"operate": _ph("pass", {"opened": 1.0, "q_primary_max": 1.0, "t_open": 1.4})}, inputs_hash="cccccccccccc")
    assert P.compare_door(inp, old_ref, old_ref, kind="usd_full")["codes"] == ["STALE_INPUTS"]
    assert P.summarize([P.compare_door(inp, good, stale)])["stale"]["n"] == 1


def test_results_stale_verdict_is_never_published():
    mj = R.normalize_record({"inputs_hash": "aaaa", "phases": {"hold": {"pass": True, "metrics": {"hold_displacement": 0.001}}}})
    px = R.normalize_record({"inputs_hash": "bbbb", "phases": {"hold": {"pass": True, "metrics": {"hold_displacement": 0.001}}}})
    kv = R.compare_kind(mj, px, {"is_hinge": True, "flags": {}}, "full")
    assert kv["status"] == "stale" and kv["grade"] == "X" and [c["code"] for c in kv["classes"]] == ["STALE_INPUTS"]
    v = R.door_verdict("db0001_test", mj, {"full": px, "rl": None}, {"is_hinge": True, "flags": {}})
    assert v["status"] == "stale" and v["ok"] is None and v["primary_class"] == "STALE_INPUTS"
    assert R.manifest_status(v) == "untested"          # withheld, not published as ok or fail
    block = MERGE.qa_block(v, {"date": "2026-09-05"})
    assert block["status"] == "stale" and block["ok"] is None


def test_normalize_record_carries_provenance():
    rec = R.normalize_record({"inputs_hash": "abc123", "protocol_version": "1.0", "metrics_version": "1.1",
                              "inputs": {"primary_joint": "leaf_hinge"}, "phases": {"settle": {"pass": True}}})
    assert rec["inputs_hash"] == "abc123" and rec["metrics_version"] == "1.1" and rec["inputs"]["primary_joint"] == "leaf_hinge"
    assert R.normalize_record({"phases": {}})["inputs_hash"] is None      # a record without the field is not "stale"
    assert R.stale_reason(R.normalize_record({"phases": {}}), rec) is None


# ---------------------------------------------------------------------------------------------------------------------
# 2. metric-definition skew: a changed formula is reported, not differenced

def test_metrics_version_skew_is_reported_not_graded():
    assert P.skewed_metrics({"metrics_version": "1.1"}, {"metrics_version": "1.1"}) == []
    assert "arrival_speed" in P.skewed_metrics({"metrics_version": "1.1"}, {})        # missing field == metrics 1.0
    inp = _inputs()
    inp["schedule"] = {k: dict(v, relatch="relatches", operate="opens") for k, v in inp["schedule"].items()}
    m = {"relatch_closed_angle": 0.0, "relatch_repush_angle": 0.001, "t_close": 0.6, "arrival_speed": 3.0, "opened_before": 1.6}
    mj = _rec({"relatch": _ph("pass", m, expected="relatches")}, metrics_version="1.1")
    px = _rec({"relatch": _ph("pass", dict(m, arrival_speed=0.02), expected="relatches")})
    v = P.compare_door(inp, mj, px, kind="usd_full")
    d = v["phases"]["relatch"]["deltas"]["arrival_speed"]
    assert d["ok"] is None and "definition changed" in d["not_comparable"]
    assert "METRICS_VERSION_SKEW" in v["codes"] and "METRIC_DELTA" not in v["codes"] and v["grade"] == "A"
    assert v["metrics_version"]["not_comparable"] == ["arrival_speed", "speed_at_latch"]


# ---------------------------------------------------------------------------------------------------------------------
# 3. arrival speed: the peak of the approach, not the sample at the impact

def _closing_curve(t0: float, hz: int, v_approach: float = 3.0) -> dict:
    """A leaf closing at ``v_approach`` and stopping dead at t = 1.0 s, sampled on a grid offset by ``t0``."""
    ts, qs, vs = [], [], []
    t = t0
    while t <= 1.4 + 1e-9:
        if t < 1.0:
            qs.append((1.0 - t) * v_approach)
            vs.append(-v_approach)
        else:
            qs.append(0.0)
            vs.append(0.0)
        ts.append(round(t, 6))
        t += 1.0 / hz
    return {"t": ts, "q": {"leaf_hinge": qs}, "v": {"leaf_hinge": vs}, "minmax": {}, "vmax": {"leaf_hinge": v_approach}, "finite": True, "warnings": []}


def test_arrival_speed_is_invariant_to_the_sample_grid():
    inp = _inputs()
    inp["schedule"] = {k: dict(v, relatch="relatches") for k, v in inp["schedule"].items()}
    old, new = [], []
    for t0 in (0.0, 0.004, 0.008, 0.012, 0.016, 0.02, 0.028):
        c = _closing_curve(t0, 30)
        m = P.phase_metrics(inp, "relatch", c)
        new.append(m["arrival_speed"])
        # metrics 1.0: |v| at the sample nearest the crossing - the crossing sample is already stopped
        k = min(range(len(c["t"])), key=lambda i: abs(c["t"][i] - m["t_close"]))
        old.append(abs(c["v"]["leaf_hinge"][k]))
    assert max(new) - min(new) < 1e-9 and new[0] == pytest.approx(3.0)      # the speed the leaf actually arrives with
    assert min(old) == pytest.approx(0.0) and max(old) == pytest.approx(3.0)  # the old definition is bimodal on the same trajectory
    # and it does not move when the *simulator's* step rate changes either (500 Hz MuJoCo vs 120 Hz PhysX, both sampled at 30 Hz)
    assert P.phase_metrics(inp, "relatch", _closing_curve(0.0, 30))["arrival_speed"] == pytest.approx(
        P.phase_metrics(inp, "relatch", _closing_curve(1.0 / 240, 30))["arrival_speed"])


def test_approach_speed_window_and_fallback():
    c = _closing_curve(0.0, 30)
    assert P.approach_speed(c, "leaf_hinge", None, 3.0 * DEG) is None
    assert P.approach_speed(c, "leaf_hinge", 1.0, 3.0 * DEG, window=0.1) == pytest.approx(3.0)
    # a crossing inside the first sample interval falls back to the last sample outside the closed band
    tiny = {"t": [0.0, 0.033], "q": {"leaf_hinge": [0.2, 0.0]}, "v": {"leaf_hinge": [-2.5, 0.0]}}
    assert P.approach_speed(tiny, "leaf_hinge", 0.033, 3.0 * DEG) == pytest.approx(2.5)


def test_speed_at_latch_uses_the_same_definition():
    inp = _inputs()
    inp["schedule"] = {k: dict(v, closer="closes") for k, v in inp["schedule"].items()}
    m = P.phase_metrics(inp, "closer", _closing_curve(0.017, 30))
    assert m["speed_at_latch"] == pytest.approx(3.0) and m["peak_closing_speed"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------------------------------------------------
# 4. the operate rebound: grade the peak, not the bounce

def test_opened_rebound_is_waived_and_the_peak_is_graded():
    inp = _inputs(rng=(0.0, 2.0944))
    # db0054_stall: MuJoCo bounces back to 0.916 rad after touching the 1.920 rad stop, PhysX stays parked on it
    mj = _rec({"operate": _ph("pass", {"opened": 0.9157, "q_primary_max": 1.9198, "primary_at_limit": True, "t_open": 1.4})})
    px = _rec({"operate": _ph("pass", {"opened": 1.9199, "q_primary_max": 1.9199, "primary_at_limit": True, "t_open": 1.4})})
    v = P.compare_door(inp, mj, px, kind="usd_full")
    d = v["phases"]["operate"]["deltas"]
    assert d["opened"]["ok"] is True and "joint limit" in d["opened"]["waived"] and d["q_primary_max"]["ok"] is True
    assert v["grade"] == "A"
    # a door that did NOT reach the stop in PhysX still disagrees, on both metrics
    short = _rec({"operate": _ph("pass", {"opened": 0.42, "q_primary_max": 0.55, "primary_at_limit": False, "t_open": 1.4})})
    v2 = P.compare_door(inp, mj, short, kind="usd_full")
    assert v2["grade"] == "B" and "METRIC_DELTA" in v2["codes"]
    assert v2["phases"]["operate"]["deltas"]["opened"]["ok"] is False and v2["phases"]["operate"]["deltas"]["q_primary_max"]["ok"] is False


def test_primary_at_limit_needs_a_range():
    inp = _inputs(rng=(0.0, 2.0944))
    assert P.primary_at_limit(inp, 2.0944) is True and P.primary_at_limit(inp, 2.08) is True and P.primary_at_limit(inp, 1.9) is False
    unlimited = _inputs()
    unlimited["joints"]["leaf_hinge"]["range"] = None
    assert P.primary_at_limit(unlimited, 99.0) is None


def test_results_path_waives_the_same_rebound():
    inputs = {"primary_joint": "leaf_hinge", "joints": {"leaf_hinge": {"type": "hinge", "range": [0.0, 1.9199]}},
              "thresholds": {"thr": 2.0 * DEG, "limit_tol": {"hinge": 2.0 * DEG, "slide": 0.01}, "v_cap_primary": 15.0}}
    mj = {"status": "pass", "metrics": {"opened": 0.9157, "q_primary_max": 1.9198}, "expected": "opens"}
    px = {"status": "pass", "metrics": {"opened": 1.9199, "q_primary_max": 1.9199}, "expected": "opens"}
    row = R.compare_phase("operate_open", mj, px, True, inputs)
    assert row["metric_deltas"]["opened"]["ok"] is True and row["within_tol"] is True
    px_short = {"status": "pass", "metrics": {"opened": 0.42, "q_primary_max": 0.55}, "expected": "opens"}
    assert R.compare_phase("operate_open", mj, px_short, True, inputs)["within_tol"] is False


# ---------------------------------------------------------------------------------------------------------------------
# 5. hold: a free door and a door with locked play are not judged by the latch slop

def test_hold_tolerance_follows_the_expectation():
    latched = P.metric_tolerances(_inputs(hold="hold"), "hold")
    free = P.metric_tolerances(_inputs(hold="free_opens"), "hold", "free_opens")
    assert latched["hold_displacement"] == (0.01, 0.0)
    assert free["hold_displacement"] == (0.1, 0.2)
    # a leaf authored with locked play may rest anywhere inside it
    play = P.metric_tolerances(_inputs(hold="hold", thr=2.0 * DEG + 0.25), "hold", "hold")
    assert play["hold_displacement"][0] == pytest.approx(0.01 + 0.25) and P.locked_play(_inputs(thr=2.0 * DEG + 0.25)) == pytest.approx(0.25)
    assert P.locked_play(_inputs()) == 0.0


def test_free_hold_overshoot_is_not_a_discrepancy():
    inp = _inputs(hold="free_opens")
    mj = _rec({"hold": _ph("pass", {"hold_displacement": 0.42, "q_at_1s": 0.21, "t_free": 0.9}, expected="free_opens")})
    px = _rec({"hold": _ph("pass", {"hold_displacement": 0.37, "q_at_1s": 0.19, "t_free": 0.95}, expected="free_opens")})
    assert P.compare_door(inp, mj, px, kind="usd_full")["grade"] == "A"
    far = _rec({"hold": _ph("pass", {"hold_displacement": 0.05, "q_at_1s": 0.02, "t_free": 3.4}, expected="free_opens")})
    v = P.compare_door(inp, mj, far, kind="usd_full")
    assert v["grade"] == "B" and "METRIC_DELTA" in v["codes"]        # a leaf that barely moves still disagrees


# ---------------------------------------------------------------------------------------------------------------------
# 6. relatch continues from operate: its timing is not like-for-like across different starting angles

def test_relatch_timing_not_compared_across_different_start_angles():
    inp = _inputs()
    inp["schedule"] = {k: dict(v, relatch="relatches") for k, v in inp["schedule"].items()}
    m = {"relatch_closed_angle": 0.0, "relatch_repush_angle": 0.001, "t_close": 0.47, "arrival_speed": 0.63, "opened_before": 0.972}
    mj = _rec({"relatch": _ph("pass", m, expected="relatches")})
    px = _rec({"relatch": _ph("pass", dict(m, t_close=1.0, arrival_speed=0.34, opened_before=1.920), expected="relatches")})
    v = P.compare_door(inp, mj, px, kind="usd_full")
    d = v["phases"]["relatch"]["deltas"]
    assert d["t_close"]["ok"] is None and "different angle" in d["t_close"]["not_comparable"] and d["arrival_speed"]["ok"] is None
    assert v["grade"] == "A" and "METRIC_DELTA" not in v["codes"]
    # same starting angle: the timing is compared again
    same = _rec({"relatch": _ph("pass", dict(m, t_close=1.07, arrival_speed=0.34), expected="relatches")})
    v2 = P.compare_door(inp, mj, same, kind="usd_full")
    assert v2["phases"]["relatch"]["deltas"]["t_close"]["ok"] is False and v2["grade"] == "B"


# ---------------------------------------------------------------------------------------------------------------------
# 7. the settle classifier reads the joint instead of assuming a spring

def _settle_phases(mj_drift: float, px_drift: float) -> dict:
    return {"settle": {"mujoco": "pass", "physx": "fail" if abs(px_drift) > 0.05 else "pass", "agree": False, "within_tol": False,
                       "expected": "settle", "metric_deltas": {}, "reason": None}}


def _mkrec(drift: float, joint: str, inputs: dict | None = None, **extra) -> dict:
    return R.normalize_record({"phases": {"settle": {"pass": abs(drift) < 0.05,
                                                     "metrics": {"settle_drift": abs(drift), "settle_drift_signed": {joint: drift}} | extra}},
                               **({"inputs": inputs} if inputs else {})})


SPRUNG = {"primary_joint": "leaf_hinge", "thresholds": {"v_cap_primary": 15.0},
          "joints": {"leaf_handle_hinge": {"type": "hinge", "stiffness": 0.35, "springref": -0.35, "frictionloss": 0.0, "range": [-0.35, 0.9]}}}
FREE_ROTOR = {"primary_joint": "rotor_hinge", "thresholds": {"v_cap_primary": 15.0},
              "joints": {"rotor_hinge": {"type": "hinge", "stiffness": 0.0, "springref": 0.0, "frictionloss": 4.17, "range": [-0.05, 0.05]}}}


def test_settle_preload_only_when_the_joint_has_a_spring():
    ctx = {"flags": {}, "is_hinge": True}
    mj = _mkrec(0.0, "leaf_handle_hinge", SPRUNG, max_v_primary=0.0, velocity_cap_hit=False)
    px = _mkrec(0.301, "leaf_handle_hinge", max_v_primary=0.4, velocity_cap_hit=False)
    codes = [c["code"] for c in R.classify(_settle_phases(0.0, 0.301), mj, px, ctx, "full", False)]
    assert codes == ["PHYSICS_PARAM_PRELOAD"]


def test_settle_velocity_explosion_is_not_called_a_preload():
    ctx = {"flags": {}, "is_hinge": True}
    mj = _mkrec(0.0, "rotor_hinge", FREE_ROTOR, max_v_primary=0.0, velocity_cap_hit=False)
    px = _mkrec(9.4036, "rotor_hinge", max_v_primary=110506.9, velocity_cap_hit=True)
    cls = R.classify(_settle_phases(0.0, 9.4036), mj, px, ctx, "full", False)
    assert [c["code"] for c in cls] == ["VELOCITY_EXPLOSION"] and "110507" in cls[0]["detail"].replace(",", "") or "1.105e+05" in cls[0]["detail"]
    # a spring-less joint held by friction that drifts without exploding is a friction problem, not a preload one
    px2 = _mkrec(0.4, "rotor_hinge", max_v_primary=0.9, velocity_cap_hit=False)
    assert [c["code"] for c in R.classify(_settle_phases(0.0, 0.4), mj, px2, ctx, "full", False)] == ["PHYSICS_PARAM_FRICTION"]


def test_settle_drift_beyond_twice_the_spring_target_is_not_a_preload():
    ctx = {"flags": {}, "is_hinge": True}
    mj = _mkrec(0.0, "leaf_handle_hinge", SPRUNG, max_v_primary=0.0, velocity_cap_hit=False)
    px = _mkrec(1.4, "leaf_handle_hinge", max_v_primary=2.0, velocity_cap_hit=False)   # 4x the 0.35 rad target: not a sag
    assert [c["code"] for c in R.classify(_settle_phases(0.0, 1.4), mj, px, ctx, "full", False)] == ["PHYSICS_PARAM"]


# ---------------------------------------------------------------------------------------------------------------------
# 8. the adaptive push is sized by the leaf

def test_push_base_scales_with_the_weight_moment():
    assert QA.push_base("hinge", 30.0, 0.9) == 60.0                       # a full-size door keeps the full authority
    assert QA.push_base("hinge", 0.895, 0.3) == 2.0                       # a pet flap gets the floor, not 60 N*m
    assert QA.push_base("hinge", 12.5013067296, 0.762) == pytest.approx(0.5 * 12.5013067296 * 9.81 * 0.762)
    assert QA.push_base("slide", 40.0) == 80.0 and QA.push_base("slide", 3.0) == pytest.approx(0.5 * 3.0 * 9.81)
    assert QA.push_base("hinge", None, None) == 60.0 and QA.push_base("hinge", 30.0, None) == 60.0   # unknown leaf: unchanged
    # the floor is high enough that the push always exceeds a small static load, the cap low enough to stay a human push
    assert QA.PUSH_BASE_MIN == 2.0 and QA.PUSH_BASE_MAX == {"hinge": 60.0, "slide": 80.0}


def test_push_base_keeps_twice_the_static_resistance():
    """The base only sets the *margin*: the push is still 2 x (gravity bias + friction + preload) on top of it."""
    for m, w, static in ((0.16, 0.15, 0.02), (12.5, 0.76, 0.18), (95.0, 1.1, 40.0)):
        push = min(2.0 * static + QA.push_base("hinge", m, w), QA.PUSH_CAP["hinge"])
        assert push > 2.0 * static


# ---------------------------------------------------------------------------------------------------------------------
# 9. metrics that measure the run, not the door, do not decide a grade

def test_bookkeeping_and_cross_phase_metrics_are_not_gating():
    for name in ("t_end", "n_samples", "max_v_primary", "max_v_any", "opened_before", "rebounds", "velocity_cap_hit", "settle_drift_other_max"):
        assert R.metric_delta(name, 1.0, 99.0, True) is None, name
    assert R.metric_delta("opened", 1.0, 99.0, True) is not None


def test_velocity_cap_is_compared_on_the_door_joint_only():
    """`velocity_cap_hit` also fires when *any* joint passes 50 rad/s, and the two files do not have the same joints."""
    inputs = {"primary_joint": "leaf_hinge", "joints": {"leaf_hinge": {"type": "hinge", "range": [0.0, 2.0]}},
              "thresholds": {"thr": 2.0 * DEG, "limit_tol": {"hinge": 2.0 * DEG, "slide": 0.01}, "v_cap_primary": 15.0}}
    # MuJoCo spins a closer-arm pinion the USD does not have: max_v_any explodes, the leaf itself agrees
    mj = {"status": "pass", "metrics": {"closer_final_angle": 0.01, "max_v_primary": 1.28, "max_v_any": 75.18, "velocity_cap_hit": True}, "expected": "closes"}
    px = {"status": "pass", "metrics": {"closer_final_angle": 0.012, "max_v_primary": 1.37, "max_v_any": 1.37, "velocity_cap_hit": False}, "expected": "closes"}
    row = R.compare_phase("closer_return", mj, px, True, inputs)
    assert "velocity_cap_hit" not in row["metric_deltas"] and "velocity_cap_hit_primary" not in row["metric_deltas"]
    assert row["within_tol"] is True
    # the leaf itself leaving the velocity range is still a disagreement
    px2 = {"status": "pass", "metrics": {"closer_final_angle": 0.012, "max_v_primary": 210.0, "max_v_any": 210.0, "velocity_cap_hit": True}, "expected": "closes"}
    row2 = R.compare_phase("closer_return", mj, px2, True, inputs)
    assert row2["metric_deltas"]["velocity_cap_hit_primary"]["ok"] is False and row2["within_tol"] is False


def test_settle_drift_is_compared_per_joint_over_the_joints_the_usd_has():
    inputs = {"primary_joint": "leaf_hinge", "thresholds": {"v_cap_primary": 15.0},
              "joints": {"leaf_hinge": {"type": "hinge", "range": [0, 2]}, "arm_pinion_hinge": {"type": "hinge", "range": None}}}
    mj = {"status": "pass", "metrics": {"settle_drift": 0.001, "settle_drift_signed": {"leaf_hinge": 0.001, "arm_pinion_hinge": 3.4}}, "expected": "settle"}
    px = {"status": "pass", "metrics": {"settle_drift": 0.001, "settle_drift_signed": {"leaf_hinge": 0.001}}, "expected": "settle"}
    row = R.compare_phase("settle", mj, px, True, inputs)
    assert "settle_drift_joint" not in row["metric_deltas"]          # the pinion is not in the USD and has no range: not compared
    px_bad = {"status": "pass", "metrics": {"settle_drift": 0.06, "settle_drift_signed": {"leaf_hinge": 0.06}}, "expected": "settle"}
    row2 = R.compare_phase("settle", mj, px_bad, True, inputs)
    assert row2["metric_deltas"]["settle_drift_joint"]["ok"] is False and "leaf_hinge" in row2["metric_deltas"]["settle_drift_joint"]["detail"]


# ---------------------------------------------------------------------------------------------------------------------
# 10. the report says which runs it compared, and how far apart the metrics are

def _write_results(d: str, px_hash: str = "aaaa", px_metrics_version: str | None = None) -> None:
    os.makedirs(d, exist_ok=True)
    def rec(opened, qmax, hash_, mv):
        r = {"inputs_hash": hash_, "phases": {"settle": {"pass": True, "metrics": {"settle_drift": 0.0}},
                                              "operate_open": {"pass": True, "expected": "opens", "metrics": {"opened": opened, "q_primary_max": qmax}}},
             "inputs": {"primary_joint": "leaf_hinge", "joints": {"leaf_hinge": {"type": "hinge", "range": [0.0, 2.0944]}},
                        "thresholds": {"thr": 2.0 * DEG, "limit_tol": {"hinge": 2.0 * DEG, "slide": 0.01}, "v_cap_primary": 15.0}}}
        if mv:
            r["metrics_version"] = mv
        return r
    mj = {f"db{i:04d}_swing_single": rec(1.6 + 0.001 * i, 1.6 + 0.001 * i, "aaaa", "1.1") for i in range(1, 21)}
    px = {k: rec(v["phases"]["operate_open"]["metrics"]["opened"] + 0.02, v["phases"]["operate_open"]["metrics"]["q_primary_max"] + 0.02, px_hash, px_metrics_version)
          for k, v in mj.items()}
    json.dump({"meta": {"engine": {"mujoco": "3.12.0"}, "protocol_version": "1.0", "metrics_version": "1.1", "generated": "2026-09-05T01:00:00",
                        "commit": "deadbeef", "dataset": {"n_doors": 1000, "version": "0.1.0"}}, "doors": mj}, open(os.path.join(d, "mujoco.json"), "w"))
    json.dump({"meta": {"engine": {"isaac_sim": None, "isaac_lab": "2.3.2"}, "protocol_version": "1.0", "generated": "2026-09-05T09:00:00"}, "doors": px},
              open(os.path.join(d, "isaac_full.json"), "w"))


def test_report_shows_provenance_versions_and_stale_counts(tmp_path):
    d = str(tmp_path / "res")
    _write_results(d)
    summary, md = build_report(d, None, str(tmp_path / "media"), plots=False)
    prov = summary["provenance"]
    assert prov["reference"]["engine"] == {"mujoco": "3.12.0"} and prov["reference"]["metrics_version"] == "1.1"
    assert prov["isaac"]["full"]["metrics_version"].startswith("1.0")
    assert "### Which runs this page compares" in md and "isaac_sim **not recorded**" in md and "isaac_lab `2.3.2`" in md
    assert "metrics whose *definition* changed" in md and summary["counts"]["full"]["metrics_skew"] == 20
    # a stale PhysX run: every door withheld, said so at the top, and never published
    d2 = str(tmp_path / "res2")
    _write_results(d2, px_hash="bbbb", px_metrics_version="1.1")
    summary2, md2 = build_report(d2, None, None, plots=False)
    assert summary2["counts"]["full"]["stale"] == 20 and summary2["counts"]["full"]["X"] == 20 and summary2["counts"]["full"]["compared"] == 0
    assert "20 doors stale" in md2 and "STALE_INPUTS" in md2
    assert all(R.manifest_status(v) == "untested" for v in summary2["doors"].values())


def test_report_metric_histograms_and_tolerance_derivations(tmp_path):
    d = str(tmp_path / "res")
    _write_results(d)
    media = str(tmp_path / "media")
    summary, md = build_report(d, None, media, plots=True)
    key = "full|operate_open|opened|hinge"          # hinge and slide bounds differ, so they are never pooled in one row
    assert summary["metric_stats"][key]["n"] == 20 and summary["metric_stats"][key]["median_abs"] == pytest.approx(0.02, abs=1e-6)
    assert summary["metric_stats"][key]["plot"] == "media/parity/hist_full_operate_open_opened_hinge.png"
    assert os.path.isfile(os.path.join(media, "hist_full_operate_open_opened_hinge.png"))
    assert "## Metric deltas" in md and "Delta histograms" in md
    # every tolerance that can decide a grade explains where it comes from
    assert "where the bound comes from" in md
    for k in R.TOLERANCES:
        assert k in R.TOLERANCE_NOTES, k
    assert plot_histogram(os.path.join(media, "empty.png"), "empty", [1.0], 0.1) is False


def test_report_lists_every_door_of_a_class(tmp_path):
    d = str(tmp_path / "res")
    _write_results(d, px_hash="bbbb", px_metrics_version="1.1")
    _summary, md = build_report(d, None, None, plots=False)
    assert "all 20 doors" in md and "`db0007_swing_single`" in md


# ---------------------------------------------------------------------------------------------------------------------
# 11. the runner can name the simulator it ran on

def test_package_version_resolves_or_says_it_cannot():
    import _common as C

    assert C.package_version("json") == json.__version__            # __version__ where there is one
    assert C.package_version("numpy") is not None                   # distribution metadata where there is not
    assert C.package_version("definitely_not_installed_xyz") is None
    eng = C.simulator_engine()
    assert set(eng) >= {"isaac_sim", "isaac_lab", "python", "platform"}
    assert eng["python"] and eng["platform"]
    # every value is either a version string or None - never a silent empty string the report would print as blank
    for k, v in eng.items():
        assert v is None or (isinstance(v, str) and v), (k, v)
