"""Isaac parity gate: result-file loading, per-door comparison and classification, the report renderer and the
qa.json / manifest merge, all on synthetic result files (no simulator, no dataset needed).

The synthetic doors mirror the first 40-door GPU probe: a door at parity, a spring-latch door whose bolt never
retracts in PhysX (EXPORT_COUPLING), a mag-lock door that swings open (EXPORT_WELD), a lever that sags at settle
(PHYSICS_PARAM_PRELOAD), a barn slider stuck on friction (PHYSICS_PARAM_FRICTION), a quantitative-only closer door that
agrees in a finer rerun (QUANT + SOLVER_SENSITIVITY), a USD that failed to spawn (LOAD_ERROR / grade X), an untested
door, and a canonical-USD-only disagreement (RL_CANON).
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

from doorbench.parity import results as R  # noqa: E402
from doorbench.parity.report import build_report, plot_curves, write_outputs  # noqa: E402
import merge_isaac_results as MERGE  # noqa: E402
import parity_report as REPORT_CLI  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# synthetic data

def _curve(q_end: float, T: float = 6.4, n: int = 40, delay: float = 1.2):
    return [[round(i * T / n, 3), round(0.0 if i * T / n < delay else q_end * min(1.0, (i * T / n - delay) / 3.0), 4)] for i in range(n + 1)]


def _rec(settle=0.0, hold=0.002, hold_pass=True, expected_hold="hold", opened=None, open_pass=None, op_travel=None, bolt=None, release=None, closer=None, curves=None,
         errors=(), is_hinge=True, settle_signed=None, settle_extra=None, inputs=None, inputs_hash=None, metrics_version=None, q_primary_max=None, at_limit=None):
    ph = {"P2 settle": {"pass": abs(settle) < 0.05, "metrics": {"settle_drift": settle} | ({"settle_drift_signed": settle_signed} if settle_signed else {}) | (settle_extra or {})},
          "P3 hold": {"pass": hold_pass, "expected": expected_hold, "metrics": {"hold_displacement": hold}},
          "P9 limits": {"pass": True, "metrics": {}}, "P10 sanity": {"pass": True, "metrics": {"max_v_primary": 1.2}}}
    if opened is not None:
        m = {"opened": opened, "operator_travel_reached": op_travel, "bolt_retract_max_frac": bolt}
        if q_primary_max is not None:
            m["q_primary_max"] = q_primary_max
        if at_limit is not None:
            m["primary_at_limit"] = at_limit
        ph["P4 operate_open"] = {"pass": open_pass, "metrics": m}
    if release is not None:
        ph["P5 release"] = {"pass": release < 0.006, "metrics": {"bolt_after_release_m": release}}
    if closer is not None:
        ph["P7 closer_return"] = {"pass": abs(closer) < math.radians(6), "metrics": {"closer_final_angle": closer}}
    rec = {"phases": ph, "curves": curves or {}, "metrics": {"is_hinge": is_hinge}, "errors": list(errors)}
    if inputs is not None:
        rec["inputs"] = inputs
    if inputs_hash is not None:
        rec["inputs_hash"] = inputs_hash
    if metrics_version is not None:
        rec["metrics_version"] = metrics_version
    return rec


def _inputs(primary: str, joints: dict, thr: float = math.radians(2.0), limit_tol: float | None = None) -> dict:
    """The slice of the protocol inputs the comparison needs (thresholds + joint parameters), as the MuJoCo runner writes it."""
    return {"primary_joint": primary, "joints": joints,
            "thresholds": {"thr": thr, "limit_tol": {"hinge": limit_tol if limit_tol is not None else math.radians(2.0), "slide": 0.01}, "v_cap_primary": 15.0}}


SPRUNG_LEVER = _inputs("leaf_hinge", {"leaf_hinge": {"type": "hinge", "stiffness": 0.0, "springref": 0.0, "frictionloss": 0.0, "range": [0.0, 1.92]},
                                      "leaf_handle_hinge": {"type": "hinge", "stiffness": 0.35, "springref": -0.35, "frictionloss": 0.0, "range": [-0.35, 0.9]}})
FREE_ROTOR = _inputs("rotor_hinge", {"rotor_hinge": {"type": "hinge", "stiffness": 0.0, "springref": 0.0, "frictionloss": 4.17, "range": [-0.05, 0.05]}}, thr=math.radians(2.0) + 0.05)


DOORS = {   # id -> (family, latch, lock, engaged, closer, operator, kinematics)
    "db0005_garage_tiltup": ("garage_tiltup", "none", "none", False, "none", "pull_d", "hinge_horizontal"),
    "db0002_swing_single": ("swing_single", "tubular_residential", "none", False, "none", "knob_round", "hinge_vertical"),
    "db0026_swing_single": ("swing_single", "none", "mag_lock", True, "none", "pull_d", "hinge_vertical"),
    "db0036_swing_single": ("swing_single", "tubular_residential_70", "none", False, "concealed_overhead", "lever_euro_backplate", "hinge_vertical"),
    "db0012_swing_single": ("swing_single", "none", "none", False, "norton_1600", "lever_straight", "hinge_vertical"),
    "db0033_gate_sliding": ("gate_sliding", "slide_bolt_heavy", "none", False, "none", "pull_d", "slide_horizontal"),
    "db0017_hatch_ceiling": ("hatch_ceiling", "none", "none", False, "none", "pull_d", "hinge_horizontal"),
    "db0029_sliding_single": ("sliding_single", "none", "none", False, "none", "pull_d", "slide_horizontal"),
    "db0019_swing_double": ("swing_double", "vertical_rods", "none", False, "lcn_4040", "panic_touchbar_svr", "hinge_vertical"),
    "db0187_turnstile_fullheight": ("turnstile_fullheight", "none", "mag_lock", True, "none", "turnstile_arm", "rotor"),
}


def make_results(out: str) -> None:
    mj, full, rl = {}, {}, {}
    mj["db0005_garage_tiltup"] = _rec(hold=0.40, expected_hold="free_opens", curves={"hatch_hinge": _curve(1.54)})
    full["db0005_garage_tiltup"] = _rec(hold=0.41, expected_hold="free_opens", curves={"door_hinge": _curve(1.53)})
    rl["db0005_garage_tiltup"] = _rec(hold=0.39, expected_hold="free_opens", curves={"door_hinge": _curve(1.52)})
    mj["db0002_swing_single"] = _rec(opened=1.69, open_pass=True, op_travel=0.87, bolt=1.0, release=0.0, curves={"leaf_hinge": _curve(1.69), "leaf_handle_hinge": _curve(0.87, delay=0.6)})
    full["db0002_swing_single"] = _rec(opened=0.002, open_pass=False, op_travel=0.86, bolt=0.0, release=0.0, curves={"P4 operate_open": {"door_hinge": _curve(0.002), "operator_hinge": _curve(0.86, delay=0.6)}})
    rl["db0002_swing_single"] = _rec(opened=0.002, open_pass=False, op_travel=0.86, bolt=0.0, release=0.0, curves={"door_hinge": _curve(0.002)})
    mj["db0026_swing_single"] = _rec(hold=1.2e-6, curves={"leaf_hinge": _curve(0.0)})
    full["db0026_swing_single"] = _rec(hold=1.571, hold_pass=False, curves={"door_hinge": _curve(1.571, delay=0.0)})
    rl["db0026_swing_single"] = _rec(hold=1.571, hold_pass=False, curves={"door_hinge": _curve(1.571, delay=0.0)})
    mj["db0036_swing_single"] = _rec(settle=0.0, opened=0.94, open_pass=True, op_travel=0.8, bolt=1.0, release=0.0, closer=0.01,
                                     settle_signed={"leaf_handle_hinge": 0.0}, inputs=SPRUNG_LEVER)
    full["db0036_swing_single"] = _rec(settle=0.301, opened=0.002, open_pass=False, op_travel=0.8, bolt=0.0, release=0.0, closer=0.012,
                                       settle_signed={"leaf_handle_hinge": 0.301})
    rl["db0036_swing_single"] = _rec(settle=0.0, opened=0.94, open_pass=True, op_travel=0.79, bolt=0.98, release=0.0, closer=0.011)
    mj["db0012_swing_single"] = _rec(hold=0.6, expected_hold="free_opens", closer=0.01, curves={"leaf_hinge": _curve(1.7)})
    full["db0012_swing_single"] = _rec(hold=0.9, expected_hold="free_opens", closer=0.012, curves={"door_hinge": _curve(1.7)})
    rl["db0012_swing_single"] = _rec(hold=0.61, expected_hold="free_opens", closer=0.011, curves={"door_hinge": _curve(1.7)})
    mj["db0033_gate_sliding"] = _rec(opened=3.6, open_pass=True, op_travel=0.08, bolt=1.0, is_hinge=False)
    full["db0033_gate_sliding"] = {"phases": {}, "errors": ["spawn: RuntimeError: Failed to load USD"], "metrics": {}, "curves": {}}
    mj["db0017_hatch_ceiling"] = _rec(hold=0.5, expected_hold="free_opens")
    # list-shaped phases, status strings, shorthand metrics, prismatic
    mj["db0029_sliding_single"] = {"phases": [{"name": "settle", "status": "pass", "settle_drift": 0.0}, {"name": "free_opens", "pass": True, "expected": "free_opens", "metrics": {"hold_displacement": 0.9, "t_free": 0.8}}],
                                   "metrics": {"is_hinge": False}, "curves": {"leaf_slide": _curve(0.9)}}
    full["db0029_sliding_single"] = {"phases": [{"name": "settle", "status": "pass", "settle_drift": 0.0}, {"name": "free_opens", "pass": False, "expected": "free_opens", "metrics": {"hold_displacement": 0.006, "t_free": None}}],
                                     "metrics": {"is_hinge": False}, "curves": {"door_slide": _curve(0.006)}}
    rl["db0029_sliding_single"] = full["db0029_sliding_single"]
    # full-height turnstile rotor: no spring anywhere, PhysX blows past the velocity cap and ends 9.4 rad away
    mj["db0187_turnstile_fullheight"] = _rec(settle=0.0, hold=0.001, settle_signed={"rotor_hinge": 0.0}, settle_extra={"max_v_primary": 0.0, "velocity_cap_hit": False},
                                             inputs=FREE_ROTOR, is_hinge=True)
    full["db0187_turnstile_fullheight"] = _rec(settle=9.4036, hold=0.001, settle_signed={"rotor_hinge": 9.4036},
                                               settle_extra={"max_v_primary": 110506.9, "velocity_cap_hit": True}, is_hinge=True)
    rl["db0187_turnstile_fullheight"] = _rec(settle=0.0, hold=0.001, settle_signed={"rotor_hinge": 0.0}, settle_extra={"max_v_primary": 0.0, "velocity_cap_hit": False}, is_hinge=True)
    # panic door: full agrees (both hold), rl disagrees (rl opened: welded exit device released) -> RL_CANON
    mj["db0019_swing_double"] = _rec(hold=0.002)
    full["db0019_swing_double"] = _rec(hold=0.0018)
    rl["db0019_swing_double"] = _rec(hold=0.9, hold_pass=False)
    os.makedirs(out, exist_ok=True)
    json.dump({"engine": {"mujoco": "3.12.0"}, "protocol": {"name": "parity_v1"}, "doors": mj}, open(os.path.join(out, "mujoco.json"), "w"))
    json.dump({"engine": {"isaac_sim": "5.1.0", "isaac_lab": "2.3.2"}, "doors": full}, open(os.path.join(out, "isaac_full.json"), "w"))
    json.dump(rl, open(os.path.join(out, "isaac_rl.json"), "w"))                      # bare {door_id: record}
    json.dump({"db0012_swing_single": _rec(hold=0.62, expected_hold="free_opens", closer=0.01)}, open(os.path.join(out, "isaac_full_dt240.json"), "w"))


def make_assets(out: str) -> None:
    """A tiny dataset: manifest + spec.json + qa.json per door (valid hardware catalogue ids so door_flags works)."""
    doors = []
    for i, (did, (fam, latch, lock, engaged, closer, op, kin)) in enumerate(DOORS.items()):
        ddir = os.path.join(out, "doors", did)
        os.makedirs(ddir, exist_ok=True)
        spec = {"id": did, "family": fam, "kinematics": {"type": kin, "max_open_deg": 90}, "latch": {"model": latch}, "lock": {"model": lock, "engaged": engaged, "robot_side_release": True},
                "closer": {"model": closer}, "operator": {"model": op, "sides": "both"}, "robot": {"robot_outside": False}, "condition": "new", "leaf": {"width": 0.9}}
        json.dump(spec, open(os.path.join(ddir, "spec.json"), "w"))
        json.dump({"checks": {"load_full": True, "settle": True, "hold": True, "actuate_opens": True}, "metrics": {"qa_push": 60.0}, "signed_off": True, "time_s": 0.1}, open(os.path.join(ddir, "qa.json"), "w"), indent=1)
        doors.append({"id": did, "index": i + 1, "family": fam, "latch": latch, "lock": lock, "lock_engaged": engaged, "closer": closer, "operator": op, "robot_side_release": True, "signed_off": True, "qa_failed": [], "thumbs": [], "files": {}})
    doors.append({"id": "db0999_swing_single", "index": 999, "family": "swing_single", "latch": "none", "lock": "none", "lock_engaged": False, "closer": "none", "operator": "pull_d", "robot_side_release": True, "signed_off": True, "qa_failed": [], "thumbs": [], "files": {}})
    json.dump({"name": "DoorBench", "version": "0.1.0", "generated": "2026-09-05", "n_doors": 1000, "n_signed_off": 10, "families": sorted({d["family"] for d in doors}), "doors": doors}, open(os.path.join(out, "manifest.json"), "w"))


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parity")
    res, assets = str(tmp / "results"), str(tmp / "assets")
    make_results(res)
    make_assets(assets)
    return {"tmp": str(tmp), "results": res, "assets": assets}


@pytest.fixture(scope="module")
def report(paths):
    summary, md = build_report(paths["results"], paths["assets"], os.path.join(paths["tmp"], "media"), top_n=20, plots=True)
    return summary, md


# ---------------------------------------------------------------------------------------------------------------------
# schema helpers

def test_canonical_phase_aliases():
    assert R.canonical_phase("P4 operate_open") == "operate_open"
    assert R.canonical_phase("operate") == "operate_open"
    assert R.canonical_phase("actuate_opens") == "operate_open"
    assert R.canonical_phase("free_opens") == "hold"
    assert R.canonical_phase("P3") == "hold"
    assert R.canonical_phase("closer_returns") == "closer_return"
    assert R.canonical_phase("latch_returns") == "release"
    assert R.canonical_phase("something_new") == "something_new"


def test_kind_from_filename():
    assert R.kind_from_filename("results/parity/isaac_full.json") == ("isaac", "full", "")
    assert R.kind_from_filename("results/parity/isaac_rl.json") == ("isaac", "rl", "")
    assert R.kind_from_filename("results/parity/isaac_usd_rl_dt240.json") == ("isaac", "rl", "dt240")
    assert R.kind_from_filename("results/parity/isaac_full_dt240.json") == ("isaac", "full", "dt240")
    assert R.kind_from_filename("results/parity/mujoco.json") == ("mujoco", None, "")
    assert R.kind_from_filename("results/parity/summary.json")[0] is None


def test_load_results_shapes(paths):
    mj, meta = R.load_results(os.path.join(paths["results"], "mujoco.json"))
    assert meta["engine"] == {"mujoco": "3.12.0"} and len(mj) == len(DOORS) - 0 or len(mj) >= 8
    rl, _ = R.load_results(os.path.join(paths["results"], "isaac_rl.json"))       # bare dict
    assert "db0002_swing_single" in rl
    rec = mj["db0029_sliding_single"]                                              # list-shaped phases + status strings + shorthand metric
    assert rec["phases"]["settle"]["status"] == "pass" and rec["phases"]["settle"]["metrics"]["settle_drift"] == 0.0
    assert rec["phases"]["hold"]["pass"] is True and rec["phases"]["hold"]["metrics"]["t_free"] == 0.8
    assert R.load_results(os.path.join(paths["results"], "does_not_exist.json")) == ({}, {})
    bad = R.normalize_record("garbage")
    assert bad["phases"] == {} and bad["errors"]
    assert R.normalize_record({"phases": {"hold": {"status": "skip", "reason": "na_env_logic"}}})["phases"]["hold"]["status"] == "skip"


def test_curves_helpers():
    nested = {"P4 operate_open": {"door_hinge": [[0, 0], [1, 1]]}, "P7 closer": {"door_hinge": [[0, 1], [1, 0]]}}
    assert R.pick_curve(nested, "primary", phase="closer_return") == [[0, 1], [1, 0]]
    assert R.pick_curve({"leaf_hinge": [[0, 0], [1, 2]], "x": [[0, 0], [1, 0.1]]}, "primary") == [[0, 0], [1, 2]]
    assert R.pick_curve({"weird": [[0, 0], [1, 5]], "small": [[0, 0], [1, 0.1]]}, "primary") == [[0, 0], [1, 5]]   # largest excursion
    assert R.pick_curve({"t": [0, 1], "q": [0, 1]}, "operator") is None
    assert R.curve_rmse([[0, 0], [1, 1], [2, 2]], [[0, 0], [2, 2]]) == pytest.approx(0.0)
    assert R.curve_rmse([[0, 0], [1, 1]], [[0, 1], [1, 2]]) == pytest.approx(1.0)
    assert R.curve_rmse([[0, 0]], [[0, 0], [1, 1]]) is None


def test_tolerances_latched_vs_free():
    latched = R.compare_phase("hold", {"status": "pass", "pass": True, "metrics": {"hold_displacement": 0.002}, "expected": "hold"}, {"status": "pass", "pass": True, "metrics": {"hold_displacement": 0.03}, "expected": "hold"}, True)
    assert latched["agree"] is True and latched["within_tol"] is False        # 28 mrad apart on a latched door is a real difference
    free = R.compare_phase("hold", {"status": "pass", "pass": True, "metrics": {"hold_displacement": 0.40}, "expected": "free_opens"}, {"status": "pass", "pass": True, "metrics": {"hold_displacement": 0.39}, "expected": "free_opens"}, True)
    assert free["within_tol"] is True                                          # 10 mrad on a freely opening door is noise
    assert R.metric_delta("opened", 1.0, 1.15, True)["ok"] is True             # 20 % relative
    assert R.metric_delta("opened", 0.2, 0.5, True)["ok"] is False
    assert R.metric_delta("qa_push", 60, 900, True) is None                    # non-gating
    assert R.metric_delta("x", None, 1.0, True) is None
    skip = R.compare_phase("hold", {"status": "skip", "pass": None, "metrics": {}}, {"status": "pass", "pass": True, "metrics": {}}, True)
    assert skip["agree"] is None


# ---------------------------------------------------------------------------------------------------------------------
# classification

def _verdicts(report):
    return report[0]["doors"]


def test_parity_door(report):
    v = _verdicts(report)["db0005_garage_tiltup"]
    assert v["grade"] == "A" and v["ok"] is True and v["classes"] == [] and v["primary_class"] == "OK"
    assert v["kinds"]["full"]["phases"]["hold"] == "agree" and v["kinds"]["rl"]["phases"]["hold"] == "agree"
    assert R.manifest_status(v) == "ok"


def test_coupling_bug(report):
    v = _verdicts(report)["db0002_swing_single"]
    assert v["grade"] == "C" and v["ok"] is False
    assert v["classes"] == ["EXPORT_COUPLING"]
    assert v["kinds"]["full"]["phases"]["operate_open"] == "disagree" and v["kinds"]["rl"]["phases"]["operate_open"] == "disagree"
    assert "H1" in v["likely_root_cause"]
    assert "operate_open.opened" in v["kinds"]["full"]["metrics"]


def test_weld_missing(report):
    v = _verdicts(report)["db0026_swing_single"]
    assert v["classes"] == ["EXPORT_WELD"] and v["grade"] == "C"
    assert v["hardware"]["lock"] == "mag_lock" and v["hardware"]["lock_engaged"] is True


def test_preload_and_coupling_full_only(report):
    v = _verdicts(report)["db0036_swing_single"]
    assert set(v["kinds"]["full"]["classes"]) == {"PHYSICS_PARAM_PRELOAD", "EXPORT_COUPLING"}
    assert v["kinds"]["rl"]["grade"] == "A" and "RL_CANON" not in v["classes"]     # rl agrees: the full kind is the odd one out
    assert v["grade"] == "C"


def test_friction_free_opens(report):
    v = _verdicts(report)["db0029_sliding_single"]
    assert v["classes"] == ["PHYSICS_PARAM_FRICTION"]
    assert v["kinds"]["full"]["phases"]["hold"] == "disagree"


def test_quant_and_solver_sensitivity(report):
    v = _verdicts(report)["db0012_swing_single"]
    assert v["kinds"]["full"]["grade"] == "B" and v["kinds"]["rl"]["grade"] == "A"
    assert v["ok"] is True and R.manifest_status(v) == "ok"
    assert "QUANT" in v["kinds"]["full"]["classes"] and "SOLVER_SENSITIVITY" in v["kinds"]["full"]["classes"]


def test_load_error_and_untested(report):
    v = _verdicts(report)["db0033_gate_sliding"]
    assert v["kinds"]["full"]["grade"] == "X" and v["kinds"]["full"]["classes"] == ["LOAD_ERROR"]
    assert v["kinds"]["rl"]["status"] == "untested"
    assert v["grade"] == "X" and R.manifest_status(v) == "fail"
    u = _verdicts(report)["db0017_hatch_ceiling"]
    assert u["status"] == "untested" and R.manifest_status(u) == "untested" and u["grade"] is None
    assert "db0999_swing_single" not in _verdicts(report)                         # a manifest door nobody ran is only counted


def test_rl_canon(report):
    v = _verdicts(report)["db0019_swing_double"]
    assert v["kinds"]["full"]["grade"] == "A" and v["kinds"]["rl"]["grade"] == "C"
    assert "RL_CANON" in v["kinds"]["rl"]["classes"]


def test_reference_qa_failure():
    ctx = {"flags": {}, "qa_checks": {"actuate_opens": True}, "is_hinge": True}
    mj = R.normalize_record({"phases": {"operate_open": {"pass": False, "metrics": {"opened": 0.0}}}})
    px = R.normalize_record({"phases": {"operate_open": {"pass": False, "metrics": {"opened": 0.0}}}})
    cmp = R.compare_kind(mj, px, ctx, "full")
    assert cmp["grade"] == "A" and any(c["code"] == "REFERENCE_QA_FAILURE" for c in cmp["classes"])


def test_operate_classes_by_hardware():
    mj = R.normalize_record({"phases": {"operate_open": {"pass": True, "metrics": {"opened": 1.5, "operator_travel_reached": 1.57}}}})
    px = R.normalize_record({"phases": {"operate_open": {"pass": False, "metrics": {"opened": 0.0, "operator_travel_reached": 1.57}}}})
    hatch = {"flags": {"spring_latch": False}, "latch_kind": "none", "lock_kind": "none", "lock_engaged": False, "is_hinge": True}
    assert [c["code"] for c in R.compare_kind(mj, px, hatch, "full")["classes"]] == ["PHYSICS_PARAM_FRICTION"]        # nothing holds it: push vs load
    hook = {"flags": {"spring_latch": False}, "latch_kind": "slide_bolt", "lock_kind": "none", "lock_engaged": False, "is_hinge": False}
    assert [c["code"] for c in R.compare_kind(mj, px, hook, "full")["classes"]] == ["CONTACT_GEOMETRY"]               # a bolt that does not clear its keeper
    px2 = R.normalize_record({"phases": {"operate_open": {"pass": False, "metrics": {"opened": 0.0, "operator_travel_reached": 0.05}}}})
    assert [c["code"] for c in R.compare_kind(mj, px2, hook, "full")["classes"]] == ["VALIDATOR_PROTOCOL"]            # operator barely moved: effort
    locked = {"flags": {"spring_latch": False}, "latch_kind": "none", "lock_kind": "deadbolt_single", "lock_engaged": True, "robot_side_release": False, "is_hinge": True}
    mj_l = R.normalize_record({"phases": {"operate_open": {"pass": False, "metrics": {"opened": 0.0}}}})
    px_l = R.normalize_record({"phases": {"operate_open": {"pass": True, "metrics": {"opened": 1.5}}}})
    assert [c["code"] for c in R.compare_kind(mj_l, px_l, locked, "full")["classes"]] == ["EXPORT_WELD"]              # a locked door PhysX opened


def test_no_reference():
    px = R.normalize_record({"phases": {"hold": {"pass": True}}})
    v = R.door_verdict("dbX", None, {"full": px, "rl": None}, {"flags": {}})
    assert v["status"] == "no_reference" and R.manifest_status(v) == "untested"


# ---------------------------------------------------------------------------------------------------------------------
# report

def test_counts_and_headline(report):
    summary, md = report
    c = summary["counts"]
    assert c["n_doors_total"] == 1000
    assert c["full"]["tested"] == 9 and c["rl"]["tested"] == 8
    assert c["full"]["A"] + c["full"]["B"] + c["full"]["C"] + c["full"]["X"] == c["full"]["tested"]
    assert c["full"]["untested"] == 1000 - 9
    assert c["door"]["ok"] + c["door"]["fail"] + c["door"]["untested"] == 1000
    assert "# Isaac parity gate" in md and "## Headline" in md
    assert f"| `full` | {c['full']['tested']} / 1000 | **{c['full']['A']} / 1000**" in md
    assert "## Discrepancy classes" in md and "`EXPORT_COUPLING`" in md and "`EXPORT_WELD`" in md
    assert "## By family" in md and "| swing_single |" in md
    assert "### latch kind" in md and "| tubular_latch |" in md
    assert "## Top offenders" in md and "db0002_swing_single" in md
    assert summary["by_class"]["EXPORT_COUPLING"]["n_doors"] == 2
    assert summary["by_family"]["swing_single"]["n_family"] == 5
    assert summary["protocol"]["names"]["mujoco"] == "parity_v1"


def test_plots_written(report, paths):
    summary, md = report
    v = summary["doors"]["db0002_swing_single"]
    assert v["plot"] == "media/parity/db0002_swing_single_operate_open.png"
    png = os.path.join(paths["tmp"], "media", "db0002_swing_single_operate_open.png")
    assert os.path.exists(png) and os.path.getsize(png) > 500
    assert "![db0002_swing_single operate_open](media/parity/db0002_swing_single_operate_open.png)" in md
    assert summary["doors"]["db0033_gate_sliding"]["plot"] is None              # no curves -> no plot, no crash
    assert plot_curves(os.path.join(paths["tmp"], "empty.png"), "x", [{"label": "a", "pts": []}]) is False


def test_write_outputs_and_cli(paths):
    tmp = paths["tmp"]
    docs, summ = os.path.join(tmp, "docs", "ISAAC_PARITY.md"), os.path.join(tmp, "results", "summary.json")
    rc = REPORT_CLI.main(["--results", paths["results"], "--assets", paths["assets"], "--docs", docs, "--media", os.path.join(tmp, "docs", "media", "parity"), "--top", "5"])
    assert rc == 0 and os.path.exists(docs) and os.path.exists(summ)
    s = json.load(open(summ))
    assert s["schema_version"] == "1" and len(s["top_offenders"]) == 5 and "doors" in s
    empty = os.path.join(tmp, "empty_results")
    os.makedirs(empty, exist_ok=True)
    summary, md = build_report(empty, None, None, plots=False)
    assert summary["counts"]["n_doors_total"] == 0 and "No door disagrees" in md
    write_outputs(summary, md, os.path.join(empty, "summary.json"), os.path.join(tmp, "empty.md"))
    assert os.path.exists(os.path.join(empty, "summary.json"))


# ---------------------------------------------------------------------------------------------------------------------
# merge (qa.json + manifest)

def test_merge_idempotent(paths):
    summ = os.path.join(paths["results"], "summary.json")
    assert os.path.exists(summ), "the CLI test writes it"
    assets = paths["assets"]
    qa_before = json.load(open(os.path.join(assets, "doors", "db0002_swing_single", "qa.json")))
    rc = MERGE.main(["--summary", summ, "--assets", assets])
    assert rc == 0
    qa = json.load(open(os.path.join(assets, "doors", "db0002_swing_single", "qa.json")))
    assert qa["signed_off"] == qa_before["signed_off"] and qa["checks"] == qa_before["checks"]
    ip = qa["isaac_parity"]
    assert ip["ok"] is False and ip["grade"] == "C" and ip["classes"] == ["EXPORT_COUPLING"] and ip["primary_class"] == "EXPORT_COUPLING"
    assert ip["kinds"]["full"]["phases"]["operate_open"] == "disagree" and ip["date"] and "commit" in ip
    assert ip["kinds"]["full"]["metrics"]["operate_open.opened"][0] == pytest.approx(1.69)
    ok = json.load(open(os.path.join(assets, "doors", "db0005_garage_tiltup", "qa.json")))["isaac_parity"]
    assert ok["ok"] is True and ok["grade"] == "A"
    un = json.load(open(os.path.join(assets, "doors", "db0017_hatch_ceiling", "qa.json")))["isaac_parity"]
    assert un["status"] == "untested" and un["ok"] is None
    man = json.load(open(os.path.join(assets, "manifest.json")))
    by = {d["id"]: d for d in man["doors"]}
    assert by["db0002_swing_single"]["isaac_parity"] == "fail" and by["db0002_swing_single"]["isaac_parity_grade"] == "C"
    assert by["db0005_garage_tiltup"]["isaac_parity"] == "ok"
    assert by["db0012_swing_single"]["isaac_parity"] == "ok"                     # grade B counts as ok
    assert by["db0033_gate_sliding"]["isaac_parity"] == "fail"                    # load error
    assert by["db0017_hatch_ceiling"]["isaac_parity"] == "untested" and by["db0999_swing_single"]["isaac_parity"] == "untested"
    # ok: db0005, db0012; fail: db0002, db0026, db0036, db0029, db0019 (rl only), db0033 (load error), db0187; untested: db0017, db0999
    assert man["isaac_parity"]["n_ok"] == 2 and man["isaac_parity"]["n_fail"] == 7 and man["isaac_parity"]["n_untested"] == 2
    assert man["n_signed_off"] == 10 and by["db0002_swing_single"]["signed_off"] is True
    # second run: nothing changes; --check agrees
    m_before = open(os.path.join(assets, "manifest.json")).read()
    q_before = open(os.path.join(assets, "doors", "db0002_swing_single", "qa.json")).read()
    stats = MERGE.merge(json.load(open(summ)), assets, verbose=False)
    assert stats["qa_written"] == 0 and stats["manifest_changed"] == 0
    assert open(os.path.join(assets, "manifest.json")).read() == m_before
    assert open(os.path.join(assets, "doors", "db0002_swing_single", "qa.json")).read() == q_before
    assert MERGE.main(["--summary", summ, "--assets", assets, "--check"]) == 0
    # a changed verdict is detected by --check and written by a plain run
    s = json.load(open(summ))
    s["doors"]["db0005_garage_tiltup"]["grade"] = "B"
    json.dump(s, open(summ, "w"))
    assert MERGE.main(["--summary", summ, "--assets", assets, "--check"]) == 1
    assert MERGE.main(["--summary", summ, "--assets", assets]) == 0
    assert json.load(open(os.path.join(assets, "doors", "db0005_garage_tiltup", "qa.json")))["isaac_parity"]["grade"] == "B"


def test_merge_recompute_without_summary(paths, tmp_path):
    assets2 = str(tmp_path / "assets")
    make_assets(assets2)
    rc = MERGE.main(["--summary", str(tmp_path / "missing.json"), "--results", paths["results"], "--assets", assets2])
    assert rc == 0
    assert json.load(open(os.path.join(assets2, "doors", "db0026_swing_single", "qa.json")))["isaac_parity"]["classes"] == ["EXPORT_WELD"]
    assert MERGE.main(["--summary", str(tmp_path / "missing.json"), "--assets", str(tmp_path / "nowhere")]) == 2
