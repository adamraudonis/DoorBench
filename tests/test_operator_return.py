"""Operator release behaviour: sprung handles snap back, unsprung ones do not.

The owner's question was "if you stop twisting a handle, does it snap back with a spring like motion?".  These tests
pin the three answers the hardware catalogue gives (spring / gravity / detent), the derivation that turns them into
joint parameters, and the QA gate that measures the result in MuJoCo.
"""
from __future__ import annotations

import math
import os

import pytest

from doorbench import hardware as H
from doorbench import physics as P
from doorbench import qa as QA
from doorbench.build import build_model, export_door
from doorbench.spec import generate_all

mujoco = pytest.importorskip("mujoco")


ROTARY_HANDLES = ("lever", "knob", "keypad_lever", "card_lever", "keypad_deadbolt")


@pytest.fixture(scope="module")
def specs():
    return list(generate_all())


def _spec(specs, **want):
    for s in specs:
        if all(_get(s, k) == v for k, v in want.items()):
            return s
    pytest.skip(f"no door matching {want}")


def _get(s, dotted):
    cur = s
    for part in dotted.split("."):
        cur = cur[part]
    return cur


# --------------------------------------------------------------------------- catalogue
def test_every_operator_declares_a_return_kind():
    for op in H.OPERATORS.values():
        assert op.return_kind in ("spring", "gravity", "detent", "none"), op.id


def test_sprung_operators_have_a_spring_and_unsprung_ones_do_not():
    for op in H.OPERATORS.values():
        if op.return_kind == "spring":
            assert op.spring_rate > 0 and op.spring_torque_preload > 0, f"{op.id} claims a return spring but has none"
            assert op.detent_friction == 0.0, op.id
        else:
            assert op.spring_torque_preload == 0.0 and op.spring_rate == 0.0, \
                f"{op.id} is {op.return_kind}, it must not carry a return spring"
        if op.return_kind == "detent":
            assert op.detent_friction > 0, f"{op.id} stays where it is put, so something must hold it"
            assert op.return_note, f"{op.id} must say why it has no return spring"


def test_the_hardware_that_stays_where_it_is_put():
    """The owner's list: thumbturns, deadbolt turns, handwheels, dog levers, cremone handles must NOT spring back."""
    for oid in ("wheel_vault", "wheel_ship_hatch", "dog_lever", "cremone_bolt",
                "slide_bolt_barrel", "slide_bolt_heavy", "cane_bolt_drop", "stall_slide_latch", "hasp_padlock"):
        op = H.OPERATORS[oid]
        assert op.return_kind == "detent", oid
        assert op.hold_friction == op.detent_friction > 0, oid


def test_levers_sit_in_the_ansi_return_band():
    """ANSI/BHMA A156.2: a lever returns to horizontal; Grade 1 sets are roughly 0.5-1.5 N*m of preload."""
    for op in H.OPERATORS.values():
        if op.kind in ("lever", "keypad_lever", "card_lever") and op.return_kind == "spring" and op.motion == "rotate_normal":
            assert 0.3 <= op.spring_torque_preload <= 1.5, (op.id, op.spring_torque_preload)
            # and the whole travel stays inside the code force limit at the grip
            tau_full = op.spring_torque_preload + op.spring_rate * op.travel
            assert tau_full <= op.operable_force_limit * op.grip_offset + 1e-9, (op.id, tau_full)


# --------------------------------------------------------------------------- derivation
def test_operator_dynamics_block_has_units_and_kind(specs):
    s = _spec(specs, **{"operator.model": "lever_straight"})
    phys = P.derive(s)
    blk = phys["operator"]
    assert blk["return_kind"] == "spring"
    assert blk["units"]["preload"] == "N*m" and blk["units"]["inertia"] == "kg*m^2"
    assert blk["spring_rate"] == H.OPERATORS["lever_straight"].spring_rate
    assert blk["return_note"] and blk["formula"]


def test_build_records_per_joint_dynamics_and_critical_damping(specs):
    s = _spec(specs, **{"operator.model": "lever_straight", "family": "swing_single"})
    phys = P.derive(s)
    build_model(s, phys)
    joints = phys["operator"]["joints"]
    assert joints, "no operator joint recorded"
    rec = next(r for r in joints.values() if r["return_kind"] == "spring")
    for key in ("inertia", "damping", "frictionloss", "spring_preload", "spring_rate",
                "expected_return_time_s", "gravity_moment_at_rest", "rest", "units"):
        assert key in rec, key
    # b = 2 zeta sqrt(k I) at zeta = 1
    assert rec["damping"] == pytest.approx(2.0 * P.OPERATOR_DAMPING_RATIO * math.sqrt(rec["spring_rate"] * rec["inertia"]), rel=1e-6)
    # the spring beats the handle's own weight with margin, and friction cannot park it short of rest
    assert rec["spring_preload"] >= P.OPERATOR_GRAVITY_MARGIN * max(rec["gravity_moment_at_rest"], 0.0) - 1e-9
    assert rec["frictionloss"] <= P.OPERATOR_FRICTION_FRACTION * rec["spring_preload"] + 1e-9
    assert 0.15 <= rec["expected_return_time_s"] <= 0.5, rec["expected_return_time_s"]


def test_a_heavy_handle_gets_a_spring_that_can_lift_it(specs):
    """The Kason cold-storage handle weighs 2.5 kg and hangs 1.35 N*m off its spindle."""
    s = _spec(specs, **{"operator.model": "cold_storage_handle"})
    phys = P.derive(s)
    build_model(s, phys)
    rec = next(iter(phys["operator"]["joints"].values()))
    assert rec["gravity_moment_at_rest"] > 1.0
    assert rec["spring_preload"] > rec["gravity_moment_at_rest"]


def test_detent_operators_get_friction_and_no_spring(specs):
    s = _spec(specs, **{"operator.model": "wheel_vault"})
    phys = P.derive(s)
    model = build_model(s, phys)
    j = next(b.joint for b in model.bodies if b.joint and b.joint.role == "operator")
    assert j.return_kind == "detent"
    assert j.stiffness == 0.0 and j.springref == 0.0
    assert j.frictionloss >= H.OPERATORS["wheel_vault"].detent_friction


def test_return_time_integration_is_monotonic_in_stiffness():
    kw = dict(I=0.012, preload=0.5, fl=0.02, grav=0.0, q0=0.9, tol=0.009)
    slow = P.operator_return_time(k=0.6, b=2 * math.sqrt(0.6 * 0.012), **kw)
    fast = P.operator_return_time(k=2.4, b=2 * math.sqrt(2.4 * 0.012), **kw)
    assert slow > fast > 0


def test_return_time_reports_a_handle_that_never_comes_home():
    # gravity moment larger than the spring: it stalls part-way and the estimate says so
    t = P.operator_return_time(I=0.03, k=2.5, preload=0.6, b=0.5, fl=0.05, grav=1.4, q0=0.5, tol=0.005)
    assert t is None


# --------------------------------------------------------------------------- the gate
@pytest.fixture(scope="module")
def lever_door(tmp_path_factory, specs):
    s = _spec(specs, **{"operator.model": "lever_straight", "family": "swing_single"})
    tmp = tmp_path_factory.mktemp("opret")
    export_door(s, str(tmp / "doors"), str(tmp / "hardware"), formats=("mjcf", "json"))
    return s, str(tmp / "doors" / s["id"])


def test_gate_measures_a_damped_return_on_a_real_lever(lever_door):
    import json
    s, door_dir = lever_door
    phys = json.load(open(os.path.join(door_dir, "spec.json")))["physics"]
    meta = json.load(open(os.path.join(door_dir, "model.json")))["meta"]
    m = mujoco.MjModel.from_xml_path(os.path.join(door_dir, "door.xml"))
    d = mujoco.MjData(m)
    pj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, meta["primary_joint"])
    res = QA.operator_release_checks(m, d, phys, pj, phys["mass"]["total_kg"], s["leaf"]["width"])
    assert res["checks"]["operator_returns"] is True
    rec = res["metrics"][meta["operator_joint"]]
    for tag in ("closed", "open"):
        t = rec[tag]
        assert t["t_return_s"] is not None and t["t_return_s"] <= 0.5
        assert abs(t["residual"]) < t["tolerance"], "the handle must not sit off its rest stop"
        assert t["bounces"] <= QA.OPERATOR_RETURN_MAX_BOUNCES, "no chatter against the rest stop"
        assert t["driven_to"] >= 0.9 * t["travel"], "the trial must actually reach full travel"


def test_gate_fails_a_handle_whose_spring_cannot_lift_it(lever_door):
    """Weaken the return spring below the handle's own weight moment: the gate must catch the residual offset."""
    s, door_dir = lever_door
    import json
    phys = json.load(open(os.path.join(door_dir, "spec.json")))["physics"]
    meta = json.load(open(os.path.join(door_dir, "model.json")))["meta"]
    m = mujoco.MjModel.from_xml_path(os.path.join(door_dir, "door.xml"))
    d = mujoco.MjData(m)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, meta["operator_joint"])
    pj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, meta["primary_joint"])
    m.jnt_stiffness[j] = 0.02                       # a dead spring
    m.qpos_spring[m.jnt_qposadr[j]] = -0.5
    m.dof_frictionloss[m.jnt_dofadr[j]] = 0.6       # gummed up
    res = QA.operator_release_checks(m, d, phys, pj, phys["mass"]["total_kg"], s["leaf"]["width"])
    assert res["checks"]["operator_returns"] is False


def test_gate_fails_an_undamped_handle_that_chatters(lever_door):
    s, door_dir = lever_door
    import json
    phys = json.load(open(os.path.join(door_dir, "spec.json")))["physics"]
    meta = json.load(open(os.path.join(door_dir, "model.json")))["meta"]
    m = mujoco.MjModel.from_xml_path(os.path.join(door_dir, "door.xml"))
    d = mujoco.MjData(m)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, meta["operator_joint"])
    pj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, meta["primary_joint"])
    m.jnt_stiffness[j] = 30.0                       # stiff and undamped: it rings off its stop
    m.qpos_spring[m.jnt_qposadr[j]] = -0.02         # equilibrium just past the stop, so it arrives fast and bounces
    m.dof_damping[m.jnt_dofadr[j]] = 0.0
    m.dof_frictionloss[m.jnt_dofadr[j]] = 0.0
    res = QA.operator_release_checks(m, d, phys, pj, phys["mass"]["total_kg"], s["leaf"]["width"])
    rec = res["metrics"][meta["operator_joint"]]["closed"]
    # it does come home, but it springs back off its stop by several degrees on the way
    assert rec["t_return_s"] is not None
    assert rec["rebound"] > max(QA.OPERATOR_REBOUND_TOL_FACTOR * rec["tolerance"],
                                QA.OPERATOR_REBOUND_FRACTION * rec["travel"])
    assert res["checks"]["operator_returns"] is False


def test_gate_is_present_on_the_shipped_dataset():
    """Every signed-off door with a sprung or detent operator carries the corresponding check."""
    import glob
    import json
    root = os.path.join(os.path.dirname(__file__), "..", "assets", "doors")
    paths = sorted(glob.glob(os.path.join(root, "*", "qa.json")))
    if not paths:
        pytest.skip("dataset not generated")
    n_ret = n_hold = 0
    for p in paths:
        q = json.load(open(p))
        rel = q["metrics"].get("operator_release") or {}
        kinds = {r.get("return_kind") for r in rel.values() if not r.get("note")}
        if kinds & {"spring", "gravity"}:
            assert "operator_returns" in q["checks"], p
            n_ret += 1
        if "detent" in kinds:
            assert "operator_holds" in q["checks"], p
            n_hold += 1
        assert q["checks"].get("operator_returns", True), p
        assert q["checks"].get("operator_holds", True), p
    assert n_ret > 500 and n_hold > 5, (n_ret, n_hold)
