"""The ``task_achievable`` and ``pair_swing`` gates (doorbench/task_qa.py).

  * synthetic doors: a leaf clamped below its scenario's pass threshold is caught, the same leaf held by a
    releasable weld instead is not, and a weld nothing can release fails only when the task needs the door to move
  * the shipped dataset: no releasable leaf keeps less than its declared travel, no locked leaf is asked to move,
    every door's qa.json carries the gate, and the gate is inside signed_off
  * the 24 doors the gate exists for: elevator landing doors, turnstiles, a sliding gate and a garage door keep
    their whole travel and are held by a constraint the environment or their own hardware releases
  * pair_swing: every double-egress pair swings one leaf each way, every other active pair both leaves the same way

Run:  pytest -q tests/test_task_achievable.py      (skipped when assets/ has not been generated)
"""
from __future__ import annotations

import json
import math
import os

import pytest

mujoco = pytest.importorskip("mujoco")

from doorbench import task_qa as T
from doorbench.benchmark.scenarios import assign_scenarios

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(ASSETS, "doors")), reason="assets not generated")


def _doors():
    return sorted(os.listdir(os.path.join(ASSETS, "doors")))


def _load(door_id):
    d = os.path.join(ASSETS, "doors", door_id)
    return (json.load(open(os.path.join(d, "spec.json"))), json.load(open(os.path.join(d, "model.json"))),
            json.load(open(os.path.join(d, "qa.json"))), d)


# ---------------------------------------------------------------------------
# synthetic doors
# ---------------------------------------------------------------------------
SYNTH = """
<mujoco model="synthetic">
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="4 4 0.1"/>
    <body name="jamb" pos="-0.62 0 1.0"><geom name="jamb_g" type="box" size="0.03 0.05 1.0"/></body>
    <body name="leaf" pos="-0.5 0 0">
      <joint name="leaf_hinge" type="hinge" axis="0 0 1" pos="0 0 0" range="{lo} {hi}" limited="true"/>
      <geom name="leaf_slab" type="box" pos="0.45 0 1.0" size="0.45 0.02 1.0" mass="30"/>
    </body>
  </worldbody>
  {equality}
</mujoco>
"""
WELD = '<equality><weld name="leaf_hold" body1="leaf" body2="world" active="true"/></equality>'


def _synthetic(tmp_path, hi_deg, weld_release=None, scenario="open_and_traverse", max_open_deg=90):
    """A one-leaf door with `hi_deg` of hinge range, optionally held by a weld with the given release kind."""
    xml = SYNTH.format(lo=0.0, hi=math.radians(hi_deg), equality=WELD if weld_release else "")
    path = tmp_path / "door.xml"
    path.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(path))
    spec = {"id": "synthetic", "family": "swing_single", "leaf": {"width": 0.9, "count": 1},
            "kinematics": {"type": "hinge_vertical", "max_open_deg": max_open_deg},
            "lock": {"model": "none", "engaged": bool(weld_release), "robot_side_release": True},
            "benchmark": {"scenarios": [{"name": scenario, "thresholds": {"clear_rad": math.radians(60), "open_rad": math.radians(30),
                                                                          "clear_m": None, "open_m": None}}]}}
    meta = {"primary_joint": "leaf_hinge"}
    if weld_release:
        meta["breakable_welds"] = [{"name": "leaf_hold", "body": "leaf", "joint": "leaf_hinge", "holding_force_N": 2670.0,
                                    "release": weld_release, "lock_model": "mag_lock", "holds_primary": True}]
    phys = {"mass": {"total_kg": 30.0}}
    return spec, meta, m, mujoco.MjData(m), phys


def test_synthetic_clamped_joint_fails(tmp_path):
    """The defect this gate exists for: a leaf whose whole range is 2.9 deg carrying `open_and_traverse`."""
    spec, meta, m, d, phys = _synthetic(tmp_path, hi_deg=2.9)
    r = T.run_task_achievable(spec, str(tmp_path), meta, m, d, phys, {"leaf_hinge": "primary"})
    assert not r["ok"]
    rules = {f["rule"] for f in r["failures"]}
    assert rules == {"travel", "reach"}
    assert "outside the primary joint's range" in [f["detail"] for f in r["failures"] if f["rule"] == "reach"][0]


def test_synthetic_full_range_passes(tmp_path):
    spec, meta, m, d, phys = _synthetic(tmp_path, hi_deg=90)
    assert T.run_task_achievable(spec, str(tmp_path), meta, m, d, phys, {"leaf_hinge": "primary"})["ok"]


def test_synthetic_releasable_weld_is_not_a_clamp(tmp_path):
    """The fix: the leaf keeps its range and a weld holds it.  The gate drops the weld and the task is possible."""
    spec, meta, m, d, phys = _synthetic(tmp_path, hi_deg=90, weld_release="env")
    r = T.run_task_achievable(spec, str(tmp_path), meta, m, d, phys, {"leaf_hinge": "primary"})
    assert r["ok"] and r["release_path"] == "env"
    assert r["scenarios"][0]["still_pinned"] == [] and r["force_leg"]["moved"] >= r["force_leg"]["threshold"]


def test_synthetic_unreleasable_weld_blocks_a_moving_task(tmp_path):
    """A weld nothing can release is a genuinely shut door: fine for `locked_recognize`, not for a task that moves."""
    spec, meta, m, d, phys = _synthetic(tmp_path, hi_deg=90, weld_release="none")
    r = T.run_task_achievable(spec, str(tmp_path), meta, m, d, phys, {"leaf_hinge": "primary"})
    assert not r["ok"] and r["failures"][0]["rule"] == "reach" and "still pins the leaf" in r["failures"][0]["detail"]
    spec["benchmark"]["scenarios"][0]["name"] = "locked_recognize"
    assert T.run_task_achievable(spec, str(tmp_path), meta, m, d, phys, {"leaf_hinge": "primary"})["ok"]


# ---------------------------------------------------------------------------
# the shipped dataset
# ---------------------------------------------------------------------------
def test_every_door_passes_the_gate_and_it_is_inside_signed_off():
    bad = []
    for i in _doors():
        qa = json.load(open(os.path.join(ASSETS, "doors", i, "qa.json")))
        if "task_achievable" not in qa["checks"] or not qa["checks"]["task_achievable"]:
            bad.append((i, qa["metrics"].get("task_achievable")))
        assert not qa["signed_off"] or qa["checks"]["task_achievable"], i
    assert not bad, bad[:5]


def test_no_releasable_leaf_is_clamped_below_its_declared_travel():
    """Rule 2 over the whole dataset: a joint range is static, so a leaf a release can free must keep its travel."""
    bad = []
    for i in _doors():
        spec, mj, _, _ = _load(i)
        meta = mj["meta"]
        pj = next((b["joint"] for b in mj["bodies"] if b.get("joint") and b["joint"]["name"] == meta.get("primary_joint")), None)
        if pj is None or not pj.get("range"):
            continue
        want, _ = T.declared_travel(spec, meta)
        span = pj["range"][1] - pj["range"][0]
        if T.lock_release_path(spec, meta) != "none" and want > 0 and span < 0.95 * want:
            bad.append((i, span, want))
    assert not bad, bad[:8]


def test_a_leaf_no_release_can_free_never_carries_a_task_that_moves():
    bad = []
    for i in _doors():
        spec, mj, _, _ = _load(i)
        if T.lock_release_path(spec, mj["meta"]) != "none":
            continue
        names = [s["name"] for s in (spec.get("benchmark") or {}).get("scenarios", [])]
        moving = [n for n in names if n in T.MOVING_SCENARIOS]
        if moving:
            bad.append((i, names))
    assert not bad, bad[:8]


def test_locked_doors_are_never_asked_to_start_open():
    """`close_only` hands the robot an OPEN door, which a lock holding the leaf shut cannot be started in."""
    for i in _doors():
        spec, _, _, _ = _load(i)
        names = [s["name"] for s in (spec.get("benchmark") or {}).get("scenarios", [])]
        assert not (spec["lock"].get("engaged") and "close_only" in names), i
        assert names == assign_scenarios(spec), i


def test_elevators_turnstiles_and_the_other_clamped_doors_keep_their_travel():
    """The 24 doors this work is about: full travel, and a holding constraint instead of a shortened range."""
    seen = {"elevator": 0, "turnstile": 0, "other": 0}
    for i in _doors():
        spec, mj, _, _ = _load(i)
        meta = mj["meta"]
        welds = [w for w in (meta.get("breakable_welds") or []) if w.get("holds_primary")]
        if not welds:
            continue
        pj = next((b["joint"] for b in mj["bodies"] if b.get("joint") and b["joint"]["name"] == meta["primary_joint"]), None)
        want, _ = T.declared_travel(spec, meta)
        rng = pj.get("range")
        assert rng is None or (rng[1] - rng[0]) >= 0.95 * want, (i, rng, want)
        for w in welds:
            assert w["release"] in ("env", "robot", "none") and w["holding_force_N"] > 0, (i, w)
            if w["release"] == "robot":
                assert w.get("release_joint"), (i, w)
        fam = spec["family"]
        seen["elevator" if fam == "elevator" else "turnstile" if fam.startswith("turnstile") else "other"] += 1
    assert seen["elevator"] == 8 and seen["turnstile"] == 13 and seen["other"] >= 3, seen


def test_locked_turnstile_rotor_is_unlimited_and_held_by_its_solenoid():
    ids = [i for i in _doors() if i.split("_", 1)[1].startswith("turnstile")]
    locked = 0
    for i in ids:
        spec, mj, _, _ = _load(i)
        if not spec["kinematics"].get("locked_until_credential"):
            continue
        locked += 1
        rotor = next(b["joint"] for b in mj["bodies"] if b.get("joint") and b["joint"]["name"] == "rotor_hinge")
        assert rotor["range"] is None, (i, rotor["range"])       # the whole 360 deg, not +-2.9 deg
        assert any(w["name"] == "rotor_solenoid_hold" and w["release"] == "env" for w in mj["meta"]["breakable_welds"]), i
    assert locked == 13, locked


def test_full_height_turnstile_draws_the_solenoid_that_holds_it():
    got = 0
    for i in _doors():
        spec, mj, _, _ = _load(i)
        if spec["family"] != "turnstile_fullheight" or not spec["kinematics"].get("locked_until_credential"):
            continue
        got += 1
        names = {g["name"] for b in mj["bodies"] for g in b["geoms"]}
        assert "rotor_solenoid_housing" in names and "rotor_solenoid_plunger_geom" in names, i
        assert sum(n.startswith("rotor_lock_dog_") for n in names) == spec["kinematics"]["wings"], i
        pl = next(b["joint"] for b in mj["bodies"] if b.get("joint") and b["joint"]["name"] == "rotor_solenoid_plunger_slide")
        assert pl["role"] == "lock" and pl["range"][1] > 0.006, i   # a real part the clearance sweep withdraws
    assert got == 6, got


def test_elevator_landing_doors_run_behind_the_wall():
    """The travel nothing had ever swept: a centre-opening pair used to drive 140 mm into its own jamb."""
    for i in _doors():
        spec, mj, _, _ = _load(i)
        if spec["family"] != "elevator":
            continue
        y = {b["name"]: b["pos"][1] for b in mj["bodies"]}
        wt = spec["opening"]["wall_thickness"]
        for leaf in ("leaf_a", "leaf_b"):
            if leaf in y:
                assert y[leaf] >= wt / 2, (i, leaf, y[leaf], wt)


# ---------------------------------------------------------------------------
# pair_swing
# ---------------------------------------------------------------------------
def test_double_egress_pairs_swing_one_leaf_each_way():
    de = same = 0
    for i in _doors():
        spec, _, qa, _ = _load(i)
        ps = qa["metrics"].get("pair_swing")
        if not ps or not ps.get("checked"):
            continue
        assert qa["checks"]["pair_swing"], (i, ps)
        if spec["kinematics"].get("double_egress"):
            de += 1
            assert ps["dy_leaf_a"] * ps["dy_leaf_b"] < 0, (i, ps)
        else:
            same += 1
            assert ps["dy_leaf_a"] * ps["dy_leaf_b"] > 0, (i, ps)
    assert de == 10 and same >= 30, (de, same)


def test_pair_swing_gate_catches_a_pair_built_the_wrong_way(tmp_path):
    """The gate measures where the leaf ENDS UP, so it fails a double-egress pair whose leaves both go one way."""
    xml = """
    <mujoco><compiler angle="radian"/><worldbody>
      <body name="leaf_a" pos="-0.5 0 0"><joint name="leaf_a_hinge" type="hinge" axis="0 0 1" range="0 1.57" limited="true"/>
        <geom name="a" type="box" pos="0.45 0 1" size="0.45 0.02 1" mass="30"/><site name="leaf_a_edge_mid" pos="0.9 0 1"/></body>
      <body name="leaf_b" pos="0.5 0 0"><joint name="leaf_b_hinge" type="hinge" axis="0 0 {sb}" range="0 1.57" limited="true"/>
        <geom name="b" type="box" pos="-0.45 0 1" size="0.45 0.02 1" mass="30"/><site name="leaf_b_edge_mid" pos="-0.9 0 1"/></body>
    </worldbody></mujoco>"""
    meta = {"pair": True, "primary_joint": "leaf_a_hinge", "secondary_joint": "leaf_b_hinge"}
    spec = {"context": "double_egress", "kinematics": {"double_egress": True}}
    for sb, ok in ((1, True), (-1, False)):        # +1 mirrors the leaves (opposite swing); -1 sends both one way
        p = tmp_path / f"pair{sb}.xml"
        p.write_text(xml.format(sb=sb))
        m = mujoco.MjModel.from_xml_path(str(p))
        r = T.run_pair_swing(spec, meta, m, mujoco.MjData(m))
        assert r["checked"] and r["ok"] is ok, (sb, r)
