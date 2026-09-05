"""Doors held shut by SEVERAL latches: every one of them must be modelled, driven and load-bearing.

Reported defect (db0168_ship_watertight, "it only does 1 of 6 hinges"): a watertight door carried six dog bodies with
six hinge joints, but ``meta`` named only the first as the operator, so QA, the benchmark and the viewer worked one dog
and the leaf swung with five still dogged.  Two more defects sat behind it - the dogs did not actually hold (a
hinge-stile wedge is 34 mm from the leaf hinge pin and had 3 mm of slop to its cleat, so 4 of the 6 individually dogged
doors swung 103-133 deg with a dog engaged), and the quick-acting variant drove 4 hard-coded dogs through nothing
visible.  The same "one of several" defect was in the cremone bolt and the surface vertical rod exit device, whose down
rods were drawn as cylinders and latched nothing.

Gates pinned here:
  meta                     operator_joints / operator_coupling on every door
  all_latches_release      each independent latch holds the leaf on its own; all of them released opens it
  rod_points_hold          a two-point rod mechanism's head bolt AND floor bolt each hold on their own
  quick-acting coupling    one handwheel joint-equality-drives every dog and both push rods

Run:  pytest -q tests/test_multi_latch.py      (~30 s)
"""
from __future__ import annotations

import json
import math
import os

import pytest

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "doors")
have_assets = os.path.isdir(ASSETS)
pytestmark = pytest.mark.skipif(not have_assets, reason="generated dataset not present")

DOG_LEVER = "db0168_ship_watertight"        # 6 individually dogged wedges
WHEEL = "db0744_ship_watertight"            # quick-acting: one handwheel drives every dog
BLAST = "db0288_blast"                      # 2 lever bolts, each throwing its own bolt into its own pocket
SVR = "db0025_swing_double"                 # surface vertical rod exit device: head latch + floor latch
CREMONE = "db0341_swing_double"             # cremone knob: shoot bolt up into the head, shoot bolt down into the floor


def _door(did):
    with open(os.path.join(ASSETS, did, "model.json")) as f:
        model = json.load(f)
    with open(os.path.join(ASSETS, did, "qa.json")) as f:
        qa = json.load(f)
    return model, qa


def _all_ids():
    return sorted(d for d in os.listdir(ASSETS) if os.path.isdir(os.path.join(ASSETS, d)))


# ---------------------------------------------------------------------------------------------------------------
# meta contract
# ---------------------------------------------------------------------------------------------------------------
def test_every_door_names_every_operator():
    """`operator_joints` lists real joints and always contains `operator_joint`; `operator_coupling` is one of two."""
    for did in _all_ids():
        model, _ = _door(did)
        meta = model["meta"]
        joints = {b["joint"]["name"] for b in model["bodies"] if b.get("joint")}
        ops = meta.get("operator_joints")
        assert isinstance(ops, list), did
        assert all(n in joints for n in ops), (did, ops)
        assert meta.get("operator_coupling") in ("individual", "coupled"), did
        if meta.get("operator_joint"):
            assert meta["operator_joint"] in ops, did
        else:
            assert ops == [], did
        if len(ops) < 2:
            assert meta["operator_coupling"] == "coupled", did


def test_individually_latched_doors_are_the_dog_and_lever_bolt_doors():
    """Only mechanisms whose latches are worked one at a time are marked "individual"."""
    ind = {}
    for did in _all_ids():
        model, _ = _door(did)
        if model["meta"].get("operator_coupling") == "individual":
            ind[did] = model["meta"]["operator_joints"]
    assert ind, "no individually latched doors found"
    for did, ops in ind.items():
        assert len(ops) >= 2, (did, ops)
        assert all(n.startswith("dog_") for n in ops), (did, ops)
        assert did.endswith(("_ship_watertight", "_vault", "_blast")), did
    assert ind[DOG_LEVER] == [f"dog_{k}_hinge" for k in range(6)]
    assert ind[BLAST] == ["dog_0_hinge", "dog_1_hinge"]


# ---------------------------------------------------------------------------------------------------------------
# the gates
# ---------------------------------------------------------------------------------------------------------------
def test_all_latches_release_gate_runs_and_passes_on_every_individually_latched_door():
    n = 0
    for did in _all_ids():
        model, qa = _door(did)
        if model["meta"].get("operator_coupling") != "individual":
            assert "all_latches_release" not in qa["checks"], did
            continue
        n += 1
        assert qa["checks"]["all_latches_release"] is True, did
        mt = qa["metrics"]
        partial, full = mt["all_latches_partial_displacement"], mt["all_latches_full_displacement"]
        assert len(partial) == len(model["meta"]["operator_joints"]), did
        thr_hold, target = mt["all_latches_thresholds"]
        # each latch on its own holds the leaf shut, and only releasing all of them opens it
        assert max(partial) < thr_hold, (did, partial)
        assert full > target, (did, full)
        assert full > 20 * max(partial), (did, partial, full)      # a real latch, not a lucky threshold
    assert n >= 13, n            # 6 watertight dog-lever + 5 blast + 2 vault today


def test_rod_points_hold_gate_covers_every_two_point_rod_mechanism():
    n = 0
    for did in _all_ids():
        model, qa = _door(did)
        joints = {b["joint"]["name"] for b in model["bodies"] if b.get("joint")}
        leaf = (model["meta"].get("primary_joint") or "").rsplit("_hinge", 1)[0]
        pairs = [(f"{leaf}_top_latch_slide", f"{leaf}_bottom_latch_slide"),
                 (f"{leaf}_cremone_top_bolt_slide", f"{leaf}_cremone_bottom_bolt_slide")]
        two_point = any(a in joints and b in joints for a, b in pairs)
        assert ("rod_points_hold" in qa["checks"]) == two_point, did
        if not two_point:
            continue
        n += 1
        assert qa["checks"]["rod_points_hold"] is True, did
        r = qa["metrics"]["rod_points"]
        assert max(r["top_only_rad"], r["bottom_only_rad"]) < math.radians(2.0), (did, r)
    assert n >= 34, n            # 31 surface-vertical-rod + 3 cremone doors today


def test_every_rod_mechanism_has_both_a_head_and_a_floor_bolt():
    """The down rod of a cremone and the bottom rod of an SVR device used to be drawn with nothing behind them."""
    seen = 0
    for did in _all_ids():
        model, _ = _door(did)
        joints = {b["joint"]["name"] for b in model["bodies"] if b.get("joint")}
        geoms = {g["name"] for b in model["bodies"] for g in b["geoms"]}
        leaf = (model["meta"].get("primary_joint") or "").rsplit("_hinge", 1)[0]
        if f"{leaf}_cremone_rod_down" in geoms:
            assert f"{leaf}_cremone_bottom_bolt_slide" in joints, did
            assert f"{leaf}_cremone_top_bolt_slide" in joints, did
            seen += 1
        if any(n.endswith("_rod_bot") for n in geoms) and f"{leaf}_top_latch_slide" in joints:
            assert f"{leaf}_bottom_latch_slide" in joints, did
            seen += 1
    assert seen >= 30, seen


# ---------------------------------------------------------------------------------------------------------------
# the quick-acting linkage
# ---------------------------------------------------------------------------------------------------------------
def test_quick_acting_wheel_drives_every_dog_and_both_push_rods():
    model, qa = _door(WHEEL)
    meta = model["meta"]
    assert meta["operator_coupling"] == "coupled" and meta["operator_joints"] == ["wheel_hinge"]
    driven = {e["a"]: e for e in model["equalities"] if e["kind"] == "joint" and e["b"] == "wheel_hinge"}
    dogs = [b["joint"]["name"] for b in model["bodies"] if b.get("joint") and b["joint"]["name"].startswith("dog_")]
    assert len(dogs) >= 4
    assert set(dogs) <= set(driven), set(dogs) - set(driven)
    assert {"linkage_rod_r_slide", "linkage_rod_l_slide"} <= set(driven), sorted(driven)
    # the linkage is visible geometry, not just a constraint
    geoms = {g["name"] for b in model["bodies"] for g in b["geoms"]}
    assert {"wheel_gearbox", "linkage_tube", "linkage_rod_r_geom", "linkage_rod_l_geom"} <= geoms
    assert all(f"dog_{k}_crank" in geoms for k in range(len(dogs)))
    assert qa["signed_off"]


def test_quick_acting_dogs_are_not_hand_worked_and_lever_dogs_are():
    wheel, _ = _door(WHEEL)
    lever, _ = _door(DOG_LEVER)
    for b in wheel["bodies"]:
        if b.get("joint") and b["joint"]["name"].startswith("dog_"):
            assert b["joint"]["robot_interactive"] is False        # the wheel works them, not the robot
    grips = {s["name"] for b in lever["bodies"] for s in b["sites"]}
    for k in range(6):
        j = next(b["joint"] for b in lever["bodies"] if b["name"] == f"dog_{k}")
        assert j["robot_interactive"] is True                      # each lever is a thing the robot turns
        assert f"dog_{k}_grip" in grips


def test_every_dog_has_its_own_cleat_slot_on_the_frame():
    """A dog with no keeper is a decoration; the cleat is a slot (inner + outer jaw) so the wedge cannot lift out."""
    for did in _all_ids():
        model, _ = _door(did)
        if not did.endswith("_ship_watertight"):
            continue
        world = next(b for b in model["bodies"] if b["name"] == "world_env")
        geoms = {g["name"] for g in world["geoms"]}
        dogs = [b for b in model["bodies"] if b.get("joint") and b["joint"]["name"].startswith("dog_")]
        assert dogs, did
        for k in range(len(dogs)):
            for part in ("", "_outer", "_base", "_bridge"):
                assert f"cleat_{k}{part}" in geoms, (did, k, part)


# ---------------------------------------------------------------------------------------------------------------
# behaviour in MuJoCo (the gate metrics are recomputed here from the shipped MJCF)
# ---------------------------------------------------------------------------------------------------------------
def test_watertight_leaf_holds_with_any_single_dog_still_dogged():
    import mujoco

    from doorbench.qa import drive_operators

    d_dir = os.path.join(ASSETS, DOG_LEVER)
    model, qa = _door(DOG_LEVER)
    m = mujoco.MjModel.from_xml_path(os.path.join(d_dir, "door.xml"))
    data = mujoco.MjData(m)
    pj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "leaf_hinge")
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in model["meta"]["operator_joints"]]
    push = qa["metrics"]["qa_push"]
    for keep in range(len(ids)):
        q = drive_operators(m, data, pj, [j for i, j in enumerate(ids) if i != keep], [], -1, push, True)
        assert q < math.radians(2.0), (keep, math.degrees(q))
    assert drive_operators(m, data, pj, ids, [], -1, push, True) > math.radians(20.0)


def test_quick_acting_wheel_retracts_every_dog_together():
    import mujoco

    m = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, WHEEL, "door.xml"))
    d = mujoco.MjData(m)
    wj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "wheel_hinge")
    dogs = [j for j in range(m.njnt) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("dog_")]
    rods = [j for j in range(m.njnt) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("linkage_rod_")]
    mujoco.mj_resetData(m, d)
    for _ in range(4000):
        d.qfrc_applied[:] = 0
        d.qfrc_applied[m.jnt_dofadr[wj]] = 12.0
        mujoco.mj_step(m, d)
    assert d.qpos[m.jnt_qposadr[wj]] > 0.9 * m.jnt_range[wj][1]
    for j in dogs:
        assert d.qpos[m.jnt_qposadr[j]] > 0.9 * m.jnt_range[j][1], mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    for j in rods:
        assert d.qpos[m.jnt_qposadr[j]] > 0.9 * m.jnt_range[j][1], mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)


def test_benchmark_release_needs_every_dog():
    """The benchmark treats a watertight door as unlatched / unlocked only once ALL of its dogs are turned.

    `latch_released` used to be True on step 0 of every watertight door: the tracker looks for role-"latch" joints and
    a dogged door has none (its dogs carry role "lock"), so "no latch parts" read as "latch already released".
    """
    from doorbench.benchmark.env import DoorEnv

    env = DoorEnv(os.path.join(ASSETS, DOG_LEVER))
    env.reset(env.core_scenarios[0])
    ops = env.meta["operator_joints"]
    assert len(ops) == 6 and set(ops) <= set(env.operator_joints)
    assert len(env.tracker.latch_joints) == 6, "the dogs ARE this door's latch"
    assert env.ojs and len(env.ojs) == 6
    assert env.tracker.L.latch_released is False
    for n in ops[:-1]:                       # five of six dogs
        for _ in range(400):
            env.apply_joint_torque(n, 20.0)
            env.step()
    assert env.tracker.L.latch_released is False, "latched released with a dog still dogged"
    assert env.tracker.L.lock_released is False
    assert "unlock" not in env._fired
    for _ in range(600):                     # and now the sixth
        for n in ops:
            env.apply_joint_torque(n, 20.0)
        env.step()
    assert env.tracker.L.latch_released is True
    assert env.tracker.L.lock_released is True
    assert "unlock" in env._fired
    env.close()
