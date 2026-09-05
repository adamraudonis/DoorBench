"""USD export fixes for the Isaac parity gate, checked on freshly exported representative doors (~20 s, no
generated assets/ needed).  One door per affected class, both files (door.usda + door_rl.usda):

  env release      mag lock / delayed egress / electric bolt / interlock: the MJCF ``<weld>`` leaf -> world becomes a
                   real breakable ``UsdPhysics.FixedJoint`` base -> leaf with ``physics:excludeFromArticulation``
                   (PhysX loop joint - the articulation tree keeps ONE parent per link), ``breakForce`` ==
                   ``breakTorque`` == the latch model's holding force and ``physics:jointEnabled`` for the
                   environment to clear.  Before: JSON metadata only, and a mag-locked leaf swung 1.14-1.78 rad open
                   in PhysX while MuJoCo held it at 1e-6.
  rl weld state    door_rl.usda welds every mechanism part that has no canonical slot; parts the operator retracts
                   (revolute hooks, cremone shoot bolts, wheel-driven dogs) are welded RELEASED, an engaged lock with
                   no coupling stays welded ENGAGED, and every decision is recorded in ``doorbench:rl`` so
                   ``doorbench.parity.protocol`` reads ground truth instead of guessing from the spec.
  self-collision   the articulation root enables self-collision (PhysX then skips joint-adjacent links only, which is
                   MuJoCo's parent/child default) and every pair MuJoCo suppresses is authored as
                   ``PhysxFilteredPairsAPI``; a lift pin / drop bolt / swing-pair latch therefore holds in PhysX too.
  couplings        ``PhysxMimicJointAPI`` only on rotational -> rotational equalities (PhysX articulation mimic
                   joints support rotational axes only and drop the rest silently); hinge -> slide and slide -> slide
                   carry ``doorbench:coupling_*`` emulation data plus the reflected inertia on the driver.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdPhysics  # noqa: E402

from doorbench import hardware as H  # noqa: E402
from doorbench.build import export_door  # noqa: E402
from doorbench.parity import protocol as P  # noqa: E402
from doorbench.spec import generate_all  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "isaaclab"))
from validate_usd_static import mj_filtered_pairs, validate_door  # noqa: E402

DEG = 180.0 / math.pi

# one door per affected class, picked by what the spec says (not by id: the generator may renumber)
PICKS = {
    "maglock": lambda s: s["lock"]["model"] in ("mag_lock", "delayed_egress") and s["lock"].get("engaged"),
    "hook_slider": lambda s: s["latch"]["model"] == "hook_slider" and s["family"] == "sliding_single" and s["lock"].get("robot_side_release", True),
    "cremone": lambda s: s["operator"]["model"] == "cremone_bolt" and s["lock"].get("robot_side_release", True),
    "wheel_dogs": lambda s: s["operator"]["model"] == "wheel_ship_hatch" and s["lock"]["model"] == "dogs",
    "welded_lock": lambda s: s["family"] == "swing_single" and s["lock"]["model"] == "slide_bolt" and s["lock"].get("engaged"),
    "swing_pair": lambda s: s["family"] == "swing_double" and s["latch"]["model"].startswith("tubular"),
    "baby_gate": lambda s: s["family"] == "baby_gate",
    "gate_swing": lambda s: s["family"] == "gate_swing" and s["latch"]["model"] == "magnalatch",   # world-mounted lift pin
    "drop_bolt": lambda s: s["family"] == "sliding_single" and s["latch"]["model"] == "electric_bolt",
    "rise": lambda s: s["family"] == "cold_storage" and s["hinge"].get("axis_tilt_deg"),
    "thumbturn": lambda s: s["lock"]["model"] == "deadbolt_single" and s["lock"].get("engaged"),
    "auto_slider": lambda s: s["family"] == "automatic_sliding",
    "keypad": lambda s: s["lock"]["model"].startswith("keypad") and s["lock"].get("engaged"),   # buttons, not bolts
    # an engaged hook lock the robot cannot reach: the canonical file must keep it engaged
    "no_release": lambda s: s["latch"]["model"] == "hook_slider" and s["lock"].get("engaged") and not s["lock"].get("robot_side_release", True),
}


@pytest.fixture(scope="module")
def doors(tmp_path_factory):
    """{class: (spec, door_dir, model_json, full stage, rl stage, rl meta)}."""
    out = tmp_path_factory.mktemp("usd_export_fixes")
    specs = generate_all()
    res = {}
    for key, pred in PICKS.items():
        s = next((x for x in specs if pred(x)), None)
        assert s is not None, f"no door in the dataset matches the {key!r} class"
        export_door(s, str(out / "doors"), str(out / "hardware"), formats=("usd", "json"))
        dd = str(out / "doors" / s["id"])
        with open(os.path.join(dd, "model.json")) as f:
            mj = json.load(f)
        with open(os.path.join(dd, "spec.json")) as f:
            spec = json.load(f)
        full = Usd.Stage.Open(os.path.join(dd, "door.usda"))
        rl = Usd.Stage.Open(os.path.join(dd, "door_rl.usda"))
        res[key] = (spec, dd, mj, full, rl, json.loads(rl.GetDefaultPrim().GetAttribute("doorbench:rl").Get()))
    return res


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _joints(stage, cls=None):
    out = {}
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint) or p.IsA(UsdPhysics.FixedJoint):
            if cls is None or p.IsA(cls):
                out[p.GetName()] = p
    return out


def _schemas(prim):
    md = prim.GetMetadata("apiSchemas")
    return set(md.GetAddedOrExplicitItems()) if md is not None else set()


def _attr(prim, name, default=None):
    a = prim.GetAttribute(name)
    return a.Get() if (a and a.IsValid() and a.Get() is not None) else default


def _env_release_joints(stage):
    return {n: p for n, p in _joints(stage).items() if _attr(p, "doorbench:role") == "env_release"}


def _filtered_pairs(stage):
    out = set()
    for p in stage.Traverse():
        if not p.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rel = p.GetRelationship("physxFilteredPairs:filteredPairs")
        for t in ([str(x) for x in rel.GetTargets()] if rel and rel.IsValid() else []):
            out.add(tuple(sorted((p.GetName(), str(t).split("/")[-1]))))
    return out


def _root(stage):
    return next(p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI))


def _inputs(spec, mj, dd):
    return P.door_inputs(spec, mj, rl_meta=P.read_rl_meta(dd))


# ---------------------------------------------------------------------------
# (1) environment-released locks: a real breakable joint, not a JSON note
# ---------------------------------------------------------------------------
def test_env_release_weld_is_a_real_joint(doors):
    spec, dd, mj, full, rl, rlm = doors["maglock"]
    welds = [e for e in mj["equalities"] if e["kind"] == "weld"]
    assert welds, "the maglock door must still carry the MJCF weld equality"
    want_force = {w["name"]: float(w["holding_force_N"]) for w in mj["meta"]["breakable_welds"]}
    assert want_force and all(f > 0 for f in want_force.values())
    for kind, stage in (("full", full), ("rl", rl)):
        er = _env_release_joints(stage)
        assert set(er) == {w["name"] for w in welds}, f"{kind}: {sorted(er)}"
        for name, p in er.items():
            assert p.IsA(UsdPhysics.FixedJoint), f"{kind} {name} must be a FixedJoint"
            # excluded from the articulation: PhysX solves it as a loop joint instead of making `base` a second
            # parent of the leaf (an articulation is a tree)
            assert _attr(p, "physics:excludeFromArticulation") is True
            assert _attr(p, "physics:jointEnabled") is True
            assert _attr(p, "doorbench:weld_body") == welds[0]["a"]
            # MuJoCo's breakaway test compares every row of the weld constraint - three force and three torque rows -
            # against the same holding force (DoorEnv._lock_logic), so PhysX gets it on both channels
            assert _attr(p, "physics:breakForce") == pytest.approx(want_force[name], rel=1e-6)
            assert _attr(p, "physics:breakTorque") == pytest.approx(want_force[name], rel=1e-6)
            assert _attr(p, "doorbench:holding_force_N") == pytest.approx(want_force[name], rel=1e-6)
        listed = json.loads(stage.GetDefaultPrim().GetAttribute("doorbench:env_release").Get())
        assert {e["name"] for e in listed} == set(er)
        assert all(e["holding_force_N"] > 0 for e in listed)


def test_env_release_holding_force_is_the_hardware_model(doors):
    spec, dd, mj, full, rl, rlm = doors["maglock"]
    lk = H.LOCKS[spec["lock"]["model"]]
    assert lk.kind in ("mag_lock", "delayed_egress")
    force = float(list(_env_release_joints(full).values())[0].GetAttribute("doorbench:holding_force_N").Get())
    assert force in (H.LATCHES["mag_lock_600"].holding_force, H.LATCHES["mag_lock_1200"].holding_force)


def test_env_release_joint_does_not_add_a_second_parent(doors):
    """The articulation must stay a tree: the leaf keeps exactly one non-loop parent joint."""
    _, _, _, full, rl, _ = doors["maglock"]
    for stage in (full, rl):
        parents = {}
        for name, p in _joints(stage).items():
            if _attr(p, "physics:excludeFromArticulation") is True:
                continue
            b1 = [str(t) for t in UsdPhysics.Joint(p).GetBody1Rel().GetTargets()]
            assert len(b1) == 1
            assert b1[0] not in parents, f"{b1[0]} is the child of {parents.get(b1[0])} and {name}"
            parents[b1[0]] = name


def test_doors_without_env_release_have_no_such_joint(doors):
    for key in ("hook_slider", "rise", "swing_pair", "baby_gate"):
        _, _, mj, full, rl, _ = doors[key]
        assert not [e for e in mj["equalities"] if e["kind"] == "weld"]
        assert _env_release_joints(full) == {} and _env_release_joints(rl) == {}


# ---------------------------------------------------------------------------
# (2) canonical RL export: welded released vs welded engaged, recorded per door
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key,role_hint", [("hook_slider", "latch"), ("cremone", "operator"), ("wheel_dogs", "operator")])
def test_rl_releases_parts_the_operator_retracts(doors, key, role_hint):
    """A revolute hook, a cremone shoot bolt and wheel-driven dogs have no canonical slot; welding them ENGAGED
    locked the door by construction while the protocol expects it to open."""
    spec, dd, mj, full, rl, rlm = doors[key]
    ir = {b["joint"]["name"]: b["joint"] for b in mj["bodies"] if b.get("joint")}
    released = {w["joint"]: w for w in rlm["released_parts"]}
    assert released, f"{key}: nothing welded released"
    holding = [w for w in rlm["released_holding"]]
    assert holding, f"{key}: the part that holds the leaf is not among the released ones"
    assert not rlm["welded_engaged"], f"{key}: {[w['joint'] for w in rlm['welded_engaged']]} still welded engaged"
    for jn, w in released.items():
        src = ir[jn]
        assert w["shift"] == pytest.approx(src["range"][1] - src["modeled_at"], abs=1e-9)
        assert w["released"] and not w["shift"] < 0
    if role_hint == "operator":
        assert any("operator" in w["reason"] for w in rlm["released_parts"])
        assert set(rlm["operator_driven_joints"]) & set(released)


def test_buttons_are_welded_but_never_count_as_holding(doors):
    """Keypad keys and REX / call buttons press INTO the leaf face: they can never reach the frame, so welding them
    in either state neither holds nor releases the leaf.  Treating them as holding parts made every keypad-locked
    door read `stays_closed` (and every unlocked one lose its hold phase), which PhysX contradicts."""
    spec, dd, mj, full, rl, rlm = doors["keypad"]
    keys = [w for w in rlm["welded"] if "keypad_key" in w["joint"]]
    assert len(keys) >= 5, [w["joint"] for w in rlm["welded"]]
    for w in keys:
        assert w["press_only"] is True and w["holding"] is False, w
    assert not [w for w in rlm["welded_engaged"] + rlm["released_holding"] if "keypad_key" in w["joint"]]
    # a button is never what makes the canonical door read "locked shut" (this door's deadbolt may still be)
    inp = _inputs(spec, mj, dd)
    assert not [j for j in P._rl_blocking(inp, inp["rl"]) if "keypad_key" in j]
    for w in rlm["welded"]:
        if w["semantic"] in ("sensor", "decor"):
            assert w["holding"] is False, w


def test_press_only_is_geometric_not_by_name(doors):
    """`press_only` is 'the slide axis is the leaf normal', so a 12.7 mm latch bolt or a 20 mm shoot bolt (which move
    in the plane of the leaf, toward an edge) is never mistaken for a button even though both are small."""
    for key, (_, _, mj, _, _, rlm) in doors.items():
        for w in rlm["welded"]:
            if w["type"] != "slide":
                assert w["press_only"] is False, (key, w)
            if w["role"] == "latch" and w["type"] == "slide":
                assert w["press_only"] is False, (key, w)


def test_no_robot_side_release_keeps_the_holding_part_engaged(doors):
    """A hook / cremone bolt / lock bar the robot cannot reach (keyed outside only, padlock, no inside trim) must NOT
    be welded released: the real door stays locked, so `locked_holds` must hold in the canonical file too."""
    spec, dd, mj, full, rl, rlm = doors["no_release"]
    assert spec["lock"]["engaged"] and not spec["lock"].get("robot_side_release", True)
    assert rlm["welded_engaged"], "the engaged lock part must stay engaged"
    assert any("no robot-side release" in w["reason"] for w in rlm["welded_engaged"])
    assert not [w for w in rlm["released_parts"] if w["role"] == "lock"]
    sched = _inputs(spec, mj, dd)["schedule"]
    assert sched["usd_rl"]["hold"] == "hold"
    assert sched["usd_rl"]["locked"] == sched["mjcf"]["locked"]


def test_rl_keeps_an_engaged_lock_without_a_release_welded_shut(doors):
    """A slide bolt the robot must throw separately has no canonical slot and no operator coupling: door_rl.usda
    cannot open, and that is recorded rather than guessed."""
    spec, dd, mj, full, rl, rlm = doors["welded_lock"]
    engaged = {w["joint"] for w in rlm["welded_engaged"]}
    assert engaged, "the engaged slide bolt must be recorded as welded engaged"
    for w in rlm["welded_engaged"]:
        assert w["shift"] == 0.0 and w["was_engaged"] and w["holding"]
    assert not (engaged & {w["joint"] for w in rlm["released_parts"]})


def test_rl_weld_record_is_complete_and_consistent(doors):
    for key, (spec, dd, mj, full, rl, rlm) in doors.items():
        ir = {b["joint"]["name"]: b["joint"] for b in mj["bodies"] if b.get("joint")}
        rec = {w["joint"] for w in rlm["welded"]}
        assert rec <= set(ir), key
        for sub in ("released_parts", "released_holding", "welded_engaged"):
            assert {w["joint"] for w in rlm[sub]} <= rec, f"{key}/{sub}"
        assert not ({w["joint"] for w in rlm["released_parts"]} & {w["joint"] for w in rlm["welded_engaged"]}), key
        # a slot joint is never welded
        slots = {info["source"] for info in rlm["joints"].values() if info.get("active") and info.get("source")}
        assert not (rec & slots), key


@pytest.mark.parametrize("key", ["hook_slider", "cremone", "wheel_dogs"])
def test_protocol_expects_the_rl_door_to_open(doors, key):
    spec, dd, mj, full, rl, rlm = doors[key]
    sched = _inputs(spec, mj, dd)["schedule"]
    assert sched["mjcf"]["operate"] == "opens"
    assert sched["usd_rl"]["operate"] == "opens", f"{key}: {sched['usd_rl']['operate']}"
    # nothing can hold the canonical leaf once the holding part is welded released: the hold phase is not comparable
    assert sched["usd_rl"]["hold"].startswith("na:"), sched["usd_rl"]["hold"]
    assert sched["mjcf"]["hold"] == "hold"


def test_protocol_expects_the_welded_lock_to_stay_closed(doors):
    spec, dd, mj, full, rl, rlm = doors["welded_lock"]
    sched = _inputs(spec, mj, dd)["schedule"]
    assert sched["mjcf"]["operate"] == "opens"
    assert sched["usd_rl"]["operate"] == "stays_closed"
    assert sched["usd_rl"]["hold"] == "hold"


def test_protocol_uses_the_export_ground_truth(doors):
    spec, dd, mj, full, rl, rlm = doors["hook_slider"]
    inp = _inputs(spec, mj, dd)
    assert inp["rl"]["weld_ground_truth"] is True
    assert inp["rl"]["released_holding"] and not inp["rl"]["welded_engaged"]


# ---------------------------------------------------------------------------
# (3) self-collision + filtered pairs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", list(PICKS))
def test_self_collision_enabled(doors, key):
    _, _, _, full, rl, _ = doors[key]
    for stage in (full, rl):
        assert _attr(_root(stage), "physxArticulation:enabledSelfCollisions") is True


@pytest.mark.parametrize("key", list(PICKS))
def test_filtered_pairs_are_exactly_mujocos_filter_set(doors, key):
    """PhysX skips joint-adjacent links on its own; everything else MuJoCo suppresses (same weld body, weld
    parent/child, contact_excludes) must be authored, and nothing more."""
    _, _, mj, full, _, _ = doors[key]
    assert _filtered_pairs(full) == mj_filtered_pairs(mj), key


@pytest.mark.parametrize("key", ["baby_gate", "gate_swing", "drop_bolt"])
def test_lift_pins_and_drop_bolts_collide_with_the_leaf(doors, key):
    """The pin / drop bolt and the leaf are both world-attached (siblings under `base`), so MuJoCo collides them -
    which is how they hold the gate.  With self-collision off in PhysX they passed straight through."""
    _, _, mj, full, _, _ = doors[key]
    leaf = next(b["name"] for b in mj["bodies"] if b.get("joint") and b["joint"]["name"] == mj["meta"]["primary_joint"])
    holder = next(b["name"] for b in mj["bodies"]
                  if not b.get("static") and b["name"] != leaf and b.get("parent") is None and b.get("joint")
                  and b["semantic"] in ("latch", "lock"))
    assert tuple(sorted((leaf, holder))) not in _filtered_pairs(full), f"{key}: {leaf}/{holder} filtered"
    # both hang off `base`: PhysX only skips joint-adjacent links, so this pair is a real contact again
    assert {leaf, holder} <= {b["name"] for b in mj["bodies"] if b.get("parent") is None}


def test_swing_pair_leaves_collide_but_leaf_and_operator_do_not(doors):
    _, _, mj, full, rl, rlm = doors["swing_pair"]
    pairs_rl = _filtered_pairs(rl)
    assert ("leaf", "leaf2") not in pairs_rl, "the two leaves of a pair must collide (that is what latches them)"
    assert ("leaf", "operator") in pairs_rl, "the handle is a child of the leaf in MuJoCo: filtered"
    assert {tuple(x) for x in rlm["filtered_pairs"]} == pairs_rl
    # the active leaf's latch bolt must be free to hit the inactive leaf
    assert ("latch", "leaf2") not in pairs_rl


@pytest.mark.parametrize("key", list(PICKS))
def test_rl_filtered_pairs_are_never_joint_adjacent(doors, key):
    adjacent = {tuple(sorted(x)) for x in (("base", "carriage"), ("carriage", "leaf"), ("leaf", "operator_pivot"),
                                           ("operator_pivot", "operator"), ("leaf", "latch"), ("base", "carriage2"), ("carriage2", "leaf2"))}
    assert not (_filtered_pairs(doors[key][4]) & adjacent), key


# ---------------------------------------------------------------------------
# (4) couplings: mimic only where PhysX honours it, emulation data for the rest
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", list(PICKS))
def test_no_mimic_on_a_prismatic_axis(doors, key):
    """PhysX articulation mimic joints support rotational axes only; a mimic on a prismatic joint (or referencing a
    prismatic axis) is parsed and silently dropped - which is what happened to every hinge -> slide coupling."""
    for stage in (doors[key][3], doors[key][4]):
        for name, p in _joints(stage).items():
            for sch in _schemas(p):
                if not sch.startswith("PhysxMimicJointAPI:"):
                    continue
                inst = sch.split(":")[1]
                assert p.IsA(UsdPhysics.RevoluteJoint) and inst.startswith("rot"), f"{key}/{name}: {sch}"
                ref = [str(t) for t in p.GetRelationship(f"physxMimicJoint:{inst}:referenceJoint").GetTargets()]
                assert len(ref) == 1 and stage.GetPrimAtPath(ref[0]).IsA(UsdPhysics.RevoluteJoint), f"{key}/{name}"
                assert str(p.GetAttribute(f"physxMimicJoint:{inst}:referenceJointAxis").Get()).startswith("rot")


@pytest.mark.parametrize("key", list(PICKS))
def test_every_equality_is_classified(doors, key):
    _, _, mj, full, _, _ = doors[key]
    ir = {b["joint"]["name"]: b["joint"] for b in mj["bodies"] if b.get("joint")}
    joints = _joints(full)
    for e in mj["equalities"]:
        if e["kind"] != "joint":
            continue
        p = joints[e["a"]]
        rot = ir[e["a"]]["type"] == "hinge" and ir[e["b"]]["type"] == "hinge"
        mode = _attr(p, "doorbench:coupling_mode")
        assert mode in ("mimic", "emulated", "servo")
        if rot:
            assert mode == "mimic" and any(s.startswith("PhysxMimicJointAPI:") for s in _schemas(p))
        else:
            assert mode in ("emulated", "servo")
            assert not any(s.startswith("PhysxMimicJointAPI:") for s in _schemas(p))
        assert _attr(p, "doorbench:coupling_driver") == e["b"]
        assert _attr(p, "doorbench:coupling_c1") == pytest.approx(e["polycoeff"][1], rel=1e-5)


def test_hook_coupling_stays_a_mimic(doors):
    """hook (revolute) <- handle (revolute): PhysX honours this one, so the mimic must still be authored."""
    _, _, mj, full, _, _ = doors["hook_slider"]
    eq = next(e for e in mj["equalities"] if e["kind"] == "joint")
    p = _joints(full)[eq["a"]]
    assert _attr(p, "doorbench:coupling_mode") == "mimic"
    assert "PhysxMimicJointAPI:rotX" in _schemas(p)
    assert _attr(p, "physxMimicJoint:rotX:gearing") == pytest.approx(-eq["polycoeff"][1], rel=1e-5)
    assert _attr(p, "physxMimicJoint:rotX:offset") == pytest.approx(-eq["polycoeff"][0], abs=1e-9)


@pytest.mark.parametrize("key", ["rise", "thumbturn"])
def test_emulated_coupling_carries_the_bilateral_law(doors, key):
    """hinge -> slide couplings PhysX drops carry everything a consumer needs to apply them WITH the reaction:
    the driven DOF's effective inertia, its passive law, its constant gravity bias, and the reflected inertia
    c1^2 * I_driven the driver adds to its armature."""
    _, _, mj, full, _, _ = doors[key]
    meta = json.loads(full.GetDefaultPrim().GetAttribute("doorbench:meta").Get())
    emulated = meta["couplings_emulated"]
    assert emulated, f"{key}: no emulated coupling"
    joints = _joints(full)
    for c in emulated:
        assert c["driven_type"] == "slide" or c["driver_type"] == "slide"
        assert c["driven_inertia"] > 0
        assert c["reflected_inertia"] == pytest.approx(c["coeff"][1] ** 2 * c["driven_inertia"], rel=1e-6)
        dp, rp = joints[c["driven"]], joints[c["driver"]]
        assert _attr(dp, "doorbench:coupling_mode") == "emulated"
        assert _attr(dp, "doorbench:coupling_driven_inertia") == pytest.approx(c["driven_inertia"], rel=1e-5)
        assert _attr(dp, "doorbench:coupling_gravity_bias") == pytest.approx(c["driven_gravity_bias"], rel=1e-5, abs=1e-9)
        assert _attr(rp, "doorbench:coupling_reflected_armature", 0.0) >= c["reflected_inertia"] * (1 - 1e-4)
        assert meta["coupling_reflected_armature"][c["driver"]] > 0


@pytest.mark.parametrize("key", list(PICKS))
def test_coupling_law_is_also_given_in_usd_joint_coordinates(doors, key):
    """PhysX reports joint positions in USD coordinates (q_usd = q_db - zero_offset); a consumer that applies the
    IR coefficients to them would be off by the two offsets, so the exporter precomputes ``coeff_usd``."""
    _, _, mj, full, _, rlm = doors[key]
    ir = {b["joint"]["name"]: b["joint"] for b in mj["bodies"] if b.get("joint")}
    meta = json.loads(full.GetDefaultPrim().GetAttribute("doorbench:meta").Get())
    entries = [c for c in json.loads(full.GetDefaultPrim().GetAttribute("doorbench:couplings").Get()) if "coeff_usd" in c]
    assert len(entries) == len([e for e in mj["equalities"] if e["kind"] == "joint"])
    for c in entries:
        c0, c1 = c["coeff"]
        off_a, off_b = ir[c["driven"]]["modeled_at"], ir[c["driver"]]["modeled_at"]
        assert c["coeff_usd"] == pytest.approx([c0 + c1 * off_b - off_a, c1], abs=1e-12)
    for c in rlm.get("couplings", []):
        assert "coeff_usd" in c and c["coeff_usd"][1] == pytest.approx(c["coeff"][1])


def test_rise_coupling_reaction_equals_the_documented_closing_torque(doors):
    """The reaction a consumer applies on the driver is c1 * tau_driven_ext; for the helical hinge that is
    c1 * (-m g) = -m g dz/dq, the closing torque docs/ISAAC_LAB.md already specifies for the locked riser."""
    _, _, mj, full, rl, rlm = doors["rise"]
    meta = json.loads(full.GetDefaultPrim().GetAttribute("doorbench:meta").Get())
    rc = meta["rise_coupling"]
    c = next(x for x in meta["couplings_emulated"] if x["driven"] == rc["rise_joint"])
    assert c["driver"] == rc["hinge_joint"]
    assert c["coeff"][1] * c["driven_gravity_bias"] == pytest.approx(rc["gravity_torque_Nm"], rel=1e-6)
    assert c["driven_gravity_bias"] == pytest.approx(-rc["carried_mass_kg"] * 9.81, rel=1e-6)
    assert rlm["rise_coupling"]["gravity_torque_Nm"] == pytest.approx(rc["gravity_torque_Nm"], rel=1e-9)


def test_automatic_pair_coupling_is_left_to_the_servos(doors):
    """Both leaves of a bi-parting slider carry their own MJCF position servo, folded into their PhysX drive: the
    drives move them together, so the equality needs neither a mimic nor an emulation."""
    _, _, mj, full, _, rlm = doors["auto_slider"]
    eqs = [e for e in mj["equalities"] if e["kind"] == "joint"]
    assert eqs
    joints = _joints(full)
    for e in eqs:
        p = joints[e["a"]]
        assert _attr(p, "doorbench:coupling_mode") == "servo"
        assert _attr(joints[e["a"]], "doorbench:servo_in_drive") and _attr(joints[e["b"]], "doorbench:servo_in_drive")
        assert _attr(joints[e["b"]], "doorbench:coupling_reflected_armature") is None
    assert all(c["mode"] == "servo" for c in rlm["couplings"])
    assert rlm["coupling_reflected_armature"] == {}


def test_authored_armature_still_matches_the_ir(doors):
    """The reflected inertia is metadata, NOT folded into physxJointAxis:*:armature - the authored armature must
    keep matching model.json (the parity structure check compares them)."""
    for key, (_, _, mj, full, _, _) in doors.items():
        ir = {b["joint"]["name"]: b["joint"] for b in mj["bodies"] if b.get("joint")}
        for name, p in _joints(full).items():
            if name not in ir:
                continue
            inst = "angular" if p.IsA(UsdPhysics.RevoluteJoint) else "linear"
            assert _attr(p, f"physxJointAxis:{inst}:armature") == pytest.approx(ir[name]["armature"], rel=1e-6, abs=1e-12), f"{key}/{name}"


# ---------------------------------------------------------------------------
# the static validator accepts these files and rejects each regression
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", list(PICKS))
def test_static_validator_accepts(doors, key):
    r = validate_door(doors[key][1])
    assert r["full"]["ok"], r["full"]["errors"]
    assert r["rl"]["ok"], r["rl"]["errors"]


def test_validator_rejects_a_missing_env_release_joint(doors, tmp_path):
    _, dd, _, _, _, _ = doors["maglock"]
    bad = tmp_path / "no_weld"
    shutil.copytree(dd, bad)
    st = Usd.Stage.Open(str(bad / "door.usda"))
    st.RemovePrim(list(_env_release_joints(st).values())[0].GetPath())
    st.GetRootLayer().Save()
    r = validate_door(str(bad))
    assert not r["full"]["ok"]
    assert any("env-release" in e or "weld equalities" in e for e in r["full"]["errors"]), r["full"]["errors"]


def test_validator_rejects_self_collision_off(doors, tmp_path):
    _, dd, _, _, _, _ = doors["swing_pair"]
    bad = tmp_path / "no_selfcol"
    shutil.copytree(dd, bad)
    st = Usd.Stage.Open(str(bad / "door.usda"))
    _root(st).GetAttribute("physxArticulation:enabledSelfCollisions").Set(False)
    st.GetRootLayer().Save()
    r = validate_door(str(bad))
    assert not r["full"]["ok"]
    assert any("enabledSelfCollisions" in e for e in r["full"]["errors"]), r["full"]["errors"]


def test_validator_rejects_a_prismatic_mimic(doors, tmp_path):
    _, dd, _, _, _, _ = doors["rise"]
    bad = tmp_path / "bad_mimic"
    shutil.copytree(dd, bad)
    st = Usd.Stage.Open(str(bad / "door.usda"))
    p = next(q for q in _joints(st).values() if q.IsA(UsdPhysics.PrismaticJoint) and _attr(q, "doorbench:coupling_mode") == "emulated")
    p.AddAppliedSchema("PhysxMimicJointAPI:transX")
    st.GetRootLayer().Save()
    r = validate_door(str(bad))
    assert not r["full"]["ok"]
    assert any("mimic" in e.lower() for e in r["full"]["errors"]), r["full"]["errors"]


def test_both_directions_of_every_filtered_pair_are_authored(doors):
    """Filtering is symmetric, and so is the authoring: a pair appears on BOTH prims' lists.

    One-sided authoring relies on the omni.physx parser reading the union of the two directions.  That is what it
    does today, but it cannot be checked without Isaac, and if it ever stopped doing it half of every door's pairs
    would silently start colliding.  The duplicate target is one line of USD per pair and removes the question.
    """
    full = doors["swing_pair"][3]
    listed = {}
    for p in full.Traverse():
        rel = p.GetRelationship("physxFilteredPairs:filteredPairs")
        if rel and rel.IsValid() and rel.GetTargets():
            listed[p.GetName()] = {t.name for t in rel.GetTargets()}
    assert listed, "this door must need filtered pairs for the test to mean anything"
    for a, others in listed.items():
        for b in others:
            assert a in listed.get(b, set()), f"{b} does not list {a}: the pair is authored one-sided"


def test_validator_rejects_a_dropped_filtered_pair(doors, tmp_path):
    """A pair removed from the union (both directions) must be an error."""
    _, dd, mj, _, _, _ = doors["swing_pair"]
    assert mj_filtered_pairs(mj), "this door must need filtered pairs for the test to mean anything"
    bad = tmp_path / "no_filter"
    shutil.copytree(dd, bad)
    st = Usd.Stage.Open(str(bad / "door.usda"))
    victim = None
    for p in st.Traverse():
        rel = p.GetRelationship("physxFilteredPairs:filteredPairs")
        if rel and rel.IsValid() and rel.GetTargets():
            victim = (p.GetName(), rel.GetTargets()[0].name)
            break
    assert victim is not None
    for p in st.Traverse():                      # drop it from BOTH sides, which is what a lost pair looks like
        if p.GetName() not in victim:
            continue
        rel = p.GetRelationship("physxFilteredPairs:filteredPairs")
        if rel and rel.IsValid():
            rel.SetTargets([t for t in rel.GetTargets() if t.name not in victim])
    st.GetRootLayer().Save()
    r = validate_door(str(bad))
    assert not r["full"]["ok"]
    assert any("filtered pairs" in e for e in r["full"]["errors"]), r["full"]["errors"]
