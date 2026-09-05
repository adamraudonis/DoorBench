"""The running-clearance gate (``doorbench.clearance.Clearance.run_running`` / ``qa.checks["running_clearance"]``).

Background: a part authored EXACTLY touching a static member (0.000 m) is free in MuJoCo at margin 0 - no
penetration, no force - so the interpenetration gate passed it for the whole dataset.  PhysX is not so forgiving:
the USD export sets ``physxCollision:contactOffset`` 5 mm, contacts are generated and resolved inside it, and the
zero-gap parts jammed, drifted or exploded in Isaac Sim.  And a real door does not touch its frame anyway: 3-5 mm
at the jambs and head, 6-20 mm of floor undercut, 10-20 mm of running clearance on a revolving/turnstile rotor.

Two halves:
  * synthetic fixtures - a zero-gap door must fail, the same door with a seal / a documented exception must pass,
    a visual-only part may touch (it is not a collider in either engine), the floor undercut is enforced;
  * the families the 2026-09 triage found - full-height turnstiles (rotor column flush on the cage roof AND the
    floor), bifold / accordion (pivot-panel heel on the jamb), sliding bypass (leaf edge flush on the jamb, closed
    and at full travel), roll-up (steel bottom bar flush on the floor), toilet partitions and centre-pivot doors
    (heel corner sweeping the pilaster / reveal), hatches (lid raking the curb), and the 40 swing doors whose leaf
    was raked by the frame's reveal arris past 90 deg.

Run:  pytest -q tests/test_running_clearance.py
"""
from __future__ import annotations

import json
import math
import os

import pytest

from doorbench import folding as F
from doorbench.build import build_model, export_door
from doorbench.clearance import (RUN_MIN, RUN_MIN_FLOOR, RUN_MIN_ROTOR, Clearance, run_running_clearance)
from doorbench.geometry import common as C
from doorbench.geometry.other import ROLLUP_ASTRAGAL, ROTOR_RUN_CLEAR
from doorbench.spec import generate_all

# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------
XML = """<mujoco model="fixture">
  <worldbody>
    <body name="world_env">
      <geom name="floor" type="box" size="2 2 0.05" pos="0 0 -0.05"/>
      <geom name="jamb" type="box" size="0.05 0.1 1.0" pos="{jamb_x} 0 1.0"/>
      {extra_world}
    </body>
    <body name="leaf" pos="-0.5 0 0">
      <joint name="leaf_hinge" type="hinge" axis="0 0 1" pos="0 0 0" limited="true" range="0 1.5708"/>
      <geom name="leaf_slab" type="box" size="0.5 0.002 0.99" pos="0.5 0 {slab_z}"/>
      {extra_leaf}
    </body>
  </worldbody>
</mujoco>
"""


def _write(tmp_path, *, jamb_x=0.55, slab_z=1.0, extra_world="", extra_leaf="", world_geoms=(), leaf_geoms=(), meta=None):
    """A one-leaf fixture: the slab runs from x=-0.5 to x=+0.5, the jamb starts at jamb_x - 0.05.

    The slab is deliberately thin (4 mm): a thick leaf hinged on its own edge line swings its LOCK edge
    hypot(W, t/2) - W towards the jamb as it opens, and that would put the fixture's own sweep, not the authored
    gap, on trial here."""
    d = tmp_path
    with open(os.path.join(d, "door.xml"), "w") as f:
        f.write(XML.format(jamb_x=jamb_x, slab_z=slab_z, extra_world=extra_world, extra_leaf=extra_leaf))
    model = {"meta": meta or {},
             "bodies": [{"name": "world_env",
                         "geoms": [{"name": "floor", "semantic": "floor", "collision": True},
                                   {"name": "jamb", "semantic": "frame", "collision": True}] + list(world_geoms)},
                        {"name": "leaf", "joint": {"name": "leaf_hinge", "role": "primary"},
                         "geoms": [{"name": "leaf_slab", "semantic": "leaf", "collision": True}] + list(leaf_geoms)}]}
    with open(os.path.join(d, "model.json"), "w") as f:
        json.dump(model, f)
    return str(d)


def test_zero_gap_leaf_on_jamb_fails(tmp_path):
    """The defect itself: a slab whose edge is authored exactly on the jamb face."""
    r = run_running_clearance(_write(tmp_path))
    assert not r["ok"]
    f = next(x for x in r["failures"] if x["static"] == "jamb")
    assert f["moving"] == "leaf_slab" and f["required"] == RUN_MIN
    assert f["gap"] <= 0.0 and abs(f["q"]) < 0.05          # touching at (or a whisker off) the closed position
    # and the gate says so at rest too, not only somewhere in the travel
    rest = next(p for p in run_running_clearance(_write(tmp_path), n_steps=0, record_all=True)["pairs"]
                if p["moving"] == "leaf_slab" and p["static"] == "jamb")
    assert rest["gap"] == pytest.approx(0.0, abs=1e-3)


def test_running_gap_passes(tmp_path):
    """The same door with the real 3 mm running clearance."""
    assert run_running_clearance(_write(tmp_path, jamb_x=0.55 + RUN_MIN))["ok"]


def test_seal_may_touch(tmp_path):
    """A door sealed against its frame: the slab runs clear, the weatherstrip is *meant* to be squashed."""
    d = _write(tmp_path, jamb_x=0.55 + RUN_MIN,
               extra_leaf='<geom name="leaf_seal" type="box" size="0.0015 0.002 0.99" pos="1.0015 0 1.0"/>',
               leaf_geoms=[{"name": "leaf_seal", "semantic": "seal", "collision": True}])
    r = run_running_clearance(d, record_all=True)
    touching = next(p for p in r["pairs"] if p["moving"] == "leaf_seal" and p["static"] == "jamb")
    assert touching["gap"] == pytest.approx(0.0, abs=1e-3) and touching["required"] == 0.0
    assert r["ok"]


def test_documented_exception_passes(tmp_path):
    """meta["running_clearance_allow"] = [[a, b, reason]] is how a model declares a contact the semantics cannot."""
    meta = {"running_clearance_allow": [["leaf_slab", "jamb", "fixture: documented contact"]]}
    assert run_running_clearance(_write(tmp_path, meta=meta))["ok"]


def test_visual_only_part_may_touch(tmp_path):
    """Trim that is not a collider in either engine cannot jam anything; its overlap is the penetration gate's job."""
    d = _write(tmp_path, jamb_x=0.55 + RUN_MIN,
               extra_world='<geom name="casing" type="box" size="0.05 0.008 1.0" pos="0.65 0.028 1.0" contype="0" conaffinity="0"/>',
               world_geoms=[{"name": "casing", "semantic": "decor", "collision": False}])
    assert run_running_clearance(d)["ok"]


def test_floor_undercut_enforced(tmp_path):
    """A leaf needs more under it than beside it: 6 mm of undercut, not 3."""
    z_ok = 0.99 + RUN_MIN_FLOOR
    assert run_running_clearance(_write(tmp_path, jamb_x=0.55 + RUN_MIN, slab_z=z_ok))["ok"]
    r = run_running_clearance(_write(tmp_path, jamb_x=0.55 + RUN_MIN, slab_z=0.99 + RUN_MIN))
    assert not r["ok"]
    f = next(x for x in r["failures"] if x["static"] == "floor")
    assert f["required"] == RUN_MIN_FLOOR and f["gap"] == pytest.approx(RUN_MIN, abs=1e-3)


def test_rotor_minimum_is_declared_by_the_model(tmp_path):
    """A rotor asks for more than the structural minimum through meta["running_clearance_min"]."""
    meta = {"running_clearance_min": RUN_MIN_ROTOR}
    assert not run_running_clearance(_write(tmp_path, jamb_x=0.55 + RUN_MIN, meta=meta))["ok"]
    assert run_running_clearance(_write(tmp_path, jamb_x=0.55 + RUN_MIN_ROTOR + 0.001, meta=meta))["ok"]


# ---------------------------------------------------------------------------
# the geometry the gate flagged, pinned at its source
# ---------------------------------------------------------------------------
def test_pivot_heel_gap_solves_the_sweep():
    """pivot_heel_gap(P, t) is exactly the gap at which the heel corner clears the face it turns against."""
    for P in (0.02, 0.035, 0.05, 0.135):
        for t in (0.018, 0.028, 0.044, 0.06):
            g = C.pivot_heel_gap(P, t)
            if g == 0.0:
                assert P <= t / 2 + RUN_MIN + 1e-9        # no gap can fix it; the pivot has to move
                continue
            assert P - math.hypot(P - g, t / 2) >= RUN_MIN - 1e-9
    # the old flat 6 mm gap is what the toilet partitions and the centre-pivot doors had
    assert C.pivot_heel_gap(0.05, 0.044) > 0.006 and C.pivot_heel_gap(0.135, 0.06) > 0.006


def test_fold_jamb_gap_solves_the_sweep():
    for t in (0.018, 0.028, 0.035, 0.045):
        g = F.fold_jamb_gap(t)
        assert F.FOLD_PIVOT_IN - math.hypot(F.FOLD_PIVOT_IN - g, t / 2) >= F.FOLD_RUN_CLEAR - 1e-9
    assert F.fold_jamb_gap(0.035) > F.FOLD_JAMB_GAP        # the 5 mm flat gap left 0.3 mm


def test_hinge_throw_reaches_the_frame_face_without_binding_the_lock_edge():
    """The pin lands on the frame's swing-side face (so the reveal arris stays inside its circle), and a throw wider
    than the standard knuckle never eats more of the leading gap than there is."""
    t, depth, W = 0.044, 0.30, 0.81
    y_wall = -max(0.0, depth / 2 - t / 2 - C.LEAF_FACE_INSET)
    throw = C.hinge_throw(t, depth, y_wall, 1.0, W)
    assert throw == pytest.approx(t / 2 + C.LEAF_FACE_INSET, abs=1e-9)
    # a leaf hung deeper than its knuckle: the throw follows the face, but only as far as the lock edge can afford
    deep = C.hinge_throw(t, depth, -(depth / 2 - t / 2 - 0.05), 1.0, W)
    assert deep > t / 2 + 0.007
    assert math.hypot(W, deep + t / 2) - W <= C.GAP


@pytest.fixture(scope="module")
def specs():
    return {s["id"]: s for s in generate_all()}


def _z_span(g):
    if g.type == "box":
        return g.pos[2] - g.size[2], g.pos[2] + g.size[2]
    if g.type == "cylinder":
        return g.pos[2] - g.size[1], g.pos[2] + g.size[1]
    return None


def test_turnstile_rotor_column_runs_clear_of_roof_and_floor(specs):
    """The 7 exploding full-height turnstiles: the column ended flush on the cage roof AND on the floor."""
    n = 0
    for s in specs.values():
        if s["family"] != "turnstile_fullheight":
            continue
        model = build_model(s)
        rotor, world = model.body("rotor"), model.body("world_env")
        col = next(g for g in rotor.geoms if g.name == "rotor_column")
        roof = next(g for g in world.geoms if g.name == "cage_roof")
        bot, top = _z_span(col)
        assert bot >= ROTOR_RUN_CLEAR - 1e-9, s["id"]
        assert _z_span(roof)[0] - top >= ROTOR_RUN_CLEAR - 1e-9, s["id"]
        assert model.meta["running_clearance_min"] == RUN_MIN_ROTOR
        n += 1
    assert n >= 7


def test_rollup_seats_on_its_astragal_not_on_the_slab(specs):
    """A roll-up closes onto a rubber bottom seal; the steel bar clears the floor."""
    n = 0
    for s in specs.values():
        if s["family"] != "rollup":
            continue
        lb = build_model(s).body("curtain")
        bar = next(g for g in lb.geoms if g.name == "bottom_bar")
        astr = next(g for g in lb.geoms if g.name == "bottom_astragal")
        assert astr.semantic == "seal" and astr.collision
        assert _z_span(astr)[0] == pytest.approx(0.0, abs=1e-9), s["id"]     # the seal is what touches the floor
        assert _z_span(bar)[0] >= ROLLUP_ASTRAGAL - 1e-9, s["id"]
        n += 1
    assert n >= 10


def test_bypass_leaves_keep_their_jamb_clearance(specs):
    """Closed AND at the end of the stroke: the end leaves used to be authored flush on the jamb face."""
    n = 0
    for s in specs.values():
        if s["family"] != "sliding_bypass":
            continue
        model = build_model(s)
        Wo, W = s["opening"]["width"], s["leaf"]["width"]
        for b in model.bodies:
            if not b.name.startswith("leaf_") or b.joint is None:
                continue
            x0, x1 = b.pos[0] - W / 2, b.pos[0] + W / 2
            lo, hi = b.joint.range
            sgn = b.joint.axis[0]
            assert x0 >= -Wo / 2 + C.GAP - 1e-9 and x1 <= Wo / 2 - C.GAP + 1e-9, (s["id"], b.name)
            assert x0 + sgn * hi >= -Wo / 2 + C.GAP - 1e-9, (s["id"], b.name)
            assert x1 + sgn * hi <= Wo / 2 - C.GAP + 1e-9, (s["id"], b.name)
        n += 1
    assert n >= 20


# doors the 2026-09 triage flagged, one per class (turnstile rotor, bifold heel, accordion, bypass jamb, roll-up
# bottom bar, stall pilaster, centre pivot, floor/ceiling hatch curb, swing leaf raked by the reveal arris)
BROKEN_BEFORE = ["db0187_turnstile_fullheight", "db0995_turnstile_fullheight", "db0004_bifold", "db0009_bifold",
                 "db0020_sliding_bypass", "db0008_sliding_bypass", "db0001_rollup", "db0557_stall", "db0617_pivot",
                 "db0449_hatch_floor", "db0357_hatch_ceiling", "db0265_swing_single", "db0114_swing_single",
                 "db0325_swing_single", "db0734_swing_single", "db0742_gate_swing"]


@pytest.mark.parametrize("door_id", BROKEN_BEFORE)
def test_previously_broken_doors_now_run_clear(door_id, specs, tmp_path):
    s = specs[door_id]
    export_door(s, os.path.join(str(tmp_path), "doors"), os.path.join(str(tmp_path), "hardware"), formats=("mjcf", "json"))
    r = run_running_clearance(os.path.join(str(tmp_path), "doors", door_id))
    assert r["ok"], (door_id, r["failures"][:4])


def test_gate_is_published_with_the_penetration_gate(tmp_path, specs):
    """run_clearance returns both gates off one compiled model; qa.py publishes the second as running_clearance."""
    from doorbench.clearance import run_clearance
    s = specs["db0187_turnstile_fullheight"]
    export_door(s, os.path.join(str(tmp_path), "doors"), os.path.join(str(tmp_path), "hardware"), formats=("mjcf", "json"))
    out = run_clearance(os.path.join(str(tmp_path), "doors", s["id"]))
    assert out["ok"] and out["running"]["ok"] and out["running"]["n_pairs"] >= 1
    import inspect
    import doorbench.qa as qa
    assert 'checks["running_clearance"]' in inspect.getsource(qa.run_qa)


# ---------------------------------------------------------------------------
# sensitivity of the gate, both ways (adversarial verification, 2026-09)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gap_mm,must_fail", [(0.0, True), (0.1, True), (0.5, True), (1.0, True), (2.0, True),
                                              (2.9, True), (3.0, False), (5.0, False)])
def test_structural_gap_sensitivity(gap_mm, must_fail, tmp_path):
    """A structural running fit is caught all the way down to a fraction of a millimetre, and 3 mm passes.

    The interesting number is 0.5 mm: too small to read as a design gap, far too big for PhysX to ignore inside its
    5 mm contact offset, and invisible to the interpenetration gate (nothing overlaps).  The only slack between
    fail and pass is ``RUN_EPS`` = 10 um of float noise on the comparison, four orders below the 3 mm it guards.
    """
    r = run_running_clearance(_write(tmp_path, jamb_x=0.55 + gap_mm / 1000.0))
    flagged = [f for f in r["failures"] if f["static"] == "jamb" and f["moving"] == "leaf_slab"]
    assert bool(flagged) is must_fail, (gap_mm, r["failures"][:3])
    if flagged:
        assert flagged[0]["gap"] == pytest.approx(gap_mm / 1000.0, abs=2e-5)
        assert flagged[0]["required"] == RUN_MIN


# name, model.json semantic, authored gap in metres (negative = pressed into the jamb), must the gate flag it?
CONTACT_PARTS = [
    ("leaf_weatherstrip", "seal", 0.0, False),      # seal semantic: meant to be in contact
    ("leaf_brush_strip", "decor", 0.0, False),      # a brush strip mis-tagged as decor - the NAME has to save it
    ("leaf_magnetic_catch", "decor", 0.0, False),   # a magnetic catch must touch its keeper or it holds nothing
    ("leaf_gasket", "leaf", -0.001, False),         # a gasket squashed 1 mm into the jamb, tagged as leaf
    ("leaf_sweep", "leaf", 0.0, False),
    ("leaf_astragal", "leaf", 0.0, False),
    ("leaf_bumper", "frame", 0.0, False),
    ("leaf_edge_rail", "leaf", 0.0, True),          # CONTROL: a structural part on the jamb face IS a failure
]


@pytest.mark.parametrize("name,sem,gap,must_fail", CONTACT_PARTS)
def test_parts_that_touch_by_design_are_not_false_positives(name, sem, gap, must_fail, tmp_path):
    """Everything a real door keeps in contact may touch (and compress); a structural part in the same place fails."""
    half = 0.005
    cx = 1.0 - gap - half        # leaf-local x; world = local - 0.5, so the outer face lands on the jamb face at 0.50
    extra = f'<geom name="{name}" type="box" size="{half} 0.002 0.05" pos="{cx} 0 1.0"/>'
    r = run_running_clearance(_write(tmp_path, extra_leaf=extra,
                                     leaf_geoms=[{"name": name, "semantic": sem, "collision": True}]))
    flagged = [f for f in r["failures"] if f["moving"] == name]
    assert bool(flagged) is must_fail, (name, sem, gap, r["failures"][:3])
