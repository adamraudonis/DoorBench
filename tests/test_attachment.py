"""The ATTACHMENT gate (qa.py ``attachment`` / doorbench/attachment.py): nothing floats.

Background: the shipped db0024 carried a rubber door stop as a single cylinder hanging 0.85 m from the wall, 0.325 m
above the floor, that the leaf never touched even at its 90 deg limit - and it was one of 593 static parts across the
dataset that were bolted to nothing (closer shoes 133 mm in front of the frame, garage rollers 10 mm off the door,
keepers screwed to thin air).  Neither clearance gate can see that defect: they fail parts that touch, not parts that
touch NOTHING.

These tests pin the gate's own behaviour on synthetic fixtures - one per rule, each a two-line MJCF - and then check
that every family representative of the generated dataset comes out clean.

Run:  pytest -q tests/test_attachment.py     (~60 s)
"""
from __future__ import annotations

import json
import math
import os

import pytest

from doorbench.attachment import (MIN_HALF_EXTENT, STOP_STRIKE, TOL_BODY, TOL_BODY_GUIDED, TOL_INTRA, TOL_STATIC,
                                  Attachment, run_attachment)
from doorbench.build import build_model, export_door
from doorbench.spec import generate_all

pytest.importorskip("mujoco")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.environ.get("DOORBENCH_ASSETS", os.path.join(ROOT, "assets"))


# ---------------------------------------------------------------------------
# synthetic fixtures: the smallest door that exhibits one defect
# ---------------------------------------------------------------------------
HEAD = """<mujoco model="fixture">
  <worldbody>
    <geom name="floor" type="box" size="3 3 0.05" pos="0 0 -0.05"/>
    <geom name="wall" type="box" size="1 0.05 1.0" pos="0 0 1.0"/>
    {world}
    {bodies}
  </worldbody>
  {extra}
</mujoco>
"""


def _door(tmp_path, world="", bodies="", extra="", ir_bodies=None, meta=None, name="fix"):
    """Write a minimal door directory (door.xml + model.json) and return its path."""
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    (d / "door.xml").write_text(HEAD.format(world=world, bodies=bodies, extra=extra))
    ir = {"name": name, "tier": "full", "bodies": ir_bodies or [], "materials": {}, "equalities": [], "tendons": [],
          "contact_excludes": [], "meta": meta or {}}
    (d / "model.json").write_text(json.dumps(ir))
    return str(d)


def _ir(name, parent=None, joint=None, geoms=(), mass=0.0):
    return {"name": name, "parent": parent, "pos": [0, 0, 0], "quat": [1, 0, 0, 0],
            "joint": joint, "geoms": [{"name": g[0], "type": g[1], "size": list(g[2]), "pos": list(g[3]),
                                       "quat": [1, 0, 0, 0], "material": "m", "collision": True, "visual": True,
                                       "density": 1000.0, "friction": [0.6, 0.005, 0.0001], "tiers": ["full"],
                                       "semantic": g[4] if len(g) > 4 else "structure", "part_label": ""} for g in geoms],
            "sites": [], "tiers": ["full"], "semantic": "structure", "label": "", "static": parent is None and joint is None,
            "mass": mass, "com": [0, 0, 0], "inertia": [[1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 1e-4]]}


def _jt(name, typ="slide", axis=(0, 0, 1), rng=(0.0, 0.2), role="primary"):
    return {"name": name, "type": typ, "axis": list(axis), "pos": [0, 0, 0], "range": list(rng), "damping": 0.0,
            "frictionloss": 0.0, "stiffness": 0.0, "springref": 0.0, "armature": 0.0, "role": role, "label": "",
            "robot_interactive": True, "initial": 0.0, "notes": "", "damping_closing": None, "damping_opening": None,
            "backcheck_angle": None, "backcheck_damping": None, "limit_solref": None, "ratchet_one_way": False,
            "modeled_at": 0.0}


def _rules(res):
    return set(res["by_rule"])


def test_static_geom_bolted_to_nothing_is_caught(tmp_path):
    """RULE static_detached: the db0024 case - a bumper 100 mm off the wall with nothing behind it."""
    d = _door(tmp_path, world='<geom name="bumper" type="cylinder" size="0.025 0.02" pos="0 -0.15 0.35" euler="90 0 0"/>',
              ir_bodies=[_ir("world_env", geoms=[("floor", "box", (3, 3, 0.05), (0, 0, -0.05), "floor"),
                                                 ("wall", "box", (1, 0.05, 1.0), (0, 0, 1.0), "wall"),
                                                 ("bumper", "cylinder", (0.025, 0.02), (0, -0.15, 0.35), "frame")])])
    res = run_attachment(d)
    assert not res["ok"] and res["by_rule"].get("static_detached") == 1, res["findings"]
    f = res["findings"][0]
    assert f["names"] == ["bumper"] and f["gap"] == pytest.approx(0.08, abs=1e-3) and f["tolerance_m"] == TOL_STATIC


def test_static_geom_on_the_wall_passes(tmp_path):
    """... and the same bumper on a base plate that reaches the wall is clean."""
    d = _door(tmp_path, world='<geom name="bumper" type="cylinder" size="0.025 0.055" pos="0 -0.105 0.35" euler="90 0 0"/>',
              ir_bodies=[_ir("world_env", geoms=[("floor", "box", (3, 3, 0.05), (0, 0, -0.05), "floor"),
                                                 ("wall", "box", (1, 0.05, 1.0), (0, 0, 1.0), "wall"),
                                                 ("bumper", "cylinder", (0.025, 0.055), (0, -0.105, 0.35), "frame")])])
    assert run_attachment(d)["ok"]


def test_body_geoms_that_form_two_islands_are_caught(tmp_path):
    """RULE intra_body_split: a chain modelled as beads with daylight between them."""
    bodies = ('<body name="leaf" pos="0 -0.2 1"><joint name="j" type="hinge" axis="0 0 1"/>'
              '<geom name="slab" type="box" size="0.4 0.02 0.9"/>'
              '<geom name="bead" type="sphere" size="0.01" pos="0.2 -0.06 0"/></body>')
    d = _door(tmp_path, bodies=bodies,
              ir_bodies=[_ir("world_env", geoms=[("floor", "box", (3, 3, 0.05), (0, 0, -0.05), "floor"),
                                                 ("wall", "box", (1, 0.05, 1.0), (0, 0, 1.0), "wall")]),
                         _ir("leaf", joint=_jt("j", "hinge", role="primary"),
                             geoms=[("slab", "box", (0.4, 0.02, 0.9), (0, 0, 0), "leaf"),
                                    ("bead", "sphere", (0.01,), (0.2, -0.06, 0), "lock")], mass=10.0)])
    res = run_attachment(d)
    assert "intra_body_split" in _rules(res), res["findings"]
    f = next(x for x in res["findings"] if x["rule"] == "intra_body_split")
    assert f["names"] == ["bead"] and f["tolerance_m"] == TOL_INTRA and f["gap"] == pytest.approx(0.03, abs=1e-3)


def test_body_detached_from_its_parent_is_caught_and_the_allow_list_clears_it(tmp_path):
    """RULE body_detached, and the documented ``meta["attachment_allow"]`` escape hatch."""
    bodies = ('<body name="leaf" pos="0 -0.06 1"><joint name="j" type="hinge" axis="0 0 1"/>'
              '<geom name="slab" type="box" size="0.4 0.02 0.9"/>'
              '<body name="knob" pos="0.3 -0.06 0"><joint name="k" type="hinge" axis="0 1 0"/>'
              '<geom name="knob_g" type="sphere" size="0.02"/></body></body>')
    ir = [_ir("world_env", geoms=[("floor", "box", (3, 3, 0.05), (0, 0, -0.05), "floor"),
                                  ("wall", "box", (1, 0.05, 1.0), (0, 0, 1.0), "wall")]),
          _ir("leaf", joint=_jt("j", "hinge"), geoms=[("slab", "box", (0.4, 0.02, 0.9), (0, 0, 0), "leaf")], mass=10.0),
          _ir("knob", parent="leaf", joint=_jt("k", "hinge", role="operator"),
              geoms=[("knob_g", "sphere", (0.02,), (0, 0, 0), "operator")], mass=0.3)]
    res = run_attachment(_door(tmp_path, bodies=bodies, ir_bodies=ir))
    assert "body_detached" in _rules(res), res["findings"]
    f = next(x for x in res["findings"] if x["rule"] == "body_detached")
    assert f["body"] == "knob" and f["tolerance_m"] == TOL_BODY and f["gap"] == pytest.approx(0.02, abs=1e-3)
    ok = run_attachment(_door(tmp_path, bodies=bodies, ir_bodies=ir, name="allowed",
                              meta={"attachment_allow": [["knob*", "a knob on a spring stalk, by design"]]}))
    assert ok["ok"], ok["findings"]


def test_a_part_that_leaves_its_housing_in_the_travel_is_caught(tmp_path):
    """RULE detached_in_motion: a bolt that is in its housing at rest and out of it at the end of its throw."""
    bodies = ('<body name="leaf" pos="0 -0.06 1"><joint name="j" type="hinge" axis="0 0 1"/>'
              '<geom name="slab" type="box" size="0.4 0.02 0.9"/>'
              '<body name="bolt" pos="0.35 0 0"><joint name="b" type="slide" axis="1 0 0" range="0 0.2"/>'
              '<geom name="bolt_g" type="capsule" size="0.008 0.02"/></body></body>')
    ir = [_ir("world_env", geoms=[("floor", "box", (3, 3, 0.05), (0, 0, -0.05), "floor"),
                                  ("wall", "box", (1, 0.05, 1.0), (0, 0, 1.0), "wall")]),
          _ir("leaf", joint=_jt("j", "hinge"), geoms=[("slab", "box", (0.4, 0.02, 0.9), (0, 0, 0), "leaf")], mass=10.0),
          _ir("bolt", parent="leaf", joint=_jt("b", "slide", (1, 0, 0), (0.0, 0.2), "lock"),
              geoms=[("bolt_g", "capsule", (0.008, 0.02), (0, 0, 0), "lock")], mass=0.1)]
    res = run_attachment(_door(tmp_path, bodies=bodies, ir_bodies=ir))
    assert "detached_in_motion" in _rules(res), res["findings"]
    f = next(x for x in res["findings"] if x["rule"] == "detached_in_motion")
    assert f["body"] == "bolt" and f["q"] == pytest.approx(0.2, abs=1e-6) and f["gap"] > TOL_BODY


def test_a_running_fit_is_attached_at_the_guided_tolerance(tmp_path):
    """A body on a slide joint is carried by a running fit, which the running-clearance gate REQUIRES to keep a gap:
    it is attached at TOL_BODY_GUIDED, and only beyond that does the gate call it floating."""
    def build(gap):
        bodies = (f'<body name="leaf" pos="0 {-0.07 - gap} 1"><joint name="j" type="slide" axis="1 0 0" range="0 0.3"/>'
                  '<geom name="slab" type="box" size="0.4 0.02 0.9"/></body>')
        ir = [_ir("world_env", geoms=[("floor", "box", (3, 3, 0.05), (0, 0, -0.05), "floor"),
                                      ("wall", "box", (1, 0.05, 1.0), (0, 0, 1.0), "wall")]),
              _ir("leaf", joint=_jt("j", "slide", (1, 0, 0), (0.0, 0.3)),
                  geoms=[("slab", "box", (0.4, 0.02, 0.9), (0, 0, 0), "leaf")], mass=10.0)]
        return run_attachment(_door(tmp_path, bodies=bodies, ir_bodies=ir, name=f"g{int(gap * 1000)}"))
    assert build(TOL_BODY_GUIDED - 0.002)["ok"]
    far = build(TOL_BODY_GUIDED + 0.01)
    assert "body_detached" in _rules(far) and far["findings"][0]["tolerance_m"] == TOL_BODY_GUIDED


def test_an_equality_authored_open_is_caught(tmp_path):
    """RULE equality_anchor: a weld whose relative pose does not match where the bodies actually are."""
    bodies = ('<body name="a" pos="0 -0.2 1"><joint name="ja" type="hinge" axis="0 0 1"/>'
              '<geom name="ga" type="box" size="0.05 0.02 0.05"/></body>'
              '<body name="b" pos="0.3 -0.2 1"><joint name="jb" type="hinge" axis="0 0 1"/>'
              '<geom name="gb" type="box" size="0.05 0.02 0.05"/></body>')
    extra = '<equality><weld name="w" body1="a" body2="b" relpose="0 0 0 1 0 0 0"/></equality>'
    ir = [_ir("world_env", geoms=[("floor", "box", (3, 3, 0.05), (0, 0, -0.05), "floor"),
                                  ("wall", "box", (1, 0.05, 1.0), (0, 0, 1.0), "wall")]),
          _ir("a", joint=_jt("ja", "hinge"), geoms=[("ga", "box", (0.05, 0.02, 0.05), (0, 0, 0), "lock")], mass=1.0),
          _ir("b", joint=_jt("jb", "hinge"), geoms=[("gb", "box", (0.05, 0.02, 0.05), (0, 0, 0), "lock")], mass=1.0)]
    res = run_attachment(_door(tmp_path, bodies=bodies, extra=extra, ir_bodies=ir))
    assert "equality_anchor" in _rules(res), res["findings"]
    assert next(x for x in res["findings"] if x["rule"] == "equality_anchor")["gap"] == pytest.approx(0.3, abs=1e-3)


def test_a_stop_the_leaf_never_reaches_is_caught(tmp_path):
    """RULE stop_not_struck: exactly the shipped db0024 bumper, which stood 14 mm off the leaf at its limit."""
    bodies = ('<body name="leaf" pos="0 -0.06 1"><joint name="j" type="hinge" axis="0 0 1" range="0 1.5708"/>'
              '<geom name="slab" type="box" size="0.4 0.02 0.9"/></body>')
    world = '<geom name="door_stop_bumper" type="cylinder" size="0.02 0.02" pos="0.5 -0.5 0.35"/>'
    ir = [_ir("world_env", geoms=[("floor", "box", (3, 3, 0.05), (0, 0, -0.05), "floor"),
                                  ("wall", "box", (1, 0.05, 1.0), (0, 0, 1.0), "wall"),
                                  ("door_stop_bumper", "cylinder", (0.02, 0.02), (0.5, -0.5, 0.35), "frame")]),
          _ir("leaf", joint=_jt("j", "hinge", rng=(0.0, 1.5708)),
              geoms=[("slab", "box", (0.4, 0.02, 0.9), (0, 0, 0), "leaf")], mass=10.0)]
    meta = {"stops": [{"geom": "door_stop_bumper", "joint": "j", "mount": "floor", "q": 1.5708}]}
    res = run_attachment(_door(tmp_path, world=world, bodies=bodies, ir_bodies=ir, meta=meta))
    assert "stop_not_struck" in _rules(res), res["findings"]
    assert next(x for x in res["findings"] if x["rule"] == "stop_not_struck")["gap"] > STOP_STRIKE


def test_degenerate_and_duplicate_content_is_caught(tmp_path):
    """RULES degenerate_geom / duplicate_geom / body_without_geoms."""
    bodies = ('<body name="leaf" pos="0 -0.06 1"><joint name="j" type="hinge" axis="0 0 1"/>'
              '<geom name="slab" type="box" size="0.4 0.02 0.9"/>'
              '<geom name="sliver" type="box" size="0.1 0.00005 0.1" pos="0 -0.02 0"/>'
              '<geom name="dup_a" type="box" size="0.02 0.02 0.02" pos="0.2 -0.04 0"/>'
              '<geom name="dup_b" type="box" size="0.02 0.02 0.02" pos="0.2 -0.04 0"/></body>')
    ir = [_ir("world_env", geoms=[("floor", "box", (3, 3, 0.05), (0, 0, -0.05), "floor"),
                                  ("wall", "box", (1, 0.05, 1.0), (0, 0, 1.0), "wall")]),
          _ir("leaf", joint=_jt("j", "hinge"),
              geoms=[("slab", "box", (0.4, 0.02, 0.9), (0, 0, 0), "leaf"),
                     ("sliver", "box", (0.1, 0.00005, 0.1), (0, -0.02, 0), "decor"),
                     ("dup_a", "box", (0.02, 0.02, 0.02), (0.2, -0.04, 0), "decor"),
                     ("dup_b", "box", (0.02, 0.02, 0.02), (0.2, -0.04, 0), "decor")], mass=10.0),
          _ir("ghost", parent="leaf", joint=_jt("g", "hinge", role="decor"), geoms=[], mass=0.4)]
    res = run_attachment(_door(tmp_path, bodies=bodies, ir_bodies=ir))
    r = _rules(res)
    assert {"degenerate_geom", "duplicate_geom", "body_without_geoms"} <= r, res["findings"]
    deg = next(x for x in res["findings"] if x["rule"] == "degenerate_geom")
    assert deg["names"] == ["sliver"] and deg["tolerance_m"] == MIN_HALF_EXTENT


def test_mesh_whose_bounding_box_contradicts_its_declared_size_is_caught(tmp_path):
    """RULE mesh_bbox: the IR marks mesh parts with the neutral scale (1, 1, 1); anything else must match the mesh."""
    obj = tmp_path / "unit.obj"
    obj.write_text("v -0.05 -0.05 -0.05\nv 0.05 -0.05 -0.05\nv 0.05 0.05 -0.05\nv -0.05 0.05 -0.05\n"
                   "v -0.05 -0.05 0.05\nv 0.05 -0.05 0.05\nv 0.05 0.05 0.05\nv -0.05 0.05 0.05\n"
                   "f 1 2 3\nf 1 3 4\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\nf 2 3 7\nf 2 7 6\nf 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n")
    world = f'<geom name="lump" type="mesh" mesh="unit" pos="0 -0.1 1.0"/>'
    xml_extra = f'<asset><mesh name="unit" file="{obj}"/></asset>'
    ir = [_ir("world_env", geoms=[("floor", "box", (3, 3, 0.05), (0, 0, -0.05), "floor"),
                                  ("wall", "box", (1, 0.05, 1.0), (0, 0, 1.0), "wall"),
                                  ("lump", "mesh", (0.5, 0.5, 0.5), (0, -0.1, 1.0), "decor")])]
    res = run_attachment(_door(tmp_path, world=world, extra=xml_extra, ir_bodies=ir))
    assert "mesh_bbox" in _rules(res), res["findings"]


# ---------------------------------------------------------------------------
# the generated dataset
# ---------------------------------------------------------------------------
REPRESENTATIVES = ["db0024_swing_single", "db0019_swing_double", "db0148_garage_sectional", "db0001_rollup",
                   "db0168_ship_watertight", "db0066_revolving", "db0187_turnstile_fullheight", "db0202_turnstile_tripod",
                   "db0004_bifold", "db0065_accordion", "db0008_sliding_bypass", "db0453_sliding_single",
                   "db0130_automatic_sliding", "db0053_elevator", "db0241_hatch_floor", "db0017_hatch_ceiling",
                   "db0045_pet_door", "db0037_strip_curtain", "db0106_gate_swing", "db0033_gate_sliding",
                   "db0176_baby_gate", "db0031_saloon", "db0054_stall", "db0179_vault", "db0003_cold_storage",
                   "db0134_sliding_single", "db0005_garage_tiltup", "db0089_automatic_swing"]


@pytest.fixture(scope="module")
def specs():
    return {s["id"]: s for s in generate_all()}


@pytest.mark.parametrize("door_id", REPRESENTATIVES)
def test_family_representatives_have_nothing_floating(tmp_path_factory, specs, door_id):
    root = str(tmp_path_factory.mktemp("att"))
    spec = specs[door_id]
    export_door(spec, os.path.join(root, "doors"), os.path.join(root, "hardware"), formats=("mjcf", "json"))
    res = run_attachment(os.path.join(root, "doors", door_id))
    assert res["ok"], (door_id, res["by_rule"], [f["detail"] for f in res["findings"][:6]])


def test_the_db0024_stop_is_mounted_and_struck(specs, tmp_path):
    """The door the owner reported: its stop is bolted to the floor and the leaf arrives on it at 90 deg."""
    spec = specs["db0024_swing_single"]
    model = build_model(spec)
    world = model.body("world_env")
    names = {g.name for g in world.geoms}
    assert {"door_stop_base", "door_stop_post", "door_stop_bumper"} <= names and "wall_bumper_stop" not in names
    base = next(g for g in world.geoms if g.name == "door_stop_base")
    assert base.pos[2] - base.size[1] <= 1e-9, "the base plate sits on the floor"
    st = model.meta["stops"][0]
    assert st["geom"] == "door_stop_bumper" and st["mount"] == "floor" and st["q"] == pytest.approx(math.radians(90), abs=1e-3)
    export_door(spec, os.path.join(str(tmp_path), "doors"), os.path.join(str(tmp_path), "hardware"), formats=("mjcf", "json"))
    d = os.path.join(str(tmp_path), "doors", spec["id"])
    assert run_attachment(d)["ok"]
    # and the leaf really is on the tip at the limit (this is what stop_not_struck measures)
    a = Attachment(d)
    findings = []
    a.check_stops(findings)
    assert findings == []


def test_a_leaf_that_folds_back_to_the_wall_gets_a_wall_bumper(specs):
    """The stop's mount is decided by the leaf's own travel, not by the name in the spec: at 90 deg the leaf stands
    perpendicular to its wall and gets the floor riser (all 130 wall_bumper doors in the dataset), and a leaf that
    folds back against the wall gets the wall plate and tip."""
    base = next(s for s in specs.values() if s["kinematics"].get("stop") == "wall_bumper")
    floor_model = build_model(base)
    assert floor_model.meta["stops"][0]["mount"] == "floor"
    folded = {**base, "kinematics": {**base["kinematics"], "max_open_deg": 180}}   # flat against the wall
    wall_model = build_model(folded)
    st = wall_model.meta["stops"][0]
    assert st["mount"] == "wall", st
    world = wall_model.body("world_env")
    base_g = next(g for g in world.geoms if g.name == "door_stop_base")
    tip = next(g for g in world.geoms if g.name == "door_stop_bumper")
    # the plate is against the wall face and the tip reaches back from it to the leaf
    wall_y = [g for g in world.geoms if g.name.startswith("wall_") and g.type == "box"]
    face = max(abs(float(g.pos[1])) + float(g.size[1]) for g in wall_y)
    assert abs(abs(float(base_g.pos[1])) + float(base_g.size[1]) - face) < 0.02, (base_g.pos, face)
    assert float(tip.size[1]) > 0.005, "the rubber tip projects from the plate to the leaf"


# ---------------------------------------------------------------------------
# rule running_gear_lands, and the three "held only by the 12 mm running-fit tolerance" classes it came from
# (verify-doors: strips 6 mm under their rail on 80 doors, bifold stacks 5 mm under the fold track on 54,
#  roller carriers 50 mm short of the header on 14 automatic sliding doors)
# ---------------------------------------------------------------------------
def _gear_fixture(tmp_path, wheel_z, name):
    """A leaf on a slide joint whose only running gear is a wheel at ``wheel_z``, under a rail at z = 2.0."""
    pz, ph = (0.88 + wheel_z - 0.008) / 2, (wheel_z - 0.008 - 0.88) / 2      # carrier plate: slab top -> wheel
    bodies = (f'<body name="leaf" pos="0 -0.2 1"><joint name="j" type="slide" axis="1 0 0"/>'
              f'<geom name="leaf_slab" type="box" size="0.4 0.02 0.9"/>'
              f'<geom name="leaf_carrier_0" type="box" size="0.02 0.005 {ph}" pos="0 0 {pz}"/>'
              f'<geom name="leaf_carrier_wheel_0" type="cylinder" size="0.014 0.006" pos="0 0 {wheel_z}" euler="90 0 0"/></body>')
    world = '<geom name="rail" type="box" size="1.0 0.075 0.03" pos="0 -0.125 2.03"/>'
    return _door(tmp_path, world=world, bodies=bodies, name=name,
                 ir_bodies=[_ir("world_env", geoms=[("floor", "box", (3, 3, 0.05), (0, 0, -0.05), "floor"),
                                                    ("wall", "box", (1, 0.05, 1.0), (0, 0, 1.0), "wall"),
                                                    ("rail", "box", (1.0, 0.075, 0.03), (0, -0.125, 2.03), "track")]),
                            _ir("leaf", joint=_jt("j", "slide", (1, 0, 0), (0.0, 0.8), "primary"),
                                geoms=[("leaf_slab", "box", (0.4, 0.02, 0.9), (0, 0, 0), "leaf"),
                                       ("leaf_carrier_0", "box", (0.02, 0.005, ph), (0, 0, pz), "track"),
                                       ("leaf_carrier_wheel_0", "cylinder", (0.014, 0.006), (0, 0, wheel_z), "track")], mass=30.0)])


def test_running_gear_that_lands_on_its_rail_passes(tmp_path):
    """The wheel's top (1 + 0.986 + 0.014 = 2.000) meets the rail's underside at 2.000."""
    res = run_attachment(_gear_fixture(tmp_path, 0.986, "gear_ok"))
    assert res["ok"], res["findings"]


def test_running_gear_that_ends_in_mid_air_is_caught(tmp_path):
    """RULE running_gear_lands: the carrier stops 60 mm short of the rail.

    body_detached alone cannot see this - the leaf slab is 20 mm from the wall, inside the running-fit tolerance -
    which is exactly how 28 carrier wheels on 14 automatic sliding doors were built up to a SIDELITE's header
    700 mm to one side and ended 50 mm from anything."""
    res = run_attachment(_gear_fixture(tmp_path, 0.926, "gear_bad"))
    assert not res["ok"] and res["by_rule"].get("running_gear_lands") == 1, res["findings"]
    f = next(x for x in res["findings"] if x["rule"] == "running_gear_lands")
    assert f["gap"] == pytest.approx(0.060, abs=1e-3) and f["tolerance_m"] == TOL_BODY_GUIDED


def test_a_body_near_a_rail_still_has_to_touch_something(tmp_path):
    """The guided tolerance is earned by the BODY, not by its neighbour.

    A PVC strip hangs on a clamp bolted to its rail; it does not "run" on anything.  While the rail's own semantic
    bought the 12 mm, every strip on all 80 strip-curtain doors hung 6 mm under the rail with nothing between."""
    bodies = ('<body name="strip_0" pos="0 -0.2 2.0"><joint name="s" type="hinge" axis="1 0 0"/>'
              '<geom name="strip_0_geom" type="box" size="0.1 0.002 0.9" pos="0 0 -0.906"/></body>')
    world = '<geom name="hanger_rail" type="box" size="1.0 0.075 0.03" pos="0 -0.125 2.03"/>'
    d = _door(tmp_path, world=world, bodies=bodies, name="strip_bad",
              ir_bodies=[_ir("world_env", geoms=[("floor", "box", (3, 3, 0.05), (0, 0, -0.05), "floor"),
                                                 ("wall", "box", (1, 0.05, 1.0), (0, 0, 1.0), "wall"),
                                                 ("hanger_rail", "box", (1.0, 0.075, 0.03), (0, -0.125, 2.03), "track")]),
                         _ir("strip_0", joint=_jt("s", "hinge", (1, 0, 0), (-1.25, 1.25), "primary"),
                             geoms=[("strip_0_geom", "box", (0.1, 0.002, 0.9), (0, 0, -0.906), "leaf")], mass=1.0)])
    res = run_attachment(d)
    assert not res["ok"] and "body_detached" in res["by_rule"], res["findings"]
    assert res["findings"][0]["tolerance_m"] == TOL_BODY


def test_dataset_running_gear_all_lands():
    """Every body in the shipped dataset that carries running gear reaches the structure it rides on."""
    man = json.load(open(os.path.join(ASSETS, "manifest.json"))) if os.path.exists(os.path.join(ASSETS, "manifest.json")) else None
    if man is None:
        pytest.skip("no generated assets")
    bad = []
    for did in [d["id"] for d in man["doors"] if d["family"] in ("strip_curtain", "bifold", "accordion", "automatic_sliding", "revolving", "ship_watertight")]:
        res = run_attachment(os.path.join(ASSETS, "doors", did))
        if not res["ok"]:
            bad.append((did, res["by_rule"]))
    assert not bad, bad
