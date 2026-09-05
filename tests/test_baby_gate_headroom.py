"""The low gate must never acquire a solid wall across its overhead passage."""
import copy

import mujoco
import pytest

from doorbench.baby_gate_qa import run_baby_gate_qa
from doorbench.build import build_model, export_door
from doorbench.geometry.common import add_floor_and_wall
from doorbench.ir import Model
from doorbench.spec import generate_all


@pytest.fixture(scope="module")
def baby_specs():
    specs = [s for s in generate_all() if s["family"] == "baby_gate"]
    assert len(specs) == 10
    return specs


def test_all_baby_gates_keep_side_walls_without_header_in_every_export_tier(baby_specs, tmp_path):
    for spec in baby_specs:
        model = build_model(spec)
        names = {g.name for b in model.bodies for g in b.geoms}
        assert {"wall_left", "wall_right", "floor"} <= names
        assert "wall_header" not in names
        summary = export_door(spec, str(tmp_path / "doors"), str(tmp_path / "hardware"),
                              formats=("mjcf", "urdf", "json"))
        for path in summary["files"]["mjcf"].values():
            report = run_baby_gate_qa(mujoco.MjModel.from_xml_path(path), spec)
            assert report["ok"], (spec["id"], path, report)
        # Both visual and collider URDF geometry derive from this same IR.
        for path in summary["files"]["urdf"].values():
            assert "wall_header" not in open(path).read()


@pytest.mark.parametrize("colliding", [True, False])
def test_gate_rejects_renamed_old_header_even_if_visual_only(baby_specs, colliding):
    spec = baby_specs[0]
    report = run_baby_gate_qa(mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <geom name="unexpected_panel" type="box" pos="0 0 1.8" size=".5 .12 .8"
        contype="{int(colliding)}" conaffinity="{int(colliding)}"/>
    </worldbody></mujoco>'''), spec)
    assert not report["ok"]
    assert report["failures"][0]["geom"] == "unexpected_panel"


def test_ordinary_door_retains_lintel_and_gate_preserves_side_geometry(baby_specs):
    spec = copy.deepcopy(baby_specs[0])
    ordinary = copy.deepcopy(spec)
    ordinary["family"] = "swing_single"
    hole = (-.55, .55, 0, 1.0)
    gate_world = add_floor_and_wall(Model("gate"), spec, hole=hole)
    door_world = add_floor_and_wall(Model("door"), ordinary, hole=hole)
    expected = [g for g in door_world.geoms if g.name != "wall_header"]
    assert [g.name for g in gate_world.geoms] == [g.name for g in expected]
    for actual, previous in zip(gate_world.geoms, expected):
        assert actual.pos == previous.pos
        assert actual.size == previous.size
    assert any(g.name == "wall_header" for g in door_world.geoms)
    assert run_baby_gate_qa(None, ordinary) == {"ok": True, "applicable": False, "failures": []}
