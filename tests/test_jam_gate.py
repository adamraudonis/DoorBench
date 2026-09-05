"""The jam gate (qa.py ``free_opens`` / ``no_jam``): a door that nothing holds shut must move under the QA push, and no
static geometry may press on a moving part while it does.

Background: every revolving door's wing stiles ended exactly at the underside of the wall header (zero gap).  The
clearance gate only fails interpenetrations, so a coplanar touch passed it - yet its contact normal is orthogonal to
the rotor's only DOF, the constraint is degenerate and MuJoCo emitted 8-17 kN of normal force; friction on that stalled
10 of 15 rotors under the 66 N*m push.  These tests pin the geometric clearance of the rotor, the dynamic behaviour, and
that the gate flags the old geometry.

Run:  pytest -q tests/test_jam_gate.py      (~40 s)
"""
from __future__ import annotations

import json
import math
import os

import pytest

from doorbench.build import build_model, export_door
from doorbench.geometry.other import REVOLVING_RUN_CLEAR
from doorbench.qa import FREE_SWING_FAMILIES, JAM_FORCE_N, door_flags, jam_sweep, run_qa
from doorbench.spec import generate_all

JAMMED_BEFORE = ("db0779_revolving", "db0108_revolving")   # crawled 0.046 rad in 6 s / did not move at all before the fix


@pytest.fixture(scope="module")
def specs():
    return {s["id"]: s for s in generate_all()}


def _z_extent(g):
    """(bottom, top) of a box / cylinder geom in its body frame (rotor and world geoms here are only rotated about z)."""
    if g.type == "box":
        return g.pos[2] - g.size[2], g.pos[2] + g.size[2]
    if g.type == "cylinder":
        return g.pos[2] - g.size[1], g.pos[2] + g.size[1]
    return None


def test_revolving_rotor_runs_clear_of_ceiling_and_header(specs):
    """Geometric: the whole rotor envelope ends >= the running clearance below the canopy ceiling, the header sits above
    the canopy, and the bearing boss keeps most of that clearance to the shaft (no zero-gap touch anywhere above)."""
    n = 0
    for s in specs.values():
        if s["family"] != "revolving":
            continue
        model = build_model(s)
        rotor, world = model.body("rotor"), model.body("world_env")
        rotor_top = max(_z_extent(g)[1] for g in rotor.geoms if _z_extent(g))
        rotor_bot = min(_z_extent(g)[0] for g in rotor.geoms if _z_extent(g))
        canopy = next(g for g in world.geoms if g.name == "drum_canopy")
        header = next(g for g in world.geoms if g.name == "wall_header")
        boss = next(g for g in world.geoms if g.name == "rotor_top_bearing")
        pivot = next(g for g in world.geoms if g.name == "rotor_floor_pivot")
        assert _z_extent(canopy)[0] - rotor_top >= REVOLVING_RUN_CLEAR - 1e-9, s["id"]
        assert _z_extent(header)[0] >= _z_extent(canopy)[1] - 1e-9, (s["id"], "header must sit on the canopy, not at wing height")
        assert _z_extent(boss)[0] - rotor_top >= 0.008 - 1e-9, s["id"]
        assert rotor_bot - _z_extent(pivot)[1] >= 0.008 - 1e-9, s["id"]
        # wing tips brush the drum glass: outer face of the stile inside the glass inner face by a seal gap, never touching
        R = s["opening"]["drum_diameter"] / 2
        drum = [g for g in world.geoms if g.name.startswith("drum_") and g.type == "box"]
        r_glass_inner = min(math.hypot(g.pos[0], g.pos[1]) - g.size[1] for g in drum)
        assert 0.005 <= r_glass_inner - R <= 0.03, s["id"]
        n += 1
    assert n == 15


def _export_and_qa(spec, root):
    out = export_door(spec, os.path.join(root, "doors"), os.path.join(root, "hardware"), formats=("mjcf", "json"))
    dd = os.path.join(root, "doors", spec["id"])
    with open(os.path.join(dd, "model.json")) as f:
        meta = json.load(f)["meta"]
    with open(os.path.join(dd, "spec.json")) as f:
        phys = json.load(f)["physics"]
    return out, dd, meta, run_qa(spec, dd, meta, out["files"], phys)


@pytest.mark.parametrize("door_id", JAMMED_BEFORE)
def test_revolving_turns_freely_under_the_qa_push(tmp_path, specs, door_id):
    pytest.importorskip("mujoco")
    _, _, _, qa = _export_and_qa(specs[door_id], str(tmp_path))
    c, mt = qa["checks"], qa["metrics"]
    assert c["free_opens"] and c["no_jam"], (c, mt)
    assert mt["hold_displacement"] > math.radians(10) and mt["jam_t_free"] is not None and mt["jam_t_free"] <= 1.0, mt
    assert mt["jam_peak_force_N"] == 0.0, mt["jam_peak_pair"]     # nothing static touches the rotor at all
    assert c["clearance"] and qa["signed_off"], [k for k, v in c.items() if not v]


def test_jam_gate_flags_a_header_coplanar_with_the_wing_tops(tmp_path, specs):
    """Regression for the shipped defect: lower the header so its underside is coplanar with the wing stile tops (the
    old geometry).  The geometric clearance gate cannot see a zero-gap touch; the jam gate must."""
    mujoco = pytest.importorskip("mujoco")
    out, dd, meta, qa = _export_and_qa(specs["db0779_revolving"], str(tmp_path))
    ms = mujoco.MjSpec.from_file(out["files"]["mjcf"]["full"])
    header = next(g for g in ms.geoms if g.name == "wall_header")
    stile = next(g for g in ms.geoms if g.name == "wing_0_stile")
    header.pos[2] = (stile.pos[2] + stile.size[2]) + header.size[2]
    m = ms.compile()
    d = mujoco.MjData(m)
    pj = m.joint(meta["primary_joint"]).id
    jam = jam_sweep(m, d, pj, qa["metrics"]["qa_push"], math.radians(10))
    assert jam["peak_force_N"] > JAM_FORCE_N, jam
    assert "wall_header" in jam["peak_pair"] and any(n.startswith("wing_") for n in jam["peak_pair"]), jam
    # the clearance gate on the same geometry sees no interpenetration - which is why the jam gate exists
    from doorbench.clearance import run_clearance
    assert run_clearance(dd)["ok"]


def test_locked_rotor_holds_and_does_not_jam(tmp_path, specs):
    pytest.importorskip("mujoco")
    s = next(s for s in specs.values() if s["family"] == "turnstile_tripod" and s["kinematics"].get("locked_until_credential"))
    _, _, _, qa = _export_and_qa(s, str(tmp_path))
    c, mt = qa["checks"], qa["metrics"]
    assert "free_opens" not in c and c["locked_holds"] and c["no_jam"], (c, mt)
    assert mt["hold_displacement"] < math.radians(4) and mt["jam_peak_force_N"] < JAM_FORCE_N


def test_jam_gate_covers_every_free_swing_family(specs):
    fams = {s["family"] for s in specs.values()}
    assert set(FREE_SWING_FAMILIES) <= fams
    for s in specs.values():
        if s["family"] in FREE_SWING_FAMILIES:
            assert door_flags(s)["free_swing"]
