"""The enclosure gates: ``checks["guided_travel"]`` and ``checks["wall_opening"]`` (doorbench/enclosure_qa.py).

Both exist because of what the 2026-09 vision review saw on doors that passed every other gate:

  * 15 of 15 roll-up curtains rose as a rigid slab, left the top of their side guides at the opening head and
    ended up hanging 2.1-3.6 m in the air above the wall, with sky between the curtain and the building;
  * 18 of 18 sectional garage doors cut the wall away over the door's whole lift envelope so that the leaf could
    travel inside the wall plane, leaving a 2.0-2.5 m hole above the door, open to the sky;
  * 7 of them hung the opener's motor on the end of a 2.9 m rail cantilevered into a scene with no ceiling.

The tests are in three groups: the real dataset must pass (every vertical-lift door, every family's wall), the
mutations that reproduce each defect must FAIL (a truncated guide, an envelope wider than its hardware, the old
lift-envelope hole in the wall), and the mechanism itself must work (the curtain shortens as it coils, the rollers
stay in their track through the travel, the door reaches the end of its travel under a lift force).

Run:  pytest -q tests/test_enclosure_gates.py
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from doorbench.build import export_door
from doorbench.enclosure_qa import OPENING_TOL, _world_aabb, run_guided_travel_qa, run_wall_opening_qa
from doorbench.geometry.other import GARAGE_JAMB_T
from doorbench.spec import generate_all

VERTICAL = ("rollup", "garage_sectional")


@pytest.fixture(scope="module")
def specs():
    return {s["id"]: s for s in generate_all()}


@pytest.fixture(scope="module")
def exported(tmp_path_factory, specs):
    """Every roll-up and sectional door, plus one door of each other walled family, built once."""
    mujoco = pytest.importorskip("mujoco")
    root = tmp_path_factory.mktemp("enclosure")
    chosen, seen = [], set()
    for s in specs.values():
        if s["family"] in VERTICAL:
            chosen.append(s)
        elif s["family"] not in seen:
            seen.add(s["family"])
            chosen.append(s)
    out = {}
    for s in chosen:
        r = export_door(s, str(root / "doors"), str(root / "hardware"), formats=("mjcf", "json"))
        meta = json.loads((root / "doors" / s["id"] / "model.json").read_text())["meta"]
        out[s["id"]] = (mujoco.MjModel.from_xml_path(r["files"]["mjcf"]["full"]), meta, s,
                        r["files"]["mjcf"]["full"])
    return out


def _vertical(exported):
    return {k: v for k, v in exported.items() if v[2]["family"] in VERTICAL}


# ---------------------------------------------------------------------------
# the dataset passes
# ---------------------------------------------------------------------------
def test_every_vertical_lift_door_declares_and_passes_guided_travel(exported):
    doors = _vertical(exported)
    assert len(doors) >= 30, sorted(doors)
    for name, (model, meta, spec, _) in doors.items():
        r = run_guided_travel_qa(model, meta)
        assert r["declared"], name
        assert r["ok"], (name, r["failures"])
        assert r["n_geoms"] >= 4 and r["n_zones"] >= 1, (name, r)


def test_no_other_family_accidentally_declares_a_guide_envelope(exported):
    for name, (model, meta, spec, _) in exported.items():
        if spec["family"] in VERTICAL:
            continue
        assert not meta.get("guided_travel"), name


def test_the_wall_shows_the_declared_opening_on_every_family(exported):
    for name, (model, meta, spec, _) in exported.items():
        r = run_wall_opening_qa(model, meta, spec)
        assert r["ok"], (name, r)
        if not r.get("declared") or not r["measured_opening"]:
            continue
        want, got = r["expected_opening"], r["measured_opening"]
        assert got[1] - got[0] <= (want[1] - want[0]) + 2 * OPENING_TOL, (name, r)
        assert got[3] <= want[3] + OPENING_TOL, (name, r)


def test_sectional_wall_opening_is_the_door_opening_not_the_lift_envelope(exported):
    """The defect was a hole Ho + Hh + 0.08 high; the wall now shows Ho + the jamb lining."""
    n = 0
    for name, (model, meta, spec, _) in exported.items():
        if spec["family"] != "garage_sectional":
            continue
        r = run_wall_opening_qa(model, meta, spec)
        Ho, Hh = spec["opening"]["height"], spec["leaf"]["height"]
        assert r["ok"] and r["measured_opening"], (name, r)
        # the measured hole is the finished opening, not the lift envelope (Ho + Hh + 0.08)
        assert r["measured_opening"][3] <= Ho + GARAGE_JAMB_T + 0.03, (name, r["measured_opening"])
        assert r["measured_opening"][3] < Ho + Hh - 1.0, (name, r["measured_opening"])
        n += 1
    assert n >= 15


# ---------------------------------------------------------------------------
# the mutations that reproduce each defect must fail
# ---------------------------------------------------------------------------
def test_a_curtain_that_rises_out_of_its_guides_is_caught(exported):
    """The original roll-up defect: guides that stop at the head, and a curtain that carries on past them."""
    name, (model, meta, spec, _) = next((k, v) for k, v in sorted(exported.items()) if v[2]["family"] == "rollup")
    bad = copy.deepcopy(meta)
    bad["guided_travel"]["zones"] = [z for z in bad["guided_travel"]["zones"] if "hood" not in z["label"]]
    r = run_guided_travel_qa(model, bad)
    assert not r["ok"], (name, r)
    assert any(f["check"] == "outside_guides" for f in r["failures"]), r["failures"]
    assert r["worst_excursion_m"] > 0.05, r


def test_an_envelope_wider_than_the_hardware_that_backs_it_is_rejected(exported):
    """A declared envelope is only as big as the guide that makes it: no declaring your way out of the gate."""
    name, (model, meta, spec, _) = next((k, v) for k, v in sorted(exported.items()) if v[2]["family"] in VERTICAL)
    bad = copy.deepcopy(meta)
    bad["guided_travel"]["zones"][0]["x"] = [bad["guided_travel"]["zones"][0]["x"][0] - 0.5,
                                             bad["guided_travel"]["zones"][0]["x"][1] + 0.5]
    r = run_guided_travel_qa(model, bad)
    assert not r["ok"], (name, r)
    assert [f for f in r["failures"] if f["check"] == "zone_face"], r["failures"]


def test_an_envelope_backed_by_nothing_is_rejected(exported):
    name, (model, meta, spec, _) = next((k, v) for k, v in sorted(exported.items()) if v[2]["family"] in VERTICAL)
    bad = copy.deepcopy(meta)
    bad["guided_travel"]["zones"][0]["backed_by"] = ["no_such_geom"]
    r = run_guided_travel_qa(model, bad)
    assert not r["ok"] and any(f["check"] == "zone_unbacked" for f in r["failures"]), r


def test_the_old_lift_envelope_hole_in_the_wall_is_caught(exported):
    """Put the header back where it was - at the top of the wall - and the gate must see the 2 m hole."""
    mujoco = pytest.importorskip("mujoco")
    name, (model, meta, spec, path) = next((k, v) for k, v in sorted(exported.items())
                                           if v[2]["family"] == "garage_sectional")
    ms = mujoco.MjSpec.from_file(path)
    Ho, Hh = spec["opening"]["height"], spec["leaf"]["height"]
    moved = False
    for g in ms.geoms:
        if g.name == "wall_header":
            top = float(g.pos[2]) + float(g.size[2])
            bottom = Ho + Hh + 0.08                       # the hole the old builder cut for the lift envelope
            g.pos = [float(g.pos[0]), float(g.pos[1]), (bottom + top) / 2]
            g.size = [float(g.size[0]), float(g.size[1]), max(0.01, (top - bottom) / 2)]
            moved = True
    assert moved, "no wall_header to move"
    r = run_wall_opening_qa(ms.compile(), meta, spec)
    assert not r["ok"], r
    assert r["stray_open_area_m2"] > 1.0, r
    assert r["worst_stray_point"][1] > Ho + 1.0, r


def test_a_wall_opening_may_only_be_declared_by_an_exempt_family(exported):
    """meta["wall_opening"] is honoured for the two families that document why; everywhere else it is ignored."""
    from doorbench.enclosure_qa import WALL_OPENING_EXEMPT

    name, (model, meta, spec, _) = next((k, v) for k, v in sorted(exported.items())
                                        if v[2]["family"] == "garage_sectional")
    assert spec["family"] not in WALL_OPENING_EXEMPT
    lying = copy.deepcopy(meta)
    lying["wall_opening"] = [-9.0, 9.0, 0.0, 9.0]          # "the whole wall is the opening"
    r = run_wall_opening_qa(model, lying, spec)
    assert not r["exempt"] and r["expected_opening"][3] == pytest.approx(spec["opening"]["height"]), r
    assert r["ok"], r                                       # the real wall is still right; the lie changed nothing
    for fam in WALL_OPENING_EXEMPT:
        assert any(v[2]["family"] == fam for v in exported.values()), fam
        m2, meta2, spec2, _ = next(v for v in exported.values() if v[2]["family"] == fam)
        assert meta2.get("wall_opening"), f"{fam} must declare the passage it leaves open"
        assert run_wall_opening_qa(m2, meta2, spec2)["exempt"] is True


# ---------------------------------------------------------------------------
# the mechanism itself
# ---------------------------------------------------------------------------
def _assembly_span(model, data, meta, mujoco):
    """(lowest, highest) z of the moving assembly - the exposed curtain, when it is a curtain."""
    ids = [g for g in range(model.ngeom)
           if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) in
           set(meta["guided_travel"]["bodies"])]
    boxes = [_world_aabb(model, data, g) for g in ids]      # rotation-aware: a grille rod's local z runs along x
    return min(float(b[0][2]) for b in boxes), max(float(b[1][2]) for b in boxes)


def test_the_curtain_shortens_as_it_coils(exported):
    """A coiling curtain's exposed length falls by the distance travelled; a rigid slab's does not."""
    mujoco = pytest.importorskip("mujoco")
    from doorbench.clearance import resolve_joint_equalities

    n = 0
    for name, (model, meta, spec, _) in sorted(exported.items()):
        if spec["family"] != "rollup":
            continue
        data = mujoco.MjData(model)
        j = model.joint(meta["primary_joint"]).id
        travel = float(model.jnt_range[j][1])
        if travel < 0.1:
            continue                                        # locked shut by an engaged slide lock
        spans = {}
        for frac in (0.0, 1.0):
            q = model.qpos0.copy()
            q[model.jnt_qposadr[j]] = frac * travel
            for _ in range(2):
                resolve_joint_equalities(model, q, mujoco)
                q[model.jnt_qposadr[j]] = frac * travel
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            spans[frac] = _assembly_span(model, data, meta, mujoco)
        Hh = spec["leaf"]["height"]
        shut, open_ = spans[0.0], spans[1.0]
        assert shut[1] - shut[0] >= 0.9 * Hh, (name, shut)           # shut: the full curtain hangs in the opening
        exposed = open_[1] - open_[0]
        assert exposed <= (Hh - travel) + Hh / 4 + 0.10, (name, exposed, Hh, travel)
        assert open_[0] >= travel - 0.02, (name, open_)              # the bottom bar is up at the head
        n += 1
    assert n >= 10


def test_sectional_rollers_stay_in_their_track_through_the_travel(exported):
    """Every roller runs inside the C-channel - between the flanges and inboard of the web - at every q."""
    mujoco = pytest.importorskip("mujoco")
    n = 0
    for name, (model, meta, spec, _) in sorted(exported.items()):
        if spec["family"] != "garage_sectional":
            continue
        gname = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)]
        data = mujoco.MjData(model)
        j = model.joint(meta["primary_joint"]).id
        travel = float(model.jnt_range[j][1])
        rollers = {s: [g for g in range(model.ngeom) if (gname[g] or "").startswith(f"roller_{s}_")] for s in "lr"}
        webs = {s: gname.index(f"track_{s}") for s in "lr"}
        flanges = {s: [gname.index(f"track_{s}_flange_{p}") for p in "pn"] for s in "lr"}
        assert all(rollers[s] for s in "lr"), name
        for frac in np.linspace(0.0, 1.0, 9):
            data.qpos[:] = model.qpos0
            data.qpos[model.jnt_qposadr[j]] = frac * travel
            mujoco.mj_forward(model, data)
            for s in "lr":
                web = data.geom_xpos[webs[s]]
                wz = model.geom_size[webs[s]]
                fy = sorted(float(data.geom_xpos[f][1]) for f in flanges[s])
                for g in rollers[s]:
                    p, r = data.geom_xpos[g], float(model.geom_size[g][0])
                    assert abs(float(p[1]) - float(web[1])) + r <= 0.033 + 0.006, (name, gname[g], frac)
                    assert fy[0] < float(p[1]) < fy[1], (name, gname[g], frac)
                    assert abs(float(p[2]) - float(web[2])) + r <= float(wz[2]), (name, gname[g], frac)
        n += 1
    assert n >= 15


def test_every_vertical_lift_door_reaches_the_end_of_its_travel(exported):
    """A lift force takes the door all the way up.

    The gate that fires here is not kinematic: a RIGID header stop moulding cleared every clearance sweep (the
    exterior lift handle sweeping it is on the clearance allow-list, because a vinyl header seal is a compliant
    part) and still stopped the door dead at 64 % of its travel in simulation, on its own handle.
    """
    mujoco = pytest.importorskip("mujoco")
    from doorbench.clearance import Clearance   # noqa: F401  (documents where released_qpos lives)

    n, stuck = 0, []
    for name, (model, meta, spec, path) in sorted(exported.items()):
        if spec["family"] not in VERTICAL:
            continue
        data = mujoco.MjData(model)
        j = model.joint(meta["primary_joint"]).id
        travel = float(model.jnt_range[j][1])
        if travel < 0.1:
            continue                              # an engaged lock pins the joint: that door is meant not to move
        # release every lock / mechanism joint (an engaged slide lock really does hold the door)
        q = model.qpos0.copy()
        for k in range(model.njnt):
            if model.jnt_limited[k] and k != j and float(model.jnt_range[k][1] - model.jnt_range[k][0]) > 0.006:
                q[model.jnt_qposadr[k]] = float(model.jnt_range[k][1])
        data.qpos[:] = q
        force = max(400.0, 12.0 * float(sum(model.body_mass)))
        adr = int(model.jnt_dofadr[j])
        for _ in range(4000):
            data.qfrc_applied[adr] = force
            mujoco.mj_step(model, data)
        reached = float(data.qpos[model.jnt_qposadr[j]]) / travel
        if reached < 0.95:
            stuck.append((name, round(reached, 3)))
        n += 1
    assert not stuck, stuck
    assert n >= 25


def test_no_opener_is_cantilevered_into_the_air(exported):
    """Finding 11: the motor hung on the end of a 2.9 m rail in a scene with no ceiling to hang it from.

    A jackshaft opener bolts to the wall beside the shaft; nothing it is made of reaches far off that wall.
    """
    mujoco = pytest.importorskip("mujoco")
    n = 0
    for name, (model, meta, spec, _) in sorted(exported.items()):
        if spec["family"] != "garage_sectional" or spec["kinematics"].get("opener", "none_manual") == "none_manual":
            continue
        wall_face = spec["opening"]["wall_thickness"] / 2
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        reach = 0.0
        for g in range(model.ngeom):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if not nm.startswith("opener"):
                continue
            reach = max(reach, float(data.geom_xpos[g][1] + model.geom_aabb[g, 4]) - wall_face)
        assert 0.0 < reach <= 0.5, (name, reach)
        n += 1
    assert n >= 4
