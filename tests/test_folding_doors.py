"""Folding doors (bifold / accordion): the zigzag mechanism must be consistent and must actually fold.

Regression for the 12 accordion doors that shipped kinematically locked (2026-09): every even panel hinge was
authored with range [-pi, 0] while its track coupling drove it to +2 q_pivot, so the fold sat on a joint limit that
the equality pushed against and did not move under any push (the QA never pushed free-swing families, and the
clearance sweep applies couplings without checking ranges).  Also covers the header clearance (panels used to touch
the head jamb and the track sat inside it) and the face-hinge stacking geometry.
"""
import json
import math

import numpy as np
import pytest

from doorbench.spec import generate_all
from doorbench.build import build_model, export_door
from doorbench import folding as F
from doorbench.clearance import coupling_range_failures
from doorbench import qa as QA


@pytest.fixture(scope="module")
def folding_specs():
    return [s for s in generate_all() if s["family"] in ("bifold", "accordion")]


def _poly(c, x):
    return sum(c[k] * x ** k for k in range(len(c)))


def test_fold_couplings_stay_inside_driven_ranges(folding_specs):
    """The image of the pivot range under every panel coupling lies inside the driven hinge's own range (IR level)."""
    assert len(folding_specs) == 42
    for s in folding_specs:
        model = build_model(s)
        joints = {b.joint.name: b.joint for b in model.bodies if b.joint is not None}
        couplings = {e.a: e for e in model.equalities if e.kind == "joint"}
        driven = [n for n, j in joints.items() if n.startswith("panel_") and j.role == "secondary"]
        assert driven and set(driven) == set(couplings), s["id"]
        for name in driven:
            e = couplings[name]
            drv, drn = joints[e.b], joints[name]
            assert drv.role == "primary" and drv.range[0] == 0.0 and drv.range[1] == pytest.approx(math.radians(85))
            lo, hi = drn.range
            for x in np.linspace(drv.range[0], drv.range[1], 25):
                y = _poly(e.polycoeff, x)
                assert lo - 1e-9 <= y <= hi + 1e-9, (s["id"], name, x, y, drn.range)
            k = int(name.split("_")[2])
            assert e.polycoeff[1] == F.fold_coupling(k) and tuple(drn.range) == F.fold_hinge_range(k)


def test_fold_geometry_clears_track_header_and_stacks_face_to_face(folding_specs):
    for s in folding_specs:
        W, Hh, t = s["leaf"]["width"], s["leaf"]["height"], s["leaf"]["thickness"]
        Ho = s["opening"]["height"]
        # panels hang below a 30 mm track mounted under the head jamb, 20 mm off the floor
        assert Ho >= F.FOLD_FLOOR_GAP + Hh + F.FOLD_TRACK_GAP + F.FOLD_TRACK_H - 1e-9, s["id"]
        model = build_model(s)
        tracks = [g for b in model.bodies for g in b.geoms
                  if g.name == "fold_track" or g.name.startswith("fold_track_")]
        assert tracks, s["id"]
        track_bottom = min(g.pos[2] - g.size[2] for g in tracks)
        assert track_bottom >= F.FOLD_FLOOR_GAP + Hh + F.FOLD_TRACK_GAP - 1e-9
        assert max(g.pos[2] + g.size[2] for g in tracks) <= Ho + 1e-9
        for b in model.bodies:
            if b.joint is None or not b.name.startswith("panel_") or b.joint.role != "secondary":
                continue
            k = int(b.name.split("_")[2])
            # face hinges: the axis lies on the face the pair closes onto, alternating between the two faces
            assert abs(abs(b.joint.pos[1]) - t / 2) < 1e-9, (s["id"], b.name, b.joint.pos)
            assert math.copysign(1, b.joint.pos[1]) == (1.0 if k % 2 == 1 else -1.0)
            # A louvered leaf consists of actual stiles, rails and angled
            # slats, so validate every structural leaf piece rather than
            # requiring the obsolete solid-slab name.
            from doorbench.ir import quat_to_mat
            pieces = [g for g in b.geoms if g.semantic in ("leaf", "glass") and g.type == "box"]
            assert pieces, (s["id"], b.name)
            for piece in pieces:
                extent = np.abs(quat_to_mat(piece.quat)) @ np.asarray(piece.size)
                assert piece.pos[2] + extent[2] <= track_bottom - F.FOLD_TRACK_GAP + 1e-9
                assert piece.pos[2] - extent[2] >= F.FOLD_FLOOR_GAP - 1e-9


LOCKED_FOLD_XML = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="a"><joint name="pivot" type="hinge" axis="0 0 1" range="0 1.4835" limited="true"/><geom type="box" size="0.1 0.01 0.5" mass="1"/>
    <body name="b" pos="0.2 0 0"><joint name="fold" type="hinge" axis="0 0 1" range="{lo} {hi}" limited="true"/><geom type="box" size="0.1 0.01 0.5" mass="1"/></body>
  </body></worldbody>
  <equality><joint name="couple" joint1="fold" joint2="pivot" polycoef="0 2 0 0 0"/></equality></mujoco>"""


def test_coupling_range_gate_catches_a_locked_fold():
    """The deterministic gate: a driven joint whose coupling image leaves its own range is a locked mechanism."""
    mujoco = pytest.importorskip("mujoco")
    bad = mujoco.MjModel.from_xml_string(LOCKED_FOLD_XML.format(lo=-math.pi, hi=0.0))
    fails = coupling_range_failures(bad)
    assert len(fails) == 1 and fails[0]["driven"] == "fold" and fails[0]["driver"] == "pivot"
    assert fails[0]["overshoot"] == pytest.approx(2 * 1.4835, abs=1e-6) and fails[0]["equality"] == "couple"
    good = mujoco.MjModel.from_xml_string(LOCKED_FOLD_XML.format(lo=0.0, hi=math.pi))
    assert coupling_range_failures(good) == []
    # a small numerical slop is tolerated, a real conflict is not
    tight = mujoco.MjModel.from_xml_string(LOCKED_FOLD_XML.format(lo=0.0, hi=2 * 1.4835 - 0.0005))
    assert coupling_range_failures(tight) == []
    short = mujoco.MjModel.from_xml_string(LOCKED_FOLD_XML.format(lo=0.0, hi=2.0))
    assert len(coupling_range_failures(short)) == 1


def test_accordion_folds_under_the_qa_push_and_signs_off(tmp_path, folding_specs):
    mujoco = pytest.importorskip("mujoco")
    s = next(x for x in folding_specs if x["id"] == "db0177_accordion")
    out = export_door(s, str(tmp_path / "doors"), str(tmp_path / "hardware"), formats=("mjcf", "json"))
    door_dir = tmp_path / "doors" / s["id"]
    meta = json.load(open(door_dir / "model.json"))["meta"]
    phys = json.load(open(door_dir / "spec.json"))["physics"]
    m = mujoco.MjModel.from_xml_path(out["files"]["mjcf"]["full"])
    d = mujoco.MjData(m)
    pj = m.joint(meta["primary_joint"]).id
    assert coupling_range_failures(m) == []
    # the qa.py free-swing push: the fold must move past 10 deg and then reach its stack stop, with no contact force
    flags = QA.door_flags(s)
    assert flags["free_swing"] and not flags["has_holding"]
    push = QA.qa_push(m, d, pj, phys["mass"]["total_kg"], s["leaf"]["width"])["push"]     # the push qa.py actually applies to this leaf
    moved = QA.push_primary(m, d, pj, push, has_holding=False, thr_free=math.radians(10))
    assert moved > math.radians(10), moved
    for _ in range(1000):
        d.qfrc_applied[:] = 0
        d.qfrc_applied[m.jnt_dofadr[pj]] = push
        mujoco.mj_step(m, d)
    q0 = float(d.qpos[m.jnt_qposadr[pj]])
    assert q0 > math.radians(80) and d.ncon == 0, (q0, d.ncon)
    # the zigzag: hinge k follows -2 / +2 times the pivot
    n = s["leaf"]["count"]
    for k in range(1, n):
        qk = float(d.qpos[m.jnt_qposadr[m.joint(f"panel_0_{k}_hinge").id]])
        assert qk == pytest.approx(F.fold_coupling(k) * q0, abs=0.02), (k, qk, q0)
    # lead edge excursion along the track: the face-hinged chain first lengthens; the closed lead gap swallows it
    Wo = s["opening"]["width"]
    # A real recessed pull splits the slab collision volume around its cup.
    # Measure the complete leading panel envelope across every retained piece.
    prefix=f"panel_0_{n - 1}_slab"
    lead=[g for g in range(m.ngeom) if m.geom(g).name.startswith(prefix)]
    assert lead
    mujoco.mj_resetData(m, d)
    xs = []
    for q in np.linspace(0.0, m.jnt_range[pj][1], 200):
        d.qpos[:] = 0.0
        d.qpos[m.jnt_qposadr[pj]] = q
        for k in range(1, n):
            d.qpos[m.jnt_qposadr[m.joint(f"panel_0_{k}_hinge").id]] = F.fold_coupling(k) * q
        mujoco.mj_forward(m, d)
        xs.append(max(d.geom_xpos[g][0] + (d.geom_xmat[g].reshape(3,3) @ (np.array(c) * m.geom_size[g]))[0]
                      for g in lead for c in [(sx,sy,0.) for sx in (-1,1) for sy in (-1,1)]))
    gap_closed = Wo / 2 - xs[0]
    excursion = max(xs) - xs[0]
    assert excursion > 0.005, excursion                                   # the effect is real for an 8-panel stack
    assert excursion == pytest.approx(F.fold_lead_excursion(n, s["leaf"]["width"], s["leaf"]["thickness"]), abs=0.003)
    assert gap_closed - excursion >= 0.002, (gap_closed, excursion)        # never touches the strike jamb
    # and the sign-off QA agrees, with the new free_opens check present
    qa = QA.run_qa(s, str(door_dir), meta, out["files"], phys)
    assert qa["checks"]["free_opens"] is True and qa["checks"]["clearance"] is True, qa["checks"]
    assert "hold" not in qa["checks"] and qa["signed_off"], qa
    assert qa["metrics"]["hold_displacement"] > math.radians(10)


def test_free_swing_flags_drive_the_qa_expectation():
    """Free-swing families are pushed: a leaf nothing holds must open, a locked rotor / bolted flap must hold."""
    from doorbench.qa import door_flags, FREE_SWING_FAMILIES
    specs = {s["id"]: s for s in generate_all()}
    acc = specs["db0177_accordion"]
    assert acc["family"] in FREE_SWING_FAMILIES and door_flags(acc)["free_swing"] and not door_flags(acc)["has_holding"]
    pet = specs["db0892_pet_door"]        # slide bolt engaged: the flap is held
    assert door_flags(pet)["free_swing"] and door_flags(pet)["has_holding"]
