"""Future native paddle correction; all exports and hardware stay in tmp_path."""
import itertools
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from doorbench import hardware as H, physics as P
from doorbench.build import build_model, export_door
from doorbench.export.mjcf import build_mjcf
from doorbench.export.urdf import build_urdf
from doorbench.geometry import common as C
from doorbench.ir import Body, Model, Site, quat_from_axis_angle, quat_to_mat


def _fixture(operator, u=1, v=1, faces=(-1, 1), backlash=None):
    ir = Model("paddle_fixture")
    mat = C.mat_from_material(ir, "stainless", "plate")
    leaf = ir.add_body(Body("leaf", None, static=True))
    leaf.geoms.append(C.box("slab", (0, 0, 1), (0.5, 0.022, 1), mat))
    spec = {"latch": {"model": "tubular_spring"}}
    # Use a catalogue latch with a real 70 mm backset, without depending on
    # dataset ids or generating the shared dataset/mesh cache on disk.
    spec["latch"]["model"] = next(k for k, value in H.LATCHES.items() if value.backset == 0.07)
    primary = C.add_rotary_operator(ir, leaf, spec, {}, H.OPERATORS[operator], u, v,
                                    u * 0.4, 1.0, 0.044, list(faces), backlash)
    return ir, primary


def _compile(ir):
    mj = pytest.importorskip("mujoco")
    return mj.MjModel.from_xml_string(ET.tostring(build_mjcf(ir), encoding="unicode"))


def _face_coordinates(m, d, row):
    g, s = m.geom(row["geom"]).id, m.site(row["site"]).id
    matrix = d.geom_xmat[g].reshape(3, 3)
    return matrix.T @ (d.site_xpos[s] - d.geom_xpos[g]), m.geom_size[g]


@pytest.mark.parametrize("operator,u,v", list(itertools.product(
    ("paddle_push_pull", "paddle_hospital_arm"), (-1, 1), (-1, 1))))
def test_actual_surface_force_has_correct_moment_and_full_sweep_clears_leaf(operator, u, v):
    mj = pytest.importorskip("mujoco")
    ir, primary = _fixture(operator, u, v)
    m, op = _compile(ir), H.OPERATORS[operator]
    d = mj.MjData(m)
    rows = ir.meta["paddle_mechanisms"][0]["faces"]
    fixed_ids = [g for g in range(m.ngeom) if "backplate" in (m.geom(g).name or "")]
    mj.mj_forward(m, d)
    fixed_start = d.geom_xpos[fixed_ids].copy()
    for angle in np.linspace(0, op.travel, 65):
        for row in rows:
            d.qpos[m.jnt_qposadr[m.joint(row["joint"]).id]] = angle
        mj.mj_forward(m, d)
        np.testing.assert_array_equal(d.geom_xpos[fixed_ids], fixed_start)
        for row in rows:
            g, s, j = m.geom(row["geom"]).id, m.site(row["site"]).id, m.joint(row["joint"]).id
            local, half = _face_coordinates(m, d, row)
            np.testing.assert_allclose(local, (0, -v * half[1], 0), atol=1.2e-6)
            # Site local +Z is the surface outward normal, including both the
            # authored rest lean and the live hinge transform.
            outward = d.site_xmat[s].reshape(3, 3)[:, 2]
            np.testing.assert_allclose(outward, d.geom_xmat[g].reshape(3, 3) @ (0, -v, 0), atol=1.2e-6)
            force = -outward * 22.2
            generalized = np.zeros(m.nv)
            mj.mj_applyFT(m, d, force, np.zeros(3), d.site_xpos[s], m.site_bodyid[s], generalized)
            assert generalized[m.jnt_dofadr[j]] == pytest.approx(22.2 * op.grip_offset, abs=3e-5)
            assert np.dot(force, (0, v, 0)) > 0  # pushes or pulls in opening direction
            for signs in itertools.product((-1, 1), repeat=3):
                corner = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ (half * signs)
                assert row["face"] * corner[1] - 0.022 >= 0.01199
    assert primary.joint.name == "handle_hinge"
    assert sum(j.robot_interactive for j in ir.joints()) == 1


def test_old_axis_and_off_surface_site_are_detected():
    mj = pytest.importorskip("mujoco")
    ir, _ = _fixture("paddle_push_pull")
    m = _compile(ir)
    row = ir.meta["paddle_mechanisms"][0]["faces"][0]
    j, s = m.joint(row["joint"]).id, m.site(row["site"]).id
    m.jnt_axis[j] = (0, -1, 0)  # former spindle-normal axis
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    generalized = np.zeros(m.nv)
    mj.mj_applyFT(m, d, -d.site_xmat[s].reshape(3, 3)[:, 2], np.zeros(3),
                  d.site_xpos[s], m.site_bodyid[s], generalized)
    assert abs(generalized[m.jnt_dofadr[j]]) < 1e-12
    m.site_pos[s, 0] += 0.042  # original anchor lay 10 mm outside 32 mm half-width
    mj.mj_forward(m, d)
    local, half = _face_coordinates(m, d, row)
    assert abs(local[0]) - half[0] == pytest.approx(0.010)


def test_mjcf_preserves_static_and_body_site_orientations():
    mj = pytest.importorskip("mujoco")
    ir, primary = _fixture("paddle_push_pull")
    quaternion = tuple(quat_from_axis_angle((1, 2, 3), 0.73))
    ir.body("leaf").sites.append(Site("fixed_frame", (0, 0, 1), quaternion))
    primary.sites.append(Site("moving_frame", (0.01, 0.02, -0.03), quaternion))
    m = _compile(ir)
    d = mj.MjData(m)
    d.qpos[m.jnt_qposadr[m.joint(primary.joint.name).id]] = 0.17
    mj.mj_forward(m, d)
    np.testing.assert_allclose(d.site_xmat[m.site("fixed_frame").id].reshape(3, 3),
                                quat_to_mat(quaternion), atol=1.2e-6)
    np.testing.assert_allclose(d.site_xmat[m.site("moving_frame").id].reshape(3, 3),
                                d.xmat[m.body(primary.name).id].reshape(3, 3) @ quat_to_mat(quaternion), atol=1.2e-6)


@pytest.mark.parametrize("actuated_face", (-1, 1))
def test_either_face_drives_the_coupled_rockers_and_returns(actuated_face):
    mj = pytest.importorskip("mujoco")
    ir, _ = _fixture("paddle_push_pull")
    m = _compile(ir)
    m.opt.gravity[:] = 0  # isolate prescribed face wrench, return spring and cam constraint
    d = mj.MjData(m)
    rows = ir.meta["paddle_mechanisms"][0]["faces"]
    row = next(row for row in rows if row["face"] == actuated_face)
    sid = m.site(row["site"]).id
    addresses = [m.jnt_qposadr[m.joint(row["joint"]).id] for row in rows]
    for _ in range(600):
        mj.mj_forward(m, d)
        d.qfrc_applied[:] = 0
        mj.mj_applyFT(m, d, -60 * d.site_xmat[sid].reshape(3, 3)[:, 2], np.zeros(3),
                      d.site_xpos[sid], m.site_bodyid[sid], d.qfrc_applied)
        mj.mj_step(m, d)
    assert min(d.qpos[addresses]) > 0.35
    # MuJoCo equality constraints are soft; bound the induced grip mismatch
    # to 0.2 mm even while overloading the end stop with 60 N.
    assert H.OPERATORS["paddle_push_pull"].grip_offset * abs(np.diff(d.qpos[addresses])[0]) < 0.0002
    d.qfrc_applied[:] = 0
    for _ in range(1500):
        mj.mj_step(m, d)
    assert np.max(np.abs(d.qpos[addresses])) < 0.001


@pytest.mark.parametrize("faces,backlash", [((-1,), None), ((1,), 0.025), ((-1, 1), 0.025)])
def test_face_selection_locked_range_and_coupling_exports(faces, backlash):
    ir, primary = _fixture("paddle_push_pull", faces=faces, backlash=backlash)
    xml, urdf = build_mjcf(ir), build_urdf(ir)
    assert len(ir.equalities) == len(faces) - 1
    assert primary.joint.range[1] == pytest.approx(backlash or 0.4)
    if len(faces) == 2:
        eq = ir.equalities[0]
        exported = xml.find(f"equality/joint[@name='{eq.name}']")
        assert exported is not None and exported.get("joint1") == eq.a and exported.get("joint2") == eq.b
        assert list(map(float, exported.get("polycoef").split())) == [0, 1, 0, 0, 0]
        mimic = urdf.find(f"joint[@name='{eq.a}']/mimic")
        assert mimic is not None and mimic.get("joint") == eq.b and mimic.get("multiplier") == "1"
        follower = next(b for b in ir.bodies if b.joint and b.joint.name == eq.a)
        assert follower.joint.role == "mechanism" and not follower.joint.robot_interactive
        # One set return spring, not duplicated on the cam follower.
        assert follower.joint.stiffness == follower.joint.frictionloss == 0


def test_full_usd_preserves_cam_and_contact_frames(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    from doorbench.export.usd import write_usd
    ir, _ = _fixture("paddle_hospital_arm")
    path = write_usd(ir, str(tmp_path / "door"), str(tmp_path / "hardware"))
    stage = Usd.Stage.Open(path)
    eq = ir.equalities[0]
    joint = next(p for p in stage.Traverse() if p.GetName() == eq.a)
    relation = next(r for r in joint.GetRelationships() if r.GetName().endswith(":referenceJoint"))
    assert relation.GetTargets()[0].name == eq.b
    gearing = next(a for a in joint.GetAttributes() if a.GetName().endswith(":gearing"))
    assert gearing.Get() == -1.0
    for body in ir.bodies:
        for site in body.sites:
            prim = next(p for p in stage.Traverse() if p.GetName() == site.name)
            orient = next(o for o in UsdGeom.Xformable(prim).GetOrderedXformOps()
                          if o.GetOpType() == UsdGeom.XformOp.TypeOrient).Get()
            value = [orient.GetReal(), *orient.GetImaginary()]
            assert abs(np.dot(value, site.quat)) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("door_id", ("db0074_swing_single", "db0116_swing_single", "db0347_swing_single"))
def test_real_fixture_export_retains_latch_and_surface_mechanics(tmp_path, door_id):
    mj = pytest.importorskip("mujoco")
    repo = Path(__file__).resolve().parents[1]
    source = repo / "assets" / "doors" / door_id / "spec.json"
    before = source.read_bytes()
    spec = json.loads(before)
    exported = export_door(spec, str(tmp_path / "doors"), str(tmp_path / "hardware"),
                           formats=("mjcf", "urdf", "json"))
    ir = build_model(spec)
    descriptor = ir.meta["paddle_mechanisms"][0]
    primary = descriptor["primary_joint"]
    m = mj.MjModel.from_xml_path(exported["files"]["mjcf"]["full"])
    assert primary == "leaf_handle_hinge"
    assert any(primary in [j for j, _ in tendon.sites] for tendon in ir.tendons)
    assert any(eq.b == primary and "paddle_cam" in eq.name for eq in ir.equalities)
    # A forward force at a real exported paddle site has positive joint moment.
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    for row in descriptor["faces"]:
        s, j = m.site(row["site"]).id, m.joint(row["joint"]).id
        qforce = np.zeros(m.nv)
        mj.mj_applyFT(m, d, -d.site_xmat[s].reshape(3, 3)[:, 2], np.zeros(3),
                      d.site_xpos[s], m.site_bodyid[s], qforce)
        assert qforce[m.jnt_dofadr[j]] == pytest.approx(H.OPERATORS[spec["operator"]["model"]].grip_offset, abs=2e-6)
    assert source.read_bytes() == before
    assert all(str(tmp_path) in str(Path(exported["files"]["mjcf"][tier]).resolve()) for tier in ("full", "simple", "minimal"))


@pytest.mark.parametrize("operator", ("paddle_push_pull", "paddle_hospital_arm"))
def test_force_estimate_does_not_become_zero_after_axis_metadata_change(operator):
    repo = Path(__file__).resolve().parents[1]
    spec = json.loads((repo / "assets/doors/db0074_swing_single/spec.json").read_text())
    spec["operator"]["model"] = operator
    op = H.OPERATORS[operator]
    force = P.derive(spec)["compliance"]["operator_force_N"]
    assert op.motion == "rotate_horizontal"
    assert force == pytest.approx((max(op.spring_torque_preload, 1.5) + op.spring_rate * op.travel) / op.grip_offset)
    assert force > 0
