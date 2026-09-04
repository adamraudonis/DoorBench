import json
import math
import os

import pytest

from doorbench.spec import generate_all
from doorbench import physics as P, hardware as H, materials as M
from doorbench.build import build_model, export_door


@pytest.fixture(scope="module")
def specs():
    return generate_all()


def test_exactly_1000_unique(specs):
    assert len(specs) == 1000
    assert len({s["id"] for s in specs}) == 1000


def test_catalog_references(specs):
    for s in specs:
        assert s["operator"]["model"] in H.OPERATORS
        assert s["latch"]["model"] in H.LATCHES
        assert s["lock"]["model"] in H.LOCKS
        assert s["closer"]["model"] in H.CLOSERS
        assert s["hinge"]["model"] in H.HINGES
        assert s["leaf"]["slab"] in M.SLABS
        assert s["seal"] in H.SEALS


def test_physics_ranges(specs):
    for s in specs:
        p = P.derive(s)
        m = p["mass"]["total_kg"]
        assert 0.05 < m < 3000, (s["id"], m)
        if "hinge" in p:
            assert 0 <= p["hinge"]["coulomb_torque_Nm"] < 200
        if p["closer"]["kind"] != "none":
            assert p["closer"]["spring_preload_Nm"] >= 0 or p["closer"]["kind"] == "gas_strut"


def test_deterministic(specs):
    again = generate_all()
    assert [s["id"] for s in again] == [s["id"] for s in specs]
    assert again[17] == specs[17]


def test_build_all_models(specs):
    for s in specs[::25]:
        model = build_model(s)
        assert model.meta.get("primary_joint")
        for b in model.bodies:
            mass, com, I = b.inertial("full")
            assert mass >= 0


def test_export_and_load(tmp_path, specs):
    mujoco = pytest.importorskip("mujoco")
    s = next(x for x in specs if x["family"] == "swing_single" and x["operator"]["model"].startswith("lever") and not x["lock"]["engaged"] and x["closer"]["model"] == "none")
    out = export_door(s, str(tmp_path / "doors"), str(tmp_path / "hardware"), formats=("mjcf", "urdf", "json"))
    m = mujoco.MjModel.from_xml_path(out["files"]["mjcf"]["full"])
    d = mujoco.MjData(m)
    meta = json.load(open(tmp_path / "doors" / s["id"] / "model.json"))["meta"]
    jd = m.joint(meta["primary_joint"]).id
    jh = m.joint(meta["operator_joint"]).id
    # latched door holds
    for _ in range(500):
        d.qfrc_applied[:] = 0
        d.qfrc_applied[m.jnt_dofadr[jd]] = 40.0
        mujoco.mj_step(m, d)
    assert abs(d.qpos[m.jnt_qposadr[jd]]) < math.radians(1.5)
    # turning the lever opens it
    for _ in range(1500):
        d.qfrc_applied[:] = 0
        d.qfrc_applied[m.jnt_dofadr[jd]] = 40.0
        d.qfrc_applied[m.jnt_dofadr[jh]] = 3.0
        mujoco.mj_step(m, d)
    assert d.qpos[m.jnt_qposadr[jd]] > math.radians(20)
    mu = mujoco.MjModel.from_xml_path(out["files"]["urdf"]["full"])
    assert mu.njnt >= 2
