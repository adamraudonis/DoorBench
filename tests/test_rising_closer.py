"""Cam-lift doors must accommodate lift without fighting their arm-closer loop."""
import json
import math

import numpy as np
import pytest
from scipy.optimize import least_squares

from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.qa import run_qa

IDS = ("db0188_cold_storage", "db0432_cold_storage", "db0549_cold_storage", "db0585_cold_storage", "db0937_cold_storage")


@pytest.fixture(scope="module", params=IDS)
def door(request, tmp_path_factory):
    mujoco = pytest.importorskip("mujoco")
    spec = next(s for s in generate_all() if s["id"] == request.param)
    root = tmp_path_factory.mktemp(request.param)
    exported = export_door(spec, str(root / "doors"), str(root / "hardware"), formats=("mjcf", "urdf", "json"))
    directory = root / "doors" / spec["id"]
    model_json = json.loads((directory / "model.json").read_text())
    phys = json.loads((directory / "spec.json").read_text())["physics"]
    model = mujoco.MjModel.from_xml_path(str(directory / "door.xml"))
    return mujoco, spec, directory, exported, model_json, phys, model


def _solve(mujoco, model, data, angle, *, fixed_shoe=False, guess=None):
    primary = model.joint("leaf_hinge").id
    rise = model.joint("leaf_rise").id
    shoe = model.joint("closer_shoe_slide").id
    coupling = model.equality("leaf_rise_couple").id
    pitch = model.eq_data[coupling][1]
    data.qpos[:] = model.qpos0
    data.qpos[model.jnt_qposadr[primary]] = angle
    data.qpos[model.jnt_qposadr[rise]] = angle * pitch
    ids = [model.joint(n).id for n in ("closer_pinion", "closer_elbow")]
    if not fixed_shoe:
        ids.append(shoe)
    addresses = [model.jnt_qposadr[j] for j in ids]
    equality = model.equality("closer_arm_connect").id
    a, b = model.eq_obj1id[equality], model.eq_obj2id[equality]

    def residual(x):
        data.qpos[addresses] = x
        mujoco.mj_kinematics(model, data)
        tip = data.xpos[a] + data.xmat[a].reshape(3, 3) @ model.eq_data[equality][:3]
        anchor = data.xpos[b] + data.xmat[b].reshape(3, 3) @ model.eq_data[equality][3:6]
        return tip - anchor

    fit = least_squares(residual, np.zeros(len(ids)) if guess is None else guess,
                        ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=100)
    return float(np.linalg.norm(residual(fit.x))), fit.x, pitch


def test_slotted_shoe_closes_loop_over_full_travel(door):
    mujoco, _, _, _, _, _, model = door
    data = mujoco.MjData(model)
    hi = model.jnt_range[model.joint("leaf_hinge").id][1]
    guess = None
    for angle in np.linspace(0.0, hi, 37):
        separation, guess, pitch = _solve(mujoco, model, data, angle, guess=guess)
        assert separation < 1e-4
        shoe_q = data.qpos[model.jnt_qposadr[model.joint("closer_shoe_slide").id]]
        assert abs(shoe_q - angle * pitch) < 1e-5
        assert -1e-6 <= shoe_q <= model.jnt_range[model.joint("closer_shoe_slide").id][1]
    # Negative fixture: fixing the same shoe reinstates the original impossible planar loop.
    separation, _, _ = _solve(mujoco, model, data, hi, fixed_shoe=True)
    assert separation > 0.01


def test_shoe_preserves_gravitational_work_and_closer_spring(door):
    mujoco, _, _, _, _, phys, model = door
    data = mujoco.MjData(model)
    _, _, pitch = _solve(mujoco, model, data, math.pi / 3)
    mujoco.mj_forward(model, data)
    hinge, rise, shoe = [model.joint(n).id for n in ("leaf_hinge", "leaf_rise", "closer_shoe_slide")]
    # Virtual work along the coupled hinge/rise/shoe path includes both elevated masses.
    gravity_torque = data.qfrc_bias[model.jnt_dofadr[hinge]] + pitch * sum(
        data.qfrc_bias[model.jnt_dofadr[j]] for j in (rise, shoe))
    # MJCF currently uses MuJoCo's 9.81 m/s² default; derive from the actual model.
    expected = phys["mass"]["total_kg"] * abs(model.opt.gravity[2]) * pitch
    assert gravity_torque == pytest.approx(expected, rel=2e-5)
    assert model.jnt_stiffness[shoe] == 0
    assert model.jnt_stiffness[hinge] == pytest.approx(phys["closer"]["spring_stiffness_Nm_per_rad"], abs=1e-5)


def test_rising_closer_still_passes_clearance_and_dynamic_qa(door):
    _, spec, directory, exported, model_json, phys, _ = door
    qa = run_qa(spec, str(directory), model_json["meta"], exported["files"], phys)
    assert qa["signed_off"], qa
    assert qa["checks"]["clearance"]
    assert qa["checks"]["closer_returns"]
