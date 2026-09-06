"""Native causal checks for the one-door prototype; no render-based assertions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "physical_human", ROOT / "scripts/physical_human/prototype.py"
)
prototype = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prototype)


def test_only_actor_actuators_and_no_world_or_hand_welds():
    model, _ = prototype.make_model()
    names = [model.joint(i).name for i in model.actuator_trnid[:, 0]]
    assert all(n.startswith(("actor_", "finger_", "thumb_")) for n in names)
    assert len([n for n in names if n.startswith(("finger_", "thumb_"))]) == 28
    assert model.nmocap == 0
    assert model.neq == 1
    assert model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
    assert model.joint("actor_root").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.nsensor == 30


def test_native_opening_requires_contact_and_released_latch(tmp_path):
    normal = prototype.run(tmp_path / "final")
    no_touch = prototype.run(tmp_path / "no-touch", no_touch=True)
    blocked = prototype.run(tmp_path / "blocked", latch_blocked=True)
    assert 40 < normal["max_door_deg"] < 65
    assert no_touch["max_door_deg"] < 0.05
    assert blocked["max_door_deg"] < 1
    assert no_touch["hand_contact_impulse_ns"] == 0
    assert normal["hand_contact_impulse_ns"] > 1
    assert normal["nonhand_door_impulse_ns"] == 0
    assert normal["min_pelvis_z"] > 0.90
    assert normal["max_foot_drift_m"] < 0.005
    assert normal["max_contact_penetration_m"] < 0.002
    assert normal["max_self_penetration_m"] < 0.002
    assert not any(normal["warnings"])
    first_open = next(r for r in normal["rows"] if r["door_deg"] > 1)
    assert first_open["latch_mm"] > 10
    assert normal["rows"][-1]["touch_n"] < 1
    # Saved state and measurement timestamps must describe the same native pose.
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "final/scene.xml"))
    trace = np.load(tmp_path / "final/trajectory.npz")
    assert model.nkey == 1
    np.testing.assert_allclose(trace["time"], [r["t"] for r in normal["rows"]])
    qi = model.joint("door_hinge").qposadr[0]
    np.testing.assert_allclose(
        -trace["qpos"][:, qi] * 180 / np.pi, [r["door_deg"] for r in normal["rows"]]
    )
