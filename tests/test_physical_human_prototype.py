"""Native mechanics and grasp checks; visual review remains a separate step."""

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
    assert all(n.startswith(("actor_", "hand_")) for n in names)
    assert len([n for n in names if n.startswith("hand_")]) == 40
    assert model.nmocap == 0
    assert model.neq == 1
    assert model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
    assert model.joint("actor_root").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.nsensor == 48


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
    assert normal["quality_passed"], normal["quality_checks"]
    assert normal["rows"][-1]["door_deg"] > 40
    assert normal["kinematics"]["passed"], normal["kinematics"]["violations"]
    assert normal["kinematics"]["samples"] > 6000
    assert normal["grasp"]["passed"], normal["grasp"]["violations"]
    assert normal["peak_total_hand_contact_n"] < 160
    # Saved state and measurement timestamps must describe the same native pose.
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "final/scene.xml"))
    trace = np.load(tmp_path / "final/trajectory.npz")
    assert model.nkey == 1
    assert trace["hand_keypoints"].shape == (len(trace["time"]), 2, 21, 3)
    np.testing.assert_allclose(trace["time"], [r["t"] for r in normal["rows"]])
    qi = model.joint("door_hinge").qposadr[0]
    np.testing.assert_allclose(
        -trace["qpos"][:, qi] * 180 / np.pi, [r["door_deg"] for r in normal["rows"]]
    )


def test_thumb_model_has_cmc_and_two_phalanges_and_mirrors_correctly():
    import xml.etree.ElementTree as ET

    from scripts.physical_human.hand_anatomy import attach_hand, target_pose

    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    for side in ["l", "r"]:
        attach_hand(ET.SubElement(world, "body", name="wrist_" + side), side)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    assert model.njnt == 40
    reference_lengths = None
    for amount in [0, 0.35, 1]:
        for side in ["l", "r"]:
            for name, value in target_pose(side, amount).items():
                data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        points = np.array(
            [
                [
                    data.site_xpos[model.site(f"hand_keypoint_{s}_{i:02}").id]
                    for i in range(21)
                ]
                for s in ["l", "r"]
            ]
        )
        # Hinge axes are axial vectors: reflecting them like positions silently
        # reverses flexion. Check every landmark in several articulated poses.
        np.testing.assert_allclose(points[0] * [-1, 1, 1], points[1], atol=1e-9)
        lengths = np.linalg.norm(np.diff(points[0, 1:5], axis=0), axis=1)
        if reference_lengths is None:
            reference_lengths = lengths
            # Thumb CMC belongs near the wrist, proximal to the index MCP.
            assert np.linalg.norm(points[0, 1]) < 0.5 * np.linalg.norm(points[0, 5])
            # Adult-sized metacarpal + proximal/distal thumb, not a finger stub.
            assert 0.075 < lengths.sum() < 0.115
            assert np.linalg.norm(points[0, 12]) > np.linalg.norm(points[0, 20]) + 0.025
        else:
            np.testing.assert_allclose(lengths, reference_lengths, atol=1e-9)


def test_kinematic_audit_rejects_an_actual_overextended_thumb():
    from scripts.physical_human.kinematics import KinematicAudit

    model, _ = prototype.make_model()
    data = prototype.initial(model)
    joint = model.joint("hand_l_cmc_abduction")
    data.qpos[joint.qposadr[0]] = joint.range[1] + np.deg2rad(35)
    mujoco.mj_forward(model, data)
    audit = KinematicAudit(model)
    audit.observe(data)
    result = audit.result()
    assert not result["passed"]
    assert any("hand_l_cmc_abduction" in v for v in result["violations"])
    assert result["max_joint_limit_excess_deg"] == pytest.approx(35)


def test_kinematic_audit_rejects_unphysical_wrist_bend_and_digit_speed():
    from scripts.physical_human.kinematics import KinematicAudit

    model, _ = prototype.make_model()
    data = prototype.initial(model)
    wrist = model.joint("actor_wrist_l_flexion")
    data.qpos[wrist.qposadr[0]] = -1.5
    data.qvel[model.joint("hand_l_ip_flexion").dofadr[0]] = 25
    mujoco.mj_forward(model, data)
    audit = KinematicAudit(model)
    audit.observe(data)
    result = audit.result()
    assert not result["passed"]
    assert any("wrist bend" in v for v in result["violations"])
    assert any("angular speed" in v for v in result["violations"])


def test_published_v2_thumb_above_handle_is_rejected():
    """The actual shipped failure must fail even while anatomical limits pass."""
    import json

    from scripts.physical_human.grasp import GraspAudit
    from scripts.physical_human.kinematics import KinematicAudit

    snapshot = json.loads(
        (
            Path(__file__).parent / "fixtures/physical_human/rejected_v2_grasp.json"
        ).read_text()
    )
    model, _ = prototype.make_model()
    assert snapshot["joint_names"] == [model.joint(i).name for i in range(model.njnt)]
    data = mujoco.MjData(model)
    anatomy = KinematicAudit(model)
    grasp = GraspAudit(model)
    for frame in snapshot["frames"]:
        data.qpos[:] = frame["qpos"]
        data.qvel[:] = frame["qvel"]
        data.ctrl[:] = frame["ctrl"]
        mujoco.mj_forward(model, data)
        anatomy.observe(data)
        achieved = grasp.observe(data, "pull")
        assert not achieved["opposed_grasp"]
        assert not achieved["thumb_below_grip"]
        assert achieved["thumb_normal_force_n"] == 0
    assert anatomy.result()["passed"]
    assert not grasp.result()["passed"]


def test_grasp_needs_opposite_loaded_contacts_on_the_usable_grip():
    from scripts.physical_human.grasp import digit_for_geom, opposed_contacts

    contacts = [
        ("thumb", [-0.05, -0.0732, -0.0096], 1.0),
        ("index", [-0.06, -0.0588, 0.0096], 2.0),
        ("middle", [-0.08, -0.0588, 0.0096], 1.0),
    ]
    degrees, fingers, force = opposed_contacts(contacts)
    assert degrees == pytest.approx(180)
    assert fingers == 2
    assert force == 1
    assert opposed_contacts([contacts[0], contacts[1], contacts[1]])[1] == 1
    # Thumb above (same side), floating, or gripping the stem cannot qualify.
    for bad in [
        ("thumb", [-0.05, -0.066, 0.012], 1.0),
        ("thumb", [-0.05, -0.066, -0.012], 0.0),
        ("thumb", [0, -0.066, -0.012], 1.0),
    ]:
        assert opposed_contacts([bad, *contacts[1:]])[0] == 0
    assert digit_for_geom("hand_l_firstmc_contact0") is None
    assert digit_for_geom("hand_l_distal_thumb_contact0") == "thumb"


def test_correct_thumb_does_not_hide_one_finger_crossing_the_handle():
    import json

    from scripts.physical_human.grasp import GraspAudit
    from scripts.physical_human.kinematics import KinematicAudit

    snapshot = json.loads(
        (
            Path(__file__).parent
            / "fixtures/physical_human/rejected_middle_finger.json"
        ).read_text()
    )
    model, _ = prototype.make_model()
    data = mujoco.MjData(model)
    frame = snapshot["frames"][0]
    data.qpos[:] = frame["qpos"]
    data.qvel[:] = frame["qvel"]
    data.ctrl[:] = frame["ctrl"]
    mujoco.mj_forward(model, data)
    anatomy = KinematicAudit(model)
    anatomy.observe(data)
    achieved = GraspAudit(model).measure(data)
    assert anatomy.result()["passed"]
    assert achieved["thumb_below_grip"]
    assert achieved["thumb_on_opposite_side"]
    assert not achieved["finger_side_checks"]["middle"]
    assert not achieved["four_fingers_together"]
    assert not achieved["opposed_grasp"]


def test_actual_abrupt_arm_reorientation_is_rejected():
    import json

    from scripts.physical_human.kinematics import KinematicAudit

    frame = json.loads(
        (
            Path(__file__).parent / "fixtures/physical_human/rejected_arm_jump.json"
        ).read_text()
    )
    model, _ = prototype.make_model()
    data = mujoco.MjData(model)
    data.qpos[:] = frame["qpos"]
    data.qvel[:] = frame["qvel"]
    mujoco.mj_forward(model, data)
    audit = KinematicAudit(model)
    audit.observe(data, "pull")
    result = audit.result()
    assert not result["passed"]
    assert any("arm angular speed" in v for v in result["violations"])
