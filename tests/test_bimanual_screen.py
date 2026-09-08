import copy

import pytest

from doorbench.dexterous.bimanual_screen import screen_checks, validate_grasp_reference


def sequence():
    return [dict(phase=phase, left_position_error_m=0., right_position_error_m=0.,
        right_pose_constrained=phase != "left_push", left_normal_error=0.,
        collisions=[], arm_static_motor_utilization=.4, max_loopback_violation_rad=0.,
        left_hand_slab_distance_m=.0005, stance_status="solved",
        planned_root_acceleration=[0.] * 6, leaf_rad=1.2 if phase == "left_push" else 0.)
        for phase in ("left_reach", "operator", "both_push", "right_release", "left_push")]


def test_reject_approach_reference_that_is_not_the_qualified_grasp():
    reference = dict(initial_root=[0.] * 7, initial_joints={"finger": 0.},
        acquisition={"path_qpos": [[0.], [1.]]})
    with pytest.raises(ValueError, match="grasp-only"):
        validate_grasp_reference(reference)


def test_static_screen_rejects_disconnected_endpoints_and_missing_contact():
    rows = sequence()
    assert all(screen_checks(rows, .1).values())
    assert not screen_checks(rows, .4)["continuous_arm_waypoints"]
    rows[-1]["left_hand_slab_distance_m"] = .03
    assert not screen_checks(rows, .1)["left_panel_proximity"]


@pytest.mark.parametrize("field,value,gate", [
    ("collisions", [{"depth_m": .004}], "collisions"),
    ("arm_static_motor_utilization", 1.01, "arm_strength"),
    ("max_loopback_violation_rad", .8, "loopback"),
    ("planned_root_acceleration", [0., 0., float("nan"), 0., 0., 0.], "finite_support_plan"),
    ("left_position_error_m", float("nan"), "positions"),
])
def test_geometric_or_physical_planning_failure_cannot_pass(field, value, gate):
    rows = copy.deepcopy(sequence())
    rows[2][field] = value
    assert not screen_checks(rows, .1)[gate]


def test_empty_or_partial_sequence_fails():
    assert not all(screen_checks([], 0.).values())
    assert not screen_checks(sequence()[-1:], .1)["complete_static_sequence"]
