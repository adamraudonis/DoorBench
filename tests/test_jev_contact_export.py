"""Synthetic archival fixtures verify timing/provenance, not robot behavior."""
import pytest

from scripts.dexterous.export_jev_contact_cases import case_from_row, select_indices


def row():
    return dict(contact_force_source="actual_mj_step_dynamics", contact_interval_start_s=1.,
        contact_interval_end_s=1.002, contact_geometry_time_s=1., finite=True,
        pre_integration_state=dict(handle_angle_rad=.2),
        operation=dict(phase="lever_operation", actual_handle_rad=.2, commanded_operator_reference_rad=.3),
        pad_grasp=dict(digit_forces_N=dict(ff=.5), contacts=[dict(digit="ff", normal_force_N=.5)],
                       lever_geom="handle"))


def convert(value):
    return case_from_row(value, digit="ff", index=5, episode_id="fixture", threshold_N=.15)


def test_actual_interval_epoch_and_missing_state_are_preserved():
    case = convert(row())
    sample = case["telemetry"]
    assert sample["simulation_time_s"] == 1.
    assert sample["angle_error_rad"] == pytest.approx(.1)
    assert sample["slip_speed_mps"] is None and sample["joint_margin_rad"] is None and sample["balance_stable"] is None
    assert sample["contact_scope"] == "privileged_digit_handle_contact"
    assert case["label"]["contact"] is True


def test_reset_samples_cannot_masquerade_as_completed_physical_intervals():
    value = row()
    value.pop("contact_force_source")
    assert convert(value) is None


def test_mismatched_force_epoch_or_patch_load_is_rejected():
    value = row()
    value["contact_geometry_time_s"] = 1.002
    with pytest.raises(ValueError, match="epoch"):
        convert(value)
    value = row()
    value["pad_grasp"]["contacts"][0]["normal_force_N"] = .8
    with pytest.raises(ValueError, match="patches"):
        convert(value)
    value = row()
    value["operation"]["actual_handle_rad"] = .21
    with pytest.raises(ValueError, match="pre-integration"):
        convert(value)


def test_missing_contact_load_is_not_zero_contact():
    value = row()
    value["pad_grasp"]["digit_forces_N"] = {}
    case = convert(value)
    assert case["telemetry"]["normal_load_N"] is None
    assert case["label"]["contact"] is None
    assert not case["telemetry"]["sensor_valid"]


def test_selection_covers_contact_transition_and_time_range():
    cases = [dict(label=dict(contact=i >= 50), provenance=dict(phase="acquisition")) for i in range(100)]
    selected = select_indices(cases, 12)
    assert len(selected) == len(set(selected)) == 12
    assert {0, 49, 50, 99} <= set(selected)
