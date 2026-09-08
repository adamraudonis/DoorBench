from types import SimpleNamespace
import pytest
from doorbench.dexterous.control_mode import validate_sensor_actor_mode


def test_sensor_actor_accepts_only_sensor_control_with_static_reset_and_audit_settings():
    validate_sensor_actor_mode(SimpleNamespace(sensor_policy_checkpoint='actor.pt',sensor_layout='actual.json',
        reset_from_acquisition_path=True,sensor_reset_preflight='reset.json',sensor_objective='partial-opening',record=True))
    validate_sensor_actor_mode(SimpleNamespace(sensor_policy_checkpoint=None,mechanism_test=True))


@pytest.mark.parametrize('name,value',[
    ('native_robot','oracle.xml'),('acquisition',True),('operate_after_acquisition',True),
    ('full_sequence_reset','teacher-reset.json'),('panel_push',True),('mechanism_test',True),
    ('full_opening',True),('left_palm_targets','oracle.json'),('right_release_screen','oracle.json'),
    ('follow_leaf_during_transfer',True),('panel_profile','plain-v1'),('palm_load_target',8.),
    ('bimanual_runtime_screen','oracle.json'),('left_planning_profile','strict-v1'),
    ('whole_body_return_path','oracle.json'),('whole_body_ungrip_path','oracle.json'),
    ('transfer_load_target',6.),
    ('upright_gain',.1),('grip_force',1.),('finger_curl',.1),('torso_damping',.1),
    ('stance_qp',True),('press_feedforward',True),('grip_reset_targets',True),
    ('arm_impedance',2.),('grip_impedance',2.),('grip_rotation_fraction',.5),
    ('operator_compliance_gain',.5),('acquisition_middle_finger_force',3.),
    ('acquisition_index_finger_force',3.),('time_scale',2.)])
def test_no_optional_teacher_or_feedback_path_can_be_combined_with_actor(name,value):
    options=SimpleNamespace(sensor_policy_checkpoint='actor.pt',sensor_layout='actual.json',**{name:value})
    with pytest.raises(ValueError,match=name):validate_sensor_actor_mode(options)


def test_sensor_calibration_is_required_before_initializing_plant():
    with pytest.raises(ValueError,match='calibration'):
        validate_sensor_actor_mode(SimpleNamespace(sensor_policy_checkpoint='actor.pt'))


@pytest.mark.parametrize('extra',[{},dict(reset_from_acquisition_path=True),dict(sensor_reset_preflight='proof.json')])
def test_actor_must_not_start_from_an_initialized_grasp(extra):
    with pytest.raises(ValueError,match='bound contact-free reset'):
        validate_sensor_actor_mode(SimpleNamespace(sensor_policy_checkpoint='actor.pt',sensor_layout='actual.json',**extra))


def test_sensor_balance_uses_the_same_no_teacher_boundary():
    options=SimpleNamespace(sensor_balance_calibration='posture.json',sensor_layout='actual.json',
        reset_from_acquisition_path=True,sensor_reset_preflight='reset.json')
    validate_sensor_actor_mode(options)
    options.native_robot='teacher.xml'
    with pytest.raises(ValueError,match='native_robot'):validate_sensor_actor_mode(options)
