import copy
import numpy as np
import pytest
from doorbench.dexterous.destination_state_binding import freeze_destination_state,admit_destination_state,ROOT_CONVENTION


@pytest.fixture
def state():
    names=['joint_'+str(i) for i in range(69)]
    names[-1]='lh_LFJ1'  # Omitted by the legacy 47-joint ungrip admission.
    return dict(motor_contract={'source_xml_sha256':'a'*64,'joint_names':names,
        'actuators':[{'name':'motor_'+str(i),'force_range':[-1.,1.]} for i in range(61)]},
        door_source_sha256='b'*64,time_s=60.,measured_time_s=60.,
        root_state_world=np.array([0,0,.9,1,0,0,0,0,0,0,0,0,0.]),
        joint_position=dict.fromkeys(names,.1),joint_velocity=dict.fromkeys(names,0.),
        door_joint_order=['leaf_hinge','leaf_handle_hinge'],door_position={'leaf_hinge':.3,'leaf_handle_hinge':0.},
        door_velocity={'leaf_hinge':.05,'leaf_handle_hinge':0.},root_state_convention=ROOT_CONVENTION)


def test_complete_bound_state_and_quaternion_double_cover(state):
    binding=freeze_destination_state(**state)
    state['root_state_world'][3:7]*=-1
    result=admit_destination_state(binding,**state)
    assert result['passed'] and result['complete_robot_joint_count']==69
    assert binding['root_state_world'][3]==1.


@pytest.mark.parametrize('field',['joint_position','joint_velocity'])
def test_previously_unbound_left_finger_cannot_change(state,field):
    binding=freeze_destination_state(**state)
    state[field]['lh_LFJ1']+=.002
    with pytest.raises(ValueError,match='exact attained'):admit_destination_state(binding,**state)


@pytest.mark.parametrize('change',['missing_joint','missing_door_joint','stale_epoch','mixed_com','motor_caps','door_speed','root_speed','corrupt_binding'])
def test_destination_cannot_reuse_partial_stale_or_different_physics(state,change):
    binding=freeze_destination_state(**state)
    if change=='missing_joint':del state['joint_position']['lh_LFJ1']
    if change=='missing_door_joint':del state['door_position']['leaf_handle_hinge']
    if change=='stale_epoch':state.update(time_s=60.002,measured_time_s=60.002)
    if change=='mixed_com':state['root_state_convention']='legacy-actor-pose-com-velocity'
    if change=='motor_caps':state['motor_contract']['actuators'][0]['force_range']=[-2.,2.]
    if change=='door_speed':state['door_velocity']['leaf_hinge']+=.01
    if change=='root_speed':state['root_state_world'][7]=.01
    if change=='corrupt_binding':binding['joint_position']['lh_LFJ1']=.3
    with pytest.raises(ValueError):admit_destination_state(binding,**state)


@pytest.mark.parametrize('bad',[True,np.nan,np.array([60.])])
def test_clock_is_explicit_scalar_and_finite(state,bad):
    state['time_s']=bad
    with pytest.raises(ValueError):freeze_destination_state(**state)


def test_same_epoch_geometry_and_input_copies(state):
    original=copy.deepcopy(state);binding=freeze_destination_state(**state)
    state['root_state_world'][0]=.5;state['joint_position']['lh_LFJ1']=.5
    assert binding['joint_position']==original['joint_position']
    assert binding['root_state_world'][0]==0.
    original['measured_time_s']-=.002
    with pytest.raises(ValueError,match='coherent'):freeze_destination_state(**original)
