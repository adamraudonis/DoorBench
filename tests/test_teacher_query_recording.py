import json
import numpy as np
import pytest
from doorbench.dexterous.teacher_query_recording import TeacherQueryRecorder


def fixture(tmp_path):
    recorder=TeacherQueryRecorder(tmp_path/'queries',joint_names=[f'j{i}' for i in range(69)],hand_body_names=['rh_palm'])
    pose=np.array([0.,0.,1.,1.,0.,0.,0.])
    row=dict(root_state=np.r_[pose,np.zeros(6)],joint_position=np.zeros(69),joint_velocity=np.zeros(69),
        handle_pose=pose.copy(),leaf_pose=pose.copy(),door_position=np.zeros(3),right_hand_forces_world=np.zeros((1,3)))
    return recorder,row


def test_measurements_are_copied_and_never_advertised_as_actor_inputs_or_teacher_labels(tmp_path):
    rec,row=fixture(tmp_path);rec.record(time_s=0.,**row);row['joint_position'][0]=99.
    rec.finish(complete=False)
    with np.load(rec.output/'pre-action-measurements.npz',allow_pickle=False) as data:
        assert data['joint_position'][0,0]==0.
        assert data['time_s'].tolist()==[0.]
    report=json.loads((rec.output/'report.json').read_text())
    assert report['capture_complete'] is False
    assert report['actor_input'] is False and report['contains_teacher_labels'] is False


@pytest.mark.parametrize('clock',[.002,-.002,.1,float('nan')])
def test_missing_or_future_reset_measurement_is_rejected(tmp_path,clock):
    rec,row=fixture(tmp_path)
    with pytest.raises(ValueError,match='causal'):rec.record(time_s=clock,**row)


def test_duplicate_clock_and_mismatched_body_force_shape_are_rejected(tmp_path):
    rec,row=fixture(tmp_path);rec.record(time_s=0.,**row)
    with pytest.raises(ValueError,match='causal'):rec.record(time_s=0.,**row)
    row['right_hand_forces_world']=np.zeros((2,3))
    with pytest.raises(ValueError,match='right_hand'):rec.record(time_s=.002,**row)
    assert len(rec.rows['time_s'])==1
