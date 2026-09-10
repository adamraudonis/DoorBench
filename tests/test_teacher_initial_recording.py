from types import SimpleNamespace
import numpy as np
import pytest
from doorbench.dexterous.isaac_sensor_recording import IsaacSensorRecorder


def test_initial_teacher_packet_is_observed_before_physics_and_separate_from_label(tmp_path):
    calls=[]
    def observe(**kwargs):
        calls.append(kwargs)
        return dict(previous_action=kwargs['previous_action'],sensor_valid=np.array([False]),sensor_time_s=np.array([-1.]))
    recorder=SimpleNamespace(control_source='privileged_teacher',times=[],output=tmp_path,
        layout={'action_order':['one','two']},builder=SimpleNamespace(observe=observe))
    forces=np.array([1.,-2.])
    IsaacSensorRecorder.record_teacher_initial_decision(recorder,forces)
    assert calls[0]['now_s']==0.
    with np.load(tmp_path/'teacher-initial-decision.npz') as z:
        assert float(z['time_s'])==0.
        np.testing.assert_array_equal(z['motor_forces'],forces)
        np.testing.assert_array_equal(z['previous_action'],[0.,0.])
        assert not z['sensor_valid'].any()
    with pytest.raises(ValueError):IsaacSensorRecorder.record_teacher_initial_decision(recorder,forces)


def test_later_or_actor_recording_cannot_claim_teacher_cold_start(tmp_path):
    for source,times in [('sensor_actor',[]),('privileged_teacher',[.002])]:
        recorder=SimpleNamespace(control_source=source,times=times,output=tmp_path)
        with pytest.raises(ValueError):IsaacSensorRecorder.record_teacher_initial_decision(recorder,[1.])
    assert not list(tmp_path.iterdir())
