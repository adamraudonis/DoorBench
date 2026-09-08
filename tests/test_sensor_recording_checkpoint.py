"""Numeric evidence survives interruption; actor cold start is reproducible."""
import json
import numpy as np
import pytest
from doorbench.dexterous.isaac_sensor_recording import IsaacSensorRecorder, _atomic_npz
from doorbench.dexterous.sensor_contract import ActorObservationBuilder, SENSOR_KEYS


def recorder(tmp_path):
    rec=IsaacSensorRecorder.__new__(IsaacSensorRecorder)
    rec.output=tmp_path;rec.control_source='sensor_actor'
    rec.layout={'sensors':[{'name':'pad','dimension':3}],'tactile_dimension':3}
    rec.cameras={'rgb_left':None,'rgb_right':None};rec.frames={k:[] for k in rec.cameras}
    rec.frame_times=[];rec.times=[.002]
    builder=ActorObservationBuilder(joint_count=2,action_count=2,tactile_dimension=3)
    builder.reset(seed=0);packet=builder.observe(now_s=.002,previous_action=np.zeros(2))
    rec.samples={k:[v] for k,v in packet.items() if not k.startswith('rgb_')}
    return rec,packet


def test_partial_checkpoint_preserves_causal_missing_camera_evidence(tmp_path):
    rec,_=recorder(tmp_path)
    partial=rec.finish(complete=False)
    assert partial['capture_complete'] is False
    assert partial['samples']==1 and partial['frames_per_eye']==0
    assert partial['rgb_mean_pixel_change']=={'rgb_left':None,'rgb_right':None}
    assert partial['all_sensor_streams_present'] is False
    with np.load(tmp_path/'actor-sensors.npz',allow_pickle=False) as data:
        assert not data['sensor_valid'].any()
        assert np.all(data['sensor_time_s']==-1)
    rec.finish(complete=True)
    assert json.loads((tmp_path/'report.json').read_text())['capture_complete'] is True


def test_interrupted_compression_keeps_previous_complete_numeric_file(tmp_path,monkeypatch):
    path=tmp_path/'values.npz';_atomic_npz(path,value=np.array([3.]))
    previous=path.read_bytes()
    def fail(stream,**arrays):
        stream.write(b'incomplete');raise OSError('Interrupted writer')
    monkeypatch.setattr(np,'savez_compressed',fail)
    with pytest.raises(OSError):_atomic_npz(path,value=np.array([4.]))
    assert path.read_bytes()==previous


def test_actor_first_action_and_all_invalid_packet_are_preserved_once(tmp_path):
    rec,packet=recorder(tmp_path);forces=np.array([.5,-.2])
    rec.record_initial_decision(packet,forces)
    with np.load(tmp_path/'actor-initial-decision.npz',allow_pickle=False) as initial:
        assert initial['time_s']==0.
        for key,value in packet.items():np.testing.assert_array_equal(initial[key],value)
        np.testing.assert_array_equal(initial['motor_forces'],forces)
    with pytest.raises(ValueError,match='already'):rec.record_initial_decision(packet,forces)
