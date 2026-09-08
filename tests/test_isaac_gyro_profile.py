"""Changed sensing stays causal and visible without changing legacy packets."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from doorbench.dexterous.isaac_sensor_recording import IsaacSensorRecorder, bind_gyro_profile
from doorbench.dexterous.isaac_sensors import enqueue_robot_sensors
from doorbench.dexterous.pose_gyro import OwnImuPoseGyroscope, DEFAULT_PROFILE, PROFILE
from doorbench.dexterous.sensor_contract import ActorObservationBuilder, SENSOR_KEYS


def test_legacy_layout_identity_and_opt_in_checkpoint_distinction():
    original={'imu':{'body_name':'torso','quaternion_wxyz_body':[1,0,0,0]}}
    assert bind_gyro_profile(original,DEFAULT_PROFILE)==original
    alternate=bind_gyro_profile(original,PROFILE)
    assert alternate['imu']['gyro_profile']==PROFILE and alternate!=original
    assert 'gyro_profile' not in original['imu']
    with pytest.raises(ValueError,match='differs'):
        bind_gyro_profile(alternate,DEFAULT_PROFILE)
    with pytest.raises(ValueError,match='Unknown'):
        bind_gyro_profile(original,'silent-correction')


class RestrictedImu:
    lin_acc_b=np.array([[1.,2.,9.81]])
    def __getattr__(self,key):
        raise AssertionError('Unexpected IMU channel: '+key)


def test_override_is_only_gyro_and_does_not_read_or_write_backend_rate():
    builder=ActorObservationBuilder(joint_count=2,action_count=2,tactile_dimension=3)
    builder.reset(seed=0)
    robot=SimpleNamespace(joint_pos=np.array([[.1,.2]]),joint_vel=np.array([[.3,.4]]))
    rate=np.array([.01,.02,.03],np.float32)
    enqueue_robot_sensors(builder,robot,RestrictedImu(),np.array([1.,2.,3.]),
        joint_indices=[1,0],capture_s=.002,gyro_override=rate)
    rate[:]=100
    packet=builder.observe(now_s=.002,previous_action=np.zeros(2))
    np.testing.assert_allclose(packet['imu_gyro'],[.01,.02,.03])
    np.testing.assert_allclose(packet['imu_accelerometer'],[1,2,9.81])
    np.testing.assert_allclose(packet['joint_position'],[.2,.1])
    assert packet['sensor_time_s'][SENSOR_KEYS.index('imu_gyro')]==.002


@pytest.mark.parametrize('bad',[np.zeros(4),np.array([0.,np.nan,0.])])
def test_bad_override_rejects_before_enqueuing_any_channel(bad):
    builder=ActorObservationBuilder(joint_count=2,action_count=2,tactile_dimension=3)
    builder.reset(seed=0)
    with pytest.raises(ValueError,match='three finite'):
        enqueue_robot_sensors(builder,object(),RestrictedImu(),np.zeros(3),
            joint_indices=[0,1],capture_s=.002,gyro_override=bad)
    assert not builder.observe(now_s=.002,previous_action=np.zeros(2))['sensor_valid'].any()


def test_recorder_observes_one_completed_interval_without_sensor_cache_mutation(tmp_path):
    class View:
        prim_paths=['/World/H1/torso']
        pose=np.array([[0.,0.,0.,0.,0.,0.,1.]])
        def get_transforms(self):return self.pose
    view=View()
    producer=OwnImuPoseGyroscope(view,expected_body_path=view.prim_paths[0],
        robot_body_paths=view.prim_paths,imu_quaternion_wxyz_body=[1,0,0,0])
    producer.reset_episode()
    rec=IsaacSensorRecorder.__new__(IsaacSensorRecorder)
    rec.pose_gyro=producer;rec.gyro_profile=PROFILE;rec.output=tmp_path
    rec.builder=ActorObservationBuilder(joint_count=2,action_count=2,tactile_dimension=3)
    rec.builder.reset(seed=0)
    initial=rec.builder.observe(now_s=0,previous_action=np.zeros(2))
    assert not initial['sensor_valid'].any()
    rec.imu=SimpleNamespace(data=RestrictedImu(),update=lambda dt:None)
    rec.adapter=SimpleNamespace(read=lambda physics_dt:np.zeros(3))
    rec.joint_indices=[0,1];rec.samples={k:[] for k in initial if not k.startswith('rgb_')}
    rec.times=[];rec.frame_times=[]
    robot=SimpleNamespace(joint_pos=np.zeros((1,2)),joint_vel=np.zeros((1,2)))
    view.pose[0,3:]=Rotation.from_rotvec([0.,.0002,0.]).as_quat()
    rec.update(robot_data=robot,dt=.002,time_s=.002,previous_action=np.zeros(2),rendered=False)
    np.testing.assert_allclose(rec.samples['imu_gyro'][0],[0,.1,0],atol=1e-8)
    assert producer.receipt()['samples']==1
    for _ in range(3):rec.builder.observe(now_s=.002,previous_action=np.zeros(2))
    assert producer.receipt()['samples']==1
    with pytest.raises(ValueError,match='consecutive'):
        rec.update(robot_data=robot,dt=.002,time_s=.002,previous_action=np.zeros(2),rendered=False)
    assert len(rec.times)==1
    rec._write_gyro_receipt()
    with np.load(tmp_path/'gyro-producer-evidence.npz',allow_pickle=False) as evidence:
        np.testing.assert_array_equal(evidence['time_s'],[0,.002])
        assert evidence['body_quaternion_xyzw_world'].shape==(2,4)


def test_first_step_failure_preserves_producer_evidence_without_fabricating_packet(tmp_path):
    class View:
        prim_paths=['/World/H1/torso']
        def get_transforms(self):return np.array([[0.,0.,0.,0.,0.,0.,1.]])
    producer=OwnImuPoseGyroscope(View(),expected_body_path='/World/H1/torso',
        robot_body_paths=View.prim_paths,imu_quaternion_wxyz_body=[1,0,0,0])
    producer.reset_episode()
    with pytest.raises(ValueError):producer.observe(now_s=.004)
    rec=IsaacSensorRecorder.__new__(IsaacSensorRecorder)
    rec.output=tmp_path;rec.pose_gyro=producer;rec.gyro_profile=PROFILE;rec.control_source='sensor_actor'
    rec.layout={'sensors':[{'name':'pad','dimension':3}],'tactile_dimension':3}
    rec.cameras={'rgb_left':None,'rgb_right':None};rec.frames={k:[] for k in rec.cameras}
    rec.frame_times=[];rec.times=[]
    builder=ActorObservationBuilder(joint_count=2,action_count=2,tactile_dimension=3)
    builder.reset(seed=0)
    rec.samples={k:[] for k in builder.observe(now_s=0,previous_action=np.zeros(2)) if not k.startswith('rgb_')}
    report=rec.finish(complete=True)
    assert not report['capture_complete'] and not report['all_sensor_streams_present']
    assert report['samples']==0 and report['gyro_producer']['failed_reason']
    with np.load(tmp_path/'gyro-producer-evidence.npz',allow_pickle=False) as evidence:
        np.testing.assert_array_equal(evidence['time_s'],[0])
