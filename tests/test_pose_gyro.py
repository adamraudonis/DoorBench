import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from doorbench.dexterous.pose_gyro import OwnImuPoseGyroscope,interval_local_gyro

class View:
    def __init__(self):self.prim_paths=['/World/H1/torso'];self.pose=np.array([[1,2,3,0,0,0,1.]],dtype=np.float32);self.reads=0
    def get_transforms(self):self.reads+=1;return self.pose
    def set_rotation(self,r):self.pose[0,3:]=Rotation.from_matrix(r).as_quat()

def make(view,mount=(1,0,0,0)):
    return OwnImuPoseGyroscope(view,expected_body_path='/World/H1/torso',robot_body_paths=['/World/H1/pelvis','/World/H1/torso'],imu_quaternion_wxyz_body=mount)

def test_only_owned_imu_body_and_three_gyro_values():
    v=View();g=make(v);g.reset_episode();v.set_rotation(Rotation.from_rotvec([.0002,0,0]).as_matrix());got=g.observe(now_s=.002)
    assert got.shape==(3,) and got.dtype==np.float32;np.testing.assert_allclose(got,[.1,0,0],atol=1e-8)
    receipt=g.receipt();assert receipt['actor_fields']==['imu_gyro'] and not receipt['actor_receives_orientation'] and receipt['physics_writes']==0
    assert not any(k in receipt for k in ('pose','orientation','root','position','rotation'))
    v.prim_paths=['/World/Door/handle']
    with pytest.raises(ValueError):g.observe(now_s=.004)
    assert g.receipt()['samples']==1
    with pytest.raises(RuntimeError):g.observe(now_s=.004)

def test_constant_world_gauge_translation_and_mount():
    world=Rotation.from_rotvec([.5,.8,-.2]).as_matrix();delta=Rotation.from_rotvec([.0002,.0003,-.0001]).as_matrix();mount=Rotation.from_rotvec([0,0,np.pi/2]);q=mount.as_quat()[[3,0,1,2]]
    v=View();v.set_rotation(world);g=make(v,q);g.reset_episode();v.pose[0,:3]=[-100,100,.1];v.set_rotation(world@delta)
    np.testing.assert_allclose(g.observe(now_s=.002),mount.as_matrix().T@np.array([.1,.15,-.05]),atol=3e-5)

def test_causal_start_future_epoch_failure_and_reset():
    v=View();g=make(v)
    with pytest.raises(ValueError):g.observe(now_s=0.)
    g.reset_episode()
    with pytest.raises(ValueError):g.observe(now_s=.004)
    g.reset_episode();np.testing.assert_array_equal(g.observe(now_s=.002),np.zeros(3,dtype=np.float32))
    v.pose[0,3:]*=-1;np.testing.assert_array_equal(g.observe(now_s=.004),np.zeros(3,dtype=np.float32))

def test_does_not_alias_previous_frame_and_rejects_bad_pose():
    v=View();g=make(v);g.reset_episode();v.pose[:]=np.nan
    with pytest.raises(ValueError):g.observe(now_s=.002)
    assert g.receipt()['samples']==0 and g.receipt()['failed_reason']
    bad=View();bad.prim_paths+=['/World/Door/leaf']
    with pytest.raises(ValueError):make(bad)
    with pytest.raises(ValueError):OwnImuPoseGyroscope(View(),expected_body_path='/World/H1/torso',robot_body_paths=['/World/H1/pelvis'],imu_quaternion_wxyz_body=[1,0,0,0])

def test_matching_nonrobot_inventory_cannot_select_door():
    v=View();v.prim_paths=['/World/Door/handle']
    with pytest.raises(ValueError):OwnImuPoseGyroscope(v,expected_body_path='/World/Door/handle',robot_body_paths=['/World/Door/handle'],imu_quaternion_wxyz_body=[1,0,0,0])

def test_interval_does_not_depend_on_future_frames():
    r0=Rotation.from_rotvec([.3,.2,-.1]).as_matrix();r1=r0@Rotation.from_rotvec([.0001,.0002,0]).as_matrix()
    np.testing.assert_allclose(interval_local_gyro(r0,r1,.002),[.05,.1,0],atol=1e-13)
    with pytest.raises(ValueError):interval_local_gyro(r0,r1*2,.002)
