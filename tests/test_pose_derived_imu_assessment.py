import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.dexterous.assess_pose_derived_imu import delta_angle_gyro

def test_interval_gyro_reconstructs_noncommuting_rotation_and_is_world_gauge_invariant():
    r=[Rotation.from_rotvec([.2,-.3,.1]).as_matrix()]
    for w in ([.01,.02,-.005],[-.03,.005,.008],[.02,-.01,.002]):r.append(r[-1]@Rotation.from_rotvec(w).as_matrix())
    r=np.array(r);g=delta_angle_gyro(r,.002);a=r[0]
    for i,w in enumerate(g):
        a=a@Rotation.from_rotvec(w*.002).as_matrix();np.testing.assert_allclose(a,r[i+1],rtol=0,atol=1e-14)
    world=Rotation.from_rotvec([1.,-.4,.7]).as_matrix();np.testing.assert_allclose(delta_angle_gyro(world@r,.002),g,rtol=0,atol=1e-12)

def test_interval_has_no_future_dependence_and_rejects_invalid_frames():
    r=Rotation.from_rotvec(np.arange(4)[:,None]*np.array([.01,.02,0])).as_matrix()
    np.testing.assert_allclose(delta_angle_gyro(r[:3],.002),delta_angle_gyro(r,.002)[:2],rtol=0,atol=0)
    with pytest.raises(ValueError):delta_angle_gyro(r,0)
    r[1,0,0]*=2
    with pytest.raises(ValueError):delta_angle_gyro(r,.002)
