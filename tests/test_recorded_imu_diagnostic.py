"""Independent frame and gravity-correction checks for the offline replay."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.dexterous.diagnose_recorded_imu import integrate_sample


def test_body_gyro_increment_multiplies_on_the_right():
    R=Rotation.from_rotvec([0,0,np.pi/2]).as_matrix()
    got=integrate_sample(R,[.1,0,0],[0,0,9.81],.002,0.)
    expected=R@Rotation.from_rotvec([.0002,0,0]).as_matrix()
    np.testing.assert_allclose(got,expected,atol=1e-15)
    assert not np.allclose(got,Rotation.from_rotvec([.0002,0,0]).as_matrix()@R,atol=1e-7)


def test_gyro_only_recovers_known_constant_body_rate():
    R=np.eye(3);w=np.array([.01,-.03,.1])
    for _ in range(500):R=integrate_sample(R,w,[0,0,9.81],.002,0.)
    np.testing.assert_allclose(R,Rotation.from_rotvec(w).as_matrix(),atol=1e-13)


def test_gravity_feedback_corrects_roll_but_acceleration_can_bias_it():
    R=Rotation.from_rotvec([.005,0,0]).as_matrix()
    corrected=integrate_sample(R,[0,0,0],[0,0,9.81],.002,.2)
    assert Rotation.from_matrix(corrected).magnitude()<.005
    biased=integrate_sample(np.eye(3),[0,0,0],[0,.1,9.81],.002,.2)
    assert Rotation.from_matrix(biased).magnitude()>0


def test_declared_first_sample_skip_is_not_replaced_by_an_actual_pose():
    R=Rotation.from_rotvec([.01,.02,.03]).as_matrix()
    got=integrate_sample(R,[1,2,3],[0,0,9.81],0.,.2)
    np.testing.assert_array_equal(got,R)
