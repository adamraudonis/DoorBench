import numpy as np
import pytest
from doorbench.dexterous.motor_target_control import MotorTargetControl


def fixture():
    names=[f'q{i}' for i in range(69)]
    motors=dict(joint_names=names,actuators=[dict(terms={names[i]:1.,names[68]:.5},kp=10.,
        bias=[0.,-10.,-2.],control_range=[-1.,1.],force_range=[-5.,5.]) for i in range(61)])
    return MotorTargetControl(motors)


def test_coupled_encoder_feedback_and_original_force_caps():
    control=fixture();q=np.zeros(69);dq=q.copy();q[68]=.4;dq[0]=.3
    force=control.forces(np.zeros(61),q,dq)
    assert force[0]==pytest.approx(-2.6)
    assert force[1]==pytest.approx(-2.)
    np.testing.assert_array_equal(control.forces(np.ones(61),q,dq),np.full(61,5.))


def test_force_targets_round_trip_and_impossible_force_rejected():
    control=fixture();q=np.zeros((3,69));dq=q.copy();q[:,68]=.3
    original=control.forces(np.full((3,61),.2),q,dq)
    target=control.targets_for_forces(original,q,dq)
    np.testing.assert_allclose(control.forces(target,q,dq),original,atol=1e-6)
    with pytest.raises(ValueError,match='not realizable'):
        control.targets_for_forces(np.full((3,61),10.),q,dq)
