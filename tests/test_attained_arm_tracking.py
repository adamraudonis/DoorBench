from types import SimpleNamespace
import numpy as np
import pytest
from doorbench.dexterous.attained_arm_tracking import AttainedArmTracking


def fixture():
    model=SimpleNamespace(actuator_trnid=np.array([[0,0]]),joint=lambda i:SimpleNamespace(name='right_wrist_yaw'),jnt_dofadr=np.array([0]),jnt_qposadr=np.array([0]))
    teacher=SimpleNamespace(arm_motors=np.array([0]),act=np.array([0]),m=model,d=SimpleNamespace(qfrc_bias=np.array([.3])),kp=np.array([10.]),gain=np.array([9.]),damping=np.array([1.]),bias=np.array([[0.,-10.,-.2]]),caps=np.array([[-5.,5.]]))
    return teacher,AttainedArmTracking(teacher,{'right_wrist_yaw':.2},np.array([1.]))


def test_retains_attained_preload_and_adjusts_gravity():
    teacher,c=fixture();q={'right_wrist_yaw':.2};v={'right_wrist_yaw':0.}
    result,_=c.force(np.zeros(1),0.,q,q,v)
    np.testing.assert_allclose(result,[1.])
    teacher.d.qfrc_bias[:]=.6
    result,_=c.force(np.zeros(1),.1,q,q,v)
    np.testing.assert_allclose(result,[1.3])


def test_original_caps_and_invalid_target_speed():
    _,c=fixture();q={'right_wrist_yaw':.2};v={'right_wrist_yaw':0.}
    c.force(np.zeros(1),0.,q,q,v)
    result,info=c.force(np.zeros(1),1.,{'right_wrist_yaw':.3},q,v)
    assert result[0]==5. and info['arm_clipped_motors']==1
    with pytest.raises(ValueError,match='2 rad/s'):c.force(np.zeros(1),1.001,{'right_wrist_yaw':.4},q,v)
    with pytest.raises(ValueError,match='Monotonic'):c.force(np.zeros(1),.1,q,q,v)


def test_velocity_uses_reference_update_clock_not_physics_substeps():
    _,c=fixture();q={'right_wrist_yaw':.2};v={'right_wrist_yaw':0.}
    c.force(np.zeros(1),0.,q,q,v)
    for t in [.002,.004,.006,.008]:c.force(np.zeros(1),t,q,q,v)
    _,info=c.force(np.zeros(1),.01,{'right_wrist_yaw':.21},q,v)
    np.testing.assert_allclose(info['arm_reference_velocity_rad_s'],[1.])
