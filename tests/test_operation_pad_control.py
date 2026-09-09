from types import SimpleNamespace
import numpy as np
import pytest
from doorbench.dexterous.operation_pad_control import OperationPadControl


def fixture():
    c=OperationPadControl.__new__(OperationPadControl)
    c.relative={'ff':np.array([0.,.007,0.]),'th':np.array([0.,-.007,0.])}
    c.teacher=SimpleNamespace(matrix=np.eye(2),d=SimpleNamespace(qpos=np.ones(2)),qa=np.arange(2),va=np.arange(2),
        kp=np.ones(2)*10,target=np.ones(2),bias=np.array([[0.,-10.,0.],[0.,-10.,0.]]),gain=np.zeros(2),
        fingers=np.array([0]),arm_motors=np.array([1]),finger_inverse=np.array([[1.,0.]]),
        arm_inverse=np.array([[0.,1.]]),caps=np.array([[-2.,2.],[-2.,2.]]))
    def force(d,targets):
        c.targets=targets
        return np.array([100.,-100.]),{}
    c.tracker=SimpleNamespace(generalized_force=force)
    return c


def test_material_points_stay_on_opposite_sides_of_commanded_bar_and_motors_remain_bounded():
    c=fixture();pose=np.array([0.,0.,0.,1.,0.,0.,0.]);before=c.teacher.d.qpos.copy()
    geometry=dict(operator_origin=np.zeros(3),operator_axis=np.array([0.,1.,0.]),
                  leaf_origin=np.zeros(3),leaf_axis=np.array([0.,0.,1.]))
    motors,_=c.force(np.array([1.,.5]),1.,pose,pose,dict(operator=0.,leaf=0.),
                     dict(operator=0.,leaf=np.pi/2),geometry)
    np.testing.assert_allclose(c.targets['ff'],[-.007,0.,0.],atol=1e-12)
    np.testing.assert_allclose(c.targets['th'],[.007,0.,0.],atol=1e-12)
    np.testing.assert_array_equal(motors,[2.,-2.])
    np.testing.assert_array_equal(c.teacher.d.qpos,before)


def test_feedback_has_no_initial_torque_step():
    c=fixture();pose=np.array([0.,0.,0.,1.,0.,0.,0.]);geometry=dict(
        operator_origin=np.zeros(3),operator_axis=np.array([0.,1.,0.]),
        leaf_origin=np.zeros(3),leaf_axis=np.array([0.,0.,1.]))
    motors,_=c.force(np.array([1.,.5]),0.,pose,pose,dict(operator=0.,leaf=0.),dict(operator=0.,leaf=0.),geometry)
    np.testing.assert_array_equal(motors,[1.,.5])


def test_actual_material_targets_follow_measured_handle_without_arm_counterforce():
    c=fixture();c.profile='actual-material-v1';pose=np.array([.2,.3,.4,1.,0.,0.,0.]);geometry=dict(operator_origin=np.zeros(3),operator_axis=np.array([0.,1.,0.]),leaf_origin=np.zeros(3),leaf_axis=np.array([0.,0.,1.]))
    motors,info=c.force(np.array([1.,.5]),1.,pose,pose,dict(operator=0.,leaf=0.),dict(operator=.8,leaf=.1),geometry)
    np.testing.assert_allclose(c.targets['ff'],[.2,.307,.4])
    np.testing.assert_allclose(c.targets['th'],[.2,.293,.4])
    assert motors[1]==.5 and info['operation_pad_control']=='actual-material-v1'
