from types import SimpleNamespace
import numpy as np
from doorbench.dexterous.attained_hand_tracking import AttainedHandTracking


def test_coupled_motor_preserves_sum_preload_and_original_cap():
    teacher=SimpleNamespace(fingers=np.array([0]),names=['rh_FFJ1','rh_FFJ2'],matrix=np.array([[1.,1.]]),finger_inverse=np.array([[.5,.5]]),d=SimpleNamespace(qfrc_bias=np.array([.1,.1])),va=np.array([0,1]),kp=np.array([1.]),damping=np.array([0.]),bias=np.array([[0.,-1.,-.1]]),caps=np.array([[-1.,1.]]))
    joints=dict(rh_FFJ1=.2,rh_FFJ2=.2);zero=dict(rh_FFJ1=0.,rh_FFJ2=0.)
    controller=AttainedHandTracking(teacher,joints,np.array([.3]))
    force,info=controller.force(np.zeros(1),joints,zero)
    np.testing.assert_allclose(force,[.3])
    force,_=controller.force(np.zeros(1),dict(rh_FFJ1=.1,rh_FFJ2=.3),zero)
    np.testing.assert_allclose(force,[.3])
    force,info=controller.force(np.zeros(1),zero,zero)
    np.testing.assert_allclose(force,[1.]);assert info['finger_clipped_motors']==1


def test_reference_velocity_tracks_coupled_tendon_without_changing_caps():
    teacher=SimpleNamespace(fingers=np.array([0]),names=['a','b'],matrix=np.array([[1.,1.]]),finger_inverse=np.array([[.5,.5]]),d=SimpleNamespace(qfrc_bias=np.zeros(2)),va=np.array([0,1]),kp=np.ones(1),damping=np.ones(1),bias=np.zeros((1,3)),caps=np.array([[-1.,1.]]))
    q=dict(a=.2,b=.3);v=dict(a=.1,b=.2)
    controller=AttainedHandTracking(teacher,q,np.array([.3]))
    moving,info=controller.force(np.zeros(1),q,v,target_joint_velocities=v)
    np.testing.assert_allclose(moving,[.3])
    assert info['finger_velocity_feedforward']
    np.testing.assert_allclose(info['finger_motor_reference_velocity_rad_s'],[.3])
    stationary,_=controller.force(np.zeros(1),q,v)
    assert stationary[0]<moving[0]
    capped,info=controller.force(np.zeros(1),q,v,target_joint_velocities=dict(a=2.,b=2.))
    np.testing.assert_allclose(capped,[1.]);assert info['finger_clipped_motors']==1
    import pytest
    for bad in [dict(a=.1),dict(a=float('nan'),b=0.),dict(a=2.01,b=0.)]:
        with pytest.raises(ValueError):controller.force(np.zeros(1),q,v,target_joint_velocities=bad)
