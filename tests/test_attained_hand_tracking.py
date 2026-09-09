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
