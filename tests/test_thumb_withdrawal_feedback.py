from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
from doorbench.dexterous.thumb_withdrawal_feedback import ThumbWithdrawalFeedback


def test_feedback_is_capped_motor_only_and_does_not_write_physics_state():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="rh_thdistal"><joint type="slide" axis="1 0 0"/><geom size=".01" mass=".1"/></body><body><joint type="slide" axis="0 1 0"/><geom size=".01" mass=".1"/></body></worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    teacher=SimpleNamespace(m=m,d=d,fingers=np.array([0]),va=np.array([0,1]),finger_inverse=np.array([[1.,0.]]),caps=np.array([[-1.,1.],[-2.,2.]]))
    feedback=ThumbWithdrawalFeedback(teacher,[0.,0.,0.])
    before=[v.copy() for v in (d.qpos,d.qvel,d.qfrc_applied,d.xfrc_applied)]
    result,info=feedback.force(np.array([0.,.7]),1.,np.array([.01,0.,0.]),1.)
    np.testing.assert_allclose(result,[1.,.7]);assert info['thumb_feedback_force_N']==pytest.approx(6.)
    for actual,expected in zip((d.qpos,d.qvel,d.qfrc_applied,d.xfrc_applied),before):np.testing.assert_array_equal(actual,expected)
    with pytest.raises(ValueError,match='clock'):feedback.force(result,1.,np.zeros(3),1.)
    with pytest.raises(ValueError,match='0.5 m/s'):feedback.force(result,1.002,np.array([.1,0.,0.]),1.)
