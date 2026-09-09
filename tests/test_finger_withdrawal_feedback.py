from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
from doorbench.dexterous.finger_withdrawal_feedback import FingerWithdrawalFeedback


def test_ring_feedback_uses_its_motor_and_preserves_other_digit_command():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="rh_rfdistal"><joint type="slide" axis="1 0 0"/><geom size=".01" mass=".1"/></body>
    <body name="rh_ffdistal" pos="0 .1 0"><joint type="slide" axis="0 1 0"/><geom size=".01" mass=".1"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    t=SimpleNamespace(m=m,d=d,fingers=np.array([0,1]),va=np.array([0,1]),finger_inverse=np.eye(2),caps=np.array([[-.8,.8],[-1.,1.]]))
    feedback=FingerWithdrawalFeedback(t,[0.,0.,0.],digit='rf')
    before=[x.copy() for x in (d.qpos,d.qvel,d.qfrc_applied,d.xfrc_applied)]
    command,info=feedback.force(np.array([0.,.25]),1.,np.array([.02,0.,0.]),1.)
    np.testing.assert_allclose(command,[.8,.25]);assert info['feedback_force_N']==pytest.approx(6.)
    for value,original in zip((d.qpos,d.qvel,d.qfrc_applied,d.xfrc_applied),before):np.testing.assert_array_equal(value,original)
    with pytest.raises(ValueError,match='digit'):FingerWithdrawalFeedback(t,[0,0,0],digit='sixth')
