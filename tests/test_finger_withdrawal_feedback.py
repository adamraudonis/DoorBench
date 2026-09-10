from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
from doorbench.dexterous.finger_withdrawal_feedback import FingerWithdrawalFeedback


@pytest.mark.parametrize('segment', ['distal','middle'])
def test_ring_feedback_uses_its_motor_and_preserves_other_digit_command(segment):
    xml='''<mujoco><worldbody>
    <body name="rh_rfdistal"><joint type="slide" axis="1 0 0"/><geom size=".01" mass=".1"/></body>
    <body name="rh_ffdistal" pos="0 .1 0"><joint type="slide" axis="0 1 0"/><geom size=".01" mass=".1"/></body>
    </worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml.replace('rh_rfdistal','rh_rf'+segment))
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    t=SimpleNamespace(m=m,d=d,fingers=np.array([0,1]),va=np.array([0,1]),finger_inverse=np.eye(2),caps=np.array([[-.8,.8],[-1.,1.]]))
    feedback=FingerWithdrawalFeedback(t,[0.,0.,0.],digit='rf',segment=segment)
    before=[x.copy() for x in (d.qpos,d.qvel,d.qfrc_applied,d.xfrc_applied)]
    command,info=feedback.force(np.array([0.,.25]),1.,np.array([.02,0.,0.]),1.)
    np.testing.assert_allclose(command,[.8,.25]);assert info['feedback_force_N']==pytest.approx(6.)
    for value,original in zip((d.qpos,d.qvel,d.qfrc_applied,d.xfrc_applied),before):np.testing.assert_array_equal(value,original)
    with pytest.raises(ValueError,match='digit'):FingerWithdrawalFeedback(t,[0,0,0],digit='sixth')


def make_feedback():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="rh_rfdistal">
      <joint type="slide" axis="1 0 0"/><geom size=".01" mass=".1"/>
      </body></worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    teacher=SimpleNamespace(m=m,d=d,fingers=np.array([0]),va=np.array([0]),
        finger_inverse=np.eye(1),caps=np.array([[-.8,.8]]))
    return FingerWithdrawalFeedback(teacher,[0,0,0],digit='rf')


def test_measured_handoff_discontinuity_is_bridged_without_clearing_speed_history():
    previous=np.array([.10653454410399225,-.10617076972982405,1.040675929259851])
    incoming=np.array([.11454635202835586,-.1148253168079548,1.0422590912116352])
    original=make_feedback();original.force([0.],1.,previous,1.)
    with pytest.raises(ValueError,match='0.5 m/s'):
        original.force([0.],1.002,incoming,1.002)
    feedback=make_feedback();feedback.force([0.],1.,previous,1.)
    _,info=feedback.force([0.],1.002,incoming,1.002,start_handoff=True)
    np.testing.assert_allclose(feedback.previous[1],previous,atol=1e-15,rtol=0)
    assert info['reference_speed_m_s']<1e-10
    speeds=[]
    for t in 1.002+np.arange(1,502)*.002:
        command,info=feedback.force([0.],float(t),incoming,float(t))
        speeds.append(info['reference_speed_m_s'])
        assert abs(command[0])<=.8 and info['feedback_force_N']<=6.000000001
    np.testing.assert_array_equal(feedback.previous[1],incoming)
    assert max(speeds)<.023
    assert speeds[-1]<1e-10


def test_handoff_rejects_unbounded_or_moving_source_and_keeps_incoming_speed_guard():
    f=make_feedback()
    with pytest.raises(ValueError,match='preceding target'):
        f.force([0.],1.,np.zeros(3),1.,start_handoff=True)
    f.force([0.],1.,np.zeros(3),1.)
    with pytest.raises(ValueError,match='3cm'):
        f.force([0.],1.002,np.array([.031,0,0]),1.002,start_handoff=True)
    f.force([0.],1.002,np.array([.0001,0,0]),1.002)
    with pytest.raises(ValueError,match='stationary'):
        f.force([0.],1.004,np.array([.001,0,0]),1.004,start_handoff=True)
    f=make_feedback();f.force([0.],1.,np.zeros(3),1.)
    f.force([0.],1.002,np.array([.01,0,0]),1.002,start_handoff=True)
    with pytest.raises(ValueError,match='0.5 m/s'):
        f.force([0.],1.004,np.array([.02,0,0]),1.004)


@pytest.mark.parametrize('jump',[.0254747593530975,.025444382020002876])
def test_measured_middle_segment_offsets_remain_slow_and_force_bounded(jump):
    f=make_feedback();f.force([0.],1.,np.zeros(3),1.)
    f.force([0.],1.002,np.array([jump,0,0]),1.002,start_handoff=True)
    peak=0.
    for t in 1.002+np.arange(1,502)*.002:
        command,info=f.force([0.],float(t),np.array([jump,0,0]),float(t))
        peak=max(peak,info['reference_speed_m_s'])
        assert abs(command[0])<=.8 and info['feedback_force_N']<=6.000000001
    assert peak<.048
    np.testing.assert_array_equal(f.previous[1],[jump,0,0])
