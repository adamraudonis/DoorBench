"""Index-only joint-goal feedback preserves the original sensor/cap boundary."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from test_sensor_balance import authored,cold,valid
from test_sensor_distal_touch_control import packet
from test_sensor_distal_touch_impedance import make as impedance
from doorbench.dexterous.sensor_distal_touch_impedance import PROTOCOL
from doorbench.dexterous.sensor_index_touch_control import (
    INDEX_PROTOCOL,SensorIndexTouchController,validate_index_protocol)


def make(authored,stub=False):
    base=impedance(authored,stub=stub)
    return SensorIndexTouchController(base.arm,base.schedule,authored[2],
                                     PROTOCOL.copy(),authored[1],INDEX_PROTOCOL.copy())


def test_frozen_scope_and_parameters_reject_target_or_gate_changes():
    path=Path(__file__).resolve().parents[1]/'configs/dexterous/sensor-index-touch-v1.json'
    assert validate_index_protocol(json.loads(path.read_text()))==INDEX_PROTOCOL
    for k,v in [('joint_name','torso'),('maximum_offset_rad',.02),('load_target_N',1.),
                ('start_after_s',18.),('maximum_offset_rate_radps',True)]:
        bad=dict(INDEX_PROTOCOL);bad[k]=v
        with pytest.raises(ValueError):validate_index_protocol(bad)


def test_unchanged_first19s_then_exact_index_only_bounded_offset(authored):
    c=make(authored,stub=True);base=impedance(authored,stub=True)
    previous=0.
    for i in range(12000):
        t=i*.002
        p=packet(c,t,[.5,2,2,2,3])
        f,info=c.force(p,now_s=t);g,old=base.force(copy.deepcopy(p),now_s=t)
        np.testing.assert_array_equal(f,c.arm.last_force)
        goals=dict(zip(info['goal_joint_names'],info['goal_joint_position_rad']))
        original=dict(zip(old['goal_joint_names'],old['goal_joint_position_rad']))
        for name in c.goal_names:
            assert abs(goals[name]-original[name]-(c.index_delta if name=='rh_FFJ3' else 0.))<1e-14
        assert 0<=c.index_delta<=.01 and abs(c.index_delta-previous)<=.01*.002+1e-14
        if t<=19.:assert c.index_delta==0.
        if t>19.:
            assert abs(info['index_feedback_decision_s']-(t-.002))<1e-10
            assert info['index_feedback_projection_N']==pytest.approx(.5)
        previous=c.index_delta
    assert c.index_delta==.01
    assert np.array_equal(c.minimum,[.8,.8,.8,.8,1.2])
    assert np.array_equal(c.maximum,[.06,.06,.06,.06,.08])
    c.reset_episode();assert c.index_delta==0. and c._index_schedule.offset==0.
    np.testing.assert_array_equal(c.arm.balance.kp,c._initial_kp)


def test_local_overload_reverses_proximal_offset_at_declared_rate(authored):
    c=make(authored,stub=True)
    for i in range(11000):c.force(packet(c,i*.002,[.5,2,2,2,3]),now_s=i*.002)
    assert c.index_delta==.01
    for i in range(11000,12000):c.force(packet(c,i*.002,[4,2,2,2,3]),now_s=i*.002)
    assert c.index_delta==0.


def test_real_commands_have_one_owner_and_oracle_packet_is_terminal(authored):
    c=make(authored);b=c.arm.balance
    for t in (0.,.002):
        p=cold(b) if t==0. else valid(b,t)
        force,_=c.force(p,now_s=t)
        np.testing.assert_array_equal(force,b.last_force)
        assert np.all(force>=b.caps[:,0]) and np.all(force<=b.caps[:,1])
    p=valid(b,.004);p['previous_action'].fill(0.)
    with pytest.raises(ValueError,match='Previous action'):c.force(p,now_s=.004)
    with pytest.raises(RuntimeError):c.force(p,now_s=.004)
    c.reset_episode();p=cold(b);p['index_contact_position']=np.zeros(3)
    with pytest.raises(ValueError):c.force(p,now_s=0.)
    with pytest.raises(RuntimeError):c.force(cold(b),now_s=0.)
