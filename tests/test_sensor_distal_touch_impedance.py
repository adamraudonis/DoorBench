"""Smooth opt-in policy stiffness; original hardware and default stay frozen."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from test_sensor_balance import authored, cold, valid
from test_sensor_distal_touch_control import create as template, Arm, packet
from doorbench.dexterous.sensor_reach_balance import SensorReachBalanceController
from doorbench.dexterous.sensor_distal_touch_control import SensorDistalTouchController
from doorbench.dexterous.sensor_distal_touch_impedance import (
    PROTOCOL, SensorDistalTouchImpedanceController, finger_gain_multiplier,
    validate_impedance_protocol)


def make(authored, stub=False):
    real = SensorReachBalanceController(*copy.deepcopy(authored), allow_torso_yaw=True,
        image_shape=(8,8,3), finger_impedance_multiplier=4.,
        finger_velocity_damping=.05, finger_target_velocity_damping=True)
    arm = real
    if stub:
        arm = Arm(real)
        for key in ('finger_impedance_multiplier', 'finger_velocity_damping',
                    'finger_target_velocity_damping', 'finger_motors'):
            setattr(arm, key, getattr(real, key))
    return SensorDistalTouchImpedanceController(arm, template(authored).schedule,
                                               authored[2], PROTOCOL.copy(), authored[1])


def test_exact_frozen_protocol_and_c2_ramp():
    path = Path(__file__).resolve().parents[1] / 'configs/dexterous/sensor-distal-touch-impedance-v1.json'
    assert validate_impedance_protocol(json.loads(path.read_text())) == PROTOCOL
    assert [finger_gain_multiplier(t) for t in (0.,18.998,19.,20.,21.,36.)] == [4.,4.,4.,10.,16.,16.]
    grid = np.array([finger_gain_multiplier(t) for t in np.arange(19.,21.0001,.002)])
    assert np.all(np.diff(grid) >= -1e-10)
    assert np.max(np.diff(grid)) < .022501
    assert finger_gain_multiplier(19.000001)-4. < 1e-12
    assert 16.-finger_gain_multiplier(20.999999) < 1e-12
    for key, value in [('final_finger_position_gain_multiplier', 17.), ('acquisition_unchanged_until_s', 18.),
                       ('ramp_duration_s', True), ('finger_velocity_damping_Nm_s_per_rad', .06)]:
        bad = dict(PROTOCOL);bad[key] = value
        with pytest.raises(ValueError):validate_impedance_protocol(bad)


def test_prefix_identical_then_only_finger_position_gains_change_and_reset(authored):
    c = make(authored, stub=True);b=c.arm.balance
    native = {key:getattr(b.m,key).copy() for key in ('actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_ctrlrange')}
    initial_kp=b.kp.copy();initial_bias=b.bias.copy()
    for i in range(10600):
        t=i*.002
        f,info=c.force(packet(c,t,[2,2,2,2,3]),now_s=t)
        if t<19.:
            np.testing.assert_array_equal(b.kp,initial_kp)
            np.testing.assert_array_equal(b.bias,initial_bias)
            assert dict(zip(info['goal_joint_names'],info['goal_joint_position_rad']))==c.schedule.goals(t)
        np.testing.assert_array_equal(f,c.arm.last_force)
    np.testing.assert_array_equal(b.kp[c._finger_mask],4.*initial_kp[c._finger_mask])
    np.testing.assert_array_equal(b.kp[~c._finger_mask],initial_kp[~c._finger_mask])
    np.testing.assert_array_equal(b.bias[:,[0,2]],initial_bias[:,[0,2]])
    np.testing.assert_array_equal(b.bias[~c._finger_mask,1],initial_bias[~c._finger_mask,1])
    for key,value in native.items():np.testing.assert_array_equal(getattr(b.m,key),value)
    c.reset_episode();np.testing.assert_array_equal(b.kp,initial_kp)
    np.testing.assert_array_equal(b.bias[:,1],initial_bias[:,1])
    assert c.virtual_s==19. and c.last_time is None


def test_real_two_decision_default_equivalence_caps_and_previous_force(authored):
    c=make(authored);b=c.arm.balance
    base_arm=SensorReachBalanceController(*copy.deepcopy(authored),allow_torso_yaw=True,
        image_shape=(8,8,3),finger_impedance_multiplier=4.,finger_velocity_damping=.05,
        finger_target_velocity_damping=True)
    old=SensorDistalTouchController(base_arm,c.schedule,authored[2])
    for t in (0.,.002):
        p=cold(b) if t==0 else valid(b,t)
        f,_=c.force(p,now_s=t);g,_=old.force(copy.deepcopy(p),now_s=t)
        np.testing.assert_array_equal(f,g);np.testing.assert_array_equal(f,b.last_force)
        assert np.all(f>=b.caps[:,0]) and np.all(f<=b.caps[:,1])
    p=valid(b,.004);p['previous_action'].fill(0.)
    with pytest.raises(ValueError,match='Previous action'):c.force(p,now_s=.004)
    with pytest.raises(RuntimeError):c.force(p,now_s=.004)
    c.reset_episode()
    with pytest.raises(ValueError):c.force(cold(b),now_s=False)


def test_wrong_initial_feedback_rejected(authored):
    wrong=SensorReachBalanceController(*copy.deepcopy(authored),allow_torso_yaw=True,image_shape=(8,8,3))
    with pytest.raises(ValueError,match='fourfold'):
        SensorDistalTouchImpedanceController(wrong,template(authored).schedule,authored[2],PROTOCOL.copy(),authored[1])
