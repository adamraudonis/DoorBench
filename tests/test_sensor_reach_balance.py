"""Explicit coordinated reach goals retain sensor-only pelvis/leg control."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from test_sensor_balance import authored,cold,valid
from doorbench.dexterous.sensor_reach_balance import SensorReachBalanceController
from doorbench.dexterous.reach_balance_schedule import ReferenceReachSchedule


def wrapper(authored,torso=True):return SensorReachBalanceController(*copy.deepcopy(authored),image_shape=(8,8,3),allow_torso_yaw=torso)
def goals(c):return dict(zip(c.goal_names,c.last_goals))


def test_torso_scope_is_explicit_and_lower_body_columns_are_excluded(authored):
    c=wrapper(authored);a=wrapper(authored,False)
    assert 'torso' in c.goal_names and 'torso' not in a.goal_names
    assert len(c.goal_names)==30 and len(c.arm_motors)==26
    assert len(a.goal_names)==29 and len(a.arm_motors)==25
    assert not any(n.startswith('left_') or 'hip_' in n or 'knee' in n or 'ankle' in n for n in c.goal_names)
    other=[i for i in range(69) if i not in c.indices]
    assert not np.any(c.balance.matrix[np.ix_(c.arm_motors,other)])


def test_coupled_finger_targets_use_authored_motor_sum(authored):
    c=wrapper(authored);b=c.balance;c.force(cold(b),now_s=0.,joint_goals=goals(c))
    g=goals(c);g['rh_FFJ1']+=.001;g['rh_FFJ2']+=.001;before=b.target.copy()
    c.force(valid(b,.002),now_s=.002,joint_goals=g)
    changed=np.flatnonzero(b.target!=before)
    assert len(changed)==1
    assert (b.target-before)[changed[0]]==pytest.approx(.002)
    assert b.actions[changed[0]]=='rh_A_FFJ0'


@pytest.mark.parametrize('kind',['loopback','left_hand','pelvis','leg','bound','slew','nan'])
def test_bad_reach_targets_require_episode_reset(authored,kind):
    c=wrapper(authored);g=goals(c)
    if kind=='loopback':g['rh_FFJ1']=g['rh_FFJ2']+.01
    if kind=='left_hand':g['lh_FFJ1']=.2
    if kind=='pelvis':g['root_yaw']=0.
    if kind=='leg':g['left_knee']=1.
    if kind=='bound':g['rh_THJ1']=100.
    if kind=='slew':g['torso']+=.01
    if kind=='nan':g['rh_FFJ1']=np.nan
    with pytest.raises(ValueError):c.force(cold(c.balance),now_s=0.,joint_goals=g)
    with pytest.raises(RuntimeError):c.force(cold(c.balance),now_s=0.)


def test_constant_limit_column_is_bitwise_preserved_by_interpolation():
    schedule=ReferenceReachSchedule(['wrist','elbow'],['wrist','elbow'],[[-.15,1.],[-.15,1.3]])
    for t in np.arange(0,11.002,.002):assert schedule.goals(t)['wrist']==-.15
    np.testing.assert_array_equal(schedule.values_at_fraction(0),schedule.path[0])


def test_route_selects_only_joint_columns_and_bounds_prefix():
    s=ReferenceReachSchedule(['a'],['a','b'],[[0.,8.],[1.,20.]],stop_fraction=.45)
    assert s.goals(0.)=={'a':0.}
    assert s.goals(11.)['a']==pytest.approx(.45)
    with pytest.raises(ValueError):ReferenceReachSchedule(['a'],['a'],[[0.],[1.]],stop_fraction=1.)
    with pytest.raises(ValueError):s.goals(float('nan'))


@pytest.mark.parametrize('kwargs',[
    {'finger_impedance_multiplier':.9},{'finger_impedance_multiplier':4.1},
    {'finger_velocity_damping':-.001},{'finger_velocity_damping':.101},
    {'finger_impedance_multiplier':float('nan')},
])
def test_declared_finger_feedback_is_bounded(authored,kwargs):
    with pytest.raises(ValueError):SensorReachBalanceController(*copy.deepcopy(authored),**kwargs)


def test_new_finger_feedback_changes_policy_only_and_preserves_previous_command(authored):
    c=wrapper(authored);strong=SensorReachBalanceController(*copy.deepcopy(authored),image_shape=(8,8,3),allow_torso_yaw=True,
        finger_impedance_multiplier=4.,finger_velocity_damping=.05)
    a,b=c.balance,strong.balance
    finger=np.array([n.startswith('rh_') and 'WRJ' not in n for n in a.actions])
    np.testing.assert_array_equal(b.kp[finger],4*a.kp[finger])
    np.testing.assert_array_equal(b.bias[finger,1],4*a.bias[finger,1])
    np.testing.assert_array_equal(b.bias[finger,2],a.bias[finger,2]-.05)
    np.testing.assert_array_equal(b.kp[~finger],a.kp[~finger])
    np.testing.assert_array_equal(b.bias[~finger],a.bias[~finger])
    for name in ('actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange',
                 'body_mass','body_inertia','dof_damping','dof_frictionloss','tendon_range','jnt_range'):
        np.testing.assert_array_equal(getattr(a.m,name),getattr(b.m,name))
    strong.force(cold(b),now_s=0.,joint_goals=goals(strong));g=goals(strong);g['rh_FFJ1']+=.001;g['rh_FFJ2']+=.001
    f,info=strong.force(valid(b,.002),now_s=.002,joint_goals=g)
    np.testing.assert_array_equal(f,strong.last_force)
    assert np.all(f>=strong.caps[:,0]) and np.all(f<=strong.caps[:,1])
    assert info['finger_impedance_multiplier']==4.
    assert info['coupled_joint_goals_are_nominal']
    assert set(info['goal_motor_names'])==set(a.actions[i] for i in strong.arm_motors)
    # No extra force-summing wrapper invalidates the next packet's previous action.
    strong.force(valid(b,.004),now_s=.004)
    strong.reset_episode()
    np.testing.assert_array_equal(strong.last_goals,b.desired[strong.indices])
    np.testing.assert_array_equal(b.target,b.matrix@b.desired)


def test_coupled_motor_coordinate_cannot_claim_independent_split_tracking(authored):
    c=wrapper(authored);b=c.balance;q=b.desired.copy();r=q.copy()
    i,j=[b.names.index('rh_FFJ'+n) for n in ('1','2')]
    r[i]-=.03;r[j]+=.03
    # Motor-transmission tracking can pass despite differing passive split:
    # the probe still independently audits all69 actual joint angles/anatomy.
    np.testing.assert_allclose(b.matrix@q,b.matrix@r,atol=1e-15)
    assert abs(q[i]-r[i])>.02


def test_optional_moving_reference_damping_uses_only_accepted_goal_difference(authored):
    c=SensorReachBalanceController(*copy.deepcopy(authored),image_shape=(8,8,3),allow_torso_yaw=True,
        finger_impedance_multiplier=4.,finger_velocity_damping=.05,finger_target_velocity_damping=True)
    b=c.balance;c.force(cold(b),now_s=0.,joint_goals=goals(c));g=goals(c)
    g['rh_FFJ1']+=.001;g['rh_FFJ2']+=.001
    f,info=c.force(valid(b,.002),now_s=.002,joint_goals=g)
    i=b.actions.index('rh_A_FFJ0')
    assert b.bias[i,0]==pytest.approx(.05)
    assert info['goal_motor_velocity_radps'][info['goal_motor_names'].index('rh_A_FFJ0')]==pytest.approx(1.)
    np.testing.assert_array_equal(f,c.last_force)
    c.force(valid(b,.004),now_s=.004)
    np.testing.assert_array_equal(b.bias[:,0],c.original_constant_bias)
    c.reset_episode();np.testing.assert_array_equal(b.bias[:,0],c.original_constant_bias)
