"""Opt-in arm commands preserve stationary behavior and sensor-only balance."""
import copy
import inspect
import numpy as np
import pytest
from test_sensor_balance import authored,create,cold,valid
from doorbench.dexterous.sensor_arm_balance import SensorArmBalanceController
from scripts.dexterous.probe_sensor_arm_balance import scripted_goals,validate_schedule


def wrapper(authored):return SensorArmBalanceController(*copy.deepcopy(authored),image_shape=(8,8,3))
def goals(c):return dict(zip(c.goal_names,c.last_goals))


def test_explicit_goals_are_separate_from_sensor_packet():
    assert list(inspect.signature(SensorArmBalanceController.force).parameters)==['self','packet','now_s','joint_goals']


def test_omitted_goals_match_qualified_stationary_forces(authored):
    c=wrapper(authored);base=create(authored)
    for i in range(20):
        p=cold(base) if i==0 else valid(base,i*.002)
        a,_=base.force(p,now_s=i*.002);b,_=c.force(p,now_s=i*.002)
        np.testing.assert_array_equal(a,b)
    assert c.balance.d.time==0.


def test_goals_change_only_arm_targets_and_reset_restores_stationary(authored):
    c=wrapper(authored);base=c.balance;c.force(cold(base),now_s=0.,joint_goals=goals(c));old=base.target.copy()
    g=goals(c);g['left_elbow']+=.0008
    c.force(valid(base,.002),now_s=.002,joint_goals=g)
    changed=np.flatnonzero(base.target!=old)
    assert changed.tolist()==[base.actions.index('left_elbow')]
    np.testing.assert_array_equal(base.stance.joint_target,base.desired[base.stance.local])
    held=base.target.copy();c.force(valid(base,.004),now_s=.004)
    np.testing.assert_array_equal(base.target,held)
    c.reset_episode();np.testing.assert_array_equal(base.target,old)


@pytest.mark.parametrize('kind',['torso','leg','finger','world','extra','missing','nan','large_step','joint_limit','array'])
def test_invalid_goal_rejected_and_episode_terminal(authored,kind):
    c=wrapper(authored);g=goals(c)
    if kind=='torso':g['torso']=0.
    if kind=='leg':g['left_knee']=1.
    if kind=='finger':g['rh_FFJ1']=.1
    if kind=='world':g['root']=0.
    if kind=='extra':g['unrecognized_joint']=0.
    if kind=='missing':g.pop('left_elbow')
    if kind=='nan':g['left_elbow']=np.nan
    if kind=='large_step':g['left_elbow']+=.01
    if kind=='joint_limit':g['left_elbow']=100.
    if kind=='array':g['left_elbow']=np.array([1.])
    with pytest.raises(ValueError):c.force(cold(c.balance),now_s=0.,joint_goals=g)
    with pytest.raises(RuntimeError,match='requires reset'):c.force(cold(c.balance),now_s=0.)


def test_script_has_zero_endpoint_slew_and_bounded_goals():
    import json
    from pathlib import Path
    s=json.loads((Path(__file__).resolve().parents[1]/'configs/dexterous/sensor-arm-balance-v1.json').read_text())
    names=tuple(s['deltas_rad']);desired=dict.fromkeys(names,0.)
    validate_schedule(s,names,6.)
    values=np.array([[*scripted_goals(t,desired,names,s).values()] for t in np.arange(0,6.002,.002)])
    assert np.max(abs(np.diff(values,axis=0)))/.002<.5
    np.testing.assert_array_equal(values[0],values[-1])
    assert np.max(abs(values))==pytest.approx(.2)
