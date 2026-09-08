import copy, inspect, json
from pathlib import Path
import numpy as np
import pytest
from doorbench.dexterous.sensor_arm_balance_runtime import SensorArmBalanceRuntime,evaluate_sensor_arm_balance,ARM_NAMES
from doorbench.dexterous.arm_balance_schedule import scripted_goals
from doorbench.dexterous.sensor_contract import ActorObservationBuilder
from test_sensor_balance import authored

ROOT=Path(__file__).parents[1]
def schedule():return json.loads((ROOT/'configs/dexterous/sensor-arm-balance-v1.json').read_text())
def make(authored,plan=None):
    robot,motors,layout,_=authored
    return SensorArmBalanceRuntime(robot,copy.deepcopy(motors),copy.deepcopy(layout),ROOT/'configs/dexterous/sensor-balance-v1.json',schedule() if plan is None else plan)
def cold(c):return ActorObservationBuilder(joint_count=69,action_count=61,tactile_dimension=c._shapes['tactile'][0],image_shape=(128,128,3)).observe(now_s=0.,previous_action=np.zeros(61))


def test_arm_runtime_has_only_packet_clock_and_declared_schedule(authored):
    assert list(inspect.signature(SensorArmBalanceRuntime.force).parameters)==['self','packet','now_s']
    c=make(authored);p=cold(c)
    with pytest.raises(ValueError,match='reset'):c.force(p,0.)
    c.reset_episode();f=c.force(p,0.)
    assert f.shape==(61,) and len(c.goal_names)==12 and not hasattr(c,'checkpoint_sha256')
    np.testing.assert_allclose(c.previous_action,f/c._caps[:,1],atol=1e-7,rtol=0)
    assert c.last_info['controller']=='sensor_arm_balance_v1'
    assert c._controller.balance.d.time==0.


@pytest.mark.parametrize('change',['root','torso','empty','nan','bool','duration'])
def test_undeclared_or_unbounded_schedule_is_rejected(authored,change):
    p=schedule()
    if change=='root':p['root']=[0,0,1]
    if change=='torso':p['deltas_rad']['torso']=.1
    if change=='empty':p['deltas_rad']={}
    if change=='nan':p['deltas_rad']['left_elbow']=float('nan')
    if change=='bool':p['start_s']=True
    if change=='duration':p['duration_s']=5.
    with pytest.raises((ValueError,TypeError)):make(authored,p)


def test_bad_runtime_packet_is_terminal_and_reset_restores(authored):
    c=make(authored);c.reset_episode();p=cold(c);p['root']=np.zeros(3)
    with pytest.raises(ValueError):c.force(p,0.)
    with pytest.raises(ValueError,match='reset'):c.force(cold(c),0.)
    c.reset_episode();c.force(cold(c),0.)
    p=cold(c)
    with pytest.raises(ValueError,match='Previous action'):c.force(p,.002)


def actual_rows():
    base={n:0. for n in ARM_NAMES};s=schedule();rows=[]
    for i in range(3000):
        goals=scripted_goals(i*.002,base,ARM_NAMES,s)
        info=dict(controller='sensor_arm_balance_v1',cold_start=i==0,qp_status='solved',qp_failures=0,calculator_time_s=0.,
            goal_joint_names=list(ARM_NAMES),goal_joint_position_rad=[goals[n] for n in ARM_NAMES])
        rows.append(dict(time_s=(i+1)*.002,root13_actororigin=[0.,0.,.87,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],torso_tilt_deg=.3,
            foot_floor_loads=[180.,180.],hand_contact_count=0,controller_info=info,actual_arm_joint_position=goals.copy()))
    return rows,base,s


def score(rows,base,s):return evaluate_sensor_arm_balance(rows,{'original_caps':True},initial_arm_joint_position=base,schedule=s)


def test_full_schedule_actual_motion_and_tracking_are_required():
    rows,base,s=actual_rows();assert score(rows,base,s)['passed']
    for row in rows:row['actual_arm_joint_position']=base.copy()
    r=score(rows,base,s)
    assert not r['passed'] and not r['checks']['physical_arm_motion_delivered'] and not r['checks']['arm_tracking']


@pytest.mark.parametrize('change',['missing','short','gap','contact','unloaded','unquiet','bad_info','next_goal','missing_initial','failed_physics'])
def test_missing_or_wrong_actual_evidence_fails_closed(change):
    rows,base,s=actual_rows();physics={'original_caps':True}
    if change=='missing':rows[0].pop('actual_arm_joint_position')
    if change=='short':rows=rows[:-1]
    if change=='gap':rows[12]['time_s']+=.002
    if change=='contact':rows[16]['hand_contact_count']=1
    if change=='unloaded':rows[-1]['foot_floor_loads'][0]=0.
    if change=='unquiet':rows[-1]['root13_actororigin'][7]=.04
    if change=='bad_info':rows[1]['controller_info']['qp_failures']=1
    if change=='next_goal':rows[900]['controller_info']['goal_joint_position_rad']=rows[901]['controller_info']['goal_joint_position_rad']
    if change=='missing_initial':base.pop(ARM_NAMES[0])
    if change=='failed_physics':physics['original_caps']=False
    result=evaluate_sensor_arm_balance(rows,physics,initial_arm_joint_position=base,schedule=s)
    assert not result['passed']


def test_stationary_protocol_stays_separate():
    from doorbench.dexterous.sensor_balance_runtime import evaluate_sensor_balance
    rows,base,s=actual_rows()
    with pytest.raises(ValueError):evaluate_sensor_arm_balance(rows,{},initial_arm_joint_position=base,schedule=s,expected_duration_s=5.)
    with pytest.raises(ValueError):evaluate_sensor_balance([],{},expected_duration_s=6.)
