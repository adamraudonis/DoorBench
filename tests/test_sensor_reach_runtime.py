import copy,hashlib,inspect,json
from pathlib import Path
import numpy as np
import pytest
from doorbench.dexterous.sensor_reach_runtime import (SensorReachBalanceRuntime,PARAMETER_VALUES,ROUTE_SCHEMA,PROTOCOL_SCHEMA,
    reach_names,load_reach_inputs,prepare_reach_runtime_inputs)
from doorbench.dexterous.sensor_reach_evaluation import ReachEvaluationRobot,evaluate_sensor_reach_balance,PHYSICS_FAMILIES
from doorbench.dexterous.sensor_contract import ActorObservationBuilder
from test_sensor_balance import authored
ROOT=Path(__file__).parents[1]
CAL=ROOT/'configs/dexterous/sensor-balance-v1.json'


def inputs(authored):
    _,motors,_,posture=authored;names=reach_names(motors['joint_names']);q=np.array([posture[n] for n in names]);path=np.tile(q,(401,1))
    path[:,names.index('torso')]+=np.linspace(0,.2,401)
    path[:,names.index('right_elbow')]-=np.linspace(0,.1,401)
    route=dict(schema=ROUTE_SCHEMA,source_reference_sha256='a'*64,joint_names=list(names),path_joint_position_rad=path.tolist())
    protocol=dict(schema=PROTOCOL_SCHEMA,source_schedule_sha256='b'*64,joint_route_sha256='',
        parameters=dict(PARAMETER_VALUES,scope='Synthetic joint-only test route',source_reference_sha256='a'*64))
    bind(protocol,route);return protocol,route


def bind(p,r):p['joint_route_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def runtime(authored,p=None,r=None):
    if p is None:p,r=inputs(authored)
    return SensorReachBalanceRuntime(*authored[:3],CAL,p,r)
def cold(c):return ActorObservationBuilder(joint_count=69,action_count=61,tactile_dimension=c._shapes['tactile'][0],image_shape=(128,128,3)).observe(now_s=0.,previous_action=np.zeros(61))


def test_boundary_clock_history_and_caps(authored):
    assert list(inspect.signature(SensorReachBalanceRuntime.force).parameters)==['self','packet','now_s']
    c=runtime(authored);p=cold(c)
    with pytest.raises(ValueError,match='reset'):c.force(p,0.)
    c.reset_episode();f=c.force(p,0.)
    assert f.shape==(61,) and len(c.goal_names)==30 and len(c.goal_motor_names)==26 and c.duration_s==11.
    assert c._controller.balance.d.time==0 and c.last_info['controller']=='sensor_reach_balance_v1'
    np.testing.assert_allclose(c.previous_action,f/c._caps[:,1],atol=1e-7,rtol=0)
    with pytest.raises(ValueError,match='Previous action'):c.force(p,.002)
    with pytest.raises(ValueError,match='reset'):c.force(p,0.)
    c.reset_episode();p['root']=np.zeros(3)
    with pytest.raises(ValueError):c.force(p,0.)
    c.reset_episode()
    with pytest.raises(ValueError,match='clock'):c.force(cold(c),11.)


@pytest.mark.parametrize('change',['root','leg','hash','calibration','duration','gain','nan','short','bool'])
def test_changed_runtime_projection_is_rejected(authored,change):
    p,r=inputs(authored)
    if change=='root':r['initial_root']=[0,0,.87]
    if change=='leg':r['joint_names'][0]='left_knee'
    if change=='hash':p['joint_route_sha256']='c'*64
    if change=='calibration':r['path_joint_position_rad'][0][0]+=.01
    if change=='duration':p['parameters']['duration_s']=12.
    if change=='gain':p['parameters']['finger_impedance_multiplier']=5.
    if change=='nan':r['path_joint_position_rad'][1][0]=float('nan')
    if change=='short':r['path_joint_position_rad'].pop()
    if change=='bool':p['parameters']['maximum_goal_speed_radps']=True
    if change not in ('hash','nan'):bind(p,r)
    with pytest.raises((ValueError,TypeError)):runtime(authored,p,r)


def test_projection_strips_root_and_keeps_exact_named_columns(authored,tmp_path):
    _,motors,_,posture=authored;names=motors['joint_names'];path=np.tile([posture[n] for n in names],(401,1))
    ref={'initial_root':[1,2,3,1,0,0,0],'door':{'angle':0},'acquisition':{'joint_names':names,'path_qpos':path.tolist()}}
    src=tmp_path/'reference.json';src.write_text(json.dumps(ref));s=dict(PARAMETER_VALUES,scope='test projection',source_reference_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
    schedule=tmp_path/'schedule.json';schedule.write_text(json.dumps(s));route=tmp_path/'route.json';protocol=tmp_path/'protocol.json'
    receipt=prepare_reach_runtime_inputs(src,schedule,motors,route,protocol);out=json.loads(route.read_text())
    assert set(out)=={'schema','source_reference_sha256','joint_names','path_joint_position_rad'}
    assert len(out['joint_names'])==30 and all(not n.startswith('left_') and not n.startswith('lh_') for n in out['joint_names'])
    assert receipt['joint_route_sha256']==hashlib.sha256(route.read_bytes()).hexdigest()
    load_reach_inputs(protocol,route,names,posture)
    with pytest.raises(FileExistsError):prepare_reach_runtime_inputs(src,schedule,motors,route,protocol)


@pytest.fixture(scope='module')
def example(authored):
    p,r=inputs(authored);m=ReachEvaluationRobot(authored[0],authored[1],CAL,p,r);root=[0.,0.,.87,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.];rows=[]
    for i in range(5500):
        q,coords=m.target(i*.002)
        info=dict(controller='sensor_reach_balance_v1',cold_start=i==0,qp_status='solved',qp_failures=0,calculator_time_s=0.,
            goal_joint_names=list(m.goal_names),goal_joint_position_rad=q[m.indices].tolist(),goal_motor_names=list(m.goal_motor_names),
            goal_motor_coordinates=coords.tolist(),coupled_joint_goals_are_nominal=True)
        rows.append(dict(time_s=(i+1)*.002,root13_actororigin=root.copy(),torso_tilt_deg=.3,foot_floor_loads=[250.,250.],hand_contact_count=0,
            controller_info=info,actual_joint_position=dict(zip(m.names,q.tolist()))))
    return rows,root,p,r


def score(authored,example,rows=None,physics=None):
    original,root,p,r=example
    return evaluate_sensor_reach_balance(original if rows is None else rows,{family[0]:True for family in PHYSICS_FAMILIES} if physics is None else physics,
        robot_xml=authored[0],motors=authored[1],initial_root13_actororigin=root,initial_joint_position=authored[3],calibration=CAL,protocol=p,joint_route=r)


def test_complete_actual_motion_and_robot_only_fk_pass(authored,example):
    report=score(authored,example);assert report['passed'],report
    assert report['maximum_motor_coordinate_tracking_error_rad']==0 and report['palm_endpoint_error_m']==0
    assert report['calculator_time_s']==0 and report['evaluator_state_is_actor_input'] is False


@pytest.mark.parametrize('change',['short','next_goal','contact','missing_joint','wrong_loopback','joint_stop','wrong_body_endpoint','no_motion','missing_physics','cold_late','unloaded','unquiet'])
def test_bad_actual_evidence_cannot_pass(authored,example,change):
    rows=copy.deepcopy(example[0]);physics=None
    if change=='short':rows.pop()
    if change=='next_goal':rows[2000]['controller_info']['goal_joint_position_rad']=rows[2001]['controller_info']['goal_joint_position_rad']
    if change=='contact':rows[1100]['hand_contact_count']=1
    if change=='missing_joint':rows[17]['actual_joint_position'].pop('left_knee')
    if change=='wrong_loopback':
        q=rows[17]['actual_joint_position'];q['rh_FFJ1']+=.04;q['rh_FFJ2']-=.04
    if change=='joint_stop':rows[17]['actual_joint_position']['lh_THJ1']=-1.
    if change=='wrong_body_endpoint':rows[-1]['root13_actororigin'][0]+=.03
    if change=='no_motion':
        for row in rows:row['actual_joint_position']=authored[3].copy()
    if change=='missing_physics':physics={'joint_stops':True}
    if change=='cold_late':rows[1]['controller_info']['cold_start']=True
    if change=='unloaded':rows[-1]['foot_floor_loads'][0]=0
    if change=='unquiet':rows[-1]['root13_actororigin'][7]=.04
    report=score(authored,example,rows,physics);assert not report['passed']
    if change=='wrong_loopback':
        assert report['checks']['motor_coordinate_tracking'] and not report['checks']['actual_unilateral_loopbacks']
    if change=='wrong_body_endpoint':
        assert report['checks']['motor_coordinate_tracking'] and not report['checks']['palm_endpoint_tracking']


def test_legal_passive_split_is_reported_not_falsely_motor_tracking_failure(authored,example):
    rows=copy.deepcopy(example[0]);q=rows[17]['actual_joint_position'];q['rh_FFJ1']-=.03;q['rh_FFJ2']+=.03
    report=score(authored,example,rows)
    assert report['passed'] and report['maximum_nominal_joint_tracking_error_rad']>.029
    assert report['maximum_nominal_passive_split_difference_rad']['rh_FF']>.059
    assert report['maximum_motor_coordinate_tracking_error_rad']<1e-12
