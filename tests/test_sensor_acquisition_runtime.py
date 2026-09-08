import copy,inspect
import numpy as np
import pytest
from test_sensor_balance import authored
from test_sensor_reach_runtime import inputs as reach_inputs,cold,CAL
from test_sensor_acquisition_evidence import raw
from doorbench.dexterous.sensor_acquisition_runtime import SensorAcquisitionBalanceRuntime,PARAMETER_VALUES,PROTOCOL_SCHEMA
from doorbench.dexterous.sensor_acquisition_evaluation import AcquisitionEvaluationRobot,evaluate_sensor_acquisition_balance,PHYSICS_FAMILIES
from doorbench.dexterous.sensor_reach_runtime import SensorReachBalanceRuntime


def inputs(authored):
    p,r=reach_inputs(authored);p['schema']=PROTOCOL_SCHEMA;p['parameters']=dict(PARAMETER_VALUES,scope='Synthetic acquisition boundary test',source_reference_sha256='a'*64)
    return p,r


def runtime(authored,p=None,r=None):
    if p is None:p,r=inputs(authored)
    return SensorAcquisitionBalanceRuntime(*authored[:3],CAL,p,r)


def test_acquisition_is_separately_declared_and_sensor_only(authored):
    p,r=inputs(authored);c=runtime(authored,p,r)
    assert list(inspect.signature(c.force).parameters)==['packet','now_s'] and c.duration_s==19 and c.grasp_profile=='distal-pad-v1'
    assert len(c.goal_names)==30 and len(c.goal_motor_names)==26
    with pytest.raises(ValueError,match='reset'):c.force(cold(c),0.)
    c.reset_episode();f=c.force(cold(c),0.)
    assert c.last_info['high_level_protocol']=='scripted-acquisition-19s-v1' and c._controller.balance.d.time==0
    np.testing.assert_allclose(c.previous_action,f/c._caps[:,1],atol=1e-7,rtol=0)
    with pytest.raises(ValueError):SensorReachBalanceRuntime(*authored[:3],CAL,p,r)
    p,r=reach_inputs(authored)
    with pytest.raises(ValueError):runtime(authored,p,r)


@pytest.mark.parametrize('key,value',[('stop_fraction',.45),('duration_s',11),('grasp_profile','volar-phalange-v1'),('required_grasp_hold_s',.1),('finger_impedance_multiplier',5.)])
def test_frozen_full_route_contact_contract_cannot_be_changed(authored,key,value):
    p,r=inputs(authored);p['parameters'][key]=value
    with pytest.raises(ValueError):runtime(authored,p,r)


@pytest.fixture(scope='module')
def episode(authored):
    p,r=inputs(authored);m=AcquisitionEvaluationRobot(authored[0],authored[1],CAL,p,r);root=[0.,0.,.87,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.];rows=[]
    for i in range(9500):
        q,coordinates=m.target(i*.002);t=(i+1)*.002;pad=raw();pad.update(interval_start_s=i*.002,interval_end_s=t,geometry_time_s=t)
        if t<18:
            pad['contacts']=[];pad['active_contact_count']=0
            pad['handle_pair_forces_world_N']={n:[0.,0.,0.] for n in pad['body_transforms_xyzw']}
        info=dict(controller='sensor_reach_balance_v1',cold_start=i==0,qp_status='solved',qp_failures=0,calculator_time_s=0.,
            goal_joint_names=list(m.goal_names),goal_joint_position_rad=q[m.indices].tolist(),goal_motor_names=list(m.goal_motor_names),
            goal_motor_coordinates=coordinates.tolist(),coupled_joint_goals_are_nominal=True)
        rows.append(dict(time_s=t,root13_actororigin=root.copy(),torso_tilt_deg=.3,foot_floor_loads=[250.,250.],hand_contact_count=len(pad['contacts']),
            unintended_hand_contact_count=0,pad_evidence=pad,controller_info=info,actual_joint_position=dict(zip(m.names,q.tolist()))))
    return rows,root,p,r


def score(authored,episode,rows=None,initial=None):
    original,root,p,r=episode
    return evaluate_sensor_acquisition_balance(original if rows is None else rows,{family[0]:True for family in PHYSICS_FAMILIES},robot_xml=authored[0],motors=authored[1],
        initial_root13_actororigin=root,initial_joint_position=authored[3],initial_hand_contact_count=0,initial_door_position={'leaf_hinge':0.,'leaf_handle_hinge':0.} if initial is None else initial,
        calibration=CAL,protocol=p,joint_route=r)


def test_original_500hz_opposed_hold_recomputed_from_raw_evidence(authored,episode):
    report=score(authored,episode);assert report['passed'],report
    assert report['strongest_qualified_hold_s']>=1 and report['grasp_profile']=='distal-pad-v1'
    assert report['final_half_second_minimum_pad_force_N']==dict.fromkeys(('ff','mf','rf','lf','th'),1.)


@pytest.mark.parametrize('change',['unload','dorsal_early','unexpected','short','same_sum_wrong_split','missing_raw','next_goal','body_endpoint','already_open'])
def test_no_contact_label_or_sum_tracking_can_mask_failed_grasp(authored,episode,change):
    rows=copy.deepcopy(episode[0]);initial=None
    if change=='unload':
        p=rows[-1]['pad_evidence'];path=p['contacts'][-1]['body'];p['contacts'][-1]['normal_force_N']=0.;p['handle_pair_forces_world_N'][path]=[0.,0.,0.]
    if change=='dorsal_early':
        p=rows[9010]['pad_evidence'];path=p['contacts'][-1]['body'];p['body_transforms_xyzw'][path][1]-=.016
    if change=='unexpected':rows[17]['hand_contact_count']=rows[17]['unintended_hand_contact_count']=1
    if change=='short':rows.pop(18)
    if change=='same_sum_wrong_split':
        q=rows[17]['actual_joint_position'];q['rh_FFJ1']+=.04;q['rh_FFJ2']-=.04
    if change=='missing_raw':rows[17].pop('pad_evidence')
    if change=='next_goal':rows[4000]['controller_info']['goal_motor_coordinates']=rows[4001]['controller_info']['goal_motor_coordinates']
    if change=='body_endpoint':rows[-1]['root13_actororigin'][0]+=.03
    if change=='already_open':initial={'leaf_hinge':.01,'leaf_handle_hinge':0.}
    report=score(authored,episode,rows,initial);assert not report['passed']
    if change=='dorsal_early':assert report['checks']['original_opposed_distal_grasp_hold'] and not report['checks']['original_distal_pad_patches']
    if change=='same_sum_wrong_split':assert report['checks']['motor_coordinate_tracking'] and not report['checks']['actual_unilateral_loopbacks']
