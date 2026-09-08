"""Synthetic evidence validates the boundary; it is not a robot result."""
import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous.sensor_actor import ActorDimensions
from doorbench.dexterous.motor_contract_identity import motor_contract_fingerprint
from doorbench.dexterous.sensor_balance_runtime import SensorBalanceRuntime,evaluate_sensor_balance


@pytest.fixture
def setup(tmp_path,monkeypatch):
    robot=tmp_path/'robot.xml';robot.write_text('<robot-fixture/>');sha=hashlib.sha256(robot.read_bytes()).hexdigest()
    names=[f'q{i}' for i in range(69)];actions=[f'a{i}' for i in range(61)]
    motors=dict(source_xml_sha256=sha,joint_names=names,actuators=[dict(name=n,force_range=[-10.,10.]) for n in actions])
    layout=dict(robot_xml_sha256=sha,tactile_dimension=24)
    cal=dict(schema='doorbench.sensor-balance-calibration.v1',scope='Stationary test',robot_xml_sha256=sha,
        motor_contract_sha256=motor_contract_fingerprint(motors),physics_dt_s=.002,gravity_correction=.2,
        initial_orientation='upright; yaw/XY arbitrary robot-local gauge',desired_posture=dict.fromkeys(names,0.),
        provenance=dict(source_reference_sha256='a'*64,selection='Constant posture only'))
    class Fake:
        def __init__(self,*a,**kw):self.count=0;self.bad=False
        def reset_episode(self):self.count=0;self.bad=False
        def force(self,packet,*,now_s):
            self.count+=1
            if self.bad:raise RuntimeError('Rejected controller state')
            return np.full(61,2.),dict(controller='sensor_balance_v1',count=self.count,values=[1.])
    monkeypatch.setitem(sys.modules,'doorbench.dexterous.sensor_balance',SimpleNamespace(SensorBalanceController=Fake))
    return robot,motors,layout,cal


def packet():
    dims=ActorDimensions(tactile=24)
    p={k:np.zeros(v,dtype=np.uint8 if k.startswith('rgb') else np.float32) for k,v in dims.shapes.items()}
    p.update(previous_action=np.zeros(61,np.float32),sensor_time_s=np.full(7,-1.),sensor_valid=np.zeros(7,bool))
    return p


def test_explicit_reset_and_owned_command_diagnostics(setup):
    c=SensorBalanceRuntime(*setup)
    with pytest.raises(ValueError,match='reset'):c.force(packet(),0.)
    c.reset_episode();p=packet();force=c.force(p,0.)
    np.testing.assert_allclose(force,2.);np.testing.assert_allclose(c.previous_action,.2)
    c.previous_action[:]=0;c.last_info['values'][0]=999
    np.testing.assert_allclose(c.previous_action,.2);assert c.last_info['values']==[1.]
    p['previous_action']=c.previous_action;c.force(p,.002)
    c.reset_episode();assert not c.previous_action.any();assert c.last_info=={}
    np.testing.assert_array_equal(c.force(packet(),0.),force)


@pytest.mark.parametrize('change',['schema','extra_root','posture_missing','posture_extra','posture_nan','posture_bool','robot_hash','motor_hash','dt','gravity','provenance','orientation'])
def test_calibration_mismatch_rejected_before_controller(setup,change):
    robot,motors,layout,cal=copy.deepcopy(setup)
    if change=='schema':cal['schema']='other'
    if change=='extra_root':cal['root_pose']=[0,0,.87]
    if change=='posture_missing':cal['desired_posture'].pop('q0')
    if change=='posture_extra':cal['desired_posture']['door_hinge']=0.
    if change=='posture_nan':cal['desired_posture']['q0']=float('nan')
    if change=='posture_bool':cal['desired_posture']['q0']=False
    if change=='robot_hash':cal['robot_xml_sha256']='a'*64
    if change=='motor_hash':motors['actuators'][0]['force_range']=[-9.,9.]
    if change=='dt':cal['physics_dt_s']=.004
    if change=='gravity':cal['gravity_correction']=True
    if change=='provenance':cal['provenance']['initial_root']=[0,0,.87]
    if change=='orientation':cal['initial_orientation']='Use actual world pose'
    with pytest.raises(ValueError):SensorBalanceRuntime(robot,motors,layout,cal)


@pytest.mark.parametrize('field',['root13_actororigin','door_pose','teacher_action','goal_pose'])
def test_no_evaluator_fields_in_runtime_and_failure_requires_reset(setup,field):
    c=SensorBalanceRuntime(*setup);c.reset_episode();p=packet();p[field]=np.zeros(7)
    with pytest.raises(ValueError):c.force(p,0.)
    with pytest.raises(ValueError,match='reset'):c.force(packet(),0.)
    c.reset_episode();c.force(packet(),0.)


def test_controller_failure_is_terminal(setup):
    c=SensorBalanceRuntime(*setup);c.reset_episode();c._controller.bad=True
    with pytest.raises(RuntimeError):c.force(packet(),0.)
    with pytest.raises(ValueError,match='reset'):c.force(packet(),0.)
    c.reset_episode();c.force(packet(),0.)


@pytest.fixture
def actual_rows():
    return [dict(time_s=(i+1)*.002,root13_actororigin=[0,0,.87,1,0,0,0,0,0,0,0,0,0],torso_tilt_deg=1.,
        foot_floor_loads=[200.,200.],hand_contact_count=0,
        controller_info=dict(controller='sensor_balance_v1',cold_start=i==0,qp_failures=0,qp_status='solved',calculator_time_s=0.)) for i in range(2500)]


def test_complete_stationary_scope_and_original_failure_preserved(actual_rows):
    r=evaluate_sensor_balance(actual_rows,dict(caps=True,mechanics=True))
    assert r['passed'] and all(r['checks'].values()) and r['duration_s']==5.
    assert not r['checkpoint_evaluated'] and not r['evaluator_state_is_actor_input']
    bad=evaluate_sensor_balance(actual_rows,dict(caps=False,mechanics=True))
    assert not bad['passed'] and bad['original_physics_checks']['caps'] is False


@pytest.mark.parametrize('change,check',[
    ('empty','actual_step_records_valid'),('short','complete_5s_at_2ms'),('duplicate','complete_5s_at_2ms'),
    ('missing','actual_step_records_valid'),('nan','actual_step_records_valid'),('tilt','torso_upright'),
    ('height','lowered_height'),('linear_speed','final_quiet'),('angular_speed','final_quiet'),
    ('foot','final_both_feet'),('contact','hands_away'),('qp','all_qp_solved'),('inaccurate','all_qp_solved'),
    ('stepped','estimator_never_stepped'),('cold','cold_start_at_t0'),('recold','cold_start_at_t0')])
def test_meaningful_actual_failure_cannot_pass(actual_rows,change,check):
    if change=='empty':actual_rows=[]
    elif change=='short':actual_rows.pop()
    elif change=='duplicate':actual_rows[100]['time_s']=actual_rows[99]['time_s']
    elif change=='missing':actual_rows[10].pop('controller_info')
    elif change=='nan':actual_rows[10]['root13_actororigin'][0]=float('nan')
    elif change=='tilt':actual_rows[50]['torso_tilt_deg']=12.01
    elif change=='height':actual_rows[50]['root13_actororigin'][2]=.819
    elif change=='linear_speed':actual_rows[-1]['root13_actororigin'][7]=.03
    elif change=='angular_speed':actual_rows[-1]['root13_actororigin'][10]=.05
    elif change=='foot':actual_rows[-1]['foot_floor_loads'][0]=30.
    elif change=='contact':actual_rows[50]['hand_contact_count']=1
    elif change=='qp':actual_rows[50]['controller_info']['qp_failures']=1
    elif change=='inaccurate':actual_rows[50]['controller_info']['qp_status']='solved inaccurate'
    elif change=='stepped':actual_rows[50]['controller_info']['calculator_time_s']=.002
    elif change=='cold':actual_rows[0]['controller_info']['cold_start']=False
    elif change=='recold':actual_rows[50]['controller_info']['cold_start']=True
    r=evaluate_sensor_balance(actual_rows,dict(physics=True))
    assert not r['passed'] and r['checks'][check] is False


@pytest.mark.parametrize('physics',[{},None,dict(caps=1),dict(caps='true')])
def test_missing_or_nonboolean_physical_evidence_fails(actual_rows,physics):
    r=evaluate_sensor_balance(actual_rows,physics)
    assert not r['passed'] and not r['checks']['original_physics_checks_valid']
