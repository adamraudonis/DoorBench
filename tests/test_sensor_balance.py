"""Sensor-only boundary tests and optional authored-model integration checks.

Set DOORBENCH_H1_V2_XML and DOORBENCH_H1_V2_MOTORS for another machine. No GPU,
active simulator, door model, privileged state or reference motion is involved.
"""
import ast
import copy
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest
mujoco=pytest.importorskip('mujoco')
from doorbench.dexterous.sensor_balance import SensorBalanceController
from doorbench.dexterous.sensor_contract import ActorObservationBuilder
from scripts.dexterous.export_sensor_layout import export_layout

ROOT=Path(__file__).resolve().parents[1]


def test_public_boundary_contains_no_world_state_or_active_plant():
    assert list(inspect.signature(SensorBalanceController.force).parameters)==['self','packet','now_s']
    tree=ast.parse(inspect.getsource(inspect.getmodule(SensorBalanceController)))
    assert not any(isinstance(n,ast.Attribute) and n.attr in {'mj_step','mj_step1','mj_step2','plant','environment','teacher'} for n in ast.walk(tree))
    assert 'initial_root' not in inspect.signature(SensorBalanceController).parameters


def test_frozen_posture_contains_only_fixed_joint_calibration():
    calibration=json.loads((ROOT/'configs/dexterous/sensor-balance-v1.json').read_text())
    assert len(calibration['desired_posture'])==69
    assert not any(k in calibration for k in ('initial_root','path_qpos','door','handle_pose'))


@pytest.fixture(scope='module')
def authored():
    robot=Path(os.environ.get('DOORBENCH_H1_V2_XML','/tmp/doorbench-shadow-loopback/final/h1-shadow-loopback-v2.xml'))
    motors=Path(os.environ.get('DOORBENCH_H1_V2_MOTORS','/tmp/doorbench-isaac-integration/out/dexterous/import-v2/h1-import.motors.json'))
    if not robot.exists() or not motors.exists():pytest.skip('Set authored v2 robot/motor asset paths for integration checks')
    return robot,json.loads(motors.read_text()),export_layout(robot),json.loads((ROOT/'configs/dexterous/sensor-balance-v1.json').read_text())['desired_posture']


def create(authored):return SensorBalanceController(*copy.deepcopy(authored),image_shape=(8,8,3))

def cold(c):
    return ActorObservationBuilder(joint_count=69,action_count=61,tactile_dimension=c.shapes['tactile'][0],image_shape=(8,8,3)).observe(now_s=0.,previous_action=np.zeros(61))


def valid(c,t):
    p=cold(c);p['joint_position'][:]=c.desired
    p['sensor_time_s'][:5]=t;p['sensor_valid'][:5]=True;p['imu_accelerometer'][:]=[0,0,9.81]
    for _,indices,n in c.foot_taxels:p['tactile'][indices.start]=80.
    p['previous_action'][:]=c.last_force/np.maximum(abs(c.caps[:,0]),abs(c.caps[:,1]))
    return p


def test_cold_invalid_values_do_not_supply_hidden_state_and_reset_is_explicit(authored):
    c=create(authored);p=cold(c);f,i=c.force(p,now_s=0.)
    assert i['cold_start'] and c.d.time==0
    c.reset_episode();poison=cold(c)
    for k in c.shapes:poison[k].fill(67)
    g,_=c.force(poison,now_s=0.)
    np.testing.assert_array_equal(g,f)
    c.force(valid(c,.002),now_s=.002);c.reset_episode();h,_=c.force(cold(c),now_s=0.)
    np.testing.assert_array_equal(h,f)
    assert (f>=c.caps[:,0]).all() and (f<=c.caps[:,1]).all()


@pytest.mark.parametrize('field',['root','gravity_body','door_pose','teacher_force','phase'])
def test_privileged_fields_are_rejected_and_failure_is_terminal(authored,field):
    c=create(authored);p=cold(c);p[field]=np.zeros(3)
    with pytest.raises(ValueError,match='forbidden'):c.force(p,now_s=0.)
    with pytest.raises(RuntimeError,match='requires reset'):c.force(cold(c),now_s=0.)
    c.reset_episode();c.force(cold(c),now_s=0.)


@pytest.mark.parametrize('kind',['future','stale','bad_valid','previous_action','missing_imu','clock','nonfinite'])
def test_invalid_or_noncausal_packet_cannot_advance_controller(authored,kind):
    c=create(authored);c.force(cold(c),now_s=0.);p=valid(c,.002);t=.002
    if kind=='future':p['sensor_time_s'][2]=.004
    if kind=='stale':p['sensor_time_s'][2]=-1.;p['sensor_valid'][2]=False
    if kind=='bad_valid':p['sensor_valid']=p['sensor_valid'].astype(np.float32)
    if kind=='previous_action':p['previous_action'].fill(0.)
    if kind=='missing_imu':p['sensor_valid'][2]=False
    if kind=='clock':t=.004
    if kind=='nonfinite':p['joint_position'][0]=np.nan
    with pytest.raises(ValueError):c.force(p,now_s=t)
    assert c.failed_reason
    with pytest.raises(RuntimeError):c.force(valid(c,.002),now_s=.002)


@pytest.mark.parametrize('kind',['hash','gain','bias','control','foot_frame','foot_body','taxel_bins','joint_extra','loopback'])
def test_authored_calibration_mismatch_fails_closed(authored,kind):
    robot,motor,layout,posture=copy.deepcopy(authored)
    foot=next(r for r in layout['sensors'] if r['name']=='left_ankle_touch')
    if kind=='hash':motor['source_xml_sha256']='0'*64
    if kind=='gain':motor['actuators'][0]['kp']+=1
    if kind=='bias':motor['actuators'][0]['bias'][2]-=1
    if kind=='control':motor['actuators'][0]['control_range'][0]-=.01
    if kind=='foot_frame':foot['position_body_m'][0]+=.01
    if kind=='foot_body':foot['body_name']='right_ankle_link'
    if kind=='taxel_bins':foot['fov_degrees'][0]-=1
    if kind=='joint_extra':posture['root']=1.
    if kind=='loopback':posture['rh_FFJ1']=.2;posture['rh_FFJ2']=0.
    with pytest.raises(ValueError):SensorBalanceController(robot,motor,layout,posture)


def test_local_calculator_never_steps_and_absent_touch_rejects_support_hypothesis(authored,monkeypatch):
    c=create(authored)
    def no_step(*args,**kwargs):raise AssertionError('A controller must never step physics')
    monkeypatch.setattr(mujoco,'mj_step',no_step)
    c.force(cold(c),now_s=0.)
    for i in range(1,26):c.force(valid(c,i*.002),now_s=i*.002)
    p=valid(c,.052);p['tactile'].fill(0.)
    with pytest.raises(RuntimeError,match='support hypothesis'):c.force(p,now_s=.052)
    assert c.d.time==0
    np.testing.assert_array_equal(c.sim.external_generalized_force,np.zeros(c.m.nv))


def test_fixed_solver_profile_is_recorded_and_replays_identical_sensor_history(authored):
    controllers=[SensorBalanceController(*copy.deepcopy(authored),image_shape=(8,8,3),
        solver_profile='fixed-rho-interval25-v1') for _ in range(2)]
    histories=[]
    for controller in controllers:
        history=[]
        for i in range(30):
            packet=cold(controller) if i==0 else valid(controller,i*.002)
            force,info=controller.force(packet,now_s=i*.002)
            assert info['qp_solver']['requested_settings']['adaptive_rho_interval']==25
            history.append(force.copy())
        histories.append(history)
    np.testing.assert_array_equal(histories[0],histories[1])
    assert all(c.d.time==0 for c in controllers)


def test_unknown_balance_solver_profile_is_rejected(authored):
    with pytest.raises(ValueError,match='solver profile'):
        SensorBalanceController(*authored,solver_profile='relaxed')
