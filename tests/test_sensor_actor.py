import copy
import json

import numpy as np
import pytest
torch=pytest.importorskip('torch')

from doorbench.dexterous.sensor_actor import ActorDimensions,SensorActor,prepare_actor_packet,native_motor_forces
from doorbench.dexterous.sensor_contract import SENSOR_KEYS
from doorbench.dexterous.sensor_demonstrations import SensorDemonstration


def packet(dimensions):
    result={key:np.zeros(shape,dtype=np.uint8 if key.startswith('rgb_') else np.float32) for key,shape in dimensions.shapes.items()}
    result.update(previous_action=np.zeros(dimensions.actions,np.float32),sensor_time_s=np.ones(len(SENSOR_KEYS)),sensor_valid=np.ones(len(SENSOR_KEYS),bool))
    return result


def test_privileged_fields_and_future_observations_rejected():
    d=ActorDimensions();p=packet(d)
    p['handle_pose']=np.zeros(7)
    with pytest.raises(ValueError,match='forbidden'):prepare_actor_packet(p,1.,d)
    del p['handle_pose'];p['sensor_time_s'][0]=1.01
    with pytest.raises(ValueError,match='future'):prepare_actor_packet(p,1.,d)


def test_absolute_clock_cannot_encode_phase_and_invalid_payload_is_zeroed():
    d=ActorDimensions();p=packet(d);p['sensor_time_s'][0]=.98
    p['sensor_valid'][SENSOR_KEYS.index('tactile')]=False;p['tactile'][:]=99
    a=prepare_actor_packet(p,1.,d)
    p['sensor_time_s']+=100
    b=prepare_actor_packet(p,101.,d)
    for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-7)
    assert not np.any(a['tactile'])


def test_state_is_explicit_episode_local_and_commands_respect_asymmetric_caps():
    torch.manual_seed(8);torch.set_num_threads(1)
    d=ActorDimensions(joints=3,actions=4,tactile=12,image_size=32,hidden=16)
    actor=SensorActor(d).eval();p=packet(d)
    action,hidden=actor.act(p,1.)
    repeat,_=actor.act(p,1.)
    np.testing.assert_array_equal(action,repeat)
    continued,_=actor.act(p,1.,hidden)
    assert not np.allclose(action,continued)
    caps=np.array([[-1,2],[-3,4],[-7,1],[0,4]])
    np.testing.assert_allclose(native_motor_forces([-5,5,0,-2],caps),[-1,4,-3,0])


def archive(tmp_path):
    sensor=tmp_path/'sensors';sensor.mkdir()
    report=dict(passed=True,runtime_robot_pose_writes=0,direct_door_commands=False,physics_dt_s=.002)
    (tmp_path/'operation-report.json').write_text(json.dumps(report))
    (tmp_path/'mechanical-audit.json').write_text(json.dumps(dict(passed=True)))
    from test_isaac_readiness_profiles import contract
    audit,motors=contract();motors.update(source_xml_sha256='fixture',actuators=[dict(name=name) for name in ['x','y','z','w']])
    (tmp_path/'passive-tendon-audit.json').write_text(json.dumps(audit))
    (tmp_path/'motor-contract.json').write_text(json.dumps(motors))
    d=ActorDimensions(joints=3,actions=4,tactile=12)
    layout=dict(interface_version='doorbench.sensors.v2',joint_order=['a','b','c'],action_order=['x','y','z','w'],tactile_dimension=12,robot_xml_sha256='fixture')
    (sensor/'layout.json').write_text(json.dumps(layout))
    rows=[packet(d) for _ in range(4)];times=np.arange(1,5)*.002
    for i,r in enumerate(rows):
        r['sensor_time_s'][:]=times[i];r['sensor_time_s'][-2:]=times[0] if i<2 else times[2]
        r['previous_action'][:]=i/10
    arrays={key:np.stack([r[key] for r in rows]) for key in rows[0] if not key.startswith('rgb_')}
    np.savez_compressed(sensor/'actor-sensors.npz',time_s=times,**arrays)
    images=np.stack([np.zeros((128,128,3),np.uint8),np.full((128,128,3),255,np.uint8)])
    np.savez_compressed(sensor/'actor-rgb.npz',time_s=times[[0,2]],rgb_left=images,rgb_right=images)
    return sensor


def test_next_force_label_is_separate_and_future_image_cannot_leak(tmp_path):
    archive(tmp_path);data=SensorDemonstration(tmp_path)
    inputs,labels=data.sequence(0,3)
    np.testing.assert_allclose(labels[:,0],[.1,.2,.3])
    assert not inputs['images'][0].any() and not inputs['images'][1].any()
    assert inputs['images'][2].all()
    p=data.packet(0)
    assert np.all(p['previous_action']==0) and set(p)==set(packet(data.dimensions))
    with pytest.raises(ValueError,match='episode'):data.sequence(1,3)
    report=tmp_path/'operation-report.json';r=json.loads(report.read_text());r['passed']=False;report.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='qualified'):SensorDemonstration(tmp_path)


def test_demonstration_binds_qualification_and_static_motor_contract(tmp_path):
    archive(tmp_path);data=SensorDemonstration(tmp_path)
    from doorbench.dexterous.motor_contract_identity import motor_contract_fingerprint
    motors=json.loads((tmp_path/'motor-contract.json').read_text())
    assert data.motor_contract_sha256==motor_contract_fingerprint(motors)
    assert data.metadata['grasp_profile']=='distal-pad-v1'
    assert set(data.metadata['qualification_files'])=={
        'operation-report.json','mechanical-audit.json','passive-tendon-audit.json','motor-contract.json'}
    old=data.motor_contract_sha256;motors['actuators'][0]['force_range']=[-2,2]
    (tmp_path/'motor-contract.json').write_text(json.dumps(motors))
    assert SensorDemonstration(tmp_path).motor_contract_sha256!=old
