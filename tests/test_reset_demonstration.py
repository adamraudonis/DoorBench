import json

import numpy as np
import pytest

from doorbench.dexterous.reset_demonstration import matched_cold_start_packet
from doorbench.dexterous.sensor_actor import ActorDimensions
from doorbench.dexterous.sensor_contract import SENSOR_KEYS
from test_sensor_actor import archive


def reset_pair(tmp_path):
    teacher=tmp_path/'teacher';actor=tmp_path/'actor';teacher.mkdir();actor.mkdir();archive(teacher)
    dims=ActorDimensions(joints=3,actions=4,tactile=12)
    (actor/'sensors').mkdir()
    for name in ('motor-contract.json','sensors/layout.json'):(actor/name).write_bytes((teacher/name).read_bytes())
    state=dict(root=[0.,0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],joints=[0.,0.,0.],door={'leaf':0.})
    for path in (teacher,actor):
        (path/'acquisition-reset.json').write_text(json.dumps(state))
        (path/'configuration.json').write_text(json.dumps(dict(robot_joint_names=['a','b','c'])))
    (actor/'sensors/report.json').write_text(json.dumps(dict(control_source='sensor_actor',capture_complete=False)))
    packet={key:np.zeros(shape,dtype=np.uint8 if key.startswith('rgb_') else np.float32) for key,shape in dims.shapes.items()}
    packet.update(previous_action=np.zeros(4,np.float32),sensor_valid=np.zeros(len(SENSOR_KEYS),bool),sensor_time_s=np.full(len(SENSOR_KEYS),-1.))
    # Deliberately wrong actor forces must never become a teacher label.
    np.savez_compressed(actor/'sensors/actor-initial-decision.npz',**packet,motor_forces=np.full(4,999.),time_s=np.asarray(0.))
    return teacher,actor,dims,packet


def test_only_recorded_observation_is_admitted_not_the_failed_actor_action(tmp_path):
    teacher,actor,dims,packet=reset_pair(tmp_path)
    result,metadata=matched_cold_start_packet(teacher,actor,dims)
    assert 'motor_forces' not in result and not metadata['failed_actor_motor_label_used']
    for key in packet:np.testing.assert_array_equal(result[key],packet[key])


@pytest.mark.parametrize('defect',['root','joint_order','motor','source','future','previous_action'])
def test_reset_augmentation_rejects_unmatched_or_noncold_start(tmp_path,defect):
    teacher,actor,dims,packet=reset_pair(tmp_path)
    if defect=='root':
        path=actor/'acquisition-reset.json';r=json.loads(path.read_text());r['root'][0]=.001;path.write_text(json.dumps(r))
    elif defect=='joint_order':(actor/'configuration.json').write_text(json.dumps(dict(robot_joint_names=['b','a','c'])))
    elif defect=='motor':
        path=actor/'motor-contract.json';r=json.loads(path.read_text());r['new_mechanics']=1;path.write_text(json.dumps(r))
    elif defect=='source':(actor/'sensors/report.json').write_text('{}')
    else:
        if defect=='future':packet['sensor_time_s'][0]=.002
        else:packet['previous_action'][0]=.2
        np.savez_compressed(actor/'sensors/actor-initial-decision.npz',**packet,motor_forces=np.zeros(4),time_s=np.asarray(0.))
    with pytest.raises(ValueError):matched_cold_start_packet(teacher,actor,dims)


def test_reset_label_is_actual_teacher_first_force_and_start_time_is_zero(tmp_path,monkeypatch):
    from doorbench.dexterous.sensor_demonstrations import SensorDemonstration
    from doorbench.dexterous import legacy_teacher_provenance
    teacher,actor,dims,packet=reset_pair(tmp_path)
    report=teacher/'acquisition-report.json';report.write_bytes((teacher/'operation-report.json').read_bytes())
    receipt=teacher/'receipt.json';receipt.write_text('{}')
    monkeypatch.setattr(legacy_teacher_provenance,'validate_legacy_teacher_receipt',
        lambda run,receipt,qualification:dict(sensor_samples=3,sample_end_time_s=.006,scope='fixture prefix'))
    data=SensorDemonstration(teacher,qualification='acquisition-report.json',legacy_teacher_receipt=receipt,reset_observation_run=actor)
    assert data.times.tolist()==[0.,.002,.004,.006] and len(data)==3
    values,labels=data.sequence(0,3)
    np.testing.assert_allclose(labels[:,0],[0.,.1,.2])
    assert not values['images'][0].any() and data.metadata['reset_observation']['failed_actor_motor_label_used'] is False
