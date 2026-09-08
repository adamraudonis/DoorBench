"""Profile selection and receipts must not silently accept the legacy hand."""
import json
import pytest
from doorbench.dexterous.isaac_readiness import (
    cache_identity,ready_directory,robot_filename,validate_passive_audit,
    load_ready_receipt,file_sha256)


def contract():
    tendons=[]
    for side in ('lh','rh'):
        for digit in ('FF','MF','RF','LF'):
            base=f'{side}_{digit}'
            tendons.append(dict(name=base+'_loopback',profile='shadow-loopback-v2',root_joint=base+'J2',
                terms={base+'J1':1.,base+'J2':-1.},range_rad=[-2.5708,0.],
                physx_limit_stiffness=10000.,spring_stiffness=0.,damping=0.,rest_length_rad=0.,offset_rad=0.))
    motors=dict(hand_mechanics_profile='shadow-loopback-v2',passive_tendons=tendons)
    audit=dict(profile='shadow-loopback-v2',count=8,tendons=[dict(name=t['name']) for t in tendons],backend_readback={})
    values=audit['backend_readback']
    for name in ('stiffnesses','dampings','rest_lengths','offsets'):values['get_fixed_tendon_'+name]=[[0.]*8]
    values['get_fixed_tendon_limit_stiffnesses']=[[10000.]*8]
    values['get_fixed_tendon_limits']=[[[-2.5708,0.]]*8]
    return audit,motors


def test_profile_separates_cache_and_preserves_legacy_directory(tmp_path):
    assert cache_identity('same-source','upstream-v1')!=cache_identity('same-source','shadow-loopback-v2')
    assert ready_directory(tmp_path,'upstream-v1')==tmp_path/'out/isaac-ready'
    assert ready_directory(tmp_path,'shadow-loopback-v2')==tmp_path/'out/isaac-ready/shadow-loopback-v2'
    assert robot_filename('upstream-v1')=='h1-shadow.xml'
    assert robot_filename('shadow-loopback-v2')=='h1-shadow-loopback-v2.xml'
    with pytest.raises(ValueError):cache_identity('same-source','unknown')


def test_v2_requires_real_backend_not_only_authored_constraints():
    audit,motors=contract();assert validate_passive_audit(audit,motors,'shadow-loopback-v2')['backend_verified']
    audit.pop('backend_readback')
    with pytest.raises(ValueError,match='live count'):validate_passive_audit(audit,motors,'shadow-loopback-v2')


@pytest.mark.parametrize('mutation',['missing','spring','limit','gear'])
def test_v2_rejects_incomplete_or_changed_mechanics(mutation):
    audit,motors=contract()
    if mutation=='missing':motors['passive_tendons'].pop()
    if mutation=='spring':audit['backend_readback']['get_fixed_tendon_stiffnesses'][0][0]=1.
    if mutation=='limit':audit['backend_readback']['get_fixed_tendon_limits'][0][0]=[-2.5708,.1]
    if mutation=='gear':motors['passive_tendons'][0]['terms']['lh_FFJ2']=1.
    with pytest.raises(ValueError):validate_passive_audit(audit,motors,'shadow-loopback-v2')


def test_legacy_receipt_cannot_launch_v2_demo(tmp_path):
    p=tmp_path/'ready.json';p.write_text(json.dumps({'ready':True}))
    with pytest.raises(ValueError,match='legacy v1'):load_ready_receipt(p,expected_profile='shadow-loopback-v2')


def test_v2_receipt_detects_input_changed_after_readiness(tmp_path):
    robot=tmp_path/'h1-shadow-loopback-v2.xml';robot.write_text('original bytes')
    audit,motors=contract();motors['source_xml_sha256']=file_sha256(robot)
    motor_path=tmp_path/'motors.json';motor_path.write_text(json.dumps(motors))
    native_audit=robot.with_suffix('.audit.json');native_audit.write_text('{}')
    robot_usd=tmp_path/'robot.usda';robot_usd.write_text('robot USD')
    door_usd=tmp_path/'door.usda';door_usd.write_text('door USD')
    paths=[robot,motor_path,native_audit,robot_usd,door_usd]
    receipt=dict(ready=True,mechanics_profile='shadow-loopback-v2',native_robot=str(robot),native_robot_sha256=file_sha256(robot),
        motor_contract=str(motor_path),robot_usd=str(robot_usd),door_usd=str(door_usd),
        passive_tendons=dict(count=8,backend_verified=True,audit=audit),input_hashes={str(p):file_sha256(p) for p in paths})
    p=tmp_path/'ready.json';p.write_text(json.dumps(receipt));assert load_ready_receipt(p,expected_profile='shadow-loopback-v2')['ready']
    receipt['input_hashes'].pop(str(door_usd));p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='required asset hashes'):load_ready_receipt(p,expected_profile='shadow-loopback-v2')
    receipt['input_hashes'][str(door_usd)]=file_sha256(door_usd);p.write_text(json.dumps(receipt))
    robot.write_text('changed bytes')
    with pytest.raises(ValueError,match='input changed'):load_ready_receipt(p,expected_profile='shadow-loopback-v2')
