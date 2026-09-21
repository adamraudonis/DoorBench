"""Synthetic CPU evidence fixtures; no physics, live stage or force claim."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous.withdrawal_source_context import (
    load_withdrawal_source_context, ISAAC_DENSE_CHECKS,
)
from doorbench.dexterous.isaac_release_geometry_audit import ORIGINAL_LIMITS
from doorbench.dexterous.qualified_isaac_grasp import digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def bind_documents(fixture):
    write(fixture['screen_path'], fixture['screen'])
    fixture['audit']['input_sha256'][str(fixture['screen_path'])] = digest(fixture['screen_path'])
    write(fixture['audit_path'], fixture['audit'])
    fixture['config']['screen_sha256'] = digest(fixture['screen_path'])
    fixture['config']['audit_sha256'] = digest(fixture['audit_path'])


@pytest.fixture
def native(tmp_path, monkeypatch):
    run = tmp_path/'native source'
    run.mkdir()
    robot, door = tmp_path/'robot.xml', tmp_path/'door'
    robot.write_text('synthetic original robot')
    door.mkdir(); (door/'door.xml').write_text('synthetic original door')
    manifest = dict(configuration=dict(robot=str(robot), door=str(door)),
                    inputs={'robot':{'sha256':digest(robot)}})
    write(run/'manifest.json', manifest)
    qpos = np.array([.1,.2,.3])
    np.savez(run/'trajectory.npz', terminal_qpos=qpos, terminal_time_s=50.)
    admission = dict(grasp_profile='volar-phalange-v1',
                     input_sha256={str(run/'manifest.json'):digest(run/'manifest.json')},
                     fixture='synthetic native admission')
    screen = dict(grasp_profile='volar-phalange-v1', source_admission=admission,
                  trials=[{'rows':[{'qpos':qpos.tolist()}]}])
    audit = dict(passed=True,samples=2001,physics_steps=0,duration_s=16.,
        grasp_profile='volar-phalange-v1',input_sha256={str(path):digest(path) for path in
            (robot,door/'door.xml',run/'trajectory.npz')})
    config = dict(schema='doorbench.standing-withdrawal.v1',source_run=str(run),
        screen_path=str(tmp_path/'screen.json'),audit_path=str(tmp_path/'audit.json'),
        measured_rest_transfer=False,grasp_profile='volar-phalange-v1',
        contact_audit_name='independent-contact-audit.json')
    motors = dict(source_xml_sha256=digest(robot),actuators=[{'name':'motor','force_range':[-1.,1.]}])
    calls=[]
    from doorbench.dexterous import release_source_admission
    def admit(*args,**kwargs):
        calls.append((args,kwargs)); return copy.deepcopy(admission)
    monkeypatch.setattr(release_source_admission,'admit_release_source',admit)
    fixture=dict(run=run,config=config,motors=motors,screen=screen,audit=audit,
        screen_path=Path(config['screen_path']),audit_path=Path(config['audit_path']),
        admission=admission,calls=calls,qpos=qpos)
    bind_documents(fixture)
    return fixture


def test_native_v1_retains_original_legacy_endpoint_and_no_extra_admission(native):
    result=load_withdrawal_source_context(native['config'],native['motors'],measured_rest=False)
    data=result.data
    assert data['source_engine']=='native-mujoco' and data['source_admission'] is None
    assert data['start_time_s']==50. and data['duration_s']==16.
    assert not native['calls'] and data['source_archive_path'].endswith('trajectory.npz')
    np.testing.assert_array_equal(result.initial_qpos,native['qpos'])
    assert data['screen']==native['screen'] and data['audit']==native['audit']
    assert data['authorized_stages']==0 and not data['physical_release_qualification']


def test_native_measured_rest_calls_unchanged_original_admission(native):
    native['config']['measured_rest_transfer']=True
    result=load_withdrawal_source_context(native['config'],native['motors'],measured_rest=True)
    assert result.data['source_admission']==native['admission']
    assert native['calls']==[((native['run'],),dict(profile='volar-phalange-v1',
        contact_audit_name='independent-contact-audit.json',measured_rest=True))]


def test_native_optional_motion_proof_still_uses_original_validator(native):
    native['config'].update(leaf_motion_audit_path=str(native['audit_path']),
                            leaf_motion_audit_sha256=digest(native['audit_path']))
    with pytest.raises(ValueError,match='moving-leaf'):
        load_withdrawal_source_context(native['config'],native['motors'],measured_rest=False)


def test_native_relative_absolute_aliases_preserved_but_conflicts_rejected(native,monkeypatch):
    monkeypatch.chdir(native['run'].parent)
    archive=native['run']/'trajectory.npz'
    alias=str(archive.relative_to(Path.cwd()))
    native['audit']['input_sha256'][alias]=digest(archive)
    bind_documents(native)
    assert load_withdrawal_source_context(native['config'],native['motors'],measured_rest=False)
    native['audit']['input_sha256'][alias]='b'*64
    bind_documents(native)
    with pytest.raises(ValueError,match='Conflicting digest aliases'):
        load_withdrawal_source_context(native['config'],native['motors'],measured_rest=False)


@pytest.mark.parametrize('change',['robot','audit_failed','count','physics','screen_hash','archive_hash','source_admission'])
def test_native_original_bindings_remain_required(native,change):
    measured=change=='source_admission'
    if measured:
        native['config']['measured_rest_transfer']=True
        native['screen']['source_admission']={}
    if change=='robot': native['motors']['source_xml_sha256']='b'*64
    if change=='audit_failed': native['audit']['passed']=False
    if change=='count': native['audit']['samples']=2000
    if change=='physics': native['audit']['physics_steps']=1
    if change=='archive_hash': native['audit']['input_sha256'][str(native['run']/'trajectory.npz')]='b'*64
    bind_documents(native)
    if change=='screen_hash': native['config']['screen_sha256']='b'*64
    with pytest.raises(ValueError):
        load_withdrawal_source_context(native['config'],native['motors'],measured_rest=measured)


@pytest.fixture
def isaac(tmp_path,monkeypatch):
    from doorbench.dexterous import isaac_release_planning
    run=tmp_path/'isaac source';trial=run/'trial';trial.mkdir(parents=True)
    robot,xml,usd=[tmp_path/name for name in ('robot.xml','door.xml','door.usda')]
    for path in (robot,xml,usd):path.write_text('synthetic '+path.name)
    archive=trial/'acquisition-physics.npz';archive.write_bytes(b'synthetic test archive only')
    motors=dict(source_xml_sha256=digest(robot),joint_names=['j'],actuators=[
        dict(name='a',terms={'j':1.},force_range=[-2.,2.])])
    motor_path=trial/'motor-contract.json';write(motor_path,motors)
    hashes={str(path):digest(path) for path in (robot,xml,usd,archive,motor_path)}
    qpos=np.array([.4,.5,.6])
    admission=dict(schema='doorbench.isaac-release-source.v1',source_engine='isaac-physx',
        source_run=str(run),grasp_profile='volar-phalange-v1',
        source_qualification={'passed':True,'state_sha256':'a'*64},
        coordinate_admission={'passed':True},measured_rest={'terminal_time_s':42.},
        initial_qpos=qpos.tolist(),input_sha256=hashes,robot_path=str(robot),
        door_xml_path=str(xml),door_usd_path=str(usd))
    def verify():
        for path,expected in hashes.items():
            if digest(path)!=expected:raise ValueError('Source changed')
    context=SimpleNamespace(admission=copy.deepcopy(admission),sha256='b'*64,qpos=qpos.copy(),
        terminal_time_s=42.,state_archive_path=archive,verify_inputs=verify)
    calls=[]
    def admit(*args,**kwargs):
        calls.append((args,kwargs));return context
    monkeypatch.setattr(isaac_release_planning,'admit_isaac_release_context',admit)
    screen=dict(schema='doorbench.isaac-profiled-release-candidate.v1',source_engine='isaac-physx',
        physics_steps=0,source_sample_playback=0,grasp_profile='volar-phalange-v1',
        source_context_sha256='b'*64,source_admission=copy.deepcopy(admission),
        initial_time_s=42.,initial_qpos=qpos.tolist(),input_sha256=hashes.copy(),
        trials=[dict(rows=[dict(qpos=qpos.tolist())])],
        runtime_route_exported=False,physical_contact_qualification=False)
    audit=dict(schema='doorbench.isaac-release-dense-geometry-audit.v1',source_engine='isaac-physx',
        passed=True,samples=2001,physics_steps=0,active_state_writes=0,source_sample_playback=0,
        source_admission=copy.deepcopy(admission),source_context_sha256='b'*64,
        initial_episode_time_s=42.,grasp_profile='volar-phalange-v1',
        original_thresholds=ORIGINAL_LIMITS.copy(),checks=dict.fromkeys(ISAAC_DENSE_CHECKS,True),
        failures=[],motor_contract_identity_bound=True,runtime_route_exported=False,
        physical_contact_qualification=False,delivered_motor_force_checked=False,
        motor_force_feasibility_inferred=False,source_physics_archive=str(archive),
        input_sha256=hashes.copy(),duration_s=16.)
    config=dict(schema='doorbench.standing-withdrawal.v1',source_engine='isaac-physx',
        source_run=str(run),screen_path=str(tmp_path/'candidate.json'),audit_path=str(tmp_path/'dense.json'),
        measured_rest_transfer=True,contact_audit_name='independent-contact-audit.json',
        grasp_profile='volar-phalange-v1',robot_path=str(robot),door_xml_path=str(xml),door_usd_path=str(usd),
        source_state_sha256='a'*64,start_time_s=42.,duration_s=16.)
    result=dict(run=run,config=config,motors=motors,screen=screen,audit=audit,admission=admission,
        context=context,screen_path=Path(config['screen_path']),audit_path=Path(config['audit_path']),
        calls=calls,qpos=qpos)
    bind_documents(result)
    return result


def load_isaac(fixture):
    return load_withdrawal_source_context(fixture['config'],fixture['motors'],measured_rest=True)


def test_explicit_isaac_mode_binds_actual_source_assets_epoch_and_full_motor_contract(isaac):
    result=load_isaac(isaac);data=result.data
    assert data['source_engine']=='isaac-physx' and data['source_admission']==isaac['admission']
    assert data['start_time_s']==42. and data['duration_s']==16.
    assert data['live_exact_prefix_required'] and data['authorized_stages']==0
    assert data['source_archive_path'].endswith('acquisition-physics.npz')
    assert data['source_state_sha256']=='a'*64
    assert len(isaac['calls'])==1
    assert not (isaac['run']/'manifest.json').exists() and not (isaac['run']/'trajectory.npz').exists()
    assert not any(hasattr(result,name) for name in ('scene','force','set_state','motor','articulation'))


def test_returned_context_is_detached_from_caller_and_property_mutation(isaac):
    result=load_isaac(isaac);before=result.data
    result.initial_qpos[:]=999
    result.data['screen']['initial_qpos'][0]=999
    isaac['config']['start_time_s']=999
    isaac['screen']['source_admission'].clear()
    isaac['motors']['actuators'][0]['force_range'][0]=-999
    assert result.data==before


@pytest.mark.parametrize('key',['leaf_motion_audit_path','leaf_motion_audit_sha256',
    'coupled_envelope_path','coupled_envelope_sha256','coupled_audit_path',
    'coupled_audit_sha256','panel_plan_path','panel_continuations'])
def test_old_native_motion_proofs_are_not_relabelled_as_isaac(isaac,key):
    isaac['config'][key]='old native passing receipt'
    with pytest.raises(ValueError,match='not admitted for Isaac'):load_isaac(isaac)
    assert not isaac['calls']


@pytest.mark.parametrize('change',['schema','failed','missing_check','false_check','threshold',
    'source','context','epoch','npz_path','npz_hash','profile','physical_claim','duration','first_qpos'])
def test_isaac_dense_evidence_is_required_as_a_conjunction(isaac,change):
    a=isaac['audit']
    if change=='schema':a['schema']='doorbench.native-standing-audit.v1'
    if change=='failed':a['passed']=False
    if change=='missing_check':a['checks'].pop(next(iter(ISAAC_DENSE_CHECKS)))
    if change=='false_check':a['checks']['selected_right_anatomy']=False
    if change=='threshold':a['original_thresholds']['joint_reference_velocity_rad_s']=3.
    if change=='source':a['source_admission']={}
    if change=='context':a['source_context_sha256']='c'*64
    if change=='epoch':a['initial_episode_time_s']=41.998
    if change=='npz_path':a['source_physics_archive']=str(isaac['run']/'trajectory.npz')
    if change=='npz_hash':a['input_sha256'][str(isaac['context'].state_archive_path)]='c'*64
    if change=='profile':a['grasp_profile']='distal-pad-v1'
    if change=='physical_claim':a['physical_contact_qualification']=True
    if change=='duration':a['duration_s']=8.
    if change=='first_qpos':isaac['screen']['trials'][0]['rows'][0]['qpos'][0]=np.nextafter(.4,1.)
    bind_documents(isaac)
    with pytest.raises(ValueError):load_isaac(isaac)


@pytest.mark.parametrize('change',['state','clock','duration','motor_cap','motor_transmission','motor_order'])
def test_isaac_runtime_cannot_change_source_or_motor_contract(isaac,change):
    if change=='state':isaac['config']['source_state_sha256']='c'*64
    if change=='clock':isaac['config']['start_time_s']=42.002
    if change=='duration':isaac['config']['duration_s']=8.
    if change=='motor_cap':isaac['motors']['actuators'][0]['force_range'][1]=3.
    if change=='motor_transmission':isaac['motors']['actuators'][0]['terms']['j']=2.
    if change=='motor_order':isaac['motors']['joint_names']=['different']
    with pytest.raises(ValueError):load_isaac(isaac)


def test_source_change_during_admission_cannot_return_context(isaac,monkeypatch):
    from doorbench.dexterous import isaac_release_planning
    def admit(*args,**kwargs):
        isaac['context'].state_archive_path.write_bytes(b'changed')
        return isaac['context']
    monkeypatch.setattr(isaac_release_planning,'admit_isaac_release_context',admit)
    with pytest.raises(ValueError):load_isaac(isaac)
