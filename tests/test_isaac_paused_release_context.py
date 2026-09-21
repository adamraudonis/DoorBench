"""Detached context seams and real material-frame arithmetic, no plant."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import doorbench.dexterous.isaac_paused_release_context as module
from doorbench.dexterous.isaac_paused_transfer_source import SOURCE_KIND
from test_isaac_release_source import source as old_source, window


@pytest.fixture
def source(tmp_path,monkeypatch):
    old=old_source.__wrapped__(tmp_path,monkeypatch)
    cfg=old['configuration'];cfg['args'].update(native_robot=str(old['robot']),door_usd=str(old['usd']))
    (old['trial']/'configuration.json').write_text(json.dumps(cfg))
    old['hashes'][str(old['trial']/'configuration.json')]=module.digest(old['trial']/'configuration.json')
    snapshot=old['trial']/'snapshot.json';snapshot.write_text('explicit synthetic snapshot fixture')
    old['hashes'][str(snapshot)]=module.digest(snapshot)
    phase_report=old['trial']/'phase.json';phase_report.write_text(json.dumps(old['report']))
    old['hashes'][str(phase_report)]=module.digest(phase_report)
    motors=old['trial']/'motors.json';motors.write_text('{}');old['hashes'][str(motors)]=module.digest(motors)
    observer=old['trial']/'observer-state.json';observer.write_text('{}');old['hashes'][str(observer)]=module.digest(observer)
    inspected=dict(source_kind=SOURCE_KIND,snapshot_path=str(snapshot),snapshot_sha256=module.digest(snapshot),
        live_pause=dict(epoch_s=1.5,pause_token='a'*64),source_state_sha256='a'*64,core_intervals=750,
        evidence_paths=dict(configuration=str(old['trial']/'configuration.json'),motor_contract=str(motors),
            provenance=str(old['trial']/'provenance.json'),phase_report=str(phase_report),transfer_route=str(old['captured']),
            physics=str(old['trial']/'acquisition-physics.npz'),observer_state=str(observer)),
        input_sha256=old['hashes'].copy(),measured_state_binding=old['extracted']['binding'],measured_bodies={})
    values=window();rest=dict(terminal_time_s=1.5,window_samples=251,endpoint_contacts=copy.deepcopy(values[2][-1]['contacts']))
    phase=dict(schema=module.PHASE_AUDIT_SCHEMA,passed=True,source_kind=SOURCE_KIND,source_engine='isaac-physx',
        source_run=str(old['run']),snapshot_path=str(snapshot),snapshot_sha256=module.digest(snapshot),
        live_pause=inspected['live_pause'].copy(),time_s=1.5,state_sha256='a'*64,physical_intervals=750,raw_intervals=750,
        invalid_loaded_patches=0,independent_raw_contact_audit_complete=True,episode_complete=False,authorized_stages=0,
        checks={k:True for k in module.CHECKS},measured_rest=rest,endpoint_pad=values[2][-1],input_sha256=old['hashes'].copy())
    monkeypatch.setattr(module,'inspect_paused_isaac_transfer_snapshot',lambda *_:SimpleNamespace(inspection=copy.deepcopy(inspected)))
    monkeypatch.setattr(module,'audit_paused_isaac_transfer',lambda *_:copy.deepcopy(phase))
    import doorbench.dexterous.isaac_release_source as legacy
    scene=legacy.LandedLeftScene(None,None);scene.m.nq=7
    monkeypatch.setattr(module,'LandedLeftScene',lambda *_:scene)
    monkeypatch.setattr(module,'admit_destination_planner',lambda *a,**k:(SimpleNamespace(qpos=np.zeros(7)),dict(passed=True)))
    return dict(old=old,snapshot=snapshot,inspected=inspected,phase=phase,scene=scene)


def admit(s,**kwargs):
    old=s['old'];return module.admit_paused_isaac_release_context(s['snapshot'],robot=old['robot'],
        door_xml=old['door'],door_usd=old['usd'],**kwargs)


def test_detached_distinct_context_preserves_raw_source_and_materials(source):
    result=admit(source);r=result.admission
    assert r['schema']==module.SCHEMA and r['source_kind']==SOURCE_KIND
    assert r['material_frame_admission']['passed'] and r['coordinate_admission']['passed']
    assert result.state_archive_path==result.physics_archive_path==Path(r['physics_archive_path'])
    assert result.motor_contract_path==Path(r['motor_contract_path'])
    assert result.configuration_path==Path(r['configuration_path'])
    assert r['episode_complete'] is False and r['resume_authorized'] is False
    assert r['authorized_stages']==r['physics_steps']==r['source_sample_playback']==0
    assert all(c['body'].startswith('robot/rh_') and c['isaac_body_path'].startswith('/World') for c in r['measured_rest']['endpoint_contacts'])
    r['initial_qpos'][0]=99;result.qpos[:]=99
    assert result.qpos[0]==0
    assert result.scene() is source['scene']


@pytest.mark.parametrize('key,value',[
    ('passed',False),('schema','completed-run'),('source_kind','native-mujoco'),('state_sha256','b'*64),
    ('snapshot_sha256','c'*64),('live_pause',{}),('physical_intervals',749),('raw_intervals',749),
    ('invalid_loaded_patches',1),('episode_complete',True),('authorized_stages',1),('checks',{}),
])
def test_fresh_complete_phase_is_mandatory(source,key,value):
    source['phase'][key]=value
    with pytest.raises(ValueError,match='Fresh full'):admit(source)


@pytest.mark.parametrize('change',['translation','rotation','raw_point','wrong_body','wrong_robot','wrong_usd','missing_hash'])
def test_original_material_and_source_boundaries_reject(source,change):
    if change=='translation':source['scene'].d.xpos[0,0]+=3e-6
    elif change=='rotation':source['scene'].d.xmat[0]=Rotation.from_rotvec([0,0,3e-6]).as_matrix().reshape(9)
    elif change=='raw_point':source['phase']['measured_rest']['endpoint_contacts'][0]['position'][0]+=.000003
    elif change=='wrong_body':source['phase']['measured_rest']['endpoint_contacts'][0]['body']='/World/rh_ffknuckle'
    elif change=='wrong_robot':source['old']['robot'].write_text('changed')
    elif change=='wrong_usd':source['old']['usd'].write_text('changed')
    else:source['phase']['input_sha256'].pop(str(source['snapshot']))
    with pytest.raises(ValueError):admit(source)


def test_failed_coordinate_admission_is_not_promoted(source,monkeypatch):
    monkeypatch.setattr(module,'admit_destination_planner',lambda *a,**k:(SimpleNamespace(qpos=np.zeros(7)),dict(passed=False)))
    with pytest.raises(ValueError,match='kinematic'):admit(source)


def test_source_mutation_during_mapping_is_not_rehashed_as_valid(source,monkeypatch):
    def mapping(*a,**k):
        source['old']['robot'].write_text('changed during private mapping')
        return SimpleNamespace(qpos=np.zeros(7)),dict(passed=True)
    monkeypatch.setattr(module,'admit_destination_planner',mapping)
    with pytest.raises(ValueError,match='changed'):admit(source)


def test_saved_phase_audit_is_freshly_verified(source,tmp_path):
    p=tmp_path/'phase-audit.json';p.write_text(json.dumps(source['phase']))
    assert str(p) in admit(source,phase_audit_path=p).admission['input_sha256']
    altered=copy.deepcopy(source['phase']);altered['state_sha256']='c'*64;p.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='reproduce'):admit(source,phase_audit_path=p)
