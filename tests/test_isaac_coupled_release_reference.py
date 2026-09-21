"""Synthetic receipts and private geometry only; never actual release success."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous import (isaac_coupled_release_geometry,coupled_release_geometry,
    isaac_coupled_release_audit,isaac_coupled_release_reference as module)
from doorbench.dexterous.coupled_release_reference import CoupledReleaseReference
from doorbench.dexterous.isaac_coupled_release_reference import (
    IsaacCoupledReleaseReference,validate_isaac_coupled_audit)
from doorbench.dexterous.isaac_coupled_release_audit import SCHEMA,ORIGINAL_LIMITS,ORIGINAL_GEOMETRY_LIMITS
from doorbench.dexterous.qualified_isaac_grasp import digest


@pytest.fixture
def proof(tmp_path):
    source=tmp_path/'actual-source-fixture.npz';source.write_bytes(b'synthetic source binding')
    envelope=tmp_path/'envelope.json';envelope.write_text('{"synthetic_fixture":true}')
    audit_path=tmp_path/'audit.json'
    hashes={str(p):digest(p) for p in (source,envelope)}
    for m in (isaac_coupled_release_geometry,coupled_release_geometry,isaac_coupled_release_audit):
        hashes[str(Path(m.__file__).resolve())]=digest(m.__file__)
    data=dict(source_admission={'grasp_profile':'volar-phalange-v1'},source_context_sha256='a'*64,
        source_state_sha256='b'*64,motor_contract_sha256='c'*64,initial_qpos=[.1,.2],
        start_time_s=44.,duration_s=16.,source_archive_path=str(source))
    plan=dict(duration_s=16.,operator_envelope_rad=[-.05,.05],admitted_leaf_upper_nodes=None)
    geometry=SimpleNamespace(source_data=copy.deepcopy(data),initial=np.asarray(data['initial_qpos']),
        plan=plan,_file_hashes={str(p):digest(p) for p in (source,envelope)})
    audit=dict(schema=SCHEMA,source_engine='isaac-physx',passed=True,coarse_diagnostic=False,
        samples=50001,failed_samples=0,failures=[],exact_initial_state=True,
        complete_original_geometry_coverage=True,limits=ORIGINAL_LIMITS.copy(),
        geometry_limits=ORIGINAL_GEOMETRY_LIMITS.copy(),**{k:data[k] for k in
            ('source_admission','source_context_sha256','source_state_sha256','motor_contract_sha256')},
        initial_episode_time_s=44.,initial_qpos=data['initial_qpos'].copy(),grasp_profile='volar-phalange-v1',
        source_physics_archive=str(source),physics_steps=0,active_state_writes=0,source_sample_playback=0,
        authorized_stages=0,physical_admission=False,physical_contact_qualification=False,
        delivered_motor_force_checked=False,dynamic_balance_qualification=False,motion_rate_qualification=False,
        post_motion_limiter_geometry_checked=False,runtime_route_exported=False,
        geometry_domain=dict(elapsed_s=[0,16.],leaf_rad=[.08,.4],admitted_leaf_upper_nodes=None,
            operator_rad=[-.05,.05],latch_m=[-.001,.001]),maximum=dict.fromkeys(ORIGINAL_LIMITS,0.),
        minimum_blend_all_handle_clearance_m=.004,minimum_final_right_hand_environment_clearance_m=.04,
        input_sha256=hashes)
    config=dict(coupled_audit_path=str(audit_path),coupled_envelope_path=str(envelope),
        coupled_envelope_sha256=digest(envelope))
    def bind():
        audit_path.write_text(json.dumps(audit));config['coupled_audit_sha256']=digest(audit_path)
    bind()
    return SimpleNamespace(config=config,data=data,geometry=geometry,audit=audit,bind=bind)


def test_distinct_source_bound_static_receipt_remains_static(proof):
    audit,hashes=validate_isaac_coupled_audit(proof.config,proof.geometry,proof.data)
    assert audit['authorized_stages']==0 and not audit['physical_contact_qualification']
    assert hashes[proof.config['coupled_audit_path']]==proof.config['coupled_audit_sha256']


@pytest.mark.parametrize('change',['native_schema','failed','coarse','count','false_initial','threshold',
    'endpoint','epoch','archive','source','contract','context','state','profile','physical','motion',
    'domain','clearance','maximum','missing_maximum','source_hash','code_hash'])
def test_no_old_or_omitted_geometry_receipt_can_authorize_reference(proof,change):
    a=proof.audit
    if change=='native_schema':a['schema']='doorbench.coupled-release-envelope-audit.v1'
    if change=='failed':a['passed']=False
    if change=='coarse':a['coarse_diagnostic']=True
    if change=='count':a['samples']=49999
    if change=='false_initial':a['exact_initial_state']=False
    if change=='threshold':a['geometry_limits']['maximum_joint_limit_violation_rad']=1.
    if change=='endpoint':a['initial_qpos'][0]+=.00001
    if change=='epoch':a['initial_episode_time_s']+=.002
    if change=='archive':a['source_physics_archive']=str(Path(proof.config['coupled_audit_path']))
    if change=='source':a['source_admission']={}
    if change=='contract':a['motor_contract_sha256']='d'*64
    if change=='context':a['source_context_sha256']='d'*64
    if change=='state':a['source_state_sha256']='d'*64
    if change=='profile':a['grasp_profile']='distal-pad-v1'
    if change=='physical':a['physical_contact_qualification']=True
    if change=='motion':a['motion_rate_qualification']=True
    if change=='domain':a['geometry_domain']['leaf_rad'][1]=.5
    if change=='clearance':a['minimum_blend_all_handle_clearance_m']=.003999
    if change=='maximum':a['maximum'][next(iter(ORIGINAL_LIMITS))]=float('nan')
    if change=='missing_maximum':a['maximum'].pop(next(iter(ORIGINAL_LIMITS)))
    if change=='source_hash':a['input_sha256'].pop(proof.data['source_archive_path'])
    if change=='code_hash':a['input_sha256'].pop(str(Path(isaac_coupled_release_audit.__file__).resolve()))
    proof.bind()
    with pytest.raises(ValueError):validate_isaac_coupled_audit(proof.config,proof.geometry,proof.data)


def test_reference_keeps_original_rate_projector_and_sticky_update():
    for name in ('update','_update','_project_reference','_install_reference','_pose_residual_jacobian'):
        assert getattr(IsaacCoupledReleaseReference,name) is getattr(CoupledReleaseReference,name)


def test_post_limiter_final_environment_gap_uses_current_private_state(monkeypatch):
    inspected=[];distances=[]
    monkeypatch.setattr(CoupledReleaseReference,'_inspect',lambda self,result:inspected.append(result) or {'original_gates':True})
    ref=IsaacCoupledReleaseReference.__new__(IsaacCoupledReleaseReference)
    ref.geometry=SimpleNamespace(m='private-model',d='post-limiter-private-data',times=np.array([0.,8.5]))
    ref.right=[1];ref.environment=[2,3]
    def gap(m,d,a,b,maximum,output):
        distances.append((m,d,a,b));return .03999 if b==3 else .05
    monkeypatch.setattr(module.mujoco,'mj_geomDistance',gap)
    assert ref._inspect({'release_clock_s':8.4})=={'original_gates':True}
    assert distances==[]
    with pytest.raises(ValueError,match='40mm'):ref._inspect({'release_clock_s':8.5})
    assert len(inspected)==2 and all(row[:2]==('private-model','post-limiter-private-data') for row in distances)


@pytest.mark.parametrize('config',[{},dict(capture_returned_motor_command=True),
    dict(capture_returned_motor_command=True,coupled_motion_projection=None)])
def test_exact_capture_and_projection_are_mandatory_before_geometry_load(config):
    with pytest.raises(ValueError,match='constraint-preserving'):IsaacCoupledReleaseReference(config,{})


def test_runtime_admission_requires_exact_lower_domain_in_independent_receipt(proof):
    nodes=[[0.,.08],[14.,.08],[15.5,.1],[16.,.1]]
    proof.geometry.plan['admitted_leaf_lower_nodes']=nodes
    with pytest.raises(ValueError,match='domain differs'):
        validate_isaac_coupled_audit(proof.config,proof.geometry,proof.data)
    proof.audit['geometry_domain']['admitted_leaf_lower_nodes']=copy.deepcopy(nodes);proof.bind()
    validate_isaac_coupled_audit(proof.config,proof.geometry,proof.data)
    proof.audit['geometry_domain']['admitted_leaf_lower_nodes'][2][1]=.099
    proof.bind()
    with pytest.raises(ValueError,match='domain differs'):
        validate_isaac_coupled_audit(proof.config,proof.geometry,proof.data)


def test_receipt_cannot_smuggle_lower_domain_into_default_plan(proof):
    proof.audit['geometry_domain']['admitted_leaf_lower_nodes']=[[0.,.08],[16.,.1]];proof.bind()
    with pytest.raises(ValueError,match='domain differs'):
        validate_isaac_coupled_audit(proof.config,proof.geometry,proof.data)
