"""Source-mode routing tests; synthetic contexts never represent real episodes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous import isaac_release_source_dispatch as dispatch
from doorbench.dexterous import isaac_release_planning as planner
from doorbench.dexterous.isaac_paused_release_context import PausedIsaacReleasePlanningContext
from test_isaac_release_planning import source
from test_isaac_withdrawal_runtime import runtime
from test_withdrawal_source_context import isaac, bind_documents, load_isaac, write


@pytest.mark.parametrize('kind',[False,'','native-mujoco','completed','paused-live-isaac-transfer-v0'])
def test_unknown_source_kind_rejected(kind):
    with pytest.raises(ValueError):dispatch.validate_source_kind(kind)


def test_explicit_paused_dispatch_passes_snapshot_and_phase_receipt_only(monkeypatch):
    import doorbench.dexterous.isaac_paused_release_context as paused
    calls=[];result=object()
    monkeypatch.setattr(paused,'admit_paused_isaac_release_context',lambda *a,**k:(calls.append((a,k)) or result))
    monkeypatch.setattr(planner,'admit_isaac_release_context',lambda *a,**k:pytest.fail('Completed-source admission called'))
    assert dispatch.admit_release_planning_source('snapshot.json',robot='r',door_xml='x',door_usd='u',
        source_kind=dispatch.PAUSED_SOURCE_KIND,phase_audit_path='phase.json') is result
    assert calls==[(('snapshot.json',),dict(robot='r',door_xml='x',door_usd='u',profile='volar-phalange-v1',phase_audit_path='phase.json'))]


def test_omitted_source_kind_preserves_completed_source_and_rejects_live_audit(monkeypatch):
    calls=[]
    monkeypatch.setattr(planner,'admit_isaac_release_context',lambda *a,**k:calls.append((a,k)))
    dispatch.admit_release_planning_source('run',robot='r',door_xml='x',door_usd='u')
    assert calls==[(('run',),dict(robot='r',door_xml='x',door_usd='u',profile='volar-phalange-v1'))]
    with pytest.raises(ValueError,match='explicit live source kind'):
        dispatch.admit_release_planning_source('run',robot='r',door_xml='x',door_usd='u',phase_audit_path='phase')
    assert len(calls)==1


def test_paused_candidate_retains_actual_context_identity_and_zero_runtime_authority(source,monkeypatch):
    admission=copy.deepcopy(source[0]);admission['source_kind']=dispatch.PAUSED_SOURCE_KIND
    value=PausedIsaacReleasePlanningContext(json.dumps(admission))
    calls=[]
    def numerical(context,preferences):
        calls.append(context)
        return [dict(qpos=context.qpos.tolist())],[],[],0.
    monkeypatch.setattr(planner,'_generate_profiled_rows',numerical)
    candidate=planner.generate_release_candidate(value,{})
    assert calls==[value] and candidate['source_kind']==dispatch.PAUSED_SOURCE_KIND
    assert candidate['source_context_sha256']==value.sha256
    assert candidate['source_admission']==admission
    assert np.array_equal(candidate['initial_qpos'],value.qpos)
    assert candidate['physics_steps']==0 and candidate['runtime_route_exported'] is False
    assert candidate['physical_contact_qualification'] is False


def test_a_claimed_kind_cannot_substitute_for_the_context_type():
    with pytest.raises(ValueError):
        dispatch.require_planning_context(SimpleNamespace(admission={'source_kind':dispatch.PAUSED_SOURCE_KIND}))


def test_paused_helpers_are_explicit_current_source_inputs():
    paths=dispatch.source_paths(dispatch.PAUSED_SOURCE_KIND)
    assert all(path.is_absolute() and path.is_file() for path in paths)
    assert {'isaac_release_source_dispatch.py','isaac_paused_transfer_source.py',
        'isaac_paused_transfer_audit.py','isaac_paused_release_context.py'} <= {p.name for p in paths}
    assert dispatch.source_paths(None)==(Path(dispatch.__file__).resolve(),)


def test_old_runtime_cannot_silently_consume_new_source_kind(runtime):
    from doorbench.dexterous.isaac_withdrawal_runtime import admit_isaac_withdrawal_runtime
    runtime.data['source_kind']=dispatch.PAUSED_SOURCE_KIND
    with pytest.raises(ValueError,match='distinct retained-controller runtime factory'):
        admit_isaac_withdrawal_runtime(runtime.path,runtime.motors)
    assert not any(call[0]=='reference' for call in runtime.calls)


@pytest.fixture
def paused_binding(isaac, monkeypatch):
    """Exercise downstream binding with synthetic admission, never real physics."""
    from doorbench.dexterous.qualified_isaac_grasp import digest
    context=isaac['context']
    snapshot=isaac['run']/'snapshot.json'
    write(snapshot,{'synthetic_fixture':True})
    configuration=isaac['run']/'configuration.json'
    write(configuration,{'synthetic_fixture':True})
    context.motor_contract_path=isaac['run']/'trial/motor-contract.json'
    context.configuration_path=configuration
    context.admission['source_kind']=dispatch.PAUSED_SOURCE_KIND
    for path in (snapshot,configuration):
        context.admission['input_sha256'][str(path)]=digest(path)
    for document in (isaac['screen'],isaac['audit']):
        document['source_kind']=dispatch.PAUSED_SOURCE_KIND
        document['source_admission']=copy.deepcopy(context.admission)
        document['input_sha256'].update(context.admission['input_sha256'])
    config=isaac['config']
    config.pop('contact_audit_name')
    config.update(source_kind=dispatch.PAUSED_SOURCE_KIND,
        source_snapshot_path=str(snapshot),source_snapshot_sha256=digest(snapshot),
        motor_contract_path=str(context.motor_contract_path),configuration_path=str(configuration))
    def admit(source,**kwargs):
        assert source==snapshot
        assert kwargs['source_kind']==dispatch.PAUSED_SOURCE_KIND
        return context
    monkeypatch.setattr(dispatch,'admit_release_planning_source',admit)
    bind_documents(isaac)
    return isaac


def test_paused_withdrawal_source_keeps_episode_snapshot_and_live_authority_distinct(paused_binding):
    data=load_isaac(paused_binding).data
    assert data['source_run']==str(paused_binding['run'])
    assert data['source_snapshot_path']==paused_binding['config']['source_snapshot_path']
    assert data['source_archive_path']==str(paused_binding['context'].state_archive_path)
    assert data['live_pause_authorization_required'] is True
    assert 'live_exact_prefix_required' not in data
    assert data['authorized_stages']==0 and data['physical_release_qualification'] is False


@pytest.mark.parametrize('field',[
    'source_snapshot_sha256','motor_contract_path','configuration_path','source_run'])
def test_paused_withdrawal_source_rejects_changed_identity(paused_binding,field):
    paused_binding['config'][field]='different'
    with pytest.raises(ValueError,match='Paused source paths/identity'):
        load_isaac(paused_binding)


@pytest.mark.parametrize('document',['screen','audit'])
def test_paused_withdrawal_source_rejects_completed_artifacts(paused_binding,document):
    paused_binding[document].pop('source_kind')
    bind_documents(paused_binding)
    with pytest.raises(ValueError):load_isaac(paused_binding)
