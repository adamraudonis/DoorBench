"""Comparison must not report gains from pending jobs or changed source/gates."""
import json
from pathlib import Path
import shutil

import pytest

from scripts.compare_planned_reference_corpora import compare,identity
from tests.test_export_planned_reference_web import corpus,sha,write


@pytest.fixture
def pair(corpus,tmp_path):
    before,_,_=corpus
    index=json.loads((before/'index.json').read_text());index['doors']=index['doors'][:1]
    index['selected_ids']=['fixture'];index['generator']['files']={
        'doorbench/reference/rig.py':'c'*64,'scripts/validate_planned_reference.py':'d'*64}
    result=json.loads((before/'fixture/result.json').read_text())
    result['provenance'].update(recording_index_sha256='e'*64,recording_sha256={'clip.json':'f'*64,'trajectory.npz':'1'*64})
    result['identity_sha256']=identity(result['provenance'])
    index['doors'][0].update(action='completed',identity_sha256=result['identity_sha256'])
    write(before/'fixture/result.json',result);write(before/'index.json',index)
    write(before/'report.json',{'snapshot_id':index['snapshot_id'],'status_counts':{'accepted_kinematic':1}})
    after=tmp_path/'after';shutil.copytree(before,after)
    return before,after


def refresh(root):
    result=json.loads((root/'fixture/result.json').read_text());index=json.loads((root/'index.json').read_text())
    result['identity_sha256']=identity(result['provenance'])
    result['artifacts']={p.name:sha(p) for p in (root/'fixture').iterdir() if p.name!='result.json'}
    index['doors'][0].update(identity_sha256=result['identity_sha256'],status=result['status'])
    write(root/'fixture/result.json',result);write(root/'index.json',index)
    write(root/'report.json',{'snapshot_id':index['snapshot_id'],'status_counts':{result['status']:1}})


def reject(root):
    result=json.loads((root/'fixture/result.json').read_text());result['status']='rejected';result['reason_code']='independent_validation_rejected'
    write(root/'fixture/result.json',result)
    audit=json.loads((root/'fixture/validation.json').read_text())
    audit.update(accepted=False,kinematic_accepted=False,status='rejected',failure_counts={'noncontact_clearance':1})
    write(root/'fixture/validation.json',audit);refresh(root)


def test_identical_complete_runs_keep_task_counts_and_do_not_infer_visual_or_dynamic_success(pair):
    r=compare(*pair)
    assert r['newly_accepted']==r['lost_acceptance']==[]
    assert r['transitions']=={'accepted_kinematic -> accepted_kinematic':1}
    assert r['before']['accepted_scenarios']=={'locked_recognize':1}
    assert r['common_accepted_durations'][0]['change_s']==0
    assert 'No new physics or visual acceptance' in r['scope']


@pytest.mark.parametrize('side',['before','after'])
def test_gains_and_regressions_retain_exact_door_and_source_task(pair,side):
    reject(pair[side=='after']);r=compare(*pair)
    assert r['newly_accepted' if side=='before' else 'lost_acceptance']==['fixture']
    assert r['changed_statuses'][0]['scenario']=='locked_recognize'


def test_pending_attempt_cannot_be_counted_as_unresolved_result(pair):
    p=pair[1]/'index.json';index=json.loads(p.read_text());index['doors'][0]['action']='run';write(p,index)
    with pytest.raises(ValueError,match='pending attempts'):compare(*pair)


@pytest.mark.parametrize('field',['recording_index_sha256','recording_sha256','source_sha256','native_resources_sha256'])
def test_changed_native_evidence_is_not_reported_as_planner_improvement(pair,field):
    # Rejected results avoid an earlier accepted-source hash check; cross-run
    # provenance still has to reject a changed native dependency.
    reject(pair[1]);p=pair[1]/'fixture/result.json';result=json.loads(p.read_text())
    result['provenance'][field]='2'*64 if field.endswith('index_sha256') else {'changed':'2'*64}
    write(p,result);refresh(pair[1])
    with pytest.raises(ValueError,match='native data or source task changed'):compare(*pair)


@pytest.mark.parametrize('file',['doorbench/reference/rig.py','scripts/validate_planned_reference.py'])
def test_changed_rig_or_validator_blocks_like_for_like_comparison(pair,file):
    p=pair[1]/'index.json';index=json.loads(p.read_text());index['generator']['files'][file]='a'*64;write(p,index)
    with pytest.raises(ValueError,match='same rig and independent validator'):compare(*pair)


def test_changed_numeric_thresholds_are_rejected_even_with_same_validator_file(pair):
    p=pair[1]/'fixture/validation.json';audit=json.loads(p.read_text());audit['settings']['clearance_m']=.001;write(p,audit);refresh(pair[1])
    with pytest.raises(ValueError,match='thresholds changed'):compare(*pair)


def test_fresh_artifact_hash_cannot_promote_failed_independent_report(pair):
    p=pair[1]/'fixture/validation.json';audit=json.loads(p.read_text());audit['accepted']=False;write(p,audit);refresh(pair[1])
    with pytest.raises(ValueError,match='did not pass independent'):compare(*pair)


def test_replaced_motion_invalidates_bound_validation(pair):
    p=pair[1]/'fixture/trajectory.npz';p.write_bytes(p.read_bytes()+b'changed');refresh(pair[1])
    with pytest.raises(ValueError,match='binds different motion'):compare(*pair)


def test_unlisted_failure_file_blocks_comparison(pair):
    (pair[1]/'fixture/failure.json').write_text('{}')
    with pytest.raises(ValueError,match='inventory'):compare(*pair)


@pytest.mark.parametrize('flag',['complete_proposal','artifact_bindings_verified','task_evidence_pass','source_success_declared'])
def test_incomplete_result_evidence_cannot_count_as_an_acceptance(pair,flag):
    p=pair[1]/'fixture/result.json';result=json.loads(p.read_text())
    result['new_completion'][flag]=False;write(p,result);refresh(pair[1])
    with pytest.raises(ValueError,match='did not pass independent'):compare(*pair)


def test_partial_task_report_cannot_count_as_an_acceptance(pair):
    p=pair[1]/'fixture/validation.json';audit=json.loads(p.read_text())
    audit['task_completion']['complete_proposal']=False;write(p,audit);refresh(pair[1])
    with pytest.raises(ValueError,match='did not pass independent'):compare(*pair)
