import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.export_planned_reference_web import export_corpus
from scripts.validate_planned_reference import validate
from tests.test_validate_planned_reference import fixture,rewrite

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value))

@pytest.fixture
def corpus(tmp_path):
    clip_path,trajectory,assets,clip,arrays=fixture(tmp_path/'source')
    directory=assets/'doors/fixture';write(directory/'model.json',{'bodies':[{'name':'world_env','geoms':[{'name':'floor','semantic':'floor'}]},{'name':'leaf','geoms':[{'name':'leaf_geom','semantic':'leaf'}]}]})
    clip['source_sha256']['model.json']=sha(directory/'model.json');clip['proposal']['source_sha256']=clip['source_sha256'].copy();clip['phases']=['hold']*3
    rewrite(clip_path,trajectory,clip,arrays);validation=validate(clip_path,trajectory,assets);assert validation['accepted']
    root=tmp_path/'corpus';door=root/'fixture';door.mkdir(parents=True)
    (door/'clip.json').write_bytes(clip_path.read_bytes());(door/'trajectory.npz').write_bytes(trajectory.read_bytes());write(door/'validation.json',validation)
    result={'door_id':'fixture','identity_sha256':'a'*64,'status':'accepted_kinematic',
        'source_outcome':clip['proposal']['source_outcome'],
        'new_completion':dict.fromkeys(['complete_proposal','artifact_bindings_verified','task_evidence_pass','source_success_declared'],True),
        'artifacts':{name:sha(door/name) for name in ['clip.json','trajectory.npz','validation.json']},
        'provenance':{'source_sha256':clip['source_sha256'],'native_resources_sha256':{}}}
    write(assets/'manifest.json',{'doors':[{'id':'fixture'},{'id':'waiting'}]})
    result['provenance'].update(generator_sha256='b'*64,manifest_sha256=sha(assets/'manifest.json'));write(door/'result.json',result)
    index={'schema':'doorbench.planned-reference-corpus.v1','snapshot_id':'test','updated_at':'2026-09-05T00:00:00Z',
        'manifest_sha256':sha(assets/'manifest.json'),'generator':{'sha256':'b'*64},'doors':[
            {'door_id':'fixture','family':'swing_single','status':'accepted_kinematic','identity_sha256':'a'*64,'result':'fixture/result.json'},
            {'door_id':'waiting','family':'rollup','status':'unresolved','result':None,'reason_code':'pending'}]}
    write(root/'index.json',index)
    return root,tmp_path/'web',assets

def test_roundtrip_exact_pose_export_deterministic_checksums_and_nonplayable_status(corpus):
    root,out,assets=corpus;index=export_corpus(root,out,assets)
    descriptor=index['doors'][0]['clip'];packed=(out/descriptor['path']).read_bytes()
    assert hashlib.sha256(packed).hexdigest()==descriptor['sha256']
    raw=gzip.decompress(packed);assert hashlib.sha256(raw).hexdigest()==descriptor['json_sha256'];clip=json.loads(raw)
    with np.load(root/'fixture/trajectory.npz',allow_pickle=False) as arrays:
        poses=np.array(clip['actor']['poses']).reshape(3,16,7)
        np.testing.assert_array_equal(poses[:,:,:3],arrays['actor_body_pos'])
        np.testing.assert_array_equal(poses[:,:,3:],arrays['actor_body_quat'])
        np.testing.assert_array_equal(clip['times'],arrays['actor_time'])
    assert index['doors'][1]['clip'] is None
    assert export_corpus(root,out,assets)==index
    for item in index['doors'][0]['audits'].values():assert sha(out/item['path'])==item['sha256']

@pytest.mark.parametrize('artifact',['clip.json','trajectory.npz','validation.json'])
def test_reject_corrupt_bound_artifact_before_publishing_index(corpus,artifact):
    root,out,assets=corpus;(root/'fixture'/artifact).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='stale'):export_corpus(root,out,assets)
    assert not (out/'index.json').exists()

def test_reject_changed_source_geometry(corpus):
    root,out,assets=corpus;p=assets/'doors/fixture/door.xml';p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='stale source'):export_corpus(root,out,assets)

def test_reject_result_bound_to_other_generator(corpus):
    root,out,assets=corpus;p=root/'fixture/result.json';result=json.loads(p.read_text());result['provenance']['generator_sha256']='c'*64;write(p,result)
    with pytest.raises(ValueError,match='different corpus generator'):export_corpus(root,out,assets)

def test_reject_validation_for_different_trajectory_even_when_artifact_hash_is_fresh(corpus):
    root,out,assets=corpus;p=root/'fixture/validation.json';v=json.loads(p.read_text());v['trajectory_sha256']='0'*64;write(p,v)
    result=json.loads((root/'fixture/result.json').read_text());result['artifacts']['validation.json']=sha(p);write(root/'fixture/result.json',result)
    with pytest.raises(ValueError,match='different artifacts'):export_corpus(root,out,assets)

def test_reject_false_acceptance_even_with_matching_result_status(corpus):
    root,out,assets=corpus;p=root/'fixture/validation.json';v=json.loads(p.read_text());v['accepted']=False;write(p,v)
    result=json.loads((root/'fixture/result.json').read_text());result['artifacts']['validation.json']=sha(p);write(root/'fixture/result.json',result)
    with pytest.raises(ValueError,match='did not accept'):export_corpus(root,out,assets)

@pytest.mark.parametrize('path',['../fixture/result.json','fixture/../fixture/result.json','/tmp/result.json'])
def test_reject_traversal_paths(corpus,path):
    root,out,assets=corpus;p=root/'index.json';index=json.loads(p.read_text());index['doors'][0]['result']=path;write(p,index)
    with pytest.raises(ValueError,match='Unsafe'):export_corpus(root,out,assets)

def test_reject_symlink_artifact(corpus,tmp_path):
    root,out,assets=corpus;p=root/'fixture/trajectory.npz';data=p.read_bytes();p.unlink();target=tmp_path/'external.npz';target.write_bytes(data);p.symlink_to(target)
    with pytest.raises(ValueError,match='symlink|escapes'):export_corpus(root,out,assets)

def test_output_cannot_mutate_input_tree(corpus):
    root,_,assets=corpus
    with pytest.raises(ValueError,match='separate'):export_corpus(root,root/'web',assets)

@pytest.mark.parametrize('listed',[True,False])
def test_accepted_failure_file_blocks_playback_even_if_hash_matches(corpus,listed):
    root,out,assets=corpus;p=root/'fixture/failure.json';write(p,{'error':'failed'})
    if listed:
        result=json.loads((p.parent/'result.json').read_bytes());result['artifacts'][p.name]=sha(p);write(p.parent/'result.json',result)
    with pytest.raises(ValueError,match='failure'):export_corpus(root,out,assets)
    assert not (out/'index.json').exists()

@pytest.mark.parametrize('mutation',['missing','corrupt','unlisted'])
def test_full_attempt_inventory_is_checked(corpus,mutation):
    root,out,assets=corpus;p=root/'fixture/attempt.log';p.write_text('safe log')
    result=json.loads((p.parent/'result.json').read_bytes())
    if mutation!='unlisted':result['artifacts'][p.name]=sha(p)
    write(p.parent/'result.json',result)
    if mutation=='missing':p.unlink()
    if mutation=='corrupt':p.write_text('changed')
    with pytest.raises(ValueError,match='inventory|stale'):export_corpus(root,out,assets)

def test_reject_success_flags_with_different_source_outcome(corpus):
    root,out,assets=corpus;p=root/'fixture/result.json';result=json.loads(p.read_bytes());result['source_outcome']['success']=False;write(p,result)
    with pytest.raises(ValueError,match='outcome binding'):export_corpus(root,out,assets)

def test_public_result_is_labeled_hash_bound_projection_without_local_paths(corpus):
    root,out,assets=corpus;p=root/'fixture/result.json';result=json.loads(p.read_bytes())
    result['error']={'type':'Example','message':'Cannot load /Users/example/private/file.xml','traceback':'SECRET traceback'}
    result['execution']={'pid':123,'command':'private command'};write(p,result);original_sha=sha(p)
    index=export_corpus(root,out,assets);data=(out/index['doors'][0]['audits']['result.json']['path']).read_bytes();public=json.loads(data)
    assert public['schema']=='doorbench.planned-reference-public-attempt.v1'
    assert public['original_result_sha256']==original_sha
    assert public['error']['message']=='Cannot load [local path]'
    assert all(word not in data.decode() for word in ['SECRET','/Users/','private command','traceback"','pid"'])

def test_missing_status_row_cannot_claim_all_door_coverage(corpus):
    root,out,assets=corpus;p=root/'index.json';index=json.loads(p.read_bytes());index['doors'].pop();write(p,index)
    with pytest.raises(ValueError,match='cover the source manifest'):export_corpus(root,out,assets)
