"""Fail-closed storyboard cache and phase coverage without requiring Blender."""
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from scripts import blender_motion_storyboards as boards
from tests.test_blender_planned_motion import inputs, external_report


def write(path,value):Path(path).write_text(json.dumps(value))


@pytest.fixture
def storyboard(inputs):
    root,job,clip,arrays=inputs
    clip['phases']=['approach','reach','operate','traverse']
    clip['proposal']={'source_outcome':{'success':True,'outcome':'success'}}
    validation=external_report(inputs)
    attempt=root/'attempt';attempt.mkdir()
    for name in ['clip.json','trajectory.npz']:shutil.copyfile(root/name,attempt/name)
    write(attempt/'validation.json',validation)
    result={'door_id':'tiny','status':'accepted_kinematic','source_outcome':clip['proposal']['source_outcome'],
            'new_completion':dict.fromkeys(['complete_proposal','artifact_bindings_verified','task_evidence_pass','source_success_declared'],True),
            'provenance':{'source_sha256':job['source_sha256']},
            'artifacts':{name:boards.sha(attempt/name) for name in ['clip.json','trajectory.npz','validation.json']}}
    write(attempt/'result.json',result)
    out=root/'boards';out.mkdir()
    config={name:str(attempt/file) for name,file in [('clip','clip.json'),('trajectory','trajectory.npz'),('validation','validation.json'),('result','result.json')]}
    config['appearance_job']=str(root/'job.json');write(out/'worker.json',config)
    _,_,expected,samples,_,_=boards.load_storyboard_inputs(out/'worker.json')
    rendered=[]
    for i,sample in enumerate(samples):
        path=out/f'frame-{i:02d}.png';Image.new('RGB',(480,360),(100+i,100,100)).save(path)
        rendered.append({**sample,'image':path.name,'sha256':boards.sha(path),'pose_check':{'max_position_error_m':0.,'max_rotation_error_rad':0.}})
    report={'schema':'doorbench.motion-storyboard.v1',**expected,'duration_s':1.5,'samples':rendered}
    write(out/'storyboard.json',report)
    return out,attempt,job,expected,samples


def test_phase_sampling_covers_every_contiguous_phase_and_dense_operation():
    phases=['approach']*4+['operate']*9+['reach']+['operate']*5+['traverse']*9+['settle']*3
    times=[i*i*.01 for i in range(len(phases))]
    samples=boards.phase_samples(phases,times);indices={s['index'] for s in samples}
    assert {0,len(phases)-1}<=indices
    assert {4,6,8,10,12,14,15,16,17,18,19,21,23,25,27}<=indices
    assert all(s['phase']==phases[s['index']] and s['time_s']==times[s['index']] for s in samples)
    assert any(s['index']==13 for s in samples),'Single-frame phase must survive sampling'


@pytest.mark.parametrize('phases,times',[
    (['hold'],[0]),(['a','b'],[0,float('nan')]),(['a','b'],[0,0]),
    (['a','b'],[1,2]),(['a',''],[0,1]),(['a'],[0,1]),
])
def test_invalid_phase_timeline_is_rejected(phases,times):
    with pytest.raises(ValueError):boards.phase_samples(phases,times)


def test_storyboard_pose_slicing_keeps_original_time_and_exact_array_values(inputs):
    import numpy as np
    arrays=inputs[3];before={k:v.copy() for k,v in arrays.items()}
    selected=boards.sampled_arrays(arrays,[{'index':0},{'index':2},{'index':3}])
    for key,value in before.items():
        np.testing.assert_array_equal(selected[key],value[[0,2,3]])
        np.testing.assert_array_equal(arrays[key],value)
    selected['body_pos'][0,0,0]=99
    assert arrays['body_pos'][0,0,0]==0,'Selected data must not mutate full original input'
    with pytest.raises(ValueError,match='endpoints'):boards.sampled_arrays(arrays,[{'index':0},{'index':2}])


def test_contact_sheet_is_hash_bound_and_resume_checks_each_frame(storyboard):
    out,_,_,expected,samples=storyboard
    boards.assemble(out)
    report=boards.checked_report(out,expected,samples,require_sheet=True)
    assert report['contact_sheet_sha256']==boards.sha(out/'contact-sheet.jpg')
    Image.new('RGB',(480,360),'red').save(out/'frame-00.png')
    with pytest.raises(ValueError,match='image changed'):boards.checked_report(out,expected,samples,require_sheet=True)


@pytest.mark.parametrize('key',['input_sha256','script_sha256','renderer_sha256','replay_helper_sha256','appearance_job_sha256','source_sha256','mesh_sha256','appearance_renderer_sha256'])
def test_resume_rejects_stale_input_and_all_renderer_dependencies(storyboard,key):
    out,_,_,expected,samples=storyboard;report=json.loads((out/'storyboard.json').read_text());report[key]='changed';write(out/'storyboard.json',report)
    with pytest.raises(ValueError,match='provenance'):boards.checked_report(out,expected,samples)


def test_missing_phase_image_cannot_be_hidden_by_valid_contact_sheet(storyboard):
    out,_,_,expected,samples=storyboard;boards.assemble(out)
    (out/'frame-01.png').unlink()
    with pytest.raises(OSError):boards.checked_report(out,expected,samples,require_sheet=True)


@pytest.mark.parametrize('damage',['source','trajectory','failed_report','failure_file','source_success'])
def test_source_or_acceptance_changes_fail_before_cache_use(storyboard,damage):
    out,attempt,job,_,_=storyboard
    result=json.loads((attempt/'result.json').read_text())
    if damage=='source':(Path(job['door_dir'])/'door.xml').write_text('changed')
    if damage=='trajectory':(attempt/'trajectory.npz').write_bytes(b'changed')
    if damage=='failed_report':
        report=json.loads((attempt/'validation.json').read_text());report['accepted']=False;write(attempt/'validation.json',report)
        result['artifacts']['validation.json']=boards.sha(attempt/'validation.json');write(attempt/'result.json',result)
    if damage=='failure_file':write(attempt/'failure.json',{'error':'failed'})
    if damage=='source_success':result['source_outcome']['success']=False;write(attempt/'result.json',result)
    with pytest.raises(ValueError):boards.load_storyboard_inputs(out/'worker.json')
    with pytest.raises(ValueError):boards.assemble(out)


@pytest.mark.parametrize('damage',['traversal','symlink','dimensions','phase','pose_check'])
def test_report_cannot_select_other_paths_dimensions_or_phase_samples(storyboard,damage):
    out,_,_,expected,samples=storyboard;report=json.loads((out/'storyboard.json').read_text())
    if damage=='traversal':report['samples'][0]['image']='../frame-00.png'
    if damage=='phase':report['samples'][0]['time_s']+=.25
    if damage=='pose_check':report['samples'][0]['pose_check']['max_position_error_m']=.001
    if damage=='symlink':
        image=out/'frame-00.png';data=image.read_bytes();image.unlink();target=out.parent/'external.png';target.write_bytes(data);image.symlink_to(target)
    if damage=='dimensions':
        Image.new('RGB',(10,10),'red').save(out/'frame-00.png');report['samples'][0]['sha256']=boards.sha(out/'frame-00.png')
    write(out/'storyboard.json',report)
    with pytest.raises(ValueError):boards.checked_report(out,expected,samples)
