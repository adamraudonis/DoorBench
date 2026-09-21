"""Protocol-only fake backend inventories; no source/physics qualification."""
import copy
import json

import numpy as np
import pytest

from doorbench.dexterous import isaac_live_planning_pause as p


class Clock:
    now=10.
    def __call__(self):return self.now


@pytest.fixture
def case(tmp_path):
    clock=Clock();objects={'transfer':object(),'operation':object(),'stance':object(),'left':object()}
    pause=p.LivePlanningPause(tmp_path/'fresh',episode_id='synthetic-episode',retained_objects=objects,timeout_seconds=60,monotonic=clock)
    state=dict(measured={'root13':np.r_[np.zeros(3),1.,np.zeros(9)].astype(np.float32),
        'joints69':np.zeros(69,np.float32),'joint_velocity69':np.zeros(69,np.float32),
        'door3':np.zeros(3,np.float32),'door_velocity3':np.zeros(3,np.float32),
        'body_poses':np.zeros((7,7),np.float32),'previous_motor61':np.arange(61,dtype=np.float64)},
        controller={'clock':.018,'qp_warm_start':np.arange(38,dtype=np.float64),'left_offset':.004,'support_previous':.018},
        physics_clock={'time':.020,'step':10,'callbacks':10,'robot_cache_time':.020},counts={'physics':10,'pad':11,'transfer':10})
    def anchor():return p.capture_pause_anchor(episode_id='synthetic-episode',step_index=10,epoch_s=.020,
        measured=state['measured'],controller=state['controller'],physics_clock=state['physics_clock'],
        retained_objects=objects,evidence_counts=state['counts'],pending_unaccepted_command=False)
    a=anchor();evidence=pause.directory/'evidence.json';evidence.write_text('[]')
    snapshot=dict(schema=p.SNAPSHOT_SCHEMA,source_kind='paused-live-isaac-transfer-v1',source_engine='isaac-physx',
        closed_prefix=True,episode_complete=False,live_pause={k:a[k] for k in ('episode_id','controller_identity','step_index','epoch_s',
        'physics_dt_s','measurement_fingerprint','controller_fingerprint')},files={'synthetic_only':dict(path=str(evidence),sha256=p._sha(evidence))})
    snapshot['live_pause']['pause_token']=pause.pause_token
    snapshot_path=pause.directory/'snapshot.json';snapshot_path.write_text(json.dumps(snapshot))
    return pause,clock,objects,state,anchor,snapshot_path,snapshot


def ready(case):
    pause,clock,objects,state,anchor,path,snapshot=case
    pause.publish(path,anchor());files={}
    for role in ('runtime','phase_audit','context'):
        target=pause.directory/(role+'.json');target.write_text(json.dumps({'synthetic_role':role}))
        files[role]=dict(path=str(target),sha256=p._sha(target))
    response=dict(schema=p.RESPONSE_SCHEMA,pause_token=pause.pause_token,episode_id=pause.episode_id,
        snapshot_path=str(path),snapshot_sha256=p._sha(path),decision='ready',source_context_sha256='a'*64,files=files)
    pause.response_path.write_text(json.dumps(response));pause.poll()
    return response


@pytest.mark.parametrize('keep_source_engine',[False,True])
def test_legacy_engine_key_rejected_in_favor_of_source_inspector_contract(case,keep_source_engine):
    pause,clock,objects,state,anchor,path,snapshot=case
    snapshot['engine']='isaac-physx'
    if not keep_source_engine:del snapshot['source_engine']
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError,match='unfinished same-live'):
        pause.publish(path,anchor())
    assert pause.state=='aborted'


def admission(pause,response):
    # Explicit fake trusted seam; this is not a real geometric/source audit.
    return dict(passed=True,snapshot_sha256=response['snapshot_sha256'],source_context_sha256=response['source_context_sha256'],
        input_sha256={**pause.snapshot['files'],pause.snapshot['path']:pause.snapshot['sha256'],
            **pause.response['files'],str(pause.response_path):pause.response['sha256']},synthetic_only=True)


def test_request_pause_and_exact_once_resume_keep_objects_and_state(case):
    pause,clock,objects,state,anchor,path,snapshot=case;before=anchor();ids={k:id(v) for k,v in objects.items()}
    request=pause.publish(path,anchor());assert pause.poll() is None
    assert request['live_pause']['epoch_s']==.020 and request['authorized_stages']==0
    assert request['snapshot_sha256']==p._sha(path)
    files={}
    for role in ('runtime','phase_audit','context'):
        q=pause.directory/(role+'.json');q.write_text('{}');files[role]=dict(path=str(q),sha256=p._sha(q))
    response=dict(schema=p.RESPONSE_SCHEMA,pause_token=pause.pause_token,episode_id=pause.episode_id,snapshot_path=str(path),
        snapshot_sha256=p._sha(path),decision='ready',source_context_sha256='a'*64,files=files)
    pause.response_path.write_text(json.dumps(response));pause.poll();calls=[]
    def capture():calls.append('read');return anchor()
    def validate(r):calls.append('admit');return admission(pause,r)
    receipt=pause.validate_resume(capture_anchor=capture,validate_plan=validate)
    assert calls==['read','admit','read'] and receipt['resume_handshake_passed']
    assert receipt['authorized_stages']==0 and receipt['episode_complete'] is False
    assert anchor()==before and {k:id(v) for k,v in objects.items()}==ids
    with pytest.raises(ValueError):pause.validate_resume(capture_anchor=capture,validate_plan=validate)
    assert calls==['read','admit','read']


@pytest.mark.parametrize('kind',['root','velocity','body','command','dtype','signed_zero','controller','clock','count','identity'])
def test_any_live_change_rejects_without_restore(case,kind):
    pause,_,objects,state,anchor,_,_=case;ready(case)
    if kind=='root':state['measured']['root13'][0]=np.nextafter(np.float32(0),np.float32(1))
    elif kind=='velocity':state['measured']['joint_velocity69'][3]=.001
    elif kind=='body':state['measured']['body_poses'][0,0]=.001
    elif kind=='command':state['measured']['previous_motor61'][0]=.001
    elif kind=='dtype':state['measured']['joints69']=state['measured']['joints69'].astype(np.float64)
    elif kind=='signed_zero':state['measured']['joints69'][0]=-0.
    elif kind=='controller':state['controller']['qp_warm_start'][0]=.001
    elif kind=='clock':state['physics_clock']['callbacks']+=1
    elif kind=='count':state['counts']['pad']+=1
    elif kind=='identity':objects['left']=object()
    altered=anchor()
    with pytest.raises(ValueError):pause.validate_resume(capture_anchor=anchor,validate_plan=lambda r:pytest.fail('Changed state admitted'))
    assert pause.state=='aborted' and anchor()==altered
    assert pause.request_path.exists() and pause.response_path.exists()


def test_mutation_during_trusted_admission_rejected(case):
    pause,_,_,state,anchor,_,_=case;ready(case)
    def mutate(response):
        result=admission(pause,response);state['controller']['support_previous']+=.002;return result
    with pytest.raises(ValueError,match='changed during'):pause.validate_resume(capture_anchor=anchor,validate_plan=mutate)
    assert pause.state=='aborted' and state['controller']['support_previous']==.018+.002


@pytest.mark.parametrize('change',['token','episode','snapshot','context','missing_file','file_hash','outside','schema','duplicate','abort'])
def test_stale_corrupt_or_failed_response_aborts(case,change,tmp_path):
    pause,_,_,_,anchor,path,_=case
    response=ready(case);pause.response=None;pause.state='waiting'  # Synthetic malformed first-read fixture only.
    if change=='token':response['pause_token']='f'*64
    elif change=='episode':response['episode_id']='old'
    elif change=='snapshot':response['snapshot_sha256']='f'*64
    elif change=='context':response['source_context_sha256']='bad'
    elif change=='missing_file':response['files'].pop('phase_audit')
    elif change=='file_hash':response['files']['runtime']['sha256']='f'*64
    elif change=='outside':
        q=tmp_path/'outside.json';q.write_text('{}');response['files']['runtime']=dict(path=str(q),sha256=p._sha(q))
    elif change=='schema':response['schema']='completed-offline-source'
    elif change=='abort':
        response['decision']='abort';response.pop('files');response.pop('source_context_sha256');response['reason']='Unchanged original geometry gates failed'
    text=json.dumps(response)
    if change=='duplicate':text=text[:-1]+',"episode_id":"old"}'
    pause.response_path.write_text(text)
    with pytest.raises(ValueError):pause.poll()
    assert pause.state=='aborted' and path.exists()


@pytest.mark.parametrize('when',['publish','waiting','ready','during_admission'])
def test_deadline_is_bounded_at_every_transition(case,when):
    pause,clock,_,_,anchor,path,_=case
    if when=='publish':
        clock.now=70.
        with pytest.raises(TimeoutError):pause.publish(path,anchor())
    elif when=='waiting':
        pause.publish(path,anchor());clock.now=70.
        with pytest.raises(TimeoutError):pause.poll()
    else:
        ready(case)
        def validate(response):
            result=admission(pause,response)
            if when=='during_admission':clock.now=70.
            return result
        if when=='ready':clock.now=70.
        with pytest.raises(TimeoutError):pause.validate_resume(capture_anchor=anchor,validate_plan=validate)
    assert pause.state=='aborted' and path.exists()


@pytest.mark.parametrize('target',['snapshot','evidence','runtime','response'])
def test_mutated_bound_files_reject_before_resume(case,target):
    pause,_,_,_,anchor,path,_=case;ready(case)
    q={'snapshot':path,'evidence':pause.directory/'evidence.json','runtime':pause.directory/'runtime.json','response':pause.response_path}[target]
    q.write_text('changed')
    with pytest.raises(ValueError,match='input changed'):pause.validate_resume(capture_anchor=anchor,validate_plan=lambda r:pytest.fail('Changed inputs admitted'))
    assert pause.state=='aborted' and q.read_text()=='changed'


def test_failed_admission_or_missing_bindings_cannot_resume(case):
    pause,_,_,_,anchor,_,_=case;ready(case)
    with pytest.raises(ValueError,match='trusted runtime admission'):
        pause.validate_resume(capture_anchor=anchor,validate_plan=lambda r:dict(passed=True))
    assert not pause.receipt()['resume_handshake_passed']


def test_terminal_write_failure_preserves_original_error_and_never_grants_resume(case):
    pause,_,_,_,anchor,_,_=case;ready(case)
    q=pause.directory/'terminal.json';q.write_text('unrelated existing file')
    with pytest.raises(FileExistsError):pause.validate_resume(capture_anchor=anchor,validate_plan=lambda r:admission(pause,r))
    assert pause.state=='aborted' and q.read_text()=='unrelated existing file'


def test_fresh_directories_tokens_and_no_implicit_plant_callbacks(case,tmp_path):
    pause,clock,objects,_,_,_,_=case
    other=p.LivePlanningPause(tmp_path/'other',episode_id=pause.episode_id,retained_objects=objects,timeout_seconds=1,monotonic=clock)
    assert len(pause.pause_token)==64 and pause.pause_token!=other.pause_token
    with pytest.raises(FileExistsError):p.LivePlanningPause(pause.directory,episode_id='x',retained_objects=objects,timeout_seconds=1)
    assert not hasattr(pause,'sim') and not hasattr(pause,'controller')


def test_pending_or_nonfinite_anchor_rejected(case):
    _,_,objects,_,_,_,_=case
    kwargs=dict(episode_id='x',step_index=1,epoch_s=.002,physics_clock={'callbacks':1},measured={'q':np.zeros(1)},
        controller={'time':0.},retained_objects=objects,evidence_counts={'physics':1},pending_unaccepted_command=True)
    with pytest.raises(ValueError,match='pending command'):p.capture_pause_anchor(**kwargs)
    kwargs['pending_unaccepted_command']=False;kwargs['measured']['q'][0]=np.nan
    with pytest.raises(ValueError,match='Finite'):p.capture_pause_anchor(**kwargs)


def test_native_serialized_bytes_are_hash_bound_without_deserialization(case):
    pause,_,_,state,anchor,_,_=case
    state['controller']['native_state']=b'opaque native serialization\x00\xff'
    a=anchor();state['controller']['native_state']+=b'changed'
    assert anchor()['controller_fingerprint']!=a['controller_fingerprint']


def test_backward_clock_aborts_and_explicit_planner_reason_is_retained(case):
    pause,clock,_,_,anchor,path,_=case;pause.publish(path,anchor());clock.now=9.
    with pytest.raises(ValueError,match='backward'):pause.poll()
    assert 'backward' in pause.receipt()['failure']
