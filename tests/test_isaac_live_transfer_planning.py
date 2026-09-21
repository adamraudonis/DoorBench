"""Clock, file-only pause orchestration and CLI; no physical source claims."""
import json
import ast
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from doorbench.dexterous import isaac_live_transfer_planning as planning
from doorbench.dexterous import isaac_live_planning_pause as protocol
from test_isaac_standing_transfer_cli import command,ROOT
from test_isaac_transfer_prefix_wiring import MAIN,execute
import numpy as np


def test_suffix_starts_at_next_index_and_replaces_unused_transfer_budget():
    clock=planning.LiveEpisodeClock(1.)
    seen=[]
    for step in clock:
        seen.append(step)
        if step==249:assert clock.append_suffix(.5,.016)==.516
    assert seen==list(range(258)) and clock.suffix
    with pytest.raises(ValueError):clock.append_suffix(.516,.016)


@pytest.mark.parametrize('start,duration',[(0.,.1),(.004,.1),(.002,0.),(.002,float('nan')),(.002,.001)])
def test_suffix_rejects_wrong_epoch_or_invalid_duration(start,duration):
    clock=planning.LiveEpisodeClock(1.);next(iter(clock))
    with pytest.raises(ValueError):clock.append_suffix(start,duration)
    assert not clock.suffix and clock.limit==500


@pytest.mark.parametrize('changed',[False,True])
def test_actual_protocol_wait_does_not_advance_or_reconstruct_controller(tmp_path,monkeypatch,changed):
    events=[];objects={'actual_retained_stub':object()};reads=[]
    observer=SimpleNamespace(require_ready=lambda t:events.append(('ready',t)),
        retained_objects=lambda:objects,snapshot=lambda:{'synthetic_observer':True})
    def anchor():
        reads.append(1)
        return protocol.capture_pause_anchor(episode_id='test',step_index=251,epoch_s=.502,
            physics_clock={'step':251},measured={'state':2 if changed and len(reads)>1 else 1},
            controller={'synthetic':True},retained_objects=objects,
            evidence_counts={'physical':251},pending_unaccepted_command=False)
    def assemble(folder,*,pause_token,anchor,**kwargs):
        folder.mkdir();evidence=folder/'evidence.json';evidence.write_text('{}')
        live={key:anchor[key] for key in ('episode_id','controller_identity','step_index','epoch_s',
            'physics_dt_s','measurement_fingerprint','controller_fingerprint')}
        live['pause_token']=pause_token
        snapshot=folder/'snapshot.json'
        snapshot.write_text(json.dumps(dict(schema=protocol.SNAPSHOT_SCHEMA,
            source_kind='paused-live-isaac-transfer-v1',source_engine='isaac-physx',
            episode_complete=False,closed_prefix=True,live_pause=live,
            files={'fixture':dict(path=str(evidence),sha256=protocol._sha(evidence))})))
        return dict(snapshot_path=str(snapshot))
    monkeypatch.setattr(planning,'assemble_paused_transfer_snapshot',assemble)
    folder=tmp_path/'pause'
    def reply(seconds):
        events.append(('wait',seconds));request=json.loads((folder/'request.json').read_text());files={}
        for role in ('runtime','phase_audit','context'):
            path=folder/(role+'.json');path.write_text('{}')
            files[role]=dict(path=str(path),sha256=protocol._sha(path))
        response=dict(schema=protocol.RESPONSE_SCHEMA,pause_token=request['pause_token'],episode_id='test',
            snapshot_path=request['snapshot_path'],snapshot_sha256=request['snapshot_sha256'],
            decision='ready',source_context_sha256='a'*64,files=files)
        Path(request['response_path']).write_text(json.dumps(response))
    def admit(path,motors):
        events.append('admit')
        def receipt(pause):
            return dict(passed=True,snapshot_sha256=pause.snapshot['sha256'],source_context_sha256='a'*64,
                input_sha256={**pause.snapshot['files'],pause.snapshot['path']:pause.snapshot['sha256'],
                    **pause.response['files'],str(pause.response_path):pause.response['sha256']})
        return SimpleNamespace(path=path,planning_receipt=receipt)
    monkeypatch.setattr(planning,'admit_paused_isaac_withdrawal_runtime',admit)
    def create(actual,*args,**kwargs):
        assert actual is observer and kwargs['pause'].state=='resumed'
        events.append('construct');return actual
    monkeypatch.setattr(planning,'create_paused_isaac_withdrawal_controller',create)
    def run():return planning.plan_from_live_transfer(directory=folder,episode_id='test',observer=observer,
        motors={},timeout_seconds=60,capture_anchor=anchor,
        snapshot_inputs={'rest_detector':{'terminal_time_s':.502}},sleep=reply)
    if changed:
        with pytest.raises(ValueError,match='changed during'):run()
        assert events==[('ready',.502)] and (folder/'producer-failure.json').exists()
        assert not (folder/'request.json').exists()
        changed_readback=json.loads((folder/'changed-anchor.json').read_text())
        assert changed_readback['measurement_fingerprint']!=json.loads((folder/'terminal.json').read_text())['anchor']['measurement_fingerprint']
        assert json.loads((folder/'anchor-changes.json').read_text())['change_count']>0
    else:
        controller,pause=run()
        assert controller is observer and pause.state=='resumed'
        assert events==[('ready',.502),('wait',.25),'admit','construct']
        assert len(reads)==4


def test_explicit_live_mode_preflight_and_invalid_combinations():
    options=['--live-transfer-planning-pause','--native-door','missing',
        '--standing-transfer-prefix-source','missing','--standing-transfer-hybrid-support',
        '--standing-transfer-stop-on-rest']
    for extra,passed in [(options,True),(options+['--live-planning-timeout-seconds','nan'],False),
            (options+['--standing-withdrawal-route','missing'],False),
            (options+['--pause-readback-probe-at-seconds','2'],False)]:
        result=subprocess.run(command(extra),cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)),
            capture_output=True,text=True)
        assert (result.returncode==0)==passed,result.stderr


def test_actual_producer_pause_boundary_installs_suffix_once(tmp_path,monkeypatch):
    from doorbench.dexterous import isaac_standing_reference_capture as capture
    block=next(n for n in ast.walk(MAIN) if isinstance(n,ast.If)
        and 'live_planning_pause is None' in ast.unparse(n.test))
    calls=[];saved=[];clock=planning.LiveEpisodeClock(.020)
    observer=SimpleNamespace(retained_objects=lambda:{'observer':object()})
    controller=SimpleNamespace(duration=.004,source_context=SimpleNamespace(data={'synthetic':True}))
    pause=SimpleNamespace(receipt=lambda:{'synthetic':True})
    def plan(**kwargs):
        calls.append(kwargs);kwargs['capture_anchor']()
        assert kwargs['snapshot_inputs']['rest_detector']['terminal_time_s']==.006
        return controller,pause
    monkeypatch.setattr(planning,'plan_from_live_transfer',plan)
    monkeypatch.setattr(planning,'transfer_phase_declarations',lambda **kwargs:{'synthetic':True})
    monkeypatch.setattr(planning,'phase_mechanical_audit',lambda *args:{'synthetic':True})
    tail=object()
    monkeypatch.setattr(capture,'StandingContinuationReferenceTail',lambda *args,**kwargs:tail)
    pending=object()
    scope=dict(a=SimpleNamespace(live_transfer_planning_pause=True,standing_transfer_route='source',
        live_planning_timeout_seconds=60),live_planning_pause=None,dt=.002,out=tmp_path,json=json,
        current_physics_checks=lambda:{},acquisition_states={'time_s':[.002,.004,.006]},
        pad_steps=[0]*4,transfer_steps=[0]*3,standing_continuation_steps=[0]*3,
        standing_transfer=SimpleNamespace(started=.002,info={}),
        operation=SimpleNamespace(started=.002,open_started=.004,info={},p_relative=np.zeros(3),r_relative=np.eye(3)),
        dnames=[],transfer_prefix=SimpleNamespace(receipt=lambda:{}),
        transfer_rest_stop=SimpleNamespace(receipt=lambda:{'terminal_time_s':.006}),
        live_handoff_observer=observer,motors={'actuators':[{'name':'motor'}]},
        mechanical_audit={},max_motor_delivery_error=0.,episode_clock=clock,
        pending_withdrawal_steps=pending,save_transfer_rest_stop=lambda:saved.append(True),
        sim=object(),time_origin=0.,pause_probe_counter=object(),robot=object(),door=object(),
        audit_contacts=object(),invariant_getters={},capture_native_pause_anchor=lambda **kwargs:kwargs)
    seen=[]
    for step in clock:
        seen.append(step);scope.update(step=step,transfer_rest_triggered=step>=2)
        execute([block],scope)
    assert seen==[0,1,2,3,4] and len(calls)==1 and saved==[True]
    assert scope['standing_controller'] is controller and scope['withdrawal_steps'] is pending
    assert scope['pending_withdrawal_steps'] is None and scope['standing_reference_tail'] is tail
    assert scope['transfer_rest_continued'] and scope['evaluation_seconds']==.010
    assert (tmp_path/'live-withdrawal-admission.json').exists()


@pytest.mark.parametrize('valid',[False,True])
def test_phase_declarations_preserve_original_tail_and_prefix_checks(valid):
    from doorbench.dexterous.isaac_paused_transfer_audit import CHECKS
    physics=dict.fromkeys(('complete_physics_steps','joint_stops','documented_loopbacks',
        'self_collision','environment_collision','working_hand_collision','plant_parameters_unchanged',
        'closed_leaf_start','resting_operator_start','initial_hand_door_contact_buffer_empty',
        'finite','upright','motor_delivery_matches_command','native_motor_caps'),True)
    times=np.arange(1,501)*.002
    positions=np.tile([.09,0.,0.],(500,1));positions[0]=[.09,.8,.011]
    pads=[dict(sim_time_s=i*.002,valid_pad_grasp=valid or i<400) for i in range(501)]
    rows=[dict(time_s=float(t),leaf_pose=[0,0,0,1,0,0,0],stance_status='solved',
        surface=dict(body_panel_forces_world_N={'lh_palm':[0,-6,0]},palm_normal_load_N=6.)) for t in times]
    checks=planning.transfer_phase_declarations(physics_checks=physics,
        states=dict(time_s=times,door=positions),pad_steps=pads,transfer_steps=rows,
        transfer=SimpleNamespace(started=.1),operation=SimpleNamespace(started=.05),
        door_names=['leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide'],epoch=1.,
        prefix=SimpleNamespace(receipt=lambda:{'passed':valid}),
        rest=SimpleNamespace(receipt=lambda:{'triggered':True,'terminal_time_s':1.}))
    assert set(checks)==CHECKS and len(checks)==26
    assert checks['sustained_pad_grasp'] is valid and checks['live_exact_source_prefix'] is valid
    assert all(checks.values()) is valid
