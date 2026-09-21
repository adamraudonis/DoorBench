"""Real handshake/observer wiring with mocked physical-source admission only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous import isaac_withdrawal_runtime as module
from doorbench.dexterous import isaac_live_planning_pause as protocol
from doorbench.dexterous.isaac_live_transfer_handoff import LiveTransferHandoffObserver
from doorbench.dexterous.isaac_release_source_dispatch import PAUSED_SOURCE_KIND
from doorbench.dexterous.qualified_isaac_grasp import digest
from test_isaac_withdrawal_runtime import runtime, write
from test_isaac_withdrawal_support import predecessor


@pytest.fixture
def live(runtime,tmp_path):
    transfer,_=predecessor()
    transfer.path=runtime.route;transfer.start_seconds=28.;transfer.started=28.
    transfer.acquisition=SimpleNamespace(caps=np.array([[-2.,2.]]))
    transfer.operation=SimpleNamespace(hub_avoidance=None)
    command=np.array([1.25]);calls=[]
    def force(t):calls.append(t);return command,{}
    transfer.force=force
    observer=LiveTransferHandoffObserver(transfer)
    for index in range(21749,22000):
        t=index*.002;returned,_=observer.force(t)
        observer.observe_completed_interval(command_time_s=t,post_step_time_s=(index+1)*.002,
            returned_command=returned,grasp_qualified=True,left_palm_load=6.,
            angles=dict(leaf=.091,operator=0.,latch=0.),submission_valid=True)
    objects=observer.retained_objects()
    pause=protocol.LivePlanningPause(tmp_path/'pause',episode_id='synthetic-local',
        retained_objects=objects,timeout_seconds=60)
    def anchor():return protocol.capture_pause_anchor(episode_id=pause.episode_id,
        step_index=22000,epoch_s=44.,physics_clock={'synthetic_step':22000},
        measured={'synthetic_only':True},controller=observer.snapshot(),retained_objects=objects,
        evidence_counts={'synthetic_observations':251},pending_unaccepted_command=False)
    a=anchor();live={key:a[key] for key in ('episode_id','controller_identity','step_index','epoch_s',
        'physics_dt_s','measurement_fingerprint','controller_fingerprint')}
    live['pause_token']=pause.pause_token
    evidence=pause.directory/'evidence.json';write(evidence,{'synthetic_only':True})
    snapshot=pause.directory/'snapshot.json'
    write(snapshot,dict(schema=protocol.SNAPSHOT_SCHEMA,source_kind=PAUSED_SOURCE_KIND,
        source_engine='isaac-physx',closed_prefix=True,episode_complete=False,live_pause=live,
        files={'synthetic_evidence':dict(path=str(evidence),sha256=digest(evidence))}))
    runtime.path=pause.directory/'runtime.json';runtime.config['schema']=module.PAUSED_SCHEMA;runtime.bind()
    runtime.data.update(source_kind=PAUSED_SOURCE_KIND,live_pause_authorization_required=True,
        source_snapshot_path=str(snapshot),source_snapshot_sha256=digest(snapshot),
        motor_contract_path=str(runtime.trial/'motor-contract.json'),
        configuration_path=str(runtime.trial/'configuration.json'),source_context_sha256='c'*64)
    runtime.source.pop('contact_audit_name')
    runtime.source.update({key:runtime.data[key] for key in ('source_kind','source_snapshot_path',
        'source_snapshot_sha256','motor_contract_path','configuration_path')})
    write(runtime.source_path,runtime.source);runtime.bind()
    write(runtime.path,runtime.config)
    runtime.data['source_admission'].update(snapshot_path=str(snapshot),snapshot_sha256=digest(snapshot),
        live_pause=live,observer_state=observer.snapshot())
    admission=module.admit_paused_isaac_withdrawal_runtime(runtime.path,runtime.motors)
    pause.publish(snapshot,anchor())
    context=pause.directory/'context.json';write(context,admission.source_admission)
    phase=pause.directory/'phase.json';write(phase,admission.source_admission['source_qualification'])
    response=dict(schema=protocol.RESPONSE_SCHEMA,pause_token=pause.pause_token,episode_id=pause.episode_id,
        snapshot_path=str(snapshot),snapshot_sha256=digest(snapshot),decision='ready',
        source_context_sha256='c'*64,files={role:dict(path=str(path),sha256=digest(path))
            for role,path in dict(runtime=runtime.path,context=context,phase_audit=phase).items()})
    write(pause.response_path,response);pause.poll()
    return SimpleNamespace(runtime=runtime,observer=observer,pause=pause,anchor=anchor,
        admission=admission,calls=calls,command=command,objects=objects)


def resume(f):
    return f.pause.validate_resume(capture_anchor=f.anchor,
        validate_plan=lambda response:f.admission.planning_receipt(f.pause))


def test_real_protocol_preserves_actual_observers_and_one_time_entry(live):
    f=live;a=f.admission;observer=f.observer
    before=copy.deepcopy(observer.snapshot());previous=observer.transfer.support_feedback.previous
    assert resume(f)['resume_handshake_passed']
    a.bind_observer(observer);a.authorize_live_pause(f.pause)
    assert a.authorized and not a.entered and len(f.calls)==251
    assert a.motor_capture_for(observer) is observer.motor_capture
    assert observer.snapshot()==before
    a.require_entry(44.)
    assert a.entered and a.inherited_support.feedback is observer.transfer.support_feedback
    assert observer.transfer.support_feedback.previous is previous
    assert observer.snapshot()==before and len(f.calls)==251
    with pytest.raises(ValueError):a.require_entry(44.)


@pytest.mark.parametrize('change',['unresumed','token','retained_object','history','runtime_hash'])
def test_changed_or_unaccepted_live_source_cannot_authorize(live,change):
    f=live;a=f.admission
    if change!='unresumed':resume(f)
    a.bind_observer(f.observer)
    if change=='token':f.pause.pause_token='d'*64
    if change=='retained_object':f.pause._objects['motor_capture']=object()
    if change=='history':f.observer.motor_capture.previous_command[0]+=.01
    if change=='runtime_hash':f.runtime.path.write_text('{}')
    with pytest.raises(ValueError):a.authorize_live_pause(f.pause)
    assert not a.authorized and not a.entered and len(f.calls)==251
    with pytest.raises(ValueError):a.authorize_live_pause(f.pause)


def test_late_factory_passes_same_observer_to_controller(live,monkeypatch):
    from doorbench.dexterous import standing_withdrawal
    f=live;resume(f)
    def construct(returned,motors,path,*,_isaac_runtime):
        assert returned is f.observer
        assert _isaac_runtime.motor_capture_for(returned) is f.observer.motor_capture
        assert _isaac_runtime.authorized
        return returned
    monkeypatch.setattr(standing_withdrawal,'StandingWithdrawalTeacher',construct)
    assert module.create_paused_isaac_withdrawal_controller(f.observer,f.runtime.motors,
        f.runtime.path,admission=f.admission,pause=f.pause) is f.observer


def test_archive_receipt_cannot_authorize_paused_runtime(live):
    with pytest.raises(ValueError,match='not an archive prefix'):
        live.admission.authorize_source_prefix({'passed':True})


def test_changed_observer_rejected_before_binding(live):
    live.observer.motor_capture.previous_command[0]+=.01
    with pytest.raises(ValueError,match='history differs'):live.admission.bind_observer(live.observer)
    assert live.admission.inherited_support is None


def test_completed_pause_cannot_authorize_two_runtime_instances(live):
    f=live;other=module.admit_paused_isaac_withdrawal_runtime(f.runtime.path,f.runtime.motors)
    resume(f)
    f.admission.bind_observer(f.observer);f.admission.authorize_live_pause(f.pause)
    other.bind_observer(f.observer)
    with pytest.raises(ValueError,match='exactly one runtime'):other.authorize_live_pause(f.pause)
    assert f.admission.authorized and not other.authorized


def test_legacy_constructor_rejects_distinct_runtime_schema(live):
    with pytest.raises(ValueError,match='runtime schema'):
        module.admit_isaac_withdrawal_runtime(live.runtime.path,live.runtime.motors)


def test_actual_late_teacher_constructor_reuses_capture_without_extra_transfer_calls(live,monkeypatch):
    from doorbench.dexterous import landed_left_planner
    from test_isaac_coupled_release_geometry import toy_scene
    f=live;scene=toy_scene();m,d=scene.m,scene.d;initial=d.qpos.copy()
    rh=m.site('robot/rh_palm_touch').id
    names=[m.joint(j).name[6:] for j in range(m.njnt)
        if m.joint(j).name.startswith('robot/') and m.jnt_type[j]==3]
    values={name:float(initial[m.joint('robot/'+name).qposadr[0]]) for name in names}
    row=dict(qpos=initial.tolist(),joints=values,finger_joints={},phase='measured_release',
        palm_position=d.site_xpos[rh].tolist(),palm_rotation=d.site_xmat[rh].reshape(3,3).tolist())
    f.runtime.data.update(initial_qpos=initial.tolist(),robot_path='synthetic-robot',
        door_xml_path='synthetic-door',screen={'trials':[{'rows':[
            dict(time_s=0.,**row),dict(time_s=8.,**row)]}]},audit={'duration_s':16.})
    f.admission=module.admit_paused_isaac_withdrawal_runtime(f.runtime.path,f.runtime.motors)
    f.admission.reference.geometry=SimpleNamespace(initial=initial,plan={'duration_s':16.})
    monkeypatch.setattr(landed_left_planner,'LandedLeftScene',lambda *args:scene)
    resume(f);before=f.observer.snapshot();capture=f.observer.motor_capture
    teacher=module.create_paused_isaac_withdrawal_controller(f.observer,f.runtime.motors,
        f.runtime.path,admission=f.admission,pause=f.pause)
    assert teacher.returned is f.observer and teacher.motor_capture is capture
    assert teacher.source_context is f.admission.source_context
    assert teacher.inherited_support.transfer is f.observer.transfer
    assert teacher.motor_capture.capture(44.).tolist()==[1.25]
    assert len(f.calls)==251 and f.observer.snapshot()==before
    assert teacher.started_withdrawal is None and f.admission.entered is False
