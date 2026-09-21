"""CPU-only launcher contracts with synthetic receipt files and fake processes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def module(monkeypatch, root):
    source=Path(__file__).parents[1]/'scripts/isaac/run_local_operation.py'
    spec=importlib.util.spec_from_file_location('local_operation_launch',source)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    monkeypatch.setattr(value,'ROOT',root)
    return value


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def native_fixture(tmp_path,monkeypatch):
    from test_isaac_readiness_profiles import contract
    m=module(monkeypatch,tmp_path)
    robot=tmp_path/'robot.xml';robot.write_text('synthetic robot')
    robot_audit=robot.with_suffix('.audit.json');write(robot_audit,{})
    audit,motors=contract();motors['source_xml_sha256']=m.sha(robot)
    motor=tmp_path/'motors.json';write(motor,motors)
    usd=tmp_path/'robot.usda';usd.write_text('synthetic robot USD')
    door=tmp_path/'door';door.mkdir()
    (door/'door.usda').write_text('synthetic door USD')
    (door/'door.xml').write_text('<mujoco><worldbody><body name="leaf_handle"><geom name="leaf_handle_hub_col_n" type="cylinder" size=".012 .022" pos="0 -.052 0" quat=".707107 -.707107 0 0"/></body></worldbody></mujoco>')
    ready=dict(ready=True,mechanics_profile='shadow-loopback-v2',native_robot=str(robot),native_robot_sha256=m.sha(robot),
        motor_contract=str(motor),robot_usd=str(usd),door_usd=str(door/'door.usda'),
        passive_tendons=dict(count=8,backend_verified=True,audit=audit),
        input_hashes={str(p.resolve()):m.sha(p) for p in (robot,robot_audit,motor,usd,door/'door.usda')})
    ready_path=tmp_path/'ready.json';write(ready_path,ready)
    native=tmp_path/'native trial';native.mkdir()
    reference=tmp_path/'screen/reference.json';write(reference,dict(synthetic=True))
    screen=reference.with_name('geometry-audit.json')
    write(screen,dict(passed=True,input_sha256={str(p):m.sha(p) for p in (robot,door/'door.xml',reference)}))
    (native/'reference.json').write_bytes(reference.read_bytes())
    cfg=dict(robot=str(robot),door=str(door),motors=str(motor),reference=str(reference),
        portable_wrapper=True,record_transitions=True,stance_profile='landed-foot-v1',press_seconds=5.,
        grasp_profile='distal-pad-v1',grasp_offset_in_handle_m=[.004,-.003,.0025],
        pressure_segment='distal',index_finger_force=3.,middle_finger_force=None,
        operator_compliance_gain=.2,min_acquisition_seconds=10.6,
        index_proximal_offset_rad=-.02,index_tendon_offset_rad=.06,
        operation_leaf_target_rad=.08,operation_opening_trigger_rad=.8,
        operation_operator_lead_limit_rad=.1,operation_fixed_pad_control=True,
        operation_pad_control_profile='actual-tangent-v1',operation_handle_hub_avoidance=True,
        operation_hub_clearance_m=.004,open_on_latch_clear=False)
    manifest=native/'manifest.json'
    write(manifest,dict(configuration=cfg,inputs=dict(robot=dict(sha256=m.sha(robot)),door={'door.xml':m.sha(door/'door.xml')})))
    write(native/'report.json',dict(passed=True,checks=dict(physical=True),grasp_profile='distal-pad-v1'))
    (native/'physics-steps.json.gz').write_bytes(b'synthetic physics bytes')
    raw=native/'raw-transitions/manifest.json';chunk=raw.with_name('chunk.npz');write(chunk,dict(synthetic=True))
    write(raw,dict(complete=True,chunks=[dict(file=chunk.name,sha256=m.sha(chunk))]))
    write(native/'independent-contact-audit.json',dict(passed=True,checks=dict(physical=True),grasp_profile='distal-pad-v1',
        input_sha256={str(p.relative_to(native)):m.sha(p) for p in (manifest,native/'report.json',native/'physics-steps.json.gz',raw)}))
    write(native/'independent-whole-handle-audit.json',dict(schema='doorbench.native-whole-handle-audit.v1',passed=True,
        extra_loaded_patches=0,affected_physics_intervals=0,maximum_extra_force_N=0.,
        input_sha256={str(p):m.sha(p) for p in (manifest,raw,robot,door/'door.xml')}))
    docs={};snapshots=native/'controller-inputs';snapshots.mkdir()
    for path in (reference,screen,motor):
        digest=m.sha(path);snapshot=digest[:16]+'-'+path.name
        (snapshots/snapshot).write_bytes(path.read_bytes())
        docs[str(path)]=dict(sha256=digest,snapshot=snapshot)
    write(snapshots/'manifest.json',dict(schema='doorbench.controller-input-documents.v1',documents=docs))
    write(native/'hub-geometry.json',dict(size=[.012,.022,0.],position=[0.,-.052,0.],quaternion_wxyz=[2**-.5,-2**-.5,0.,0.]))
    args=SimpleNamespace(ready=ready_path,native_trial=native,output=tmp_path/'fresh output',seconds=24.,
        isaac_python=tmp_path/'short alias/python.exe',asset_python=tmp_path/'asset python.exe',
        open_on_latch_clear=False,review_render_profile=None,jev_progress_plan=None,jev_sample_period=.2,timeout_seconds=3600.)
    return m,args,ready,cfg


def test_verified_native_and_ready_bind_all_selected_assets_and_derived_hub(tmp_path,monkeypatch):
    m,args,ready,cfg=native_fixture(tmp_path,monkeypatch)
    selected,configuration,reference,profile,declaration,hashes=m.verified_inputs(args)
    assert selected==ready and configuration==cfg and declaration is None and profile=='distal-pad-v1'
    for path in (Path(ready['motor_contract']),args.native_trial/'hub-geometry.json',args.native_trial/'raw-transitions/chunk.npz'):
        assert hashes[str(path.resolve())]==m.sha(path)


@pytest.mark.parametrize('mutation',['audit_from_other_run','changed_raw_chunk','motor_snapshot','hub','profile','missing_ready_hash','missing_loopbacks'])
def test_stale_or_mismatched_prerequisites_fail_before_launch(tmp_path,monkeypatch,mutation):
    m,args,ready,cfg=native_fixture(tmp_path,monkeypatch)
    if mutation=='audit_from_other_run':
        write(args.native_trial/'report.json',dict(passed=True,checks=dict(physical=True),changed=True))
    elif mutation=='changed_raw_chunk':
        (args.native_trial/'raw-transitions/chunk.npz').write_bytes(b'changed contact evidence')
    elif mutation=='motor_snapshot':
        for path in (args.native_trial/'controller-inputs').glob('*-motors.json'):path.write_bytes(b'changed native motor document')
    elif mutation=='hub':
        hub=args.native_trial/'hub-geometry.json';value=json.loads(hub.read_text());value['position'][1]=-.051;write(hub,value)
    elif mutation=='profile':
        path=args.native_trial/'independent-contact-audit.json';value=json.loads(path.read_text());value['grasp_profile']='volar-phalange-v1';write(path,value)
    elif mutation=='missing_ready_hash':
        ready['input_hashes'].pop(ready['robot_usd']);write(args.ready,ready)
    elif mutation=='missing_loopbacks':
        ready['passive_tendons']['backend_verified']=False;write(args.ready,ready)
    with pytest.raises(ValueError):m.verified_inputs(args)


def test_ready_motor_contract_must_match_the_frozen_native_controller(tmp_path,monkeypatch):
    m,args,ready,cfg=native_fixture(tmp_path,monkeypatch)
    other=tmp_path/'alternative-motors.json';value=json.loads(Path(ready['motor_contract']).read_text());value['unrelated_change']=1;write(other,value)
    ready['motor_contract']=str(other);ready['input_hashes'][str(other.resolve())]=m.sha(other);write(args.ready,ready)
    with pytest.raises(ValueError,match='motor contract differs'):m.verified_inputs(args)


def test_exact_argument_mapping_preserves_spaces_offsets_profiles_and_native_latch_flag(tmp_path,monkeypatch):
    m,args,ready,cfg=native_fixture(tmp_path,monkeypatch)
    cfg['open_on_latch_clear']=True
    args.review_render_profile='native-materials-v1';args.jev_progress_plan=tmp_path/'Astra plan.json'
    command,audit=m.commands(args,ready,cfg,Path(cfg['reference']),'volar-phalange-v1',tmp_path/'grasp declaration.json')
    expected={'--seconds':'24.0','--operator-compliance-gain':'0.2','--acquisition-pressure-segment':'distal',
        '--operation-index-proximal-offset-rad':'-0.02','--operation-index-tendon-offset-rad':'0.06',
        '--operation-operator-lead-limit-rad':'0.1','--operation-material-profile':'actual-tangent-v1',
        '--operation-hub-clearance-m':'0.004','--grasp-profile':'volar-phalange-v1',
        '--jev-progress-plan':str(args.jev_progress_plan),'--reference':cfg['reference'],
        '--review-render-profile':'native-materials-v1'}
    for flag,value in expected.items():assert command[command.index(flag)+1]==value
    offset=command.index('--operation-grasp-offset-in-handle-m')
    assert command[offset+1:offset+4]==['0.004','-0.003','0.0025']
    assert '--open-on-latch-clear' in command and '--operation-actual-pad-control' in command
    assert '--acquisition-middle-finger-force' not in command
    assert command[0]==str(args.isaac_python) and audit[0]==str(args.asset_python)
    assert audit[-2:]==['--grasp-profile','volar-phalange-v1']


def execution_fixture(tmp_path,monkeypatch,*,preflight=0,process_code=0,audit_code=0,timeout=False,audit_error=False,task_passed=True):
    m=module(monkeypatch,tmp_path)
    for name in ('isaac_opening.py','audit_isaac_acquisition_contacts.py'):
        path=tmp_path/'scripts/dexterous'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic executable')
    args=SimpleNamespace(output=tmp_path/'experiment',jev_progress_plan=None,timeout_seconds=3600.,seconds=24.)
    proof=tmp_path/'proof.json';write(proof,dict(synthetic=True));args.hashes={str(proof):m.sha(proof)}
    args.argv=['isaac'];args.input_files={}
    for flag in ('robot-usd','door-usd','motors','native-robot','reference'):
        path=tmp_path/(flag+'.json');write(path,dict(synthetic=flag))
        args.hashes[str(path.resolve())]=m.sha(path);args.input_files[flag.replace('-','_')]=str(path)
        args.argv+=['--'+flag,str(path)]
    args.argv+=['--grasp-profile','distal-pad-v1']
    calls=[]
    def run(command,**kw):
        calls.append(('run',command,kw))
        if '--validate-arguments-only' in command:
            if isinstance(preflight,Exception):raise preflight
            return SimpleNamespace(returncode=preflight)
        if audit_error:raise subprocess.TimeoutExpired(command,600)
        trial=args.output/'trial'
        data=dict(schema='doorbench.isaac-acquisition-contact-audit.v1',accounting_passed=True,
            independent_raw_contact_audit_complete=True,invalid_loaded_patches=0,checks=dict(raw=True),
            input_sha256={name:m.sha(trial/name) for name in ('operation-report.json','configuration.json','provenance.json','acquisition-pad-steps.json.gz')})
        write(args.output/'independent-contact-audit.json',data)
        return SimpleNamespace(returncode=audit_code)
    class Process:
        pid=123
        def __init__(self,command,**kw):
            calls.append(('popen',command,kw));self.waits=0
            trial=args.output/'trial'
            write(trial/'operation-report.json',dict(passed=task_passed,duration_s=24.,checks=dict(physical=task_passed)))
            write(trial/'configuration.json',dict(args=dict(args.input_files,grasp_profile='distal-pad-v1',open_on_latch_clear=False)))
            operation=tmp_path/'scripts/dexterous/isaac_opening.py'
            write(trial/'provenance.json',dict(files=dict(args.hashes,**{str(operation.resolve()):m.sha(operation)})))
            (trial/'acquisition-pad-steps.json.gz').write_bytes(b'synthetic interval bytes')
        def wait(self,timeout=None):
            self.waits+=1
            if timeout==3600. and timeout_flag:raise subprocess.TimeoutExpired('synthetic command',timeout)
            return process_code
    timeout_flag=timeout
    monkeypatch.setattr(m.subprocess,'run',run);monkeypatch.setattr(m.subprocess,'Popen',Process)
    return m,args,calls


def test_preview_never_launches_and_preserves_short_interpreter_alias(tmp_path,monkeypatch,capsys):
    m,args,ready,cfg=native_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(sys,'argv',['run_local_operation.py',*[item for name in ('ready','native_trial','output','asset_python','isaac_python') for item in ('--'+name.replace('_','-'),str(getattr(args,name)))]])
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:pytest.fail('Preview must not launch'))
    assert m.main()==0 and not args.output.exists()
    result=json.loads(capsys.readouterr().out)
    assert result['physics_started'] is False and result['operation'][0]==str(args.isaac_python)


@pytest.mark.parametrize('failure',[2,subprocess.TimeoutExpired('preflight',90)])
def test_preflight_failure_always_writes_failed_result_without_physics(tmp_path,monkeypatch,failure):
    m,args,calls=execution_fixture(tmp_path,monkeypatch,preflight=failure)
    assert m.execute(args,args.argv,['audit'],args.hashes)==1
    result=json.loads((args.output/'result.json').read_text())
    assert result['passed'] is False and result['physics_started'] is False
    assert not any(call[0]=='popen' for call in calls)
    assert result['launch_error_type'] in ('RuntimeError','TimeoutExpired')


@pytest.mark.parametrize('options',[dict(process_code=1),dict(audit_code=1),dict(timeout=True),dict(audit_error=True),dict(task_passed=False)])
def test_failed_runtime_audit_or_timeout_is_never_upgraded(tmp_path,monkeypatch,options):
    m,args,calls=execution_fixture(tmp_path,monkeypatch,**options)
    assert m.execute(args,args.argv,['audit'],args.hashes)==1
    result=json.loads((args.output/'result.json').read_text())
    assert result['passed'] is False and result['report_emitted']
    assert any(call[1]==['audit'] for call in calls)
    if options.get('timeout'):assert (args.output/'trial/stop.request').exists()


def test_clean_result_requires_both_passes_and_keeps_credentials_out_of_offline_subprocesses(tmp_path,monkeypatch):
    m,args,calls=execution_fixture(tmp_path,monkeypatch)
    monkeypatch.setenv('TYPESAFE_API_KEY','synthetic-key-not-for-use')
    # Opt-in live child receives the environment, argument preflight/scoring do not.
    args.jev_progress_plan=tmp_path/'plan.json'
    assert m.execute(args,args.argv,['audit'],args.hashes)==0
    result=json.loads((args.output/'result.json').read_text())
    assert result['passed'] and result['runtime_passed'] and result['independent_passed']
    for kind,command,kw in calls:
        assert ('TYPESAFE_API_KEY' in kw['env']) == (kind=='popen')
    assert 'synthetic-key-not-for-use' not in (args.output/'result.json').read_text()


def test_disabled_jev_removes_key_from_all_children(tmp_path,monkeypatch):
    m,args,calls=execution_fixture(tmp_path,monkeypatch)
    monkeypatch.setenv('TYPESAFE_API_KEY','synthetic-key-not-for-use')
    assert m.execute(args,args.argv,['audit'],args.hashes)==0
    assert all('TYPESAFE_API_KEY' not in call[2]['env'] for call in calls)


def test_preflight_cannot_race_changed_verified_input_into_physics(tmp_path,monkeypatch):
    m,args,calls=execution_fixture(tmp_path,monkeypatch)
    original=m.subprocess.run
    def mutate(command,**kw):
        result=original(command,**kw)
        if '--validate-arguments-only' in command:Path(next(iter(args.hashes))).write_text('changed during preflight')
        return result
    monkeypatch.setattr(m.subprocess,'run',mutate)
    assert m.execute(args,args.argv,['audit'],args.hashes)==1
    assert not any(call[0]=='popen' for call in calls)


@pytest.mark.parametrize('mutation',['source','profile','duration'])
def test_runtime_report_must_bind_actual_source_mode_and_requested_duration(tmp_path,monkeypatch,mutation):
    m,args,calls=execution_fixture(tmp_path,monkeypatch)
    original=m.subprocess.Popen
    def altered(command,**kw):
        process=original(command,**kw);trial=args.output/'trial'
        if mutation=='source':
            path=trial/'provenance.json';value=json.loads(path.read_text())
            value['files'][str((tmp_path/'scripts/dexterous/isaac_opening.py').resolve())]='0'*64
        elif mutation=='profile':
            path=trial/'configuration.json';value=json.loads(path.read_text());value['args']['grasp_profile']='volar-phalange-v1'
        else:
            path=trial/'operation-report.json';value=json.loads(path.read_text());value['duration_s']=12.
        write(path,value);return process
    monkeypatch.setattr(m.subprocess,'Popen',altered)
    assert m.execute(args,args.argv,['audit'],args.hashes)==1
    result=json.loads((args.output/'result.json').read_text())
    assert not result['runtime_passed'] and result['independent_passed']


def test_audit_receipt_from_another_trial_cannot_satisfy_success(tmp_path,monkeypatch):
    m,args,calls=execution_fixture(tmp_path,monkeypatch)
    original=m.subprocess.run
    def altered(command,**kw):
        result=original(command,**kw)
        if command==['audit']:
            path=args.output/'independent-contact-audit.json';value=json.loads(path.read_text())
            value['input_sha256']['operation-report.json']='0'*64;write(path,value)
        return result
    monkeypatch.setattr(m.subprocess,'run',altered)
    assert m.execute(args,args.argv,['audit'],args.hashes)==1
    result=json.loads((args.output/'result.json').read_text())
    assert result['runtime_passed'] and not result['independent_passed']
    assert result['audit_error_type']=='ValueError'


def test_controller_interruption_stops_child_and_finalizes_failure(tmp_path,monkeypatch):
    m,args,calls=execution_fixture(tmp_path,monkeypatch)
    original=m.subprocess.Popen
    def interrupted(command,**kw):
        process=original(command,**kw)
        wait=process.wait
        def stop(timeout=None):
            if timeout==args.timeout_seconds:raise KeyboardInterrupt()
            return wait(timeout=timeout)
        process.wait=stop;return process
    monkeypatch.setattr(m.subprocess,'Popen',interrupted)
    assert m.execute(args,args.argv,['audit'],args.hashes)==1
    result=json.loads((args.output/'result.json').read_text())
    assert result['interrupted'] and result['launch_error_type']=='KeyboardInterrupt'
    assert (args.output/'trial/stop.request').exists()
