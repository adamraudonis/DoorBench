import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

SOURCE=Path(__file__).parents[1]/'scripts/isaac/prepare_sensor_demo.py'
spec=importlib.util.spec_from_file_location('prepare_sensor_demo',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def setup_plan(tmp_path,monkeypatch,*,rebound=False,task='reach'):
    from doorbench.dexterous import robot_design_identity as design
    from doorbench.dexterous import motor_contract_identity as identity
    root=tmp_path/'repo';root.mkdir();assets=tmp_path/'ready';assets.mkdir()
    for name in ('robot.xml','robot.usda','door.xml','door.usda'):(assets/name).write_text('original '+name)
    names=['joint'+str(i) for i in range(69)]
    reference={'acquisition':dict(joint_names=names,path_qpos=[[0.]*69])}
    def write(name,value):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    ref=write(m.REFERENCE,reference);refsha=m.file_sha256(ref)
    robot=assets/'robot.xml';motors=assets/'motors.json'
    motors.write_text(json.dumps(dict(joint_names=names,actuators=[{}]*61)))
    calibration=dict(desired_posture=dict.fromkeys(names,0.),provenance=dict(source_reference_sha256=refsha),
        robot_xml_sha256='old' if rebound else m.file_sha256(robot),motor_contract_sha256='same')
    write(m.CALIBRATION,calibration)
    for path in m.SCHEDULES.values():write(path,dict(source_reference_sha256=refsha))
    write('doorbench/dexterous/dummy.py',{'source':'preserved'})
    ready=assets/'ready.json';ready.write_text('{}')
    receipt=dict(native_robot=str(robot),motor_contract=str(motors),robot_usd=str(assets/'robot.usda'),
        door_usd=str(assets/'door.usda'),input_hashes={str(p):m.file_sha256(p) for p in [robot,motors,assets/'robot.usda',assets/'door.usda']})
    def load(path,*,expected_profile):
        assert expected_profile=='shadow-loopback-v2';return receipt
    monkeypatch.setattr(m,'load_ready_receipt',load)
    monkeypatch.setattr(m,'usd_dependencies',lambda paths:dict(file_sha256={str(p):m.file_sha256(p) for p in paths}))
    monkeypatch.setattr(identity,'motor_contract_fingerprint',lambda contract:'same')
    monkeypatch.setattr(design,'robot_design_identity',lambda path:{'sha256':m.file_sha256(path)})
    monkeypatch.setattr(design,'verify_robot_design_identity',lambda path,expected:
        (_ for _ in ()).throw(ValueError('asset changed')) if expected['sha256']!=m.file_sha256(path) else expected)
    plan=m.build_plan(ready,tmp_path/'output',task=task,root=root,native_python='/native/python',isaac_python='/isaac/python')
    return plan


@pytest.mark.parametrize('task,seconds,flag',[('reach',11,'--sensor-reach-protocol'),('grasp',19,'--sensor-acquisition-protocol')])
def test_plan_is_read_only_and_keeps_actor_boundary(tmp_path,monkeypatch,task,seconds,flag):
    plan=setup_plan(tmp_path,monkeypatch,task=task)
    assert not Path(plan['output']).exists()
    assert plan['duration_s']==seconds and flag in plan['isaac_argv']
    assert plan['joint_passive_profile']=='legacy-tanh-v1'
    assert '--acquisition' not in plan['isaac_argv'] and '--native-robot' not in plan['isaac_argv']
    assert '--reset-from-acquisition-path' in plan['isaac_argv']
    assert not plan['isaac_launched'] and not plan['isaac_qualified']
    assert [p['name'] for p in plan['phases']]==['project_joint_only_inputs','fresh_native_component_qualification',
        'independent_actual_state_qualification','complete_causal_packet_replay','fresh_closed_contact_free_reset_preflight']


def test_path_rebinding_cannot_skip_native_qualification(tmp_path,monkeypatch):
    plan=setup_plan(tmp_path,monkeypatch,rebound=True)
    assert plan['calibration_rebound'] and plan['native_qualification_required']
    assert plan['candidate_calibration']['desired_posture']==dict.fromkeys(['joint'+str(i) for i in range(69)],0.)
    assert not plan['native_preparation_passed']


def test_explicit_gyro_profile_is_frozen_in_future_command_only(tmp_path,monkeypatch):
    baseline=setup_plan(tmp_path,monkeypatch)
    assert '--sensor-gyro-profile' not in baseline['isaac_argv']
    plan=m.build_plan(tmp_path/'ready/ready.json',tmp_path/'new-output',root=tmp_path/'repo',
                      gyro_profile='pose-delta-angle-v1')
    assert plan['isaac_sensor_gyro_profile']=='pose-delta-angle-v1'
    assert plan['isaac_argv'][-2:]==['--sensor-gyro-profile','pose-delta-angle-v1']
    assert '--sensor-gyro-profile' not in [arg for phase in plan['phases'] for arg in phase['argv']]
    assert not plan['isaac_qualified']
    with pytest.raises(ValueError,match='gyro profile'):
        m.build_plan(tmp_path/'ready/ready.json',tmp_path/'another-output',root=tmp_path/'repo',gyro_profile='undeclared')


def test_virtualenv_python_symlink_is_preserved_in_every_command(tmp_path,monkeypatch):
    setup_plan(tmp_path,monkeypatch)
    base=tmp_path/'system-python';base.write_text('interpreter')
    native=tmp_path/'native-venv/bin/python';native.parent.mkdir(parents=True);native.symlink_to(base)
    isaac=tmp_path/'isaac-venv/bin/python';isaac.parent.mkdir(parents=True);isaac.symlink_to(base)
    plan=m.build_plan(tmp_path/'ready/ready.json',tmp_path/'new-output',root=tmp_path/'repo',
        native_python=str(native),isaac_python=str(isaac))
    assert all(p['argv'][0]==str(native) for p in plan['phases'])
    assert plan['isaac_argv'][0]==str(isaac)
    assert plan['isaac_audit_argv'][0]==str(native)


def test_exit_zero_with_failed_native_report_stops_and_preserves_failure(tmp_path,monkeypatch):
    plan=setup_plan(tmp_path,monkeypatch,rebound=True);calls=[]
    def run(argv,**kwargs):
        calls.append(argv);assert argv[0]=='/native/python' and kwargs['env']['CUDA_VISIBLE_DEVICES']==''
        if 'probe_sensor_reach_balance.py' in argv[1]:
            out=Path(argv[argv.index('--output')+1]);out.mkdir();(out/'report.json').write_text('{"passed":false}')
    monkeypatch.setattr(m.subprocess,'run',run)
    with pytest.raises(ValueError,match='Qualification failed'):m.execute(plan)
    output=Path(plan['output']);progress=json.loads((output/'progress.json').read_text())
    assert len(calls)==2 and progress['completed_phases']==['project_joint_only_inputs']
    assert not progress['passed'] and not progress['isaac_launched']
    assert not (output/'isaac-launch.json').exists()
    binding=json.loads((output/'calibration-binding.json').read_text())
    assert binding['candidate'] and not binding['historical_qualification_reused']


def test_modified_source_stops_before_any_phase(tmp_path,monkeypatch):
    plan=setup_plan(tmp_path,monkeypatch);Path(plan['source_files'][0]).write_text('changed')
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:pytest.fail('Cannot run after changed frozen source'))
    with pytest.raises(ValueError,match='Frozen preparation'):m.execute(plan)
    assert not json.loads((Path(plan['output'])/'progress.json').read_text())['passed']


@pytest.mark.parametrize('value',[False,1,'true',None])
def test_phase_requires_literal_qualified_boolean(tmp_path,value):
    path=tmp_path/'report.json';path.write_text(json.dumps({'independent_evaluation':{'passed':value}}))
    with pytest.raises(ValueError):m.require_phase_report(dict(name='independent',report=str(path),gate='independent_evaluation.passed'))


def test_completed_native_phases_only_emit_a_future_isaac_command(tmp_path,monkeypatch):
    plan=setup_plan(tmp_path,monkeypatch);calls=[]
    def run(argv,**kwargs):
        calls.append(argv);output=Path(plan['output'])
        assert argv[0]=='/native/python' and 'isaac_opening.py' not in argv[1]
        phase=plan['phases'][len(calls)-1]
        if phase['name']=='project_joint_only_inputs':
            target=output/'inputs/runtime';target.mkdir()
            for name in ('protocol.json','joint-route.json'):(target/name).write_text('{}')
        if phase['name']=='fresh_native_component_qualification':
            (output/'native').mkdir();(output/'native/sensor-layout.json').write_text('{}')
        if 'report' in phase:
            seconds=phase.get('expected_duration_s',11)
            if phase['gate']=='independent_evaluation.passed':
                val={'independent_evaluation':dict(passed=True,checks={'complete':True},duration_s=seconds,physics_steps=round(seconds/.002))}
            elif phase['gate']=='replay_passed':val=dict(replay_passed=True,replayed_decisions=round(seconds/.002),failure=None,physics_steps=0)
            else:val=dict(passed=True,checks={'complete':True},duration_s=seconds,physics_steps=round(seconds/.002))
            Path(phase['report']).write_text(json.dumps(val))
    monkeypatch.setattr(m.subprocess,'run',run);launch=m.execute(plan)
    assert len(calls)==5 and launch['native_preparation_passed']
    assert not launch['isaac_launched'] and not launch['isaac_qualified']
    assert len(launch['generated_input_sha256'])==9
    assert (Path(plan['output'])/'isaac-command.txt').exists()
    assert '--verify-launch' in (Path(plan['output'])/'isaac-command.txt').read_text()
    m.verify_launch(Path(plan['output'])/'isaac-launch.json')
    Path(plan['candidate_calibration_path']).write_text('changed after qualification')
    with pytest.raises(ValueError,match='Generated qualification'):m.verify_launch(Path(plan['output'])/'isaac-launch.json')


def test_claimed_success_cannot_hide_short_native_evidence(tmp_path):
    path=tmp_path/'report.json';path.write_text(json.dumps(dict(passed=True,checks={'complete':True},duration_s=10.,physics_steps=5000)))
    with pytest.raises(ValueError,match='Incomplete native'):
        m.require_phase_report(dict(name='native',report=str(path),gate='passed',expected_duration_s=11))


def test_generated_calibration_edit_between_phases_is_rejected(tmp_path,monkeypatch):
    plan=setup_plan(tmp_path,monkeypatch);calls=[]
    def run(argv,**kwargs):
        calls.append(argv);Path(plan['candidate_calibration_path']).write_text('concurrent edit')
    monkeypatch.setattr(m.subprocess,'run',run)
    with pytest.raises(ValueError,match='Generated qualification'):m.execute(plan)
    assert len(calls)==1 and not (Path(plan['output'])/'isaac-launch.json').exists()


def test_usd_dependency_closure_binds_referenced_layer_bytes(tmp_path):
    child=tmp_path/'child.usda';child.write_text('#usda 1.0\ndef Xform "Child" {}\n')
    root=tmp_path/'root.usda';root.write_text('#usda 1.0\ndef Xform "Root" (prepend references = @child.usda@</Child>) {}\n')
    closure=m.usd_dependencies([root]);assert closure['file_sha256']=={str(p):m.file_sha256(p) for p in [root,child]}
    child.write_text('#usda 1.0\ndef Xform "Child" { double changed = 1 }\n')
    with pytest.raises(ValueError,match='Generated qualification'):m.verify_generated(closure['file_sha256'])


def test_unresolved_hardware_payload_is_not_hidden_by_root_hash(tmp_path):
    root=tmp_path/'root.usda';root.write_text('#usda 1.0\ndef Xform "Root" (prepend references = @missing.usdc@</Child>) {}\n')
    with pytest.raises(ValueError,match='Unresolved USD dependency'):m.usd_dependencies([root])
