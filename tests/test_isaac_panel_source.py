"""Synthetic evidence/admission seams, not a physical release qualification."""
import copy
import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

import doorbench.dexterous.isaac_panel_source as panel
VALIDATE_COMPLETION = panel.validate_local_completion


POSE = [0., 0., 0., 1., 0., 0., 0.]


def window():
    return [dict(sim_time_s=float(t), leaf_pose=POSE.copy(), door_q=.2,
        left_surface=dict(palm_normal_load_N=3., body_panel_forces_world_N={'lh_palm':[0.,-3.,0.]}),
        pad_grasp=dict(valid_pad_grasp=False, contacts=[]),
        right_environment_clearance_m=.05,
        geometry=dict(time_s=float(t),geometry_time_s=float(t),native_mirror_steps=0,
            right_environment_clearance_m=.05,
            body_poses_xyz_wxyz={'rh_ffdistal':POSE.copy(),'leaf':POSE.copy()}),
        teacher=dict(stance_status='solved',phase='standing_withdrawal'))
        for t in .5+np.arange(251)*.002]


def test_released_window_uses_actual_palm_and_clearance_without_final_grasp():
    rows=window();before=copy.deepcopy(rows)
    result=panel.validate_released_window(rows,terminal=1.)
    assert result['window_samples']==251
    assert result['minimum_palm_load_N']==3.
    assert result['minimum_right_environment_clearance_m']==.05
    assert result['final_grasp_required'] is False
    assert rows==before


def test_original_stance_solved_inaccurate_is_preserved():
    rows=window();rows[125]['teacher']['stance_status']='solved inaccurate'
    assert panel.validate_released_window(rows,terminal=1.)['window_samples']==251


@pytest.mark.parametrize('kind',['count','gap','clock_nan','palm','palm_lie','palm_finger_substitute',
    'pose','force_nan','clearance','clearance_nan','clearance_mismatch','stale_geometry',
    'stepped_geometry','closed_leaf','stance','invalid_contact'])
def test_original_released_window_gates_are_not_relaxed(kind):
    rows=window();row=rows[125]
    if kind=='count':rows.pop()
    if kind=='gap':row['sim_time_s']+=.002
    if kind=='clock_nan':row['sim_time_s']=float('nan')
    if kind=='palm':row['left_surface']=dict(palm_normal_load_N=1.99,body_panel_forces_world_N={'lh_palm':[0.,-1.99,0.]})
    if kind=='palm_lie':row['left_surface']['palm_normal_load_N']=4.
    if kind=='palm_finger_substitute':row['left_surface']['body_panel_forces_world_N']={'lh_ffdistal':[0.,-3.,0.]}
    if kind=='pose':row['leaf_pose'][3]=.5
    if kind=='force_nan':row['left_surface']['body_panel_forces_world_N']['lh_palm'][0]=float('nan')
    if kind=='clearance':row['right_environment_clearance_m']=row['geometry']['right_environment_clearance_m']=.039999
    if kind=='clearance_nan':row['right_environment_clearance_m']=float('nan')
    if kind=='clearance_mismatch':row['geometry']['right_environment_clearance_m']=.06
    if kind=='stale_geometry':row['geometry']['geometry_time_s']-=.002
    if kind=='stepped_geometry':row['geometry']['native_mirror_steps']=1
    if kind=='closed_leaf':row['door_q']=.074999
    if kind=='stance':row['teacher']['stance_status']='failed'
    if kind=='invalid_contact':row['pad_grasp']['contacts']=[dict(pad_qualified=False,normal_force_N=0.)]
    with pytest.raises((ValueError,KeyError)):panel.validate_released_window(rows,terminal=1.)


def motor_fixture():
    motors=dict(actuators=[dict(name='motor'+str(i),force_range=[-2.,2.]) for i in range(61)])
    values=np.zeros((500,61));values[-1]=np.linspace(-1.,1.,61)
    return dict(time_s=(np.arange(500)+1)*.002,motor_forces=values),motors


def test_actual_predecessor_command_preserves_order_epoch_and_values():
    physics,motors=motor_fixture();result=panel.actual_motor_handoff(physics,motors,terminal=1.)
    assert result['command_time_s']==.998 and result['interval_end_s']==1.
    assert result['command_Nm']==physics['motor_forces'][-1].tolist()
    assert result['motor_names']==[m['name'] for m in motors['actuators']]
    assert result['controller_internal_state_reconstructed'] is False
    result['command_Nm'][0]=90.
    assert physics['motor_forces'][-1,0]==-1.


@pytest.mark.parametrize('kind',['order_duplicate','width','nan','cap','range','time'])
def test_motor_handoff_rejects_bad_original_command_contract(kind):
    physics,motors=motor_fixture()
    if kind=='order_duplicate':motors['actuators'][1]['name']='motor0'
    if kind=='width':physics['motor_forces']=physics['motor_forces'][:,:60]
    if kind=='nan':physics['motor_forces'][0,0]=float('nan')
    if kind=='cap':physics['motor_forces'][0,0]=2.001
    if kind=='range':motors['actuators'][0]['force_range']=[2.,-2.]
    if kind=='time':physics['time_s'][-1]=1.002
    with pytest.raises(ValueError):panel.actual_motor_handoff(physics,motors,terminal=1.)


def test_complete_body_join_requires_actual_unique_inventory():
    core=dict(geometry_time_s=1.,body_names=['robot/rh_palm'],body_poses_xyz_wxyz=[POSE.copy()])
    required={'robot/rh_palm','robot/rh_ffdistal','leaf'}
    extra={'rh_ffdistal':POSE.copy(),'rh_palm':POSE.copy(),'leaf':POSE.copy()}
    result=panel.complete_measured_bodies(core,POSE,extra,required_names=required)
    assert set(result['body_names'])==required
    extra['rh_palm'][0]=1e-9
    with pytest.raises(ValueError,match='disagree'):panel.complete_measured_bodies(core,POSE,extra,required_names=required)
    with pytest.raises(ValueError,match='Complete'):panel.complete_measured_bodies(core,POSE,{},required_names=required)
    with pytest.raises(ValueError,match='Unambiguous'):panel.complete_measured_bodies(core,POSE,{'invented':POSE},required_names=required)


def test_geometry_leaf_cannot_overwrite_independently_recorded_leaf():
    core=dict(geometry_time_s=1.,body_names=['robot/rh_palm'],body_poses_xyz_wxyz=[POSE.copy()])
    other=POSE.copy();other[0]=1e-9
    with pytest.raises(ValueError,match='disagree'):
        panel.complete_measured_bodies(core,POSE,{'leaf':other},required_names={'robot/rh_palm','leaf'})


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path,monkeypatch):
    """Mock independent audit/model seams, keep actual evidence/clock checks."""
    run=tmp_path/'synthetic released source';trial=run/'trial';trial.mkdir(parents=True)
    robot,door,usd=[tmp_path/name for name in ('robot.xml','door.xml','door.usda')]
    for path in (robot,door,usd):path.write_text('Synthetic '+path.name)
    declaration=tmp_path/'profile.json'
    write(declaration,dict(profile='volar-phalange-v1',robot_xml_sha256=panel.digest(robot)))
    (trial/'grasp-profile-definition.json').write_bytes(declaration.read_bytes())
    args=dict(native_robot=str(robot),native_door=str(door),door_usd=str(usd),
        grasp_profile='volar-phalange-v1',grasp_profile_definition=str(declaration),standing_withdrawal_route='synthetic')
    configuration=dict(args=args)
    provenance=dict(files={str(p):panel.digest(p) for p in (robot,door,usd,declaration)})
    report=dict(passed=True,checks={k:True for k in panel.REQUIRED_CHECKS},
        standing_withdrawal=dict(completed=True),physics_dt_s=.002,duration_s=1.,
        runtime_robot_pose_writes=0,direct_door_commands=False,grasp_profile='volar-phalange-v1',
        observed_final_pad_grasp_hold=False)
    fresh=dict(schema='doorbench.isaac-standing-withdrawal-audit.v1',passed=True,
        producer_matches=True,checks=report['checks'].copy(),input_sha256={})
    physics,motors=motor_fixture();physics['standing_leaf_pose']=np.tile(POSE,(500,1))
    for name,value in [('configuration.json',configuration),('provenance.json',provenance),
            ('operation-report.json',report),('motor-contract.json',motors)]:write(trial/name,value)
    write(run/'isaac-withdrawal-audit.json',fresh);write(run/'independent-contact-audit.json',dict(synthetic=True))
    np.savez_compressed(trial/'acquisition-physics.npz',**physics)
    rows=window()
    with gzip.open(trial/'standing-withdrawal-steps.json.gz','wt') as f:json.dump(rows,f)
    calls=[]
    def audit(*a):calls.append(a);return copy.deepcopy(fresh)
    monkeypatch.setattr(panel,'audit_withdrawal',audit)
    monkeypatch.setattr(panel,'validate_local_completion',lambda *a:dict(passed=True,synthetic_fixture=True))
    monkeypatch.setattr(panel,'_historical_inputs',lambda *a:({},{}))
    binding=dict(sha256='a'*64,time_s=1.,robot_source_sha256=panel.digest(robot),door_source_sha256=panel.digest(usd))
    monkeypatch.setattr(panel,'extract_attained_state',lambda **k:copy.deepcopy(binding))
    core=dict(geometry_time_s=1.,body_names=['robot/rh_palm'],body_poses_xyz_wxyz=[POSE.copy()])
    monkeypatch.setattr(panel,'extract_standing_body_poses',lambda *a,**k:copy.deepcopy(core))
    m=SimpleNamespace(geom_bodyid=[1,2],body=lambda i:SimpleNamespace(name=['world','robot/rh_ffdistal','leaf'][i]))
    monkeypatch.setattr(panel,'LandedLeftScene',lambda *a:SimpleNamespace(m=m))
    monkeypatch.setattr(panel,'clearance_pairs',lambda m:[(0,1)])
    admissions=[]
    def coordinate(m,extracted,**kwargs):
        admissions.append(copy.deepcopy(extracted))
        return SimpleNamespace(qpos=np.array([1.,2.]),qvel=np.array([3.])),dict(passed=True,profile='isaac-float32-2um-2urad-v1')
    monkeypatch.setattr(panel,'admit_destination_planner',coordinate)
    return dict(run=run,trial=trial,robot=robot,door=door,usd=usd,rows=rows,report=report,
        fresh=fresh,configuration=configuration,provenance=provenance,binding=binding,
        calls=calls,admissions=admissions,physics=physics,declaration=declaration)


def admit(source):
    return panel.admit_isaac_panel_source(source['run'],robot=source['robot'],door_xml=source['door'],door_usd=source['usd'])


def test_released_source_requires_fresh_proof_and_original_complete_body_admission(source):
    result=admit(source)
    assert len(source['calls'])==1 and len(source['admissions'])==1
    assert result['schema']==panel.SCHEMA and result['source_engine']=='isaac-physx'
    assert result['source_time_s']==1. and result['initial_qvel']==[3.]
    assert result['initial_qpos']==[1.,2.]
    assert set(result['actual_body_poses_xyz_wxyz'])=={'robot/rh_palm','robot/rh_ffdistal','leaf'}
    assert source['admissions'][0]==result['extracted_state']
    assert result['authorized_stages']==result['source_sample_playback']==result['physics_steps']==0
    assert result['motor_handoff']['command_time_s']==.998
    assert all(panel.digest(path)==expected for path,expected in result['input_sha256'].items())
    assert 'native_manifest' not in result


@pytest.mark.parametrize('kind',['failed_report','incomplete','missing_gate','failed_gate','pose_write',
    'door_command','wrong_profile','failed_audit','different_audit','wrong_asset','changed_asset',
    'changed_declaration','different_state_asset','gap','missing_leaf','leaf_mismatch','low_palm'])
def test_released_source_does_not_promote_failed_or_mismatched_evidence(source,kind):
    report=source['report'];trial=source['trial']
    if kind=='failed_report':report['passed']=False
    if kind=='incomplete':report['standing_withdrawal']['completed']=False
    if kind=='missing_gate':report['checks'].pop('final_hand_clear_of_environment')
    if kind=='failed_gate':report['checks']['final_hand_clear_of_environment']=False
    if kind=='pose_write':report['runtime_robot_pose_writes']=1
    if kind=='door_command':report['direct_door_commands']=True
    if kind=='wrong_profile':report['grasp_profile']='distal-pad-v1'
    if kind=='failed_audit':source['fresh']['passed']=False
    if kind=='different_audit':source['fresh']['extra']='changed'
    if kind=='wrong_asset':source['configuration']['args']['native_door']=str(source['robot'])
    if kind=='changed_asset':source['door'].write_text('changed')
    if kind=='changed_declaration':source['declaration'].write_text('{}')
    if kind=='different_state_asset':source['binding']['robot_source_sha256']='b'*64
    if kind=='gap':source['physics']['time_s'][10]+=.001
    if kind=='missing_leaf':source['physics'].pop('standing_leaf_pose')
    if kind=='leaf_mismatch':source['physics']['standing_leaf_pose'][-1,0]=1e-9
    if kind=='low_palm':source['rows'][100]['left_surface']['palm_normal_load_N']=1.
    write(trial/'operation-report.json',report);write(trial/'configuration.json',source['configuration'])
    np.savez_compressed(trial/'acquisition-physics.npz',**source['physics'])
    with gzip.open(trial/'standing-withdrawal-steps.json.gz','wt') as f:json.dump(source['rows'],f)
    with pytest.raises((ValueError,KeyError)):admit(source)


def test_source_changed_during_fresh_audit_is_rejected(source,monkeypatch):
    def audit(*args):
        (source['trial']/'provenance.json').write_text('{}')
        return source['fresh']
    monkeypatch.setattr(panel,'audit_withdrawal',audit)
    with pytest.raises(ValueError,match='changed'):admit(source)


def test_optional_continuation_capture_is_bound_without_promoting_its_authority(source):
    path=source['trial']/'standing-continuation-steps.json.gz'
    with gzip.open(path,'wt') as f:json.dump([],f)
    result=admit(source)
    extra=result['additional_continuation_evidence']
    assert extra['sha256']==panel.digest(path)==result['input_sha256'][str(path)]
    assert extra['independently_admitted'] is False


@pytest.fixture
def completion(source,monkeypatch):
    """Original local launch receipt with mocked independent prefix reduction."""
    import scripts.isaac.run_local_operation as launcher
    run=source['run'];trial=source['trial'];args=source['configuration']['args']
    producer=run.parent/'original project/scripts/dexterous/isaac_opening.py'
    producer.parent.mkdir(parents=True);producer.write_text('original synthetic producer')
    (run/'source-isaac_opening.py').write_bytes(producer.read_bytes())
    original_hash=panel.digest(producer)
    source['provenance']['files'][str(producer)]=original_hash
    args.update(output=str(trial),seconds=1.,standing_transfer_route='synthetic-transfer',
        standing_transfer_prefix_source='synthetic-prefix',standing_transfer_start_seconds=.2)
    argv=['python','-u',str(producer),'--standing-transfer-route','synthetic-transfer',
        '--standing-transfer-prefix-source','synthetic-prefix',
        '--standing-transfer-start-seconds','.2','--standing-withdrawal-route','synthetic',
        '--native-door',str(source['door']),'--output',str(trial),'--seconds','1']
    launch=dict(argv=argv,input_sha256={str(source['door']):panel.digest(source['door'])},
        source_sha256={str(producer):original_hash},runtime_source_sha256={str(producer):original_hash},
        started_unix=10.,pid=1234,physics_started=True,passed=False,runtime_passed=False,independent_passed=False)
    result=copy.deepcopy(launch)
    for key in ('passed','runtime_passed','runtime_binding_passed','independent_passed',
        'independent_transfer_passed','source_prefix_passed','independent_withdrawal_passed',
        'withdrawal_source_prefix_passed','report_emitted'):result[key]=True
    for key in ('preflight_returncode','returncode','audit_returncode','transfer_audit_returncode','withdrawal_audit_returncode'):result[key]=0
    result.update(finished_unix=11.,report_sha256=panel.digest(trial/'operation-report.json'),
        audit_sha256=panel.digest(run/'independent-contact-audit.json'))
    write(trial/'standing-withdrawal-admission.json',dict(source_context=dict(source_run='synthetic-withdrawal-prefix',start_time_s=.5)))
    for name in ('live-prefix-witness.json','live-withdrawal-prefix-witness.json'):write(trial/name,dict(synthetic=True))
    prefix=dict(passed=True,intervals=100,input_sha256={str(trial/'acquisition-physics.npz'):panel.digest(trial/'acquisition-physics.npz')})
    monkeypatch.setattr(launcher,'audit_transfer_prefix',lambda *a:copy.deepcopy(prefix))
    monkeypatch.setattr(launcher,'audit_withdrawal_prefix',lambda *a:copy.deepcopy(prefix))
    for name in ('source-prefix-audit.json','withdrawal-source-prefix-audit.json'):write(run/name,prefix)
    with gzip.open(trial/'standing-transfer-steps.json.gz','wt') as f:json.dump([],f)
    from doorbench.dexterous.qualified_isaac_grasp import TRANSFER_CHECKS
    transfer=dict(passed=True,producer_matches=True,checks={k:True for k in TRANSFER_CHECKS},
        input_sha256={str(trial/name):panel.digest(trial/name) for name in ('operation-report.json','standing-transfer-steps.json.gz')})
    write(run/'isaac-transfer-audit.json',transfer)
    write(run/'launch.json',launch);write(run/'result.json',result)
    source.update(launch=launch,result=result,producer=producer,prefix=prefix)
    return source


def completed(item):
    hashes={}
    result=VALIDATE_COMPLETION(item['run'],item['configuration'],item['provenance'],item['report'],hashes)
    return result,hashes


def test_original_finalized_launch_and_prefixes_are_rechecked(completion):
    # Today's producer can differ; original captured bytes remain authoritative.
    completion['producer'].write_text('a later producer edit')
    result,hashes=completed(completion)
    assert result['full_prefixes_recomputed'] is True
    for name in ('launch.json','result.json','source-isaac_opening.py',
            'source-prefix-audit.json','withdrawal-source-prefix-audit.json','isaac-transfer-audit.json'):
        assert str(completion['run']/name) in hashes


@pytest.mark.parametrize('kind',['process_failure','incomplete','missing_raw','missing_transfer','missing_prefix',
    'missing_withdrawal','missing_full_prefix','timeout','launch_error','stale_report','stale_contact',
    'changed_argv','config_mismatch','captured_source','input_hash','runtime_inventory',
    'changed_prefix','changed_transfer','early_stop','wrong_duration'])
def test_incomplete_or_mismatched_original_launch_is_ineligible(completion,kind):
    item=completion;result=item['result'];run=item['run'];trial=item['trial']
    if kind=='process_failure':result['returncode']=1
    if kind=='incomplete':result.pop('finished_unix')
    flags={'missing_raw':'independent_passed','missing_transfer':'independent_transfer_passed',
        'missing_prefix':'source_prefix_passed','missing_withdrawal':'independent_withdrawal_passed',
        'missing_full_prefix':'withdrawal_source_prefix_passed'}
    if kind in flags:result[flags[kind]]=False
    if kind=='timeout':result['timeout']=True
    if kind=='launch_error':result['launch_error_type']='RuntimeError'
    if kind=='stale_report':result['report_sha256']='a'*64
    if kind=='stale_contact':result['audit_sha256']='a'*64
    if kind=='changed_argv':result['argv']=result['argv']+['--other']
    if kind=='config_mismatch':item['configuration']['args']['standing_transfer_route']='different'
    if kind=='captured_source':(run/'source-isaac_opening.py').write_text('changed captured producer')
    if kind=='input_hash':item['door'].write_text('changed original input')
    if kind=='runtime_inventory':item['provenance']['files'].pop(str(item['producer']))
    if kind=='changed_prefix':write(run/'withdrawal-source-prefix-audit.json',dict(item['prefix'],intervals=99))
    if kind=='changed_transfer':write(run/'isaac-transfer-audit.json',dict(passed=False))
    if kind=='early_stop':write(trial/'early-stop.json',dict(stopped=True))
    if kind=='wrong_duration':item['report']['duration_s']=1.002
    write(run/'result.json',result)
    with pytest.raises((ValueError,KeyError)):completed(item)
