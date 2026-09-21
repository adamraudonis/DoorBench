"""Synthetic complete saved observation/reference accounting; no physical run."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import test_isaac_standing_continuation_audit as observations
import test_isaac_standing_reference_capture as capture_fixture
from doorbench.dexterous import isaac_standing_reference_audit as module
from doorbench.dexterous.isaac_standing_reference_capture import StandingContinuationReferenceTail
from doorbench.dexterous.motor_contract_identity import motor_contract_fingerprint
from doorbench.dexterous.qualified_isaac_grasp import digest


@pytest.fixture
def source(tmp_path):
    s=observations.source.__wrapped__(tmp_path);trial=s['trial']
    for key,value in s['arrays'].items():s['arrays'][key]=np.concatenate((value,value))
    a=s['arrays'];a['time_s']=np.arange(1,7)*.002
    a['pre_step_joint_velocity'][1:]=a['joint_velocity'][:-1]
    damping=np.r_[np.full(61,.1),np.zeros(8)];friction=np.r_[np.full(61,.03),np.zeros(8)];matrix=np.eye(61,69)
    for i,v in enumerate(a['pre_step_joint_velocity']):
        a['actual_joint_effort'][i]=matrix.T@a['motor_forces'][i]-damping*v-friction*np.tanh(v/.001)
        d=v.astype(float)
        a['actual_motor_forces'][i]=matrix@(a['actual_joint_effort'][i].astype(float)+damping*d+friction*np.tanh(d/.001))
    s['records']=[copy.deepcopy(row) for row in s['records']*2]
    s['surfaces']=[copy.deepcopy(row) for row in s['surfaces']*2]
    for i,(row,surface) in enumerate(zip(s['records'],s['surfaces'])):
        t=(i+1)*.002;row.update(time_s=t,pose_time_s=t,contact_interval_s=[t-.002,t]);surface['time_s']=t
    s['report']['duration_s']=.012;s['report']['standing_continuation_capture']['observations']=6
    runtime=tmp_path/'synthetic-withdrawal-runtime.json';observations.write(runtime,dict(
        schema='doorbench.isaac-standing-withdrawal-runtime.v1',capture_returned_motor_command=True,
        inherit_transfer_support=True,left_arm_only=True,left_full_orientation=True,coupled_motion_projection='fixed-poses-v1',
        scope='Synthetic data binding only; not an admitted physical route'))
    binding=dict(runtime_path=str(runtime),runtime_sha256=digest(runtime),
        motor_contract_sha256=motor_contract_fingerprint(s['motors']),source_state_sha256='c'*64)
    context=dict(**binding,start_time_s=.006)
    observations.write(trial/'standing-withdrawal-admission.json',{'source_context':context,'synthetic_fixture_only':True})
    (trial/'standing-withdrawal-runtime.json').write_bytes(runtime.read_bytes())
    s['config']['standing_continuation_reference_capture']='accepted-command-tail-v1'
    s['config']['args']['standing_withdrawal_route']=str(runtime)
    original=tmp_path/'isaac_standing_reference_capture.py';original.write_text('# Synthetic captured source fixture\n')
    (trial/'source-isaac_standing_reference_capture.py').write_bytes(original.read_bytes())
    s['provenance']['files'].update({str(runtime):digest(runtime),str(original):digest(original)})
    for name,key in [('configuration','config'),('operation-report','report'),('provenance','provenance')]:observations.write(trial/(name+'.json'),s[key])
    observations.save_streams(s)
    c,args=capture_fixture.fixture(.006);c.source_context.data.update(binding)
    c.acquisition.caps=np.asarray([m['force_range'] for m in s['motors']['actuators']])
    c.coupled.motion.value[:]=0.;c.coupled.motion.value[3]=.01
    c.coupled.motion.velocity[:]=0.;c.coupled.motion.velocity[4]=.005;c.coupled.motion.velocity[6]=.01
    c.left.started=0.;c.left.loaded_since=.002;c.inherited_support.started=.006;c.inherited_support.feedback.started=0.
    observer=StandingContinuationReferenceTail(c,motor_names=[m['name'] for m in s['motors']['actuators']])
    for i in range(3):
        t=(3+i)*.002;post=(4+i)*.002;before=3+i-1;after=4+i-1
        c.coupled.motion.time=t;c.coupled.motion.value[4]=i*.00001;c.coupled.motion.value[6]=i*.00002
        c.inherited_support.previous=t;c.arm.last_call_time=t;c.palm.last_time=t;c.arm.previous_time=t;c.left.last_update=t
        c.inherited_support.feedback.previous=(t,a['continuation_body_poses'][before,4,:3].copy(),np.eye(3))
        nominal=dict(zip(c.coupled.names,c.coupled.motion.value[6:]));targets=nominal.copy()
        for n,v in zip(c.palm.names,c.palm.offset):targets[n]+=v
        c.acquisition.stance.target_root[:]=c.coupled.motion.value[:3]+c.stance_root_bias
        c.acquisition.stance.target_rotation=c.stance_rotation_bias@Rotation.from_rotvec(c.coupled.motion.value[3:6]).as_matrix()
        c.acquisition.stance.joint_target[:]=[nominal[n]+v for n,v in zip(c.stance_names,c.stance_joint_bias)]
        c.left.path[-1]['nominal']=np.array([nominal[n] for n in c.left.names]);c.left.normal=np.array([0.,1.,0.])
        c.arm.previous_target=np.array([targets[n] for n in c.arm.names]);c.hand.targets=np.array([targets[n] for n in c.acquisition.names])[:18]
        args.update(post_feedback_targets=targets,returned_command=a['motor_forces'][after].copy(),
            measured_root=a['root'][before].copy(),measured_joints=dict(zip(s['config']['robot_joint_names'],a['joints'][before])),
            measured_velocities=dict(zip(s['config']['robot_joint_names'],a['joint_velocity'][before])),
            measured_leaf_pose=a['continuation_body_poses'][before,4].copy(),
            measured_handle_pose=a['continuation_body_poses'][before,5].copy(),measured_angles=dict(leaf=0.,operator=0.,latch=0.))
        c.acquisition.last_force=args['returned_command'].copy()
        observer.capture_command(c,t,**args)
        observer.observe_completed_interval(command_time_s=t,post_step_time_s=post,returned_command=args['returned_command'])
    tail=observer.receipt();tail_path=trial/'standing-continuation-reference-tail.json';observations.write(tail_path,tail)
    audit=observations.audit.audit_standing_continuation(trial);audit_path=trial/'observations-audit.json';observations.write(audit_path,audit)
    s.update(tail=tail,tail_path=tail_path,observation_audit=audit,audit_path=audit_path,context=context)
    return s


def admit(s,**kw):
    args=dict(continuation_audit=s['audit_path'],expected_epoch_s=.012,
        expected_physics_sha256=s['observation_audit']['source_physics_sha256']);args.update(kw)
    return module.admit_standing_reference_tail(s['trial'],**args)


def save_tail(s):observations.write(s['tail_path'],s['tail'])


def test_real_observation_and_reference_accounting_do_not_grant_physical_authority(source):
    result=admit(source);d=result['diagnostics']
    assert result['passed'] and result['reference_tail_accounting_passed']
    assert not result['source_report_passed'] and result['requires_separate_physical_source_qualification']
    assert result['authorized_stages']==0 and not result['physical_task_qualification'] and not result['bridge_feasibility_qualified']
    assert d['command_times_s']==[.006,.008,.010]
    np.testing.assert_allclose(np.array(d['finite_difference_coordinate_velocity'])[:,6],.01,atol=1e-15)
    assert result['terminal_nominal_delta']['reference_command_epoch_s']==.010
    assert result['terminal_nominal_delta']['actual_state_epoch_s']==.012
    assert not d['first_stored_velocity_before_tail_independently_verified']
    # Rotvec derivative is not silently relabeled as world angular velocity.
    w=np.array(d['root_angular_velocity_world_interval_average'])
    assert np.max(abs(w[:,2]))>1e-6
    assert result['input_sha256'][str(source['tail_path'])]==digest(source['tail_path'])
    assert result['attained_tracking_contract']['hand']['stiffness_scale']==5.
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('kind',['command','motor_dtype','input_root','input_joint','input_velocity','leaf_pose','angle','reference_clock',
    'post_time','missing_row','pending','failure','count','marker','source_binding','runtime_copy','capture_source','caps','motor_names',
    'stored_velocity','post_target','stance_root','stance_rotation','left_nominal','arm_target','hand_target','hand_preload','support_epoch','support_normal'])
def test_corrupt_source_or_nominal_history_rejected(source,kind):
    tail=source['tail'];row=tail['tail'][-1]
    if kind=='command':row['returned_motor_command'][0]+=1e-12
    elif kind=='motor_dtype':row['returned_motor_dtype']='<f4'
    elif kind=='input_root':row['observed_input']['root13_actor_origin'][0]+=.001
    elif kind=='input_joint':row['observed_input']['joint_position'][0]+=.001
    elif kind=='input_velocity':row['observed_input']['joint_velocity'][0]+=.001
    elif kind=='leaf_pose':row['observed_input']['leaf_pose_xyz_wxyz'][0]+=.001
    elif kind=='angle':row['observed_input']['angles']['operator']=.1
    elif kind=='reference_clock':row['coupled']['time_s']-=.002
    elif kind=='post_time':row['post_step_time_s']-=.002
    elif kind=='missing_row':tail['tail'].pop()
    elif kind=='pending':tail['pending_unaccepted_command']={'command_time_s':.012}
    elif kind=='failure':tail['failure']='capture failure'
    elif kind=='count':tail['completed_reference_intervals']=4
    elif kind=='marker':source['config']['standing_continuation_reference_capture']='other';observations.write(source['trial']/'configuration.json',source['config'])
    elif kind=='source_binding':tail['source_binding']['source_state_sha256']='d'*64
    elif kind=='runtime_copy':(source['trial']/'standing-withdrawal-runtime.json').write_text('{}')
    elif kind=='capture_source':(source['trial']/'source-isaac_standing_reference_capture.py').write_text('# changed')
    elif kind=='caps':tail['static_controller_data']['motor_caps'][0][1]+=1
    elif kind=='motor_names':tail['static_controller_data']['motor_names'].reverse()
    elif kind=='stored_velocity':row['coupled']['velocity'][6]+=.1
    elif kind=='post_target':row['right']['post_feedback_joint_targets'][0]+=.001
    elif kind=='stance_root':row['stance']['target_root'][0]+=.001
    elif kind=='stance_rotation':row['stance']['target_rotation'][0][0]=.9
    elif kind=='left_nominal':row['left']['nominal_joint_target'][0]+=.001
    elif kind=='arm_target':row['right']['arm']['previous_target'][0]+=.001
    elif kind=='hand_target':row['hand']['targets'][0]+=.001
    elif kind=='hand_preload':row['hand']['preload'][0]=1.
    elif kind=='support_epoch':row['support']['previous_s']-=.002
    else:row['left']['normal_world']=[0.,0.,1.]
    save_tail(source)
    with pytest.raises((ValueError,KeyError)):admit(source)


def test_full_joint_order_is_mapped_by_names_not_array_coincidence(source):
    for row in source['tail']['tail']:
        actual=row['observed_input'];actual['joint_names']=actual['joint_names'][::-1];actual['joint_position'].reverse();actual['joint_velocity'].reverse()
    save_tail(source);assert admit(source)['passed']
    source['tail']['tail'][-1]['observed_input']['joint_velocity'].reverse();save_tail(source)
    with pytest.raises(ValueError,match='velocity'):admit(source)


def test_tampered_same_endpoint_archive_needs_new_fresh_observation_admission(source):
    source['arrays']['motor_forces'][0,0]+=.001;observations.save_streams(source)
    with pytest.raises(ValueError):admit(source)


def test_original_motion_excess_is_reported_without_bridge_promotion(source):
    # First stored velocity is not independently observed before this 3-row tail.
    # Its enormous rate still must fail the unchanged motion-limit diagnostic.
    source['tail']['tail'][0]['coupled']['velocity'][6]=2.;save_tail(source)
    result=admit(source)
    assert result['reference_tail_accounting_passed'] and not result['passed']
    assert not result['diagnostics']['within_original_coupled_motion_limits']
    assert result['diagnostics']['tail_maximum']['joint_speed_rad_s']==2.
    assert result['authorized_stages']==0


def test_expected_source_epoch_and_sha_are_exact(source):
    for override in ({'expected_epoch_s':.010},{'expected_physics_sha256':'f'*64}):
        with pytest.raises(ValueError):admit(source,**override)


def test_reference_file_cannot_change_during_analysis(source,monkeypatch):
    original=module.iter_npz_records
    def changed(*args,**kwargs):
        yield from original(*args,**kwargs)
        with source['tail_path'].open('a') as stream:stream.write('\n')
    monkeypatch.setattr(module,'iter_npz_records',changed)
    with pytest.raises(ValueError,match='changed during'):admit(source)


@pytest.mark.parametrize('path',[
    ('coupled','velocity'),('stance','qp_warm_start'),('stance','solver_metadata'),
    ('right','arm','preload'),('left','target_velocity'),('left','filtered_palm_load_N'),
    ('support','previous_ik_target'),('withdrawal','handoff_offset'),
    ('hand','original_preload'),('private_generalized_bias',),
])
def test_required_history_is_never_replaced_by_an_invented_default(source,path):
    owner=source['tail']['tail'][-1]
    for key in path[:-1]:owner=owner[key]
    del owner[path[-1]];save_tail(source)
    with pytest.raises((ValueError,KeyError)):admit(source)


@pytest.mark.parametrize('change',[
    'bool_authority','row_bool_authority','invented_absent_velocity',
    'future_ik_seed','nonfinite','duplicate_key','missing_static_gains',
])
def test_strict_state_and_json_contract(source,change):
    tail=source['tail'];row=tail['tail'][-1]
    if change=='bool_authority':tail['authorized_stages']=False
    elif change=='row_bool_authority':row['authorized_stages']=False
    elif change=='invented_absent_velocity':row['left']['target_velocity']={'available':False,'value':[0.]*7}
    elif change=='future_ik_seed':row['support']['previous_ik_target']={'available':True,'value':[.012,[0.]*7]}
    elif change=='missing_static_gains':del tail['attained_tracking_contract']['hand']['kp']
    save_tail(source)
    if change=='nonfinite':
        raw=source['tail_path'].read_text();source['tail_path'].write_text(raw.replace('"completed_reference_intervals": 3','"completed_reference_intervals": 1e309'))
    elif change=='duplicate_key':
        raw=source['tail_path'].read_text();source['tail_path'].write_text(raw.replace('"tail_samples": 3','"tail_samples": 3, "tail_samples": 3'))
    with pytest.raises((ValueError,KeyError)):admit(source)


def cli_args(source,output):
    return ['--trial',str(source['trial']),'--continuation-audit',str(source['audit_path']),
        '--expected-epoch-s','.012','--expected-physics-sha256',source['observation_audit']['source_physics_sha256'],
        '--output',str(output)]


def test_cli_real_synthetic_admission_and_no_overwrite(source,tmp_path,capsys):
    from scripts.dexterous.audit_isaac_standing_reference import main
    output=tmp_path/'analysis.json'
    assert main(cli_args(source,output))==0
    receipt=json.loads(output.read_text())
    assert receipt['passed'] and receipt['authorized_stages']==0
    assert not receipt['physical_task_qualification']
    original=output.read_bytes()
    with pytest.raises(ValueError,match='existing analysis'):main(cli_args(source,output))
    assert output.read_bytes()==original
    assert json.loads(capsys.readouterr().out)['reference_tail_accounting_passed']


def test_cli_corruption_keeps_failed_receipt(source,tmp_path):
    from scripts.dexterous.audit_isaac_standing_reference import main
    del source['tail']['tail'][-1]['private_generalized_bias'];save_tail(source)
    output=tmp_path/'failed-analysis.json'
    assert main(cli_args(source,output))==1
    receipt=json.loads(output.read_text())
    assert not receipt['passed'] and not receipt['reference_tail_accounting_passed']
    assert receipt['authorized_stages']==0 and not receipt['bridge_feasibility_qualified']
    assert 'private_generalized_bias' in receipt['error']


def test_cli_valid_accounting_with_rate_failure_is_still_nonzero(source,tmp_path):
    from scripts.dexterous.audit_isaac_standing_reference import main
    source['tail']['tail'][0]['coupled']['velocity'][6]=2.;save_tail(source)
    output=tmp_path/'rate-failure.json'
    assert main(cli_args(source,output))==1
    receipt=json.loads(output.read_text())
    assert receipt['reference_tail_accounting_passed'] and not receipt['passed']
    assert receipt['authorized_stages']==0
