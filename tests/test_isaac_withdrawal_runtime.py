"""Synthetic detached admissions; no simulator, source promotion or motor run."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from doorbench.dexterous import isaac_withdrawal_runtime as module
from doorbench.dexterous.isaac_prefix_witness import PREFIX_FIELDS
from doorbench.dexterous.motor_contract_identity import motor_contract_fingerprint
from doorbench.dexterous.qualified_isaac_grasp import digest
from doorbench.dexterous.standing_body_record import POSE_CONVENTION
from doorbench.dexterous.withdrawal_source_context import WithdrawalSourceContext
from test_isaac_withdrawal_support import predecessor


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    run=tmp_path/'synthetic source';trial=run/'trial';trial.mkdir(parents=True)
    motors=dict(source_xml_sha256='a'*64,joint_names=['j'],actuators=[
        dict(name='a',terms={'j':1.},force_range=[-2.,2.])])
    write(trial/'motor-contract.json',motors)
    route=tmp_path/'transfer.json';write(route,{'synthetic_fixture':True})
    recorded=dict(standing_transfer_support_load_target=6.,standing_transfer_hybrid_support=True,
        standing_transfer_route=str(route),standing_transfer_start_seconds=28.)
    write(trial/'configuration.json',dict(args=recorded))
    qualification_path=run/'qualification.json';write(qualification_path,{'synthetic_fixture':True})
    qualification=dict(passed=True,state_sha256='b'*64,
        input_sha256={str(qualification_path):digest(qualification_path)})
    hashes={str(p):digest(p) for p in (trial/'motor-contract.json',trial/'configuration.json',route,qualification_path)}
    data=dict(source_engine='isaac-physx',source_run=str(run),start_time_s=44.,duration_s=16.,
        motor_contract_sha256=motor_contract_fingerprint(motors),input_sha256=hashes,
        source_admission=dict(source_qualification=qualification,grasp_profile='volar-phalange-v1'),
        initial_qpos=[0,0,0,1,0,0,0],source_state_sha256='b'*64,authorized_stages=0)
    source_path=tmp_path/'source.json';envelope=tmp_path/'map.json';audit=tmp_path/'audit.json'
    source=dict(schema='doorbench.standing-withdrawal.v1',source_engine='isaac-physx',source_run=str(run),
        screen_path='bound-elsewhere',screen_sha256='c'*64,audit_path='bound-elsewhere',audit_sha256='d'*64,
        measured_rest_transfer=True,grasp_profile='volar-phalange-v1',contact_audit_name='independent-contact-audit.json',
        robot_path='bound-robot',door_xml_path='bound-xml',door_usd_path='bound-usd',source_state_sha256='b'*64,
        start_time_s=44.,duration_s=16.)
    write(source_path,source);write(envelope,{'synthetic_fixture':True});write(audit,{'synthetic_fixture':True})
    config=dict(schema=module.SCHEMA,**module.REQUIRED_OPTIONS,
        source_config_path=str(source_path),coupled_envelope_path=str(envelope),coupled_audit_path=str(audit))
    path=tmp_path/'runtime.json'
    def bind():
        for key in module.PATH_KEYS:config[key.removesuffix('_path')+'_sha256']=digest(config[key])
        write(path,config)
    bind();calls=[]
    def load(p):
        calls.append(('load',p));return WithdrawalSourceContext(json.dumps(data)),{str(source_path):digest(source_path)}
    def reference(c,d):
        calls.append(('reference',copy.deepcopy(d)))
        return SimpleNamespace(input_sha256={str(p):digest(p) for p in (source_path,envelope,audit)})
    monkeypatch.setattr(module,'load_isaac_coupled_source',load)
    monkeypatch.setattr(module,'IsaacCoupledReleaseReference',reference)
    return SimpleNamespace(path=path,config=config,source=source,source_path=source_path,bind=bind,
        data=data,motors=motors,recorded=recorded,route=route,trial=trial,calls=calls)


def admit(f):return module.admit_isaac_withdrawal_runtime(f.path,f.motors)


def receipt(admission):
    d=admission.source_context.data;t=d['start_time_s'];count=round(t/.002)
    hashes=d['source_admission']['source_qualification']['input_sha256'].copy()
    common=dict(passed=True,stage_entry_authorized=True,prefix_complete=True,
        intervals_verified=count,intervals_required=count,last_verified_time_s=t,input_sha256=hashes)
    core=dict(**common,schema='doorbench.live-isaac-prefix-witness.v1',fields=list(PREFIX_FIELDS),failure=None,
        source_run=d['source_run'],source_terminal_time_s=t,source_qualification=d['source_admission']['source_qualification'],
        runtime_motor_contract_sha256=d['motor_contract_sha256'])
    leaf=dict(**common,pose_convention=POSE_CONVENTION,
        comparison='Exact bytes after explicit float64 canonicalization; historical tensor dtype unrecorded',
        historical_tensor_dtype_compared=False)
    return dict(schema='doorbench.live-isaac-withdrawal-prefix-witness.v1',passed=True,
        stage_entry_authorized=True,prefix_complete=True,failure=None,source_run=d['source_run'],
        source_terminal_time_s=t,source_qualification=d['source_admission']['source_qualification'],
        core=core,leaf_pose=leaf,input_sha256=hashes)


def test_launcher_admission_and_factory_share_one_validation_without_live_plant(runtime):
    a=module.admit_isaac_withdrawal_runtime(runtime.path)
    assert a.runtime_path==runtime.path and a.start_time==44. and a.duration==16.
    assert a.inherited_support is None and a.target_N==6. and not a.authorized
    assert a.source_admission==runtime.data['source_admission']
    assert a.source_context.data['authorized_stages']==0
    assert set(a.input_sha256)>=set(runtime.data['input_sha256'])
    assert a.input_sha256[str(runtime.path)]==digest(runtime.path)
    assert not any(hasattr(a,n) for n in ('force','plant','articulation','step'))
    assert [name for name,_ in runtime.calls]==['load','reference']
    assert not (runtime.trial.parent/'manifest.json').exists()
    assert not (runtime.trial.parent/'trajectory.npz').exists()


def test_base_numeric_legacy_flags_cannot_turn_into_new_runtime_options(runtime):
    runtime.source.update(hybrid_support=True,left_support_target_N=3.,panel_aperture_feedback_gain_N_per_rad=20.)
    write(runtime.source_path,runtime.source);runtime.bind()
    a=admit(runtime)
    assert a.controller_config['left_support_target_N']==6.
    assert 'hybrid_support' not in a.controller_config
    assert 'panel_aperture_feedback_gain_N_per_rad' not in a.controller_config
    assert a.controller_config['capture_returned_motor_command'] is True


@pytest.mark.parametrize('change',['unknown','missing_capture','disable_support','old_schema','relative','hash','contract','unbound_args','target','hybrid','epoch'])
def test_outer_runtime_and_predecessor_source_bindings_fail_closed(runtime,change):
    if change=='unknown':runtime.config['plant_override']=True
    if change=='missing_capture':runtime.config.pop('capture_returned_motor_command')
    if change=='disable_support':runtime.config['inherit_transfer_support']=False
    if change=='old_schema':runtime.config['schema']='doorbench.standing-withdrawal.v1'
    if change=='relative':runtime.config['source_config_path']='relative.json'
    if change=='hash':runtime.config['source_config_sha256']='e'*64
    if change=='contract':runtime.motors['actuators'][0]['force_range']=[-3.,3.]
    if change=='unbound_args':runtime.data['input_sha256'].pop(str(runtime.trial/'configuration.json'))
    if change in ('target','hybrid','epoch'):
        runtime.recorded.update({'target':{'standing_transfer_support_load_target':9.},
            'hybrid':{'standing_transfer_hybrid_support':False},'epoch':{'standing_transfer_start_seconds':44.}}[change])
        write(runtime.trial/'configuration.json',{'args':runtime.recorded})
        runtime.data['input_sha256'][str(runtime.trial/'configuration.json')]=digest(runtime.trial/'configuration.json')
    write(runtime.path,runtime.config)
    with pytest.raises(ValueError):admit(runtime)


def test_failed_actual_source_does_not_reach_reference(runtime,monkeypatch):
    def reject(*args):raise ValueError('Actual source final grasp failed')
    monkeypatch.setattr(module,'load_isaac_coupled_source',reject)
    with pytest.raises(ValueError,match='final grasp'):admit(runtime)
    assert runtime.calls==[]


def test_live_predecessor_identity_and_six_newton_state_required(runtime):
    a=admit(runtime);transfer,_=predecessor();transfer.path=runtime.route;transfer.start_seconds=28.
    a.bind_predecessor(transfer)
    assert a.inherited_support.transfer is transfer and a.inherited_support.feedback is None
    with pytest.raises(ValueError):a.bind_predecessor(transfer)
    b=admit(runtime);transfer.start_seconds=28.002
    with pytest.raises(ValueError):b.bind_predecessor(transfer)


def test_complete_core_and_leaf_witness_authorizes_once_then_exact_entry(runtime):
    a=admit(runtime);transfer,_=predecessor();transfer.path=runtime.route;transfer.start_seconds=28.
    a.bind_predecessor(transfer);r=receipt(a);a.authorize_source_prefix(r)
    r['core']['intervals_verified']=0
    assert a.prefix_receipt['core']['intervals_verified']==22000
    a.require_entry(44.)
    assert a.entered and a.inherited_support.feedback is transfer.support_feedback
    with pytest.raises(ValueError):a.require_entry(44.002)


@pytest.mark.parametrize('change',['old_core_only','partial','failed','source','epoch','motor','field','core_count',
    'leaf_count','leaf_epoch','leaf_convention','leaf_dtype_claim','leaf_comparison','component_hash','qualification','duplicate'])
def test_incomplete_or_relabelled_prefix_is_terminal(runtime,change):
    a=admit(runtime);r=receipt(a)
    if change=='old_core_only':r=r['core']
    if change=='partial':r['prefix_complete']=False
    if change=='failed':r['failure']={'bad':'diverged'}
    if change=='source':r['source_run']=str(runtime.trial)
    if change=='epoch':r['source_terminal_time_s']+=.002
    if change=='motor':r['core']['runtime_motor_contract_sha256']='f'*64
    if change=='field':r['core']['fields'].remove('motor_forces')
    if change=='core_count':r['core']['intervals_verified']-=1
    if change=='leaf_count':r['leaf_pose']['intervals_required']-=1
    if change=='leaf_epoch':r['leaf_pose']['last_verified_time_s']-=.002
    if change=='leaf_convention':r['leaf_pose']['pose_convention']='XYWZ'
    if change=='leaf_dtype_claim':r['leaf_pose']['historical_tensor_dtype_compared']=True
    if change=='leaf_comparison':r['leaf_pose']['comparison']='approximate'
    if change=='component_hash':r['leaf_pose']['input_sha256']={str(runtime.route):digest(runtime.route)}
    if change=='qualification':r['source_qualification']={}
    if change=='duplicate':a.authorize_source_prefix(r)
    with pytest.raises(ValueError):a.authorize_source_prefix(r)
    with pytest.raises(ValueError):a.authorize_source_prefix(receipt(a))
    assert not a.authorized and a.failure


@pytest.mark.parametrize('mode',['unauthorized','late','evidence_changed'])
def test_runtime_entry_cannot_skip_the_live_witness_or_changed_evidence(runtime,mode):
    a=admit(runtime);transfer,_=predecessor();transfer.path=runtime.route;transfer.start_seconds=28.;a.bind_predecessor(transfer)
    if mode=='evidence_changed':
        runtime.path.write_text('{}')
        with pytest.raises(ValueError):a.authorize_source_prefix(receipt(a))
    else:
        if mode=='late':a.authorize_source_prefix(receipt(a))
        with pytest.raises(ValueError):a.require_entry(44.002 if mode=='late' else 44.)
    assert not a.entered


def test_controller_registry_is_local_existing_absolute_and_unique():
    paths=module.withdrawal_runtime_source_paths()
    assert isinstance(paths,tuple) and len(paths)==len(set(paths))
    assert all(isinstance(p,Path) and p.is_absolute() and p.is_file() for p in paths)
    assert Path(module.__file__).resolve() in paths
    assert 'isaac_withdrawal_support.py' in [p.name for p in paths]


def test_factory_uses_fresh_admission_then_detached_bridge(runtime,monkeypatch):
    from doorbench.dexterous import standing_withdrawal
    transfer,_=predecessor();transfer.path=runtime.route;transfer.start_seconds=28.
    transfer.acquisition=object();transfer.operation=object()
    def construct(bridge,motors,path,*,_isaac_runtime):
        assert bridge.transfer is transfer and bridge.requires_measured_rest
        assert _isaac_runtime.inherited_support.transfer is transfer
        _isaac_runtime.verify(path,motors)
        return _isaac_runtime
    monkeypatch.setattr(standing_withdrawal,'StandingWithdrawalTeacher',construct)
    a=module.create_isaac_withdrawal_controller(transfer,runtime.motors,runtime.path)
    assert not a.authorized and not a.entered and a.target_N==6.


@pytest.mark.parametrize('authorize',[False,True])
def test_actual_constructor_uses_detached_isaac_state_and_prefix_delegates_exact_command(runtime,monkeypatch,authorize):
    from doorbench.dexterous import landed_left_planner
    from doorbench.dexterous.resting_transfer import RestingTransferBridge
    from doorbench.dexterous.standing_withdrawal import StandingWithdrawalTeacher
    from test_isaac_coupled_release_geometry import toy_scene
    scene=toy_scene();m,d=scene.m,scene.d;initial=d.qpos.copy();rh=m.site('robot/rh_palm_touch').id
    names=[m.joint(j).name[6:] for j in range(m.njnt) if m.joint(j).name.startswith('robot/') and m.jnt_type[j]==3]
    values={name:float(initial[m.joint('robot/'+name).qposadr[0]]) for name in names}
    row=dict(qpos=initial.tolist(),joints=values,finger_joints={},phase='measured_release',
        palm_position=d.site_xpos[rh].tolist(),palm_rotation=d.site_xmat[rh].reshape(3,3).tolist())
    runtime.data.update(initial_qpos=initial.tolist(),robot_path='synthetic-robot',door_xml_path='synthetic-door',
        screen={'trials':[{'rows':[dict(time_s=0.,**row),dict(time_s=8.,**row)]}]},audit={'duration_s':16.})
    a=admit(runtime)
    a.reference.geometry=SimpleNamespace(initial=initial,plan={'duration_s':16.})
    transfer,_=predecessor();transfer.path=runtime.route;transfer.start_seconds=28.;transfer.started=28.
    transfer.acquisition=SimpleNamespace(caps=np.array([[-2.,2.]]))
    transfer.operation=SimpleNamespace(hub_avoidance=None)
    command=np.array([1.25]);calls=[]
    def source_force(*args,**kwargs):
        calls.append((args,kwargs));return command,{'phase':'source-transfer'}
    transfer.force=source_force;a.bind_predecessor(transfer)
    scene_calls=[]
    monkeypatch.setattr(landed_left_planner,'LandedLeftScene',lambda *args:scene_calls.append(args) or scene)
    teacher=StandingWithdrawalTeacher(RestingTransferBridge(transfer),runtime.motors,runtime.path,_isaac_runtime=a)
    assert scene_calls==[(Path('synthetic-robot'),Path('synthetic-door'))]
    assert teacher.start_time==44. and teacher.duration==16. and teacher.support_target==6.
    assert teacher.source_context is a.source_context and teacher.source_admission==runtime.data['source_admission']
    assert teacher.coupled is a.reference and teacher.started_withdrawal is None
    assert not teacher.hybrid_support and teacher.inherited_support is a.inherited_support
    root=np.r_[initial[scene.root:scene.root+7],np.zeros(6)];angles={'leaf':.091,'operator':0.,'latch':0.}
    feedback=transfer.support_feedback;previous=feedback.previous
    for step in range(21750,22000):
        t=step*.002
        result,info=teacher.force(t,root,values,{},None,None,angles,{},grasp_qualified=True,left_panel_load=8.,left_palm_load=6.)
        assert result is command and info['phase']=='source-transfer'
    assert len(calls)==250 and feedback.previous is previous
    assert teacher.motor_capture.capture(44.).tolist()==[1.25]
    # Before any new arm tracker, force, or posture is built, entry needs the
    # independently completed current-episode exact core+leaf witness.
    if not authorize:
        with pytest.raises(ValueError,match='authorized'):teacher.force(44.,root,values,{},None,None,angles,{},
            grasp_qualified=True,left_panel_load=8.,left_palm_load=6.)
        assert teacher.started_withdrawal is None and len(calls)==250
        return
    from doorbench.dexterous import standing_withdrawal
    # Exercise the actual outer force/handoff/support logic. Synthetic inner
    # trackers supply deliberately different candidate commands; no plant or
    # physical torque is claimed by this wiring fixture.
    monkeypatch.setattr(standing_withdrawal,'AttainedArmTracking',lambda *args:SimpleNamespace(
        force=lambda force,*args,**kwargs:(force+.1,{})))
    monkeypatch.setattr(standing_withdrawal,'AttainedHandTracking',lambda *args:SimpleNamespace(
        preload=np.array([.1]),indices=np.array([0]),force=lambda force,*args,**kwargs:(force+.3,{})))
    monkeypatch.setattr(standing_withdrawal,'ReturnPalmFeedback',lambda *args,**kwargs:SimpleNamespace(
        world_targets=lambda t,targets,*args:(targets,{})))
    acq=transfer.acquisition;acq.last_force=np.array([-.5]);acq.names=names;acq.m=m
    acq.matrix=np.zeros((1,len(names)))
    acq.stance=SimpleNamespace(target_root=root[:3].copy(),target_rotation=np.eye(3),
        joints=np.array([],int),joint_target=np.zeros(0))
    transfer.operation.geometry={};transfer.operation.force=lambda *args,**kwargs:(np.array([-.7]),{})
    left=transfer.left;left.names=names;left.path=[{'nominal':np.zeros(len(names))}];left.info={}
    left.isolate_left_arm=lambda *args,**kwargs:None
    left.update_targets=lambda *args,**kwargs:None
    left.apply_forces=lambda force,*args:force+.2
    a.reference.geometry.localr=np.eye(3)
    a.reference.update=lambda t,elapsed,*args:(values,root[:3],np.eye(3),d.site_xpos[rh],np.eye(3),
        {'coupled_reference_release_clock_s':0.})
    teacher.authorize_source_prefix(receipt(a))
    force,info=teacher.force(44.,root,values,{},[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],angles,{},
        grasp_qualified=True,left_panel_load=8.,left_palm_load=6.)
    assert force.tolist()==[1.25] and teacher.previous_force.tolist()==[1.25]
    assert info['captured_command_Nm']==[1.25] and info['maximum_cache_disagreement_Nm']==1.75
    assert info['inherited_support']['feedback_epoch']==44.
    assert info['inherited_support']['previous_feedback_epoch']==43.998
    assert info['inherited_support']['palm_only_load_N']==6. and left.hybrid_normal_target==6.
    assert feedback.started==32. and teacher.started_withdrawal==44. and len(calls)==250


def test_native_constructor_does_not_gain_the_isaac_six_newton_exception(tmp_path):
    from doorbench.dexterous.standing_withdrawal import StandingWithdrawalTeacher
    path=tmp_path/'native.json';write(path,dict(schema='doorbench.standing-withdrawal.v1',left_support_target_N=6.))
    transfer=SimpleNamespace(left=SimpleNamespace(support_load_target=4.))
    returned=SimpleNamespace(acquisition=object(),operation=object(),transfer=transfer)
    with pytest.raises(ValueError,match='support target'):StandingWithdrawalTeacher(returned,{},path)
