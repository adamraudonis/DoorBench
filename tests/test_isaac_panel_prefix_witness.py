"""Synthetic admission seams and actual streaming bytes; never physical proof."""
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

import doorbench.dexterous.isaac_panel_prefix_witness as panel
from doorbench.dexterous.isaac_attained_state import RECORDED_ROOT_CONVENTION
from doorbench.dexterous.standing_body_record import PLANNER_BODIES, POSE_CONVENTION


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path, monkeypatch):
    run=tmp_path/'synthetic released source'; trial=run/'trial'; trial.mkdir(parents=True)
    assets=[tmp_path/name for name in ('robot.xml','door.xml','door.usda')]
    for path in assets:path.write_text('Synthetic CPU fixture only')
    names=['joint_'+str(i) for i in range(69)]
    motors=dict(joint_names=names,source_xml_sha256=panel.digest(assets[0]),actuators=[
        dict(name='motor_'+str(i),terms={names[i]:1.},force_range=[-10.,10.]) for i in range(61)])
    configuration=dict(args=dict(acquisition=True,operate_after_acquisition=True,
        acquisition_stance_profile='landed-foot-v1'),dt=.002,runtime_pose_writes=0,
        direct_door_commands=False,robot_joint_names=names,door_joint_names=['leaf','handle','latch'],
        root_state_convention=RECORDED_ROOT_CONVENTION,standing_planner_body_names=list(PLANNER_BODIES),
        standing_planner_body_pose_convention=POSE_CONVENTION,standing_leaf_pose_convention=POSE_CONVENTION,
        standing_continuation_body_names=list(panel.BODY_NAMES))
    write(trial/'configuration.json',configuration)
    shape=dict(root=(13,),joints=(69,),joint_velocity=(69,),motor_forces=(61,),
        door=(3,),door_velocity=(3,),standing_body_poses=(6,7),standing_leaf_pose=(7,),
        actual_motor_forces=(61,),actual_joint_effort=(69,),pre_step_joint_velocity=(69,),
        continuation_body_poses=(6,7),actual_foot_loads=(2,),legacy_root_state_w=(13,))
    arrays={key:np.zeros((3,*size),dtype=np.float32) for key,size in shape.items()}
    arrays['time_s']=np.arange(1,4)*.002
    arrays['motor_forces']=arrays['motor_forces'].astype(np.float64)
    for key in ('root','legacy_root_state_w','standing_leaf_pose'):arrays[key][:,3]=1.
    for key in ('standing_body_poses','continuation_body_poses'):arrays[key][:,:,3]=1.
    np.savez_compressed(trial/'acquisition-physics.npz',**arrays)
    contract=dict(schema='synthetic capture fixture',body_pose_order=list(panel.BODY_NAMES),
        normal_and_friction_slots_are_independent=True)
    readback=dict(schema='synthetic readback fixture',pre_step_velocity=True)
    write(trial/'standing-continuation-contract.json',contract)
    write(trial/'motor-readback-contract.json',readback)
    rows=[dict(time_s=(i+1)*.002,pose_time_s=(i+1)*.002,
        raw=dict(normal=dict(slots=[0,1],force_N=[[1.],[2.]]),
                 friction=dict(slots=[3],force_N=[[.1,.2,.3]])),
        evidence=dict(physics_qualified=True),authorized_stages=0) for i in range(3)]
    stream=trial/'standing-continuation-steps.json.gz'
    with gzip.open(stream,'wt') as output:json.dump(rows,output)
    audit=run/'continuation-audit.json';write(audit,dict(synthetic=True))
    hashes={str(path):panel.digest(path) for path in [*trial.iterdir(),*assets,audit]}
    admission=dict(schema='doorbench.isaac-released-panel-source.v1',source_engine='isaac-physx',
        source_run=str(run),source_time_s=.006,
        source_qualification=dict(kind='actual-completed-isaac-withdrawal',passed=True,time_s=.006,state_sha256='a'*64),
        extracted_state=dict(binding=dict(sha256='a'*64)),coordinate_admission=dict(passed=True),
        authorized_stages=0,physics_steps=0,source_sample_playback=0,motor_contract=motors,input_sha256=hashes)
    calls=[]
    def admit(*args,**kwargs):calls.append((args,kwargs));return copy.deepcopy(admission)
    monkeypatch.setattr(panel,'admit_isaac_panel_source',admit)
    def observations(trial_arg,audit_arg,epoch,sha):
        assert Path(trial_arg)==trial and Path(audit_arg)==audit and epoch==.006
        assert sha==panel.digest(trial/'acquisition-physics.npz')
        return dict(schema='doorbench.isaac-standing-continuation-admission.v1',
            passed=True,accounting_passed=True,source_report_passed=True,
            source_trial=str(trial),source_run=str(run),source_terminal_time_s=.006,
            physics_dt_s=.002,physical_intervals=3,observation_intervals=3,
            source_physics_sha256=sha,motor_contract_sha256=panel.motor_contract_fingerprint(motors),
            body_pose_order=list(panel.BODY_NAMES),robot_joint_names=names,
            door_joint_names=configuration['door_joint_names'],authorized_stages=0,
            physical_task_qualification=False,input_sha256=hashes.copy(),synthetic_fixture_only=True)
    monkeypatch.setattr(panel,'_admit_observations',observations)
    return dict(run=run,trial=trial,assets=assets,arrays=arrays,rows=rows,audit=audit,
        admission=admission,hashes=hashes,motors=motors,configuration=configuration,
        contract=contract,readback=readback,calls=calls,stream=stream)


def witness(s,**overrides):
    kwargs=dict(robot=s['assets'][0],door_xml=s['assets'][1],door_usd=s['assets'][2],
        expected_source_state_sha256='a'*64,stage_start_s=.006,
        runtime_configuration=copy.deepcopy(s['configuration']),runtime_motor_contract=copy.deepcopy(s['motors']),
        runtime_continuation_contract=copy.deepcopy(s['contract']),
        runtime_motor_readback_contract=copy.deepcopy(s['readback']),continuation_audit=s['audit'])
    kwargs.update(overrides)
    return panel.LiveIsaacPanelPrefixWitness(s['run'],**kwargs)


def sample(s,i):return {key:value[i].copy() for key,value in s['arrays'].items()}


def observe(w,s,i):return w.observe(sample(s,i),continuation_observation=copy.deepcopy(s['rows'][i]))


def refresh(s):
    np.savez_compressed(s['trial']/'acquisition-physics.npz',**s['arrays'])
    with gzip.open(s['stream'],'wt') as output:json.dump(s['rows'],output)
    for name in s['hashes']:s['hashes'][name]=panel.digest(name)


def test_complete_released_source_and_exact_fresh_inputs_authorize_once(source):
    s=source;before=copy.deepcopy(s['rows']);w=witness(s)
    assert len(s['calls'])==1 and not w.receipt()['passed']
    assert not hasattr(w,'_arrays')
    for i in range(3):assert observe(w,s,i) is (i==2)
    assert w.complete and not w.receipt()['stage_entry_authorized']
    receipt=w.require_stage_entry(.006)
    assert receipt['passed'] and receipt['intervals_verified']==3
    assert receipt['core']['fields']==list(panel.FIELDS)
    assert receipt['continuation']['independently_admitted'] is True
    assert 'no multiset' in receipt['continuation']['contact_order']
    assert not receipt['continuation']['historical_tensor_dtype_compared']
    receipt['input_sha256'].clear();assert w.receipt()['input_sha256']
    assert s['rows']==before
    with pytest.raises(panel.PrefixDivergenceError):w.require_stage_entry(.006)
    assert w.failed and not w.receipt()['passed']


@pytest.mark.parametrize('field',panel.FIELDS)
def test_one_bit_in_every_original_or_additional_physics_field_rejects(source,field):
    w=witness(source);row=sample(source,0);value=np.asarray(row[field]).copy()
    value.flat[0]=np.nextafter(value.flat[0],np.array(np.inf,dtype=value.dtype));row[field]=value
    with pytest.raises(panel.PrefixDivergenceError):w.observe(row,continuation_observation=source['rows'][0])
    assert w.failed and w.intervals_verified==0
    with pytest.raises(panel.PrefixDivergenceError):observe(w,source,0)
    with pytest.raises(panel.PrefixDivergenceError):w.require_stage_entry(.006)


@pytest.mark.parametrize('kind',['missing','dtype','shape','nan'])
def test_actual_field_format_failure_is_sticky(source,kind):
    w=witness(source);row=sample(source,0)
    if kind=='missing':row.pop('pre_step_joint_velocity')
    elif kind=='dtype':row['root']=row['root'].astype(np.float64)
    elif kind=='shape':row['root']=row['root'][:-1]
    else:row['root'][0]=np.nan
    with pytest.raises(panel.PrefixDivergenceError):w.observe(row,continuation_observation=source['rows'][0])
    assert w.failed and not w.complete


@pytest.mark.parametrize('kind',['normal_order','friction_slot','normal_force','bool_int','negative_zero',
    'missing','extra','nan','epoch','physical_false'])
def test_ordered_contact_or_scalar_difference_is_never_a_multiset_pass(source,kind):
    w=witness(source);row=copy.deepcopy(source['rows'][0])
    if kind=='normal_order':
        row['raw']['normal']['slots'].reverse();row['raw']['normal']['force_N'].reverse()
    elif kind=='friction_slot':row['raw']['friction']['slots']=[0]
    elif kind=='normal_force':row['raw']['normal']['force_N'][0][0]=np.nextafter(1.,np.inf)
    elif kind=='bool_int':row['evidence']['physics_qualified']=1
    elif kind=='negative_zero':row['authorized_stages']=-0.0
    elif kind=='missing':row.pop('raw')
    elif kind=='extra':row['invented']=1
    elif kind=='nan':row['raw']['normal']['force_N'][0][0]=float('nan')
    elif kind=='epoch':row['pose_time_s']=0.
    else:row['evidence']['physics_qualified']=False
    with pytest.raises(panel.PrefixDivergenceError):w.observe(sample(source,0),continuation_observation=row)
    assert w.failed and w.intervals_verified==0


def test_dictionary_key_order_only_is_canonicalized(source):
    w=witness(source);row=source['rows'][0]
    reordered={key:row[key] for key in reversed(row)}
    assert w.observe(sample(source,0),continuation_observation=reordered) is False
    w.close()


@pytest.mark.parametrize('indices',[(1,),(0,0),(0,2)])
def test_skipped_duplicate_or_out_of_order_live_interval_fails(source,indices):
    w=witness(source)
    with pytest.raises(panel.PrefixDivergenceError):
        for i in indices:observe(w,source,i)


@pytest.mark.parametrize('count,epoch',[(0,.006),(2,.006),(3,.004),(3,float('nan')),(3,True)])
def test_premature_or_stale_entry_fails_closed(source,count,epoch):
    w=witness(source)
    for i in range(count):observe(w,source,i)
    with pytest.raises(panel.PrefixDivergenceError):w.require_stage_entry(epoch)


@pytest.mark.parametrize('kind',['core_clock','core_count','core_float32_clock','core_fortran','core_missing',
    'short_json','extra_json','duplicate_json','json_nonfinite'])
def test_even_rebound_bad_stream_never_authorizes(source,kind):
    s=source;live_rows=copy.deepcopy(s['rows'])
    if kind=='core_clock':s['arrays']['time_s'][1]=.002
    elif kind=='core_count':s['arrays']['root']=s['arrays']['root'][:-1]
    elif kind=='core_float32_clock':s['arrays']['time_s']=s['arrays']['time_s'].astype(np.float32)
    elif kind=='core_fortran':s['arrays']['root']=np.asfortranarray(s['arrays']['root'])
    elif kind=='core_missing':s['arrays'].pop('standing_leaf_pose')
    elif kind=='short_json':s['rows'].pop()
    elif kind=='extra_json':s['rows'].append(copy.deepcopy(s['rows'][-1]))
    elif kind=='duplicate_json':s['rows'][1]['time_s']=.002
    else:s['rows'][1]['raw']['normal']['force_N'][0][0]=float('nan')
    refresh(s);w=witness(s)
    with pytest.raises(panel.PrefixDivergenceError):
        for i in range(3):w.observe(sample(s,i),continuation_observation=live_rows[i])
    assert not w.complete and not w.receipt()['passed']


@pytest.mark.parametrize('name',['configuration.json','acquisition-physics.npz',
    'standing-continuation-steps.json.gz','standing-continuation-contract.json','motor-readback-contract.json'])
def test_evidence_changed_after_comparison_rejects_entry(source,name):
    w=witness(source)
    for i in range(3):observe(w,source,i)
    with (source['trial']/name).open('ab') as output:output.write(b'changed')
    with pytest.raises(panel.PrefixDivergenceError):w.require_stage_entry(.006)


@pytest.mark.parametrize('kind',['schema','engine','not_released','failed','epoch','state','coordinate',
    'authority','motor','missing_binding','conflicting_hash','relative_hash'])
def test_source_cannot_be_substituted_or_claim_prior_stage_authority(source,kind):
    a=source['admission']
    if kind=='schema':a['schema']='native'
    elif kind=='engine':a['source_engine']='mujoco'
    elif kind=='not_released':a['source_qualification']['kind']='actual-transfer'
    elif kind=='failed':a['source_qualification']['passed']=False
    elif kind=='epoch':a['source_time_s']=.004
    elif kind=='state':a['extracted_state']['binding']['sha256']='b'*64
    elif kind=='coordinate':a['coordinate_admission']['passed']=False
    elif kind=='authority':a['authorized_stages']=1
    elif kind=='motor':a['motor_contract']=copy.deepcopy(a['motor_contract']);a['motor_contract']['actuators'][0]['terms']={'other':1.}
    elif kind=='missing_binding':a['input_sha256'].pop(str(source['stream']))
    elif kind=='conflicting_hash':a['input_sha256'][str(source['trial']/'acquisition-physics.npz')]='b'*64
    else:a['input_sha256']['relative']='b'*64
    with pytest.raises(ValueError):witness(source)


@pytest.mark.parametrize('kind',['coordinate','body_order','leaf_convention','capture_contract','readback_contract'])
def test_live_sensor_and_coordinate_contracts_cannot_change(source,kind):
    config=copy.deepcopy(source['configuration']);extra={}
    if kind=='coordinate':config['robot_joint_names'].reverse()
    elif kind=='body_order':config['standing_continuation_body_names'].reverse()
    elif kind=='leaf_convention':config['standing_leaf_pose_convention']='unknown'
    elif kind=='capture_contract':extra['runtime_continuation_contract']={}
    else:extra['runtime_motor_readback_contract']={}
    with pytest.raises(ValueError):witness(source,runtime_configuration=config,**extra)


def test_close_early_prevents_all_later_authorization(source):
    w=witness(source);observe(w,source,0);w.close()
    with pytest.raises(panel.PrefixDivergenceError):observe(w,source,1)
    with pytest.raises(panel.PrefixDivergenceError):w.require_stage_entry(.006)


def test_source_and_observation_admission_are_mandatory(source,monkeypatch):
    def fail(*a,**k):raise ValueError('original evidence failed')
    monkeypatch.setattr(panel,'_admit_observations',fail)
    with pytest.raises(ValueError,match='original evidence'):witness(source)


@pytest.mark.parametrize('key,value',[
    ('schema','native'),('passed',False),('accounting_passed',False),
    ('source_report_passed',False),('source_trial','elsewhere'),('source_run','elsewhere'),
    ('source_terminal_time_s',.004),('physics_dt_s',.004),('physical_intervals',2),
    ('observation_intervals',True),('source_physics_sha256','b'*64),
    ('motor_contract_sha256','b'*64),('body_pose_order',[]),('robot_joint_names',[]),
    ('door_joint_names',[]),('authorized_stages',1),('physical_task_qualification',True)])
def test_observation_receipt_cannot_hide_another_epoch_or_invent_authority(source,monkeypatch,key,value):
    original=panel._admit_observations
    def changed(*args,**kwargs):
        result=original(*args,**kwargs);result[key]=value;return result
    monkeypatch.setattr(panel,'_admit_observations',changed)
    with pytest.raises(ValueError,match='Independent observations'):witness(source)


def test_witness_comparison_never_uses_episode_sized_np_load(source,monkeypatch):
    def no_bulk(*args,**kwargs):raise AssertionError('Witness must stream NPZ members')
    monkeypatch.setattr(np,'load',no_bulk)
    w=witness(source)
    for i in range(3):observe(w,source,i)
    assert w.require_stage_entry(.006)['passed']


def test_saved_continuation_audit_changed_before_entry_fails(source):
    w=witness(source)
    for i in range(3):observe(w,source,i)
    source['audit'].write_text('changed')
    with pytest.raises(panel.PrefixDivergenceError):w.require_stage_entry(.006)


@pytest.mark.parametrize('field',['time_s','normal'])
def test_duplicate_source_json_keys_cannot_silently_collapse(source,field):
    encoded=json.dumps(source['rows'])
    if field=='time_s':encoded=encoded.replace('"time_s": 0.002','"time_s": 999, "time_s": 0.002',1)
    else:encoded=encoded.replace('"normal": {','"normal": {}, "normal": {',1)
    with gzip.open(source['stream'],'wt') as output:output.write(encoded)
    source['hashes'][str(source['stream'])]=panel.digest(source['stream'])
    w=witness(source)
    with pytest.raises(panel.PrefixDivergenceError):observe(w,source,0)
    assert w.failed and w.intervals_verified==0
