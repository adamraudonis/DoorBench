"""Original numerical/raw gates; synthetic fixtures carry no physical claim."""
import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from doorbench.dexterous.isaac_paused_transfer_audit import (
    audit_raw_pad, _mechanical, validate_retained_observers,
)
from test_isaac_release_source import window
from test_isaac_paused_transfer_source import snapshot


@pytest.fixture
def raw():return json.loads(json.dumps(window()[2][-1]))


def test_actual_raw_reduction_uses_original_classifier_and_does_not_mutate(raw):
    before=copy.deepcopy(raw)
    result=audit_raw_pad(raw,1.5,'volar-phalange-v1')
    assert result['valid'] and result['invalid_loaded_patches']==0
    assert raw==before


@pytest.mark.parametrize('change',['reported_flag','capacity','truncated','dropped','clock','pair','point',
    'derived','missing_digit','force','nondigit','profile','rawmissing'])
def test_raw_corruption_and_load_failures_reject(raw,change):
    r=raw['raw_evidence']
    if change=='reported_flag':raw['valid_pad_grasp']=False
    elif change=='capacity':r['active_contact_count']=r['contact_capacity']
    elif change=='truncated':r['truncated']=True
    elif change=='dropped':r['dropped_contacts']=1
    elif change=='clock':r['geometry_time_s']-=.002
    elif change=='pair':r['handle_pair_forces_world_N'][r['contacts'][0]['body']][0]=.0011
    elif change=='point':r['contacts'][0]['position'][0]+=.01
    elif change=='derived':raw['contacts'][0]['axial_clearance_m']+=2e-12
    elif change=='missing_digit':raw['qualified_pad_forces_N'].pop('ff')
    elif change=='force':raw['qualified_pad_forces_N']['ff']+=1.1e-6
    elif change=='nondigit':raw['non_digit_handle_force_N']=1e-8
    elif change=='profile':raw['grasp_profile']='distal-pad-v1'
    else:r['contacts'].pop()
    with pytest.raises((ValueError,KeyError)):audit_raw_pad(raw,1.5,'volar-phalange-v1')


def mechanical():
    return dict(max_joint_stop_penetration_rad=0.,max_loopback_violation_rad=0.,max_self_penetration_m=0.,
        max_nonfoot_environment_penetration_m=0.,max_hand_door_penetration_m=0.,contact_samples=251,capacity=16384,
        checks=dict(joint_stops=True,documented_loopbacks=True,self_collision=True,environment_collision=True,
            working_hand_collision=True,plant_parameters_unchanged=True),passed=True)


def test_original_mechanical_limits_are_strict():
    assert all(_mechanical(mechanical(),251).values())
    for key,limit in [('max_joint_stop_penetration_rad',.02),('max_loopback_violation_rad',.02),
                      ('max_self_penetration_m',.003),('max_nonfoot_environment_penetration_m',.003),
                      ('max_hand_door_penetration_m',.003)]:
        value=mechanical();value[key]=limit
        with pytest.raises(ValueError):_mechanical(value,251)


@pytest.mark.parametrize('change',['missing','count','plant','nan','negative'])
def test_mechanical_producer_declaration_cannot_skip_original_gate(change):
    value=mechanical()
    if change=='missing':value['checks'].pop('self_collision')
    elif change=='count':value['contact_samples']-=1
    elif change=='plant':value['checks']['plant_parameters_unchanged']=False
    elif change=='nan':value['max_hand_door_penetration_m']=float('nan')
    else:value['max_hand_door_penetration_m']=-.0001
    with pytest.raises(ValueError):_mechanical(value,251)


def observers():
    command=[.1]*61;caps=[[-2.,2.]]*61
    s=dict(live_pause=dict(epoch_s=1.5),terminal_command=dict(command_time_s=1.498,values=command))
    motors=dict(actuators=[dict(force_range=c) for c in caps])
    state=dict(schema='doorbench.paused-transfer-observers.v1',
        rest_window=dict(previous=1.5,ready=True,since=1.,verified_at=1.5),
        motor_capture=dict(previous_time=1.498,previous_command=command.copy(),step_seconds=.002,caps=caps))
    return state,s,motors


def test_original_retained_observer_state_epoch_and_command_required():
    state,s,motors=observers();before=copy.deepcopy(state)
    assert validate_retained_observers(state,s,motors)
    assert state==before


@pytest.mark.parametrize('change',['empty_rest','new_rest','rest_epoch','fresh_command','stale_command','force','caps','dt'])
def test_empty_late_constructor_or_stale_outer_cache_rejected(change):
    state,s,motors=observers()
    if change=='empty_rest':state['rest_window']['since']=None
    elif change=='new_rest':state['rest_window']['since']=1.498
    elif change=='rest_epoch':state['rest_window']['previous']=1.498
    elif change=='fresh_command':state['motor_capture']['previous_time']=1.5
    elif change=='stale_command':state['motor_capture']['previous_time']=1.496
    elif change=='force':state['motor_capture']['previous_command'][2]+=.001
    elif change=='caps':state['motor_capture']['caps']=[[-3.,3.]]*61
    else:state['motor_capture']['step_seconds']=.004
    with pytest.raises(ValueError):validate_retained_observers(state,s,motors)


def test_full_new_phase_auditor_reduces_entire_prefix_without_completed_report(snapshot,monkeypatch):
    import doorbench.dexterous.isaac_paused_transfer_audit as module
    from doorbench.dexterous.isaac_transfer_rest_stop import TransferRestStop
    # Only the earlier operation-source witness is substituted at its explicit
    # boundary; original raw, transfer, rest and complete-core reducers all run.
    marker=dict(source_qualification=dict(state_sha256='0'*64),passed=True)
    class Prefix:
        def __init__(self,*a,**k):self.complete=False
        def observe(self,row):self.complete=True
        def require_stage_entry(self,t):assert t==.002
        def receipt(self):return copy.deepcopy(marker)
    monkeypatch.setattr(module,'LiveIsaacPrefixWitness',Prefix)
    cfg=snapshot['configuration'];cfg['args'].update(standing_transfer_start_seconds=.002,
        standing_transfer_prefix_source='synthetic-prior-source')
    snapshot['write_role']('configuration',cfg)
    n=4251;t=n*.002;v=snapshot['value'];v['live_pause'].update(step_index=n,epoch_s=t)
    p=snapshot['physics']
    for key,value in list(p.items()):p[key]=np.repeat(value[:1],n,axis=0)
    p['time_s']=np.arange(1,n+1)*.002;p['root'][:,2]=1.
    p['door'][:,0]=.09;p['door'][0,1:]=[.81,.012]
    v['terminal_command'].update(command_time_s=(n-1)*.002,post_step_time_s=t)
    snapshot['write_physics']()
    template=json.loads(json.dumps(window()[2][0]));template['normal_pair_force_consistency_error_N']=0.
    detector=TransferRestStop(.002)
    padfile=snapshot['folder']/'pads.json.gz';surfacefile=snapshot['folder']/'surfaces.json.gz'
    with gzip.open(padfile,'wt') as pads,gzip.open(surfacefile,'wt') as surfaces:
        pads.write('[{"sim_time_s":0.0,"active_contact_count":0}');surfaces.write('[')
        for tick in range(1,n+1):
            now=tick*.002;pad=copy.deepcopy(template);pad.update(sim_time_s=now,physics_dt_s=.002)
            pad['raw_evidence'].update(interval_start_s=now-.002,interval_end_s=now,geometry_time_s=now)
            surface=dict(time_s=now,leaf_pose=[0.,0.,0.,1.,0.,0.,0.],stance_status='solved',started_s=.002,
                surface=dict(palm_normal_load_N=3.,body_panel_forces_world_N={'lh_palm':[0.,-3.,0.]}))
            detector.observe(now,pad,surface,dict(zip(('leaf','operator','latch'),map(float,p['door'][tick-1]))))
            pads.write(','+json.dumps(pad));surfaces.write((',' if tick>1 else '')+json.dumps(surface))
        pads.write(']');surfaces.write(']')
    for role,path in [('pad_steps',padfile),('transfer_steps',surfacefile)]:
        v['files'][role]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert detector.receipt()['terminal_time_s']==t
    phase=dict(schema=module.PHASE_SCHEMA,episode_complete=False,duration_s=t,physics_dt_s=.002,
        source_run=str(snapshot['folder'].parent/'synthetic-live-episode'),passed=True,
        checks={k:True for k in module.CHECKS},standing_transfer=dict(route=cfg['args']['standing_transfer_route'],started_s=.002),
        operation_reference=dict(operation_start_s=0.),max_motor_delivery_error_Nm=0.)
    m=mechanical();m['contact_samples']=n
    state=dict(schema='doorbench.paused-transfer-observers.v1',
        rest_window=dict(previous=t,ready=True,since=t-.5,verified_at=t),
        motor_capture=dict(previous_time=(n-1)*.002,previous_command=[0.]*61,step_seconds=.002,caps=[[-2.,2.]]*61))
    for role,document in [('phase_report',phase),('mechanical_audit',m),('source_prefix_witness',marker),
        ('reset_state',dict(door=dict(leaf_hinge=0.,leaf_handle_hinge=0.))),('observer_state',state),
        ('rest_stop',dict(schema=module.REST_SCHEMA,episode_complete=False,detector=detector.receipt()))]:
        snapshot['write_role'](role,document)
    result=module.audit_paused_isaac_transfer(snapshot['path'])
    assert result['passed'] and result['physical_intervals']==result['raw_intervals']==4251
    assert result['measured_rest']['window_samples']==251 and result['invalid_loaded_patches']==0
    assert not result['episode_complete'] and not result['whole_task_qualified']
    assert result['authorized_stages']==result['physics_steps']==0
    assert not (snapshot['folder']/'operation-report.json').exists()
    # Exercise actual writer snapshot -> new source inspection -> full phase
    # audit -> protocol publication together, using the same synthetic prefix.
    from doorbench.dexterous.bounded_evidence import BoundedEvidence
    from doorbench.dexterous.isaac_live_planning_pause import LivePlanningPause,capture_pause_anchor
    from doorbench.dexterous.isaac_live_snapshot_assembly import assemble_paused_transfer_snapshot
    folder=snapshot['folder'];files=v['files']
    route=Path(files['transfer_route']['path'])
    cfg['args']['standing_transfer_route']=str(route)
    phase['standing_transfer']['route']=str(route)
    snapshot['write_role']('configuration',cfg)
    provenance=json.loads(Path(files['provenance']['path']).read_text())
    provenance['files'][str(route)]=hashlib.sha256(route.read_bytes()).hexdigest()
    snapshot['write_role']('provenance',provenance)
    writers={name:BoundedEvidence(folder/(name+'-live')) for name in ('pad_steps','transfer_steps')}
    for name in writers:
        with gzip.open(files[name]['path'],'rt') as stream:
            for row in json.load(stream):writers[name].append(row)
    objects={'synthetic_retained_controller':object()}
    pause=LivePlanningPause(folder.parent/'planning',episode_id='synthetic-live',retained_objects=objects,timeout_seconds=60)
    anchor=capture_pause_anchor(episode_id='synthetic-live',step_index=n,epoch_s=t,physics_clock={'steps':n},
        measured={'root':p['root'][-1]},controller={'previous_command':p['motor_forces'][-1]},
        retained_objects=objects,evidence_counts=dict(physical=n,pad=n+1,transfer=n),pending_unaccepted_command=False)
    receipt=assemble_paused_transfer_snapshot(pause.directory/'source',pause_token=pause.pause_token,anchor=anchor,
        acquisition_states=p,writers=writers,source_files={name:files[name]['path'] for name in (
            'configuration','motor_contract','provenance','reset_state','transfer_route','grasp_profile_definition')},
        source_capture_directory=folder,source_run=Path(phase['source_run']),phase_checks=phase['checks'],
        standing_transfer=phase['standing_transfer'],operation_reference=phase['operation_reference'],
        max_motor_delivery_error_Nm=0.,mechanical_audit=m,rest_detector=detector.receipt(),observer_state=state,
        source_prefix_witness=marker)
    assembled=module.audit_paused_isaac_transfer(receipt['snapshot_path'])
    assert assembled['passed'] and assembled['raw_intervals']==4251
    assert not receipt['phase_qualified'] and not receipt['episode_complete']
    request=pause.publish(receipt['snapshot_path'],anchor)
    assert request['authorized_stages']==0 and pause.state=='waiting'
    for writer in writers.values():
        assert writer.exported_path is None
        writer.append({'next_episode_row':'synthetic append remains possible'})
