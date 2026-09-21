"""Original transfer gates over a sealed prefix of an unfinished live episode.

The contact/rest/transfer/core reductions are independent. Physical collision
and plant-property maxima remain captured producer measurements, as in the
original completed-run qualification; they are not inferred from an image.
No process, plant, reset, command or completed-run report is created here.
"""
from collections import deque
import gzip
from pathlib import Path

import numpy as np

from .continuation_record_stream import iter_continuation_records
from .isaac_pad_audit import shadow_physx_pad_grasp
from .isaac_paused_transfer_source import inspect_paused_isaac_transfer_snapshot, _read, _verify
from .isaac_prefix_witness import LiveIsaacPrefixWitness, PREFIX_FIELDS
from .isaac_release_source import _same_contact_diagnostics, validate_rest_window
from .isaac_transfer_rest_stop import TransferRestStop
from .npz_record_stream import iter_npz_records
from .qualified_isaac_grasp import digest
from .standing_transfer_evaluation import standing_transfer_checks

SCHEMA='doorbench.paused-isaac-transfer-audit.v1'
PHASE_SCHEMA='doorbench.paused-isaac-transfer-phase.v1'
REST_SCHEMA='doorbench.paused-transfer-rest-stop.v1'
MECHANICAL_LIMITS=dict(joint_stops=('max_joint_stop_penetration_rad',.02),
    documented_loopbacks=('max_loopback_violation_rad',.02),self_collision=('max_self_penetration_m',.003),
    environment_collision=('max_nonfoot_environment_penetration_m',.003),
    working_hand_collision=('max_hand_door_penetration_m',.003))
CHECKS=frozenset((*MECHANICAL_LIMITS,'plant_parameters_unchanged','complete_physics_steps',
    'closed_leaf_start','resting_operator_start','initial_hand_door_contact_buffer_empty','finite','upright',
    'motor_delivery_matches_command','native_motor_caps','sustained_pad_grasp','acquisition_precedes_operation',
    'operator_driven_to_release','partial_leaf_opening_held','opening_bounded_for_transfer',
    'complete_transfer_clock','standing_transfer_started','measured_palm_load_accounting',
    'final_left_palm_support','stance_solves_every_interval','live_exact_source_prefix','qualified_transfer_rest_endpoint'))


def _number(value):
    if type(value) not in (int,float) or not np.isfinite(value):raise ValueError('Finite measured scalar required')
    return float(value)


def audit_raw_pad(row,t,profile):
    """Original full-handle classification/load gates, including invalid history."""
    raw=row['raw_evidence'];lever=raw['lever']
    if (row.get('grasp_profile')!=profile or row.get('hand')!='rh'
            or abs(_number(row['sim_time_s'])-t)>1e-10 or row.get('physics_dt_s')!=.002
            or raw.get('schema')!='doorbench.shadow-raw-pad-evidence.v1'
            or raw.get('clock')!='physx-interval-end' or raw.get('scope')!='complete-handle-body'
            or any(abs(_number(raw[k])-v)>=1e-9 for k,v in
                (('interval_start_s',t-.002),('interval_end_s',t),('geometry_time_s',t)))):
        raise ValueError('Complete same-epoch original selected raw pad evidence required')
    capacity,count=raw.get('contact_capacity'),raw.get('active_contact_count')
    if (type(capacity) is not int or type(count) is not int or not 0<=count<capacity
            or row.get('contact_capacity')!=capacity or row.get('active_contact_count')!=count
            or raw.get('truncated',False) or row.get('truncated',False)
            or raw.get('dropped_contacts',0) or row.get('dropped_contacts',0)
            or not isinstance(raw['contacts'],list) or len(raw['contacts'])>count):
        raise ValueError('Complete original contact buffer required')
    actual=shadow_physx_pad_grasp(raw['contacts'],raw['body_transforms_xyzw'],lever['center'],lever['axis'],
        half_length=lever['half_length'],radius=lever['radius'],profile=profile)
    if (type(row.get('valid_pad_grasp')) is not bool or actual['valid_pad_grasp']!=row['valid_pad_grasp']
            or not _same_contact_diagnostics(actual['contacts'],row.get('contacts'))):
        raise ValueError('Recorded classification differs from independent original raw reduction')
    error=0.
    for key in ('digit_forces_N','qualified_pad_forces_N'):
        if set(row.get(key,{}))!=set(actual[key]):raise ValueError('Complete digit-force accounting required')
        error=max(error,max(abs(_number(row[key][d])-v) for d,v in actual[key].items()))
    if error>=1e-6:raise ValueError('Original per-digit load accounting failed')
    pairs=raw['handle_pair_forces_world_N'];pair_error=0.
    if any(c['body'] not in pairs for c in raw['contacts']):raise ValueError('Raw patch has no pair-force counterpart')
    for body,wanted in pairs.items():
        wanted=np.asarray(wanted,float)
        force=sum((np.asarray(c['normal'])*c['normal_force_N'] for c in raw['contacts'] if c['body']==body),start=np.zeros(3))
        if wanted.shape!=(3,) or not np.isfinite(wanted).all():raise ValueError('Finite complete pair-force vector required')
        pair_error=max(pair_error,float(np.linalg.norm(force-wanted)))
    if pair_error>=1e-3:raise ValueError('Original raw patch/pair force accounting failed')
    invalid=sum(not c['pad_qualified'] and c['normal_force_N']>1e-6 for c in actual['contacts'])
    if invalid:raise ValueError('Original zero invalid-loaded-material criterion failed')
    if _number(row['non_digit_handle_force_N'])!=actual['non_digit_handle_force_N']:
        raise ValueError('Non-digit handle force accounting differs')
    return dict(valid=bool(actual['valid_pad_grasp']),invalid_loaded_patches=int(invalid),
        maximum_load_error_N=error,maximum_pair_error_N=pair_error)


def _mechanical(document,count):
    checks={key:0<=_number(document[field])<limit for key,(field,limit) in MECHANICAL_LIMITS.items()}
    checks['plant_parameters_unchanged']=document.get('checks',{}).get('plant_parameters_unchanged') is True
    if (type(document.get('contact_samples')) is not int or document['contact_samples']!=count
            or document.get('capacity')!=16384 or document.get('checks')!=checks
            or document.get('passed') is not all(checks.values())):
        raise ValueError('Original complete physical maxima and plant-invariant declaration required')
    return checks


def validate_retained_observers(document,inspection,motors):
    """Cross-bind copied state only; protocol separately verifies live objects."""
    t=inspection['live_pause']['epoch_s'];command=inspection['terminal_command']
    if document.get('schema')!='doorbench.paused-transfer-observers.v1':
        raise ValueError('Explicit retained live observer state required')
    rest=document['rest_window'];capture=document['motor_capture']
    if (rest.get('previous')!=t or rest.get('ready') is not True
            or _number(rest['since'])>t-.5+1e-8 or rest['since']<0
            or _number(rest['verified_at'])>t or rest['verified_at']<rest['since']
            or capture.get('previous_time')!=command['command_time_s']
            or capture.get('step_seconds')!=.002
            or not np.array_equal(np.asarray(capture.get('previous_command')),np.asarray(command['values']))
            or not np.array_equal(np.asarray(capture.get('caps')),np.asarray([m['force_range'] for m in motors['actuators']]))):
        raise ValueError('True retained rest and preceding outer-command observer state required')
    return True


def audit_paused_isaac_transfer(snapshot_path):
    inspected=inspect_paused_isaac_transfer_snapshot(snapshot_path);s=inspected.inspection
    paths={k:Path(v) for k,v in s['evidence_paths'].items()};hashes=s['input_sha256'].copy()
    phase=_read(paths['phase_report']);cfg=_read(paths['configuration']);args=cfg['args'];motors=_read(paths['motor_contract'])
    n=s['core_intervals'];t=s['live_pause']['epoch_s'];profile=args['grasp_profile']
    if (phase.get('schema')!=PHASE_SCHEMA or phase.get('episode_complete') is not False
            or phase.get('duration_s')!=t or phase.get('physics_dt_s')!=.002
            or set(phase.get('checks',{}))!=CHECKS or phase.get('passed') is not True
            or any(v is not True for v in phase['checks'].values())
            or type(phase.get('source_run')) is not str or not Path(phase['source_run']).is_absolute()):
        raise ValueError('Distinct passing actual transfer-prefix phase declaration required')
    checks=_mechanical(_read(paths['mechanical_audit']),n)
    reset=_read(paths['reset_state']);stage=phase['standing_transfer']
    start=args['standing_transfer_start_seconds']
    if stage['started_s']!=start or stage['route']!=args['standing_transfer_route']:
        raise ValueError('Actual declared transfer route and start must match')
    recorded_prefix=_read(paths['source_prefix_witness'])
    prefix=LiveIsaacPrefixWitness(args['standing_transfer_prefix_source'],
        expected_source_state_sha256=recorded_prefix['source_qualification']['state_sha256'],
        stage_start_s=start,runtime_configuration=cfg,runtime_motor_contract=motors)
    detector=TransferRestStop(start);rest=_read(paths['rest_stop'])
    if rest.get('schema')!=REST_SCHEMA or rest.get('episode_complete') is not False:
        raise ValueError('Actual pause rest detector, not an episode-termination receipt, required')
    shapes=dict(time_s=(),root=(13,),joints=(69,),joint_velocity=(69,),motor_forces=(61,),
        door=(3,),door_velocity=(3,),standing_body_poses=(6,7),torso_tilt_deg=())
    names=cfg['door_joint_names'];columns=[names.index(v) for v in ('leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide')]
    tail=deque(maxlen=251);maximum_leaf=-np.inf;maximum_operator=-np.inf;maximum_latch=-np.inf
    minimum_height=np.inf;maximum_tilt=0.;raw_count=0;max_load=0.;max_pair=0.;initial_empty=False
    with gzip.open(paths['pad_steps'],'rt') as pstream,gzip.open(paths['transfer_steps'],'rt') as sstream:
        pads=iter_continuation_records(pstream);surfaces=iter_continuation_records(sstream)
        initial=next(pads)
        if initial.get('sim_time_s')!=0.:raise ValueError('Original time-zero pad observation required')
        initial_empty=initial.get('active_contact_count')==0
        for index,row in enumerate(iter_npz_records(paths['physics'],shapes,expected_rows=n),start=1):
            now=index*.002
            try:pad=next(pads);surface=next(surfaces)
            except StopIteration as error:raise ValueError('Missing same-epoch contact or transfer interval') from error
            if not prefix.complete:
                prefix.observe({key:row[key] for key in PREFIX_FIELDS})
                if prefix.complete:prefix.require_stage_entry(now)
            result=audit_raw_pad(pad,now,profile);raw_count+=1
            max_load=max(max_load,result['maximum_load_error_N']);max_pair=max(max_pair,result['maximum_pair_error_N'])
            values=row['door'][columns];angles=dict(zip(('leaf','operator','latch'),map(float,values)))
            detector.observe(now,pad,surface,angles)
            tail.append((now,row['door'].copy(),pad,surface))
            maximum_leaf=max(maximum_leaf,float(values[0]));maximum_operator=max(maximum_operator,float(values[1]));maximum_latch=max(maximum_latch,float(values[2]))
            minimum_height=min(minimum_height,float(row['root'][2]));maximum_tilt=max(maximum_tilt,float(row['torso_tilt_deg']))
        if next(pads,None) is not None or next(surfaces,None) is not None:raise ValueError('Extra contact or transfer intervals')
    if prefix.receipt()!=recorded_prefix:raise ValueError('Original independently reconstructed operation prefix differs')
    actual_rest=detector.receipt()
    if (actual_rest!=rest.get('detector') or not actual_rest['triggered']
            or actual_rest['terminal_time_s']!=t or actual_rest['observed_samples']!=n):
        raise ValueError('Snapshot must end at the actual first whole joint rest window')
    measured_rest=validate_rest_window(np.asarray([r[0] for r in tail]),np.asarray([r[1] for r in tail]),names,
        [r[2] for r in tail],[r[3] for r in tail],terminal=t,profile=profile)
    with gzip.open(paths['transfer_steps'],'rt') as stream:
        checks.update(standing_transfer_checks(iter_continuation_records(stream),seconds=t,dt=.002,started_s=stage['started_s']))
    operation_start=phase['operation_reference']['operation_start_s']
    checks.update(complete_physics_steps=raw_count==n,closed_leaf_start=abs(_number(reset['door']['leaf_hinge']))<=.001,
        resting_operator_start=abs(_number(reset['door']['leaf_handle_hinge']))<=.001,
        initial_hand_door_contact_buffer_empty=initial_empty,finite=True,upright=maximum_tilt<12 and minimum_height>.7,
        motor_delivery_matches_command=0<=_number(phase['max_motor_delivery_error_Nm'])<1e-4,
        native_motor_caps=s['original_recorded_command_caps_checked'],sustained_pad_grasp=True,
        acquisition_precedes_operation=operation_start is not None and 0<=_number(operation_start)<start,
        operator_driven_to_release=maximum_operator>=.80 and maximum_latch>=.011,
        partial_leaf_opening_held=all(.075<=r[1][columns[0]]<=.10 for r in tail),opening_bounded_for_transfer=maximum_leaf<=.12,
        live_exact_source_prefix=prefix.receipt()['passed'],qualified_transfer_rest_endpoint=True)
    if set(checks)!=CHECKS or checks!=phase['checks'] or not all(checks.values()):
        raise ValueError('Every original actual transfer-prefix gate must independently match')
    validate_retained_observers(_read(paths['observer_state']),s,motors)
    for name in ('isaac_paused_transfer_audit.py','continuation_record_stream.py','isaac_pad_audit.py',
                 'isaac_release_source.py','isaac_transfer_rest_stop.py','standing_transfer_evaluation.py','grasp_verification.py'):
        path=Path(__file__).with_name(name);actual=digest(path)
        if str(path) in hashes and hashes[str(path)]!=actual:raise ValueError('Consumed audit helper changed')
        hashes[str(path)]=actual
    _verify(hashes)
    return dict(schema=SCHEMA,passed=True,source_kind=s['source_kind'],source_engine='isaac-physx',
        source_run=phase['source_run'],snapshot_path=s['snapshot_path'],snapshot_sha256=s['snapshot_sha256'],
        live_pause=s['live_pause'],time_s=t,state_sha256=s['source_state_sha256'],checks=checks,
        physical_intervals=n,raw_intervals=raw_count,invalid_loaded_patches=0,
        independent_raw_contact_audit_complete=True,maximum_load_accounting_error_N=max_load,
        maximum_raw_pair_force_error_N=max_pair,first_rest_detector=actual_rest,measured_rest=measured_rest,
        endpoint_pad=tail[-1][2],original_operation_prefix=recorded_prefix,
        episode_complete=False,whole_task_qualified=False,live_process_state_verified=False,
        authorized_stages=0,physics_steps=0,active_state_writes=0,input_sha256=hashes,
        scope='Fresh original transfer-prefix physical-summary/contact/rest/endpoint accounting only; no completed-run receipt, live resume authority, release or full-task success')
