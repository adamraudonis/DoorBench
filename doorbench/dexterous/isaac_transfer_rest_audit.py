"""Reconstruct the first prospective transfer hold from recorded observations."""
import gzip
import json
from pathlib import Path

import numpy as np

from .json_record_stream import iter_json_object_array
from .qualified_isaac_grasp import digest


REST_FILES=('operation-report.json','configuration.json','standing-transfer-steps.json.gz',
    'acquisition-pad-steps.json.gz','acquisition-physics.npz','standing-transfer-rest-stop.json')
AUDIT_CHECKS=frozenset(('prospective_receipt_matches','first_joint_rest_triggered',
    'declared_transfer_entry','actual_stage_outcome','source_unchanged'))


def audit_transfer_rest_stop(trial):
    from .isaac_transfer_rest_stop import TransferRestStop
    trial=Path(trial)
    inputs={str((trial/name).resolve()):digest(trial/name) for name in REST_FILES}
    configuration=json.loads((trial/'configuration.json').read_text())
    args=configuration['args']
    report=json.loads((trial/'operation-report.json').read_text())
    recorded=json.loads((trial/'standing-transfer-rest-stop.json').read_text())
    if args.get('standing_transfer_stop_on_rest') is not True:
        raise ValueError('A prospective transfer stop declaration is required')
    if recorded!=report.get('standing_transfer_rest_stop'):
        raise ValueError('Report and actual transfer stop receipt differ')
    expected_mode='prefix-only' if args.get('standing_withdrawal_route') else 'terminate'
    if (recorded.get('schema')!='doorbench.isaac-transfer-rest-stop-run.v1'
            or recorded.get('maximum_seconds')!=args['seconds'] or recorded.get('mode')!=expected_mode):
        raise ValueError('Transfer stop declaration differs from the executed mode and deadline')
    detector=TransferRestStop(args['standing_transfer_start_seconds'],dt=report['physics_dt_s'])
    names=configuration['door_joint_names']
    columns=[names.index(name) for name in ('leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide')]
    verified=0
    with np.load(trial/'acquisition-physics.npz',allow_pickle=False) as physics:
        times=physics['time_s'];doors=physics['door']
        if (len(times)!=round(report['duration_s']/report['physics_dt_s'])
                or len(doors)!=len(times) or not np.isfinite(times).all()
                or not np.isfinite(doors).all() or times[-1]!=report['duration_s']):
            raise ValueError('Complete actual physical clock required for transfer stop audit')
        with gzip.open(trial/'acquisition-pad-steps.json.gz','rt') as pads_file, \
                gzip.open(trial/'standing-transfer-steps.json.gz','rt') as surfaces_file:
            pads=iter_json_object_array(pads_file);surfaces=iter_json_object_array(surfaces_file)
            initial=next(pads)
            if initial.get('sim_time_s')!=0.:
                raise ValueError('Original initial grasp observation required')
            for t,door in zip(times,doors):
                pad=next(pads);surface=next(surfaces)
                angles=dict(zip(('leaf','operator','latch'),map(float,door[columns])))
                triggered=detector.observe(float(t),pad,surface,angles)
                verified+=1
                if triggered:break
    reconstructed=detector.receipt()
    matches=reconstructed==recorded.get('detector')
    terminal=reconstructed.get('terminal_time_s')
    actual_start=report.get('standing_transfer',{}).get('started_s')
    declared_start=args['standing_transfer_start_seconds']
    entry_matches=actual_start==declared_start
    if expected_mode=='terminate':
        outcome=(recorded.get('terminated_on_qualified_rest') is True
            and recorded.get('continued_to_withdrawal') is False and terminal==report['duration_s'])
    else:
        outcome=(recorded.get('terminated_on_qualified_rest') is False
            and recorded.get('continued_to_withdrawal') is True
            and terminal==report.get('standing_withdrawal',{}).get('started_s'))
    unchanged=inputs=={str((trial/name).resolve()):digest(trial/name) for name in REST_FILES}
    checks=dict(prospective_receipt_matches=matches,first_joint_rest_triggered=reconstructed.get('triggered') is True,
        declared_transfer_entry=entry_matches,actual_stage_outcome=outcome,source_unchanged=unchanged)
    return dict(schema='doorbench.isaac-transfer-rest-stop-audit.v1',passed=all(checks.values()),
        checks=checks,producer_matches=matches,intervals_reconstructed=verified,
        detector=reconstructed,recorded=recorded,input_sha256=inputs,physics_steps=0,
        scope='First qualifying measured hold replay; full raw-contact and transfer audits remain separately required')


def validate_transfer_rest_evidence(report,audit,trial):
    """Bind optional termination proof to the original, unchanged source files."""
    if 'standing_transfer_rest_stop' not in report:return
    from .isaac_transfer_rest_stop import CHECK_NAMES,SCHEMA
    evidence=audit.get('rest_stop',{})
    checks=evidence.get('checks',{})
    detector=evidence.get('detector',{})
    terminal=detector.get('terminal_time_s')
    complete_detector=(detector.get('schema')==SCHEMA and detector.get('triggered') is True
        and detector.get('failure') is None and detector.get('dt')==.002
        and detector.get('window_samples')==251 and detector.get('required_window_samples')==251
        and set(detector.get('checks',{}))==set(CHECK_NAMES)
        and all(value is True for value in detector.get('checks',{}).values())
        and isinstance(terminal,(int,float)) and not isinstance(terminal,bool)
        and np.isfinite(terminal) and terminal>0
        and type(detector.get('observed_samples')) is int
        and detector['observed_samples']==round(terminal/.002)
        and evidence.get('intervals_reconstructed')==detector['observed_samples'])
    if (evidence.get('schema')!='doorbench.isaac-transfer-rest-stop-audit.v1'
            or evidence.get('passed') is not True or evidence.get('producer_matches') is not True
            or set(checks)!=AUDIT_CHECKS or any(value is not True for value in checks.values())
            or evidence.get('recorded')!=report['standing_transfer_rest_stop']
            or not complete_detector
            or detector!=report['standing_transfer_rest_stop'].get('detector')):
        raise ValueError('Independently reconstructed prospective transfer stop required')
    recorded={str(Path(name).resolve()):sha for name,sha in evidence.get('input_sha256',{}).items()}
    expected={str((Path(trial)/name).resolve()):digest(Path(trial)/name) for name in REST_FILES}
    if recorded!=expected:raise ValueError('Transfer stop proof differs from the actual source evidence')
