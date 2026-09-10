"""Bind a planning endpoint to a completed, independently audited Isaac grasp.

This proves source qualification and recorded-state identity only. Original-model
coordinate admission, new-route clearance and motor-driven continuation remain
separate requirements.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .isaac_attained_state import extract_attained_state
from .standing_body_record import extract_standing_body_poses

EXTRACTED_FILES=frozenset(('configuration.json','motor-contract.json','provenance.json','acquisition-physics.npz'))
AUDITED_FILES=frozenset(('operation-report.json','acquisition-pad-steps.json.gz','configuration.json','provenance.json'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def validate_reports(report,audit,coordinator,extracted):
    if report.get('passed') is not True or not report.get('checks') or any(v is not True for v in report['checks'].values()):
        raise ValueError('Complete passing physical operation report required')
    for key in ('accounting_passed','task_passed','independent_raw_contact_audit_complete'):
        if audit.get(key) is not True:raise ValueError('Complete independent contact qualification required')
    if not audit.get('checks') or any(v is not True for v in audit['checks'].values()) or audit.get('invalid_loaded_patches')!=0:
        raise ValueError('All original contact checks and zero invalid loaded patches required')
    for key in ('passed','isaac_runtime_passed','independent_audit_passed','all_loaded_handle_patches_qualified'):
        if coordinator.get(key) is not True:raise ValueError('Completed coordinator qualification required')
    duration=report['duration_s'];dt=report['physics_dt_s'];binding=extracted['binding']
    if not np.isfinite([duration,dt,binding['time_s']]).all() or min(duration,dt)<=0:
        raise ValueError('Finite positive source clocks required')
    expected=duration/dt
    if abs(expected-round(expected))>1e-7 or audit.get('physical_intervals')!=round(expected) or audit.get('raw_intervals')!=round(expected):
        raise ValueError('Complete physical and raw interval coverage required')
    if abs(binding['time_s']-duration)>1e-10:
        raise ValueError('Only the audited terminal endpoint is admitted')
    if audit.get('final_half_second_failed_samples')!=[]:
        raise ValueError('The source endpoint must retain the original qualified hold')
    if set(extracted.get('input_sha256',{}))!=EXTRACTED_FILES or set(audit.get('input_sha256',{}))!=AUDITED_FILES:
        raise ValueError('Complete original source file bindings required')


def load_qualified_isaac_grasp(run,extracted_path):
    run=Path(run);trial=run/'trial';extracted_path=Path(extracted_path)
    report_path=trial/'operation-report.json';audit_path=run/'isaac-independent-audit.json';coordinator_path=run/'coordinator-result.json'
    tracked=[extracted_path,report_path,audit_path,coordinator_path]+[trial/n for n in sorted(EXTRACTED_FILES|AUDITED_FILES)]
    before={str(p):digest(p) for p in tracked}
    extracted=json.loads(extracted_path.read_text());report=json.loads(report_path.read_text());audit=json.loads(audit_path.read_text());coordinator=json.loads(coordinator_path.read_text())
    validate_reports(report,audit,coordinator,extracted)
    for record in (extracted,audit):
        for name,sha in record['input_sha256'].items():
            if before[str(trial/name)]!=sha:raise ValueError('Source evidence hash mismatch: '+name)
    configuration=json.loads((trial/'configuration.json').read_text());motors=json.loads((trial/'motor-contract.json').read_text());provenance=json.loads((trial/'provenance.json').read_text())
    with np.load(trial/'acquisition-physics.npz',allow_pickle=False) as physics:
        actual=extract_attained_state(configuration=configuration,motor_contract=motors,provenance=provenance,physics=physics,time_s=extracted['binding']['time_s'])
        bodies=extract_standing_body_poses(configuration,physics,time_s=actual['time_s'])
    if actual!=extracted['binding'] or bodies!=extracted.get('measured_bodies'):
        raise ValueError('Planning state differs from the exact recorded source endpoint')
    if before!={str(p):digest(p) for p in tracked}:raise ValueError('Source evidence changed during admission')
    return extracted,dict(passed=True,state_sha256=actual['sha256'],time_s=actual['time_s'],input_sha256=before,scope=__doc__.strip())
