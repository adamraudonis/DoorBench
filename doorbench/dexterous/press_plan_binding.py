"""Bind offline press geometry to the exact acquisition that attained its grip."""
import hashlib
from pathlib import Path


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def acquisition_binding(provenance):
    parameters=provenance['parameters']
    return dict(calibration_sha256=provenance['calibration_sha256'],
        arm_schedule_sha256=provenance['arm_schedule_sha256'],
        motor_contract_sha256=provenance['motor_contract_sha256'],
        gravity_correction=parameters['gravity_correction'],
        initial_velocity=parameters['initial_velocity'],
        controller_source_sha256={n:h for n,h in provenance['sources'].items() if n.startswith('doorbench/')})


def verify_acquisition_audit(source, audit):
    """A passed audit must bind the actual states and control/provenance files."""
    result=audit['independent_evaluation'];files=audit['source_sha256']
    required={'report.json','provenance.json','trajectory.npz','controller.jsonl.gz'}
    if (result.get('passed') is not True or audit['physics_steps_in_evaluator']!=0
            or not required.issubset(files) or not result.get('checks')
            or not all(v is True for v in result['checks'].values())):
        raise ValueError('Require a complete independently passed acquisition audit')
    for name,wanted in files.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or file_sha(Path(source)/relative)!=wanted:
            raise ValueError('Changed acquisition evidence: '+name)


def require_press_binding(plan, *, calibration, schedule, motors,
                          gravity_correction, initial_velocity):
    recorded=plan.get('acquisition_binding',{}).get('controller_source_sha256',{})
    required={'doorbench/dexterous/'+n+'.py' for n in ('sensor_reach_balance','sensor_balance','sensor_acquisition_schedule','locomotion_manipulation','stance')}
    if not required.issubset(recorded) or any(not n.startswith('doorbench/') or '..' in Path(n).parts for n in recorded):
        raise ValueError('Require the original acquisition controller source closure')
    root=Path(__file__).resolve().parents[2]
    expected=dict(calibration_sha256=file_sha(calibration),arm_schedule_sha256=file_sha(schedule),
        motor_contract_sha256=file_sha(motors),gravity_correction=gravity_correction,
        initial_velocity=initial_velocity,controller_source_sha256={n:file_sha(root/n) for n in recorded})
    if plan.get('schema')!='doorbench.offline-sensor-press-plan.v2' or plan.get('acquisition_binding')!=expected:
        raise ValueError('Press plan requires the exact attained acquisition configuration; regenerate after controller changes')
