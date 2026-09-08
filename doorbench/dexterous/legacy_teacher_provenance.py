"""Narrow, explicit admission of the audited legacy Isaac002 acquisition prefix.

The original capture predates control_source. Its exact reviewed prelaunch source
package has only privileged teacher dispatch, and acquisition switches to operation
at the beginning of a physics tick. Therefore the post-step sample at the switch
clock still contains an acquisition action; the following sample is excluded.
No original report is rewritten and no failed opening label is admitted.
"""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

SCHEMA='doorbench.legacy-privileged-acquisition-prefix.v1'
APPROVED_SOURCE_ARCHIVE='08f5908c866b95a6263243e026eb60ebdb3dd556dd22229f865919a28d14672e'
SOURCE_HASHES={
    'scripts/dexterous/isaac_opening.py':'49bfdd6b8d093badcd0fdd2b2f14fb4a42001e5c20abcc40be6db062c9a89378',
    'doorbench/dexterous/acquisition_teacher.py':'71d9daccdb84ee28d62befa44e7df5b3a040c4bd983b19d6eb34c91b846cc9d0',
    'doorbench/dexterous/operation_teacher.py':'593c441f3a9e7bff8e7b8b52437e601dcfe6bc78421c64438004f7dddaa4aabe',
    'doorbench/dexterous/isaac_sensor_recording.py':'a653cb930c531690ff7969b7ce207b227a2d49437bd818c0cd4341bb50101f13',
}
FILES=('acquisition-report.json','operation-report.json','mechanical-audit.json',
    'passive-tendon-audit.json','motor-contract.json','configuration.json','provenance.json',
    'launch-source/source.tar.gz','launch-source/manifest.json','acquisition-pad-steps.json.gz',
    'acquisition-physics.npz','trace.json','sensors/report.json','sensors/layout.json',
    'sensors/actor-sensors.npz','sensors/actor-rgb.npz')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def audit_legacy_acquisition_prefix(run):
    """Recompute the source identity, physical admission, and causal cutoff."""
    run=Path(run)
    read=lambda name:json.loads((run/name).read_text())
    fingerprints={name:sha(run/name) for name in FILES}
    if fingerprints['launch-source/source.tar.gz']!=APPROVED_SOURCE_ARCHIVE:
        raise ValueError('Legacy receipt only supports the explicitly reviewed teacher source package')
    manifest=read('launch-source/manifest.json');provenance=read('provenance.json')
    if (manifest['source_archive_sha256']!=APPROVED_SOURCE_ARCHIVE or
            not manifest['captured_at_unix']<provenance['captured_before_steps_unix']):
        raise ValueError('Legacy teacher source must be captured before physical execution')
    with tarfile.open(run/'launch-source/source.tar.gz') as archive:
        hashes={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest()
            for m in archive.getmembers() if m.isfile()}
    if any(hashes.get(k)!=v for k,v in manifest['source_hashes'].items()):
        raise ValueError('Prelaunch source manifest does not match its immutable package')
    for name,expected in SOURCE_HASHES.items():
        copied=run/('source-'+Path(name).name)
        remote=[digest for path,digest in provenance['files'].items() if path.endswith('/'+name)]
        if hashes.get(name)!=expected or sha(copied)!=expected or remote!=[expected]:
            raise ValueError('Actual legacy teacher dispatch differs from the reviewed source')
        fingerprints[copied.name]=expected
    sensor=read('sensors/report.json');acq=read('acquisition-report.json');op=read('operation-report.json')
    mechanics=read('mechanical-audit.json');config=read('configuration.json');args=config['args']
    if (sensor.get('control_source') not in (None,'privileged_teacher') or
            sensor.get('capture_complete') is False or op.get('closed_loop_evaluated') is True or
            acq.get('closed_loop_evaluated') is True):
        raise ValueError('Sensor actor or incomplete capture is not legacy teacher data')
    if (acq.get('passed') is not True or mechanics.get('passed') is not True or
            not acq.get('checks') or any(v is not True for v in acq['checks'].values()) or
            op.get('checks',{}).get('motor_delivery_matches_command') is False or
            acq.get('runtime_robot_pose_writes')!=0 or acq.get('direct_door_commands') is not False or
            config.get('runtime_pose_writes')!=0 or config.get('direct_door_commands') is not False):
        raise ValueError('Legacy acquisition must independently pass physical and motor-delivery gates')
    if (args.get('acquisition') is not True or args.get('operate_after_acquisition') is not True or
            any(args.get(name,False) for name in ('mechanism_test','panel_push','grip_reset_targets','upright_gain'))):
        raise ValueError('Legacy runtime dispatch is not the reviewed unassisted acquisition')
    trace=read('trace.json')
    delivery_error=max(float(np.max(np.abs(np.asarray(r['joint_torque_command'])-r['joint_torque_sent']))) for r in trace)
    if not np.isfinite(delivery_error) or delivery_error>=1e-4:
        raise ValueError('Legacy recorded motor delivery differs from teacher commands')
    dt=float(acq['physics_dt_s']);cutoff=float(op['operation_reference']['operation_start_s'])
    if dt!=.002 or not np.isfinite(cutoff) or not .5<=cutoff<acq['duration_s']:
        raise ValueError('Invalid legacy acquisition-prefix time boundary')
    with np.load(run/'acquisition-physics.npz',allow_pickle=False) as z:physics_times=z['time_s']
    with np.load(run/'sensors/actor-sensors.npz',allow_pickle=False) as z:sensor_times=z['time_s']
    expected=np.arange(1,round(acq['duration_s']/dt)+1)*dt
    if not np.array_equal(physics_times,expected) or not np.array_equal(sensor_times,expected):
        raise ValueError('Legacy prefix requires the complete matching physical and sensor clocks')
    with gzip.open(run/'acquisition-pad-steps.json.gz','rt') as f:pads=json.load(f)
    pad_times=np.asarray([r['sim_time_s'] for r in pads])
    if not np.array_equal(pad_times,np.arange(len(expected)+1)*dt):
        raise ValueError('Legacy grasp evidence must cover every physics step')
    hold=[r for r in pads if cutoff-.5-1e-9<=r['sim_time_s']<=cutoff+1e-9]
    if len(hold)!=251 or not all(r['valid_pad_grasp'] is True for r in hold):
        raise ValueError('Legacy acquisition prefix lacks the frozen continuous 0.5-second pad hold')
    count=int(np.searchsorted(sensor_times,cutoff+1e-9,side='right'))
    if count<2 or abs(sensor_times[count-1]-cutoff)>1e-9:
        raise ValueError('Acquisition cutoff is not an actual sensor timestamp')
    return dict(schema=SCHEMA,control_source='privileged_teacher',qualification='acquisition-report.json',
        source_admission='Exact independently reviewed immutable Isaac002 teacher package; no actor dispatch exists',
        sample_end_time_s=cutoff,sensor_samples=count,supervised_examples=count-1,
        qualified_hold_start_s=hold[0]['sim_time_s'],qualified_hold_end_s=hold[-1]['sim_time_s'],
        qualified_hold_physics_samples=len(hold),excluded_first_operation_action_time_s=cutoff+dt,
        motor_delivery_sampling='50Hz recorded command-versus-sent torque; original bounded motor forces are recorded every2ms',
        maximum_recorded_delivery_error_Nm=delivery_error,
        original_operation_passed=op['passed'],files_sha256=fingerprints,
        scope='Acquisition-only immutable prefix; no opening/traversal or student-control claim')


def validate_legacy_teacher_receipt(run,receipt,*,qualification):
    if qualification!='acquisition-report.json':
        raise ValueError('Legacy receipt only admits acquisition curriculum, never failed operation data')
    recorded=json.loads(Path(receipt).read_text())
    audited=audit_legacy_acquisition_prefix(run)
    if recorded!=audited:
        raise ValueError('Legacy teacher provenance receipt differs from the original archived evidence')
    return audited
