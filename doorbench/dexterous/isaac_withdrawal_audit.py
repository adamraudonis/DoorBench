"""Independent reduction and geometry reconstruction of an actual Isaac release."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from .isaac_withdrawal_evaluation import withdrawal_checks
from .isaac_withdrawal_measurements import IsaacWithdrawalMeasurements
from .json_record_stream import iter_json_object_array
from .operation_teacher import smooth_phase
from .qualified_isaac_grasp import AUDITED_FILES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_record(row, pad, surface_row, physics, index, configuration):
    """Bind each compact release record back to the original actual streams."""
    t = float(physics['time_s'][index])
    if any(abs(float(v)-t) > 1e-8 for v in (row['sim_time_s'], pad['sim_time_s'], surface_row['time_s'])):
        raise ValueError('Release/contact/surface/physics epochs differ')
    if (row['pad_grasp'] != {k:pad[k] for k in ('valid_pad_grasp', 'contacts')}
            or row['left_surface'] != surface_row['surface']
            or row['leaf_pose'] != surface_row['leaf_pose']):
        raise ValueError('Release record differs from actual contact/surface streams')
    if not np.array_equal(np.asarray(row['leaf_pose']), physics['standing_leaf_pose'][index]):
        raise ValueError('Release leaf pose differs from synchronized physics archive')
    names = configuration['door_joint_names']
    for key, joint in (('door_q','leaf_hinge'), ('handle_angle_rad','leaf_handle_hinge'), ('bolt_slide_m','leaf_latch_bolt_slide')):
        if row[key] != float(physics['door'][index, names.index(joint)]):
            raise ValueError('Release mechanism state differs from physics archive')
    geometry = row.get('geometry')
    if geometry is not None:
        for path, xyz_xyzw in pad['raw_evidence']['body_transforms_xyzw'].items():
            name = path.rsplit('/',1)[-1]
            if name not in geometry['body_poses_xyz_wxyz']:continue
            actual = np.asarray(xyz_xyzw,float)
            measured = np.asarray(geometry['body_poses_xyz_wxyz'][name],float)
            if (np.linalg.norm(actual[:3]-measured[:3]) > 2e-6
                    or np.linalg.norm((Rotation.from_quat(actual[3:])*Rotation.from_quat(measured[[4,5,6,3]]).inv()).as_rotvec()) > 2e-6):
                raise ValueError('Geometry hand pose disagrees with synchronized raw contact body transform')


def validate_contact_receipt(contact,trial,configuration,expected):
    if (contact.get('schema') != 'doorbench.isaac-acquisition-contact-audit.v1'
            or contact.get('accounting_passed') is not True
            or contact.get('independent_raw_contact_audit_complete') is not True
            or not contact.get('checks') or not all(v is True for v in contact['checks'].values())
            or set(contact.get('input_sha256',{})) != AUDITED_FILES
            or contact.get('grasp_profile') != configuration['args']['grasp_profile']
            or contact.get('physical_intervals') != expected or contact.get('raw_intervals') != expected):
        raise ValueError('Complete trial-bound independently reconstructed contact accounting required')
    hashes={}
    for name,expected_hash in contact['input_sha256'].items():
        path=Path(trial)/name
        if sha(path)!=expected_hash:raise ValueError('Contact audit belongs to changed or different trial evidence')
        hashes[str(path.resolve())]=expected_hash
    return hashes


def validate_phase_record(row,*,start,duration,release):
    t=row['sim_time_s']-.002;teacher=row['teacher']
    if t<start-1e-8:
        if (teacher.get('withdrawal_started_s') is not None or teacher.get('release_started_s') is not None
                or teacher.get('withdrawal_progress') not in (None,0.)):
            raise ValueError('Withdrawal progress recorded before admitted source epoch')
        return release
    progress=teacher.get('withdrawal_progress')
    if (teacher.get('withdrawal_started_s')!=start or teacher.get('phase')!='standing_withdrawal'
            or type(progress) not in (int,float) or not np.isfinite(progress)
            or abs(progress-float(smooth_phase((t-start)/duration)))>1e-10):
        raise ValueError('Recorded phase clock differs from admitted actual-source route')
    actual=teacher.get('release_started_s')
    if actual is not None:
        if (type(actual) not in (int,float) or not np.isfinite(actual) or actual<start
                or actual>t+1e-8 or (release is None and abs(actual-t)>1e-8)
                or (release is not None and actual!=release)):
            raise ValueError('Intentional release epoch is stale or changes within the episode')
        release=actual
    elif release is not None:raise ValueError('Intentional release marker disappeared')
    return release


def audit_withdrawal(trial, contact_audit):
    trial = Path(trial).resolve()
    contact_audit = Path(contact_audit).resolve()
    report_path = trial/'operation-report.json'
    paths = [report_path, trial/'configuration.json', trial/'provenance.json',
        trial/'acquisition-physics.npz', trial/'acquisition-pad-steps.json.gz',
        trial/'standing-transfer-steps.json.gz', trial/'standing-withdrawal-steps.json.gz',
        trial/'live-withdrawal-prefix-witness.json', trial/'standing-withdrawal-admission.json', contact_audit]
    hashes = {str(p):sha(p) for p in paths}
    report = json.loads(report_path.read_text())
    configuration = json.loads((trial/'configuration.json').read_text())
    args = configuration['args']
    stage = report['standing_withdrawal']
    from .isaac_withdrawal_runtime import admit_isaac_withdrawal_runtime
    admission=admit_isaac_withdrawal_runtime(args['standing_withdrawal_route'])
    context=admission.source_context.data
    captured=json.loads((trial/'standing-withdrawal-admission.json').read_text())
    if (captured!={'source_context':context,'source_admission':admission.source_admission}
            or stage['source_admission']!=admission.source_admission
            or stage['started_s']!=admission.start_time
            or report['duration_s']!=admission.start_time+admission.duration
            or type(stage['completed']) is not bool):
        raise ValueError('Reported withdrawal must retain its exact admitted source, epoch and duration')
    hashes.update(admission.input_sha256)
    contact = json.loads(contact_audit.read_text())
    hashes.update(validate_contact_receipt(contact,trial,configuration,round(report['duration_s']/report['physics_dt_s'])))
    prefix = json.loads((trial/'live-withdrawal-prefix-witness.json').read_text())
    admission.authorize_source_prefix(prefix)
    prefix_pass=True
    robot = Path(args['native_robot']).resolve()
    door = Path(args['native_door']).resolve()
    provenance=json.loads((trial/'provenance.json').read_text())['files']
    bound={str(Path(p).resolve()):value for p,value in provenance.items()}
    for path,key in ((robot,'robot_path'),(door,'door_xml_path')):
        if (path!=Path(context[key]).resolve() or bound.get(str(path))!=sha(path)
                or context['input_sha256'].get(str(path))!=sha(path)):
            raise ValueError('Authored withdrawal assets differ from actual admitted/provenance identity')
    calculator = IsaacWithdrawalMeasurements(door, robot, configuration['robot_joint_names'])
    hashes.update(calculator.sources)
    maximum_clearance_error = 0.
    reconstructed = 0
    observed_release=None;last_progress=None
    with np.load(trial/'acquisition-physics.npz', allow_pickle=False) as stored:
        physics = {k:stored[k] for k in ('time_s','root','joints','door','standing_body_poses','standing_leaf_pose')}
    if len(physics['time_s']) != round(report['duration_s']/report['physics_dt_s']):
        raise ValueError('Complete physical archive required')
    with gzip.open(trial/'acquisition-pad-steps.json.gz','rt') as pad_file, \
            gzip.open(trial/'standing-transfer-steps.json.gz','rt') as surface_file, \
            gzip.open(trial/'standing-withdrawal-steps.json.gz','rt') as release_file:
        pads = iter_json_object_array(pad_file)
        initial = next(pads)
        if initial['sim_time_s'] != 0:raise ValueError('Original zero-time contact reset required')
        surfaces = iter_json_object_array(surface_file)
        releases = iter_json_object_array(release_file)
        def validated_rows():
            nonlocal maximum_clearance_error, reconstructed, observed_release,last_progress
            for index, row in enumerate(releases):
                if index >= len(physics['time_s']):raise ValueError('Extra release interval')
                try:pad, surface = next(pads), next(surfaces)
                except StopIteration as exc:raise ValueError('Truncated supporting contact stream') from exc
                validate_record(row, pad, surface, physics, index, configuration)
                observed_release=validate_phase_record(row,start=admission.start_time,duration=admission.duration,release=observed_release)
                last_progress=row['teacher'].get('withdrawal_progress')
                geometry = row.get('geometry')
                if geometry is not None:
                    t = float(physics['time_s'][index])
                    measured = calculator.read(time_s=t, pose_time_s=t, root=physics['root'][index],
                        joints=dict(zip(configuration['robot_joint_names'], physics['joints'][index])),
                        angles=dict(leaf=row['door_q'], operator=row['handle_angle_rad'], latch=row['bolt_slide_m']),
                        body_poses=geometry['body_poses_xyz_wxyz'],
                        handle_pose=physics['standing_body_poses'][index,5], leaf_pose=physics['standing_leaf_pose'][index])
                    error = abs(measured['right_environment_clearance_m']-geometry['right_environment_clearance_m'])
                    maximum_clearance_error = max(maximum_clearance_error,error)
                    if error > 1e-9 or row['right_environment_clearance_m'] != geometry['right_environment_clearance_m']:
                        raise ValueError('Reported clearance does not reproduce from actual state')
                    reconstructed += 1
                yield row
            if next(pads,None) is not None or next(surfaces,None) is not None:
                raise ValueError('Release stream omits supporting intervals')
            if (observed_release!=stage['release_started_s']
                    or stage['completed']!=(last_progress is not None and last_progress>=.999)):
                raise ValueError('Summary release epoch/completion differs from every-interval controller evidence')
        checks = withdrawal_checks(report['checks'], validated_rows(), dt=report['physics_dt_s'],
            duration=report['duration_s'], started=stage['started_s'],
            release_started=stage['release_started_s'], completed=stage['completed'])
    checks['live_exact_withdrawal_source_prefix'] = bool(prefix_pass)
    matches = all(report['checks'].get(k) == v for k,v in checks.items())
    for name, expected in hashes.items():
        if sha(name) != expected:raise ValueError('Evidence changed during withdrawal audit')
    return dict(schema='doorbench.isaac-standing-withdrawal-audit.v1',
        passed=bool(report['passed'] and matches and all(v is True for v in checks.values())
            and contact['invalid_loaded_patches'] == 0), checks=checks, producer_matches=matches,
        reconstructed_geometry_intervals=reconstructed, maximum_clearance_reconstruction_error_m=maximum_clearance_error,
        physics_steps=0, input_sha256=hashes,
        scope='Independent actual-stream reduction and unstepped authored-geometry reconstruction; initialized-standing release only, no full opening or traversal claim')
