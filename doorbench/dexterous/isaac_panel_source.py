"""Admit a completed actual Isaac release to detached panel geometry only.

This source kind intentionally has no final-grasp requirement. The original
phase-specific withdrawal audit, raw contacts, support and clearance still have
to pass in full. No native history, source playback or stage permission exists.
"""
from collections import deque
import copy
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .destination_planner_admission import admit_destination_planner
from .grasp_verification import grasp_profile
from .isaac_attained_state import extract_attained_state
from .isaac_prefix_witness import _historical_inputs, COMMAND_SEMANTICS
from .isaac_withdrawal_audit import audit_withdrawal
from .json_record_stream import iter_json_object_array
from .landed_left_planner import LandedLeftScene
from .motor_contract_identity import motor_contract_fingerprint
from .qualified_isaac_grasp import digest, validate_transfer_evidence
from .standing_body_record import extract_standing_body_poses
from .standing_withdrawal_audit import clearance_pairs


SCHEMA = 'doorbench.isaac-released-panel-source.v1'
DT = .002
REQUIRED_CHECKS = frozenset((
    'joint_stops', 'documented_loopbacks', 'self_collision', 'environment_collision',
    'working_hand_collision', 'plant_parameters_unchanged', 'complete_physics_steps',
    'closed_leaf_start', 'resting_operator_start', 'initial_hand_door_contact_buffer_empty',
    'finite', 'upright', 'acquisition_precedes_operation', 'operator_driven_to_release',
    'complete_transfer_clock', 'standing_transfer_started', 'live_exact_source_prefix',
    'partial_opening_held_before_intentional_release', 'opening_bounded_before_intentional_release',
    'resting_grip_before_withdrawal', 'opposed_grip_before_intentional_release',
    'left_support_before_intentional_release', 'no_invalid_loaded_right_surfaces',
    'withdrawal_route_completed', 'final_hand_clear_of_environment',
    'leaf_remains_open_after_withdrawal', 'complete_withdrawal_clock',
    'measured_palm_load_accounting', 'synchronized_withdrawal_geometry',
    'resting_palm_support_before_withdrawal', 'final_left_palm_support',
    'live_exact_withdrawal_source_prefix', 'native_motor_caps',
    'motor_delivery_matches_command', 'stance_solves_every_interval'))


def _finite(value):
    return type(value) in (int, float) and np.isfinite(value)


def _track(hashes, path, expected=None):
    path = Path(path).resolve()
    actual = digest(path)
    if (expected is not None and actual != expected) or (str(path) in hashes and hashes[str(path)] != actual):
        raise ValueError('Released-source input changed or has conflicting identity: ' + str(path))
    hashes[str(path)] = actual
    return actual


def _pose(value):
    pose = np.asarray(value, float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1.) > 2e-6:
        raise ValueError('Actual finite normalized body pose required')
    return pose


def validate_released_window(rows, *, terminal):
    """Recheck the original inclusive 0.5 s window without a grasp predicate."""
    if not _finite(terminal) or terminal < .5 or len(rows) != 251:
        raise ValueError('Complete actual 251-sample released endpoint required')
    loads, clearances = [], []
    for index, row in enumerate(rows):
        time = row['sim_time_s']
        if not _finite(time) or abs(time-(terminal-.5+index*DT)) > 1e-8:
            raise ValueError('Exact contiguous released-window epochs required')
        pose = _pose(row['leaf_pose'])
        force = np.asarray(row['left_surface']['body_panel_forces_world_N']['lh_palm'], float)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError('Actual palm-only force vector required')
        normal = -Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()[:, 1]
        load = max(0., float(normal@force))
        declared = row['left_surface']['palm_normal_load_N']
        if not _finite(declared) or abs(load-declared) > 1e-8 or load < 2.:
            raise ValueError('Original actual palm-only support >=2 N required')
        geometry = row['geometry']
        if (not isinstance(geometry, dict) or geometry.get('native_mirror_steps') != 0
                or any(not _finite(geometry.get(k)) or abs(geometry[k]-time) > 1e-8
                       for k in ('time_s', 'geometry_time_s'))):
            raise ValueError('Same-epoch unstepped actual clearance evidence required')
        clearance = row['right_environment_clearance_m']
        if (not _finite(clearance) or clearance < .04
                or clearance != geometry.get('right_environment_clearance_m')):
            raise ValueError('Original complete RH/environment clearance >=40 mm required')
        if not _finite(row['door_q']) or row['door_q'] < .075:
            raise ValueError('Original leaf-remains-open endpoint required')
        if row['teacher']['stance_status'] not in ('solved', 'solved inaccurate'):
            raise ValueError('Original solved stance required throughout the endpoint')
        # Full independent reconstruction remains mandatory in admission.
        if any(c.get('pad_qualified') is not True for c in row['pad_grasp']['contacts']):
            raise ValueError('Original no-invalid-right-surfaces gate required')
        loads.append(load); clearances.append(clearance)
    return dict(window_samples=251, window_start_s=terminal-.5, terminal_time_s=terminal,
        minimum_palm_load_N=min(loads), minimum_right_environment_clearance_m=min(clearances),
        final_grasp_required=False, scope='Actual inclusive original released support and clearance window')


def complete_measured_bodies(core, leaf_pose, geometry_poses, *, required_names):
    """Join recorded origins, rejecting conflicting duplicate measurements."""
    if (len(core['body_names']) != len(set(core['body_names']))
            or len(core['body_names']) != len(core['body_poses_xyz_wxyz'])):
        raise ValueError('Unique complete core body inventory required')
    values = dict(zip(core['body_names'], core['body_poses_xyz_wxyz']))
    def add(name, pose):
        actual = _pose(pose)
        if name in values and not np.array_equal(actual, _pose(values[name])):
            raise ValueError('Same-epoch measured body records disagree: ' + name)
        values[name] = actual.tolist()
    add('leaf', leaf_pose)
    for name, pose in geometry_poses.items():
        matches = [key for key in (name, 'robot/'+name) if key in required_names]
        if len(matches) != 1:
            raise ValueError('Unambiguous original measured body identity required: ' + name)
        add(matches[0], pose)
    if not set(required_names) <= set(values):
        raise ValueError('Complete relevant measured body inventory required')
    names = sorted(values)
    return dict(geometry_time_s=core['geometry_time_s'], body_names=names,
        body_poses_xyz_wxyz=[_pose(values[name]).tolist() for name in names])


def actual_motor_handoff(physics, motors, *, terminal):
    """The final submitted command is evidence, never a playback controller."""
    time = np.asarray(physics['time_s'])
    values = np.asarray(physics['motor_forces'])
    actuators = motors.get('actuators', [])
    names = [motor['name'] for motor in actuators]
    if (len(names) != 61 or len(set(names)) != 61 or time.ndim != 1 or not len(time)
            or abs(float(time[-1])-terminal) > 1e-8
            or values.shape != (len(time), 61) or not np.isfinite(values).all()):
        raise ValueError('Complete original 61-command archive required')
    ranges = np.asarray([m['force_range'] for m in actuators], float)
    if (ranges.shape != (61, 2) or not np.isfinite(ranges).all()
            or np.any(ranges[:, 0] >= ranges[:, 1])
            or np.any(values < ranges[:, 0]-1e-5) or np.any(values > ranges[:, 1]+1e-5)):
        raise ValueError('Original motor force caps required for every command')
    return dict(command_time_s=terminal-DT, interval_end_s=terminal,
        command_Nm=values[-1].tolist(), motor_names=names,
        motor_contract_sha256=motor_contract_fingerprint(motors),
        semantics=COMMAND_SEMANTICS, controller_internal_state_reconstructed=False,
        scope='Recorded predecessor output only; continuation must inherit live feedback/stance state')


def _launched_arguments(argv, configuration, trial, duration):
    """Bind all explicitly declared arguments to the recorded producer values."""
    if not isinstance(argv, list) or len(argv) < 3 or not all(type(v) is str for v in argv):
        raise ValueError('Original explicit Isaac producer command required')
    producer=2 if argv[1]=='-u' else 1
    if Path(argv[producer]).name != 'isaac_opening.py':
        raise ValueError('Original explicit Isaac producer command required')
    seen=set(); i=producer+1
    while i < len(argv):
        flag=argv[i]
        if not flag.startswith('--') or flag in seen:
            raise ValueError('Unique explicit launch flags required')
        seen.add(flag); key=flag[2:].replace('-', '_'); i+=1; tokens=[]
        while i<len(argv) and not argv[i].startswith('--'):
            tokens.append(argv[i]); i+=1
        if key not in configuration:raise ValueError('Launch option absent from actual configuration: '+flag)
        actual=configuration[key]
        if type(actual) is bool:
            matches=not tokens and actual is True
        elif type(actual) in (int,float):
            matches=len(tokens)==1 and _finite(actual) and float(tokens[0])==actual
        elif isinstance(actual,list):
            matches=len(tokens)==len(actual) and all(
                (type(v) in (int,float) and _finite(v) and float(token)==v)
                or (type(v) is str and token==v) for token,v in zip(tokens,actual))
        elif type(actual) is str:
            matches=len(tokens)==1 and (tokens[0]==actual or (
                Path(actual).is_absolute() and Path(tokens[0]).is_absolute()
                and Path(actual).resolve()==Path(tokens[0]).resolve()))
        else:matches=False
        if not matches:raise ValueError('Actual producer configuration differs from launch: '+flag)
    required={'--standing-transfer-route','--standing-transfer-prefix-source',
        '--standing-withdrawal-route','--native-door','--output','--seconds'}
    if (not required<=seen or Path(configuration['output']).resolve()!=trial
            or configuration['seconds']!=duration):
        raise ValueError('Complete explicit continuous-withdrawal launch and duration required')
    for flag in ('standing_transfer_stop_on_rest','standing_transfer_hybrid_support'):
        if configuration.get(flag,False) != ('--'+flag.replace('_','-') in seen):
            raise ValueError('Prospective transfer mode differs from actual launch')
    return str(Path(argv[producer]).resolve())


def validate_local_completion(source, configuration, provenance, report, hashes):
    """Require the actual completed local process and every original conjunction."""
    trial=source/'trial'; launch_path=source/'launch.json'; result_path=source/'result.json'
    for path in (launch_path,result_path):_track(hashes,path)
    launch=json.loads(launch_path.read_text()); result=json.loads(result_path.read_text())
    required=('passed','physics_started','runtime_passed','runtime_binding_passed',
        'independent_passed','independent_transfer_passed','source_prefix_passed',
        'independent_withdrawal_passed','withdrawal_source_prefix_passed','report_emitted')
    codes=('preflight_returncode','returncode','audit_returncode',
        'transfer_audit_returncode','withdrawal_audit_returncode')
    if (any(result.get(k) is not True for k in required)
            or any(type(result.get(k)) is not int or result[k]!=0 for k in codes)
            or any(v for k,v in result.items() if k.endswith('_error_type') or k in ('timeout','interrupted'))
            or not _finite(result.get('finished_unix')) or not _finite(result.get('started_unix'))
            or result['finished_unix']<result['started_unix']
            or any((trial/name).exists() for name in ('early-stop.json','error.txt'))):
        raise ValueError('Successful finalized local process and all independent stage gates required')
    mutable={'passed','runtime_passed','independent_passed'}
    if (not launch.get('physics_started') or not {'argv','input_sha256','source_sha256',
            'runtime_source_sha256','started_unix','pid'}<=set(launch)
            or any(result.get(k)!=v for k,v in launch.items() if k not in mutable)):
        raise ValueError('Final result must retain the exact actual launch identity')
    producer=_launched_arguments(result['argv'],configuration['args'],trial,report['duration_s'])
    for key,name in (('report_sha256','trial/operation-report.json'),
                     ('audit_sha256','independent-contact-audit.json')):
        if result.get(key)!=digest(source/name):raise ValueError('Final receipt binds different actual evidence')
    sources=result['source_sha256']; runtime=result['runtime_source_sha256']; inputs=result['input_sha256']
    if not sources or not runtime or not inputs or not set(runtime)<=set(sources):
        raise ValueError('Complete original launch input and helper inventory required')
    recorded={str(Path(p).resolve()):value for p,value in provenance['files'].items()}
    if {p:v for p,v in recorded.items() if Path(p).suffix=='.py'}!=runtime:
        raise ValueError('Captured producer runtime inventory differs from launch')
    captures={}; basenames=set()
    for original,expected in sources.items():
        path=Path(original)
        if not path.is_absolute() or path.suffix!='.py' or path.name in basenames:
            raise ValueError('Unambiguous absolute original launch source inventory required')
        basenames.add(path.name); captured=source/('source-'+path.name)
        if captured.is_symlink() or captured.resolve().parent!=source:
            raise ValueError('Original launcher source must be captured locally')
        _track(hashes,captured,expected); captures[str(path.resolve())]=captured
        if original in runtime and runtime[original]!=expected:
            raise ValueError('Runtime source digest differs within original launch')
    if producer not in runtime:
        raise ValueError('Actual producer source is absent from runtime inventory')
    for original,expected in inputs.items():
        path=Path(original)
        if not path.is_absolute():raise ValueError('Absolute original launch input identities required')
        canonical=str(path.resolve())
        if canonical in recorded and recorded[canonical]!=expected:
            raise ValueError('Actual shared input differs from launch')
        _track(hashes,captures.get(canonical,path),expected)
    from scripts.isaac.run_local_operation import audit_transfer_prefix, audit_withdrawal_prefix
    from .isaac_transfer_rest_audit import validate_transfer_rest_evidence
    args=configuration['args']
    captured=json.loads((trial/'standing-withdrawal-admission.json').read_text())['source_context']
    prefix_calls=(('source-prefix-audit.json',audit_transfer_prefix,args['standing_transfer_prefix_source'],args['standing_transfer_start_seconds']),
        ('withdrawal-source-prefix-audit.json',audit_withdrawal_prefix,captured['source_run'],captured['start_time_s']))
    for name,audit,before,start in prefix_calls:
        path=source/name;_track(hashes,path);saved=json.loads(path.read_text())
        actual=audit(before,trial,start)
        if actual.get('passed') is not True or actual!=saved:
            raise ValueError('Original independently repeated full physical prefix required')
        for name,expected in actual['input_sha256'].items():_track(hashes,name,expected)
    path=source/'isaac-transfer-audit.json';_track(hashes,path);transfer=json.loads(path.read_text())
    expected={str(trial/name):digest(trial/name) for name in ('operation-report.json','standing-transfer-steps.json.gz')}
    validate_transfer_evidence(report,result,transfer,expected)
    for name,value in expected.items():_track(hashes,name,value)
    if args.get('standing_transfer_stop_on_rest'):
        validate_transfer_rest_evidence(report,transfer,trial)
    for name in ('standing-withdrawal-admission.json','live-prefix-witness.json','live-withdrawal-prefix-witness.json'):
        _track(hashes,trial/name)
    _track(hashes,Path(__file__).resolve().parents[2]/'scripts/isaac/run_local_operation.py')
    return dict(passed=True,launch_sha256=hashes[str(launch_path)],result_sha256=hashes[str(result_path)],
        full_prefixes_recomputed=True,physics_process_returncode=0,
        scope='Original completed local launch, archived source inventory and independent stage conjunction')


def admit_isaac_panel_source(source, *, robot, door_xml, door_usd):
    source = Path(source).resolve(); trial = source/'trial'
    assets = dict(robot_path=Path(robot).resolve(), door_xml_path=Path(door_xml).resolve(),
        door_usd_path=Path(door_usd).resolve())
    contact_path = source/'independent-contact-audit.json'
    audit_path = source/'isaac-withdrawal-audit.json'
    paths = [trial/name for name in ('operation-report.json', 'configuration.json', 'provenance.json',
        'motor-contract.json', 'acquisition-physics.npz', 'standing-withdrawal-steps.json.gz')]
    hashes = {}
    for path in [*paths, contact_path, audit_path]: _track(hashes, path)
    report = json.loads(paths[0].read_text())
    configuration = json.loads(paths[1].read_text()); args = configuration['args']
    provenance = json.loads(paths[2].read_text())
    motors = json.loads(paths[3].read_text())
    checks = report.get('checks', {})
    stage = report.get('standing_withdrawal', {})
    if (report.get('passed') is not True or not REQUIRED_CHECKS <= set(checks)
            or any(value is not True for value in checks.values())
            or stage.get('completed') is not True or not args.get('standing_withdrawal_route')
            or report.get('physics_dt_s') != DT or report.get('runtime_robot_pose_writes') != 0
            or report.get('direct_door_commands') is not False):
        raise ValueError('Complete original passing actual Isaac withdrawal required')
    terminal = report['duration_s']
    if not _finite(terminal) or terminal < .5 or abs(terminal/DT-round(terminal/DT)) > 1e-7:
        raise ValueError('Original finite 500 Hz terminal epoch required')
    profile = args['grasp_profile']; grasp_profile(profile)
    if report.get('grasp_profile') != profile:
        raise ValueError('Actual prospective contact profile must remain unchanged')
    completion=validate_local_completion(source,configuration,provenance,report,hashes)
    # Recompute the entire actual phase-specific proof; a saved passing label is
    # never authority. This fresh audit admits runtime+source+exact live prefix,
    # reconstructs raw/contact accounting and every authored-clearance epoch.
    fresh = audit_withdrawal(trial, contact_path)
    recorded = json.loads(audit_path.read_text())
    if (fresh.get('schema') != 'doorbench.isaac-standing-withdrawal-audit.v1'
            or fresh.get('passed') is not True or fresh.get('producer_matches') is not True
            or fresh.get('checks') != checks or fresh != recorded):
        raise ValueError('Matching fresh independent withdrawal qualification required')
    for path, expected in fresh['input_sha256'].items(): _track(hashes, path, expected)
    historical, captures = _historical_inputs(trial, provenance)
    for path, expected in historical.items(): _track(hashes, path, expected)
    bound = {str(Path(p).resolve()): value for p, value in provenance['files'].items()}
    for key, arg in (('robot_path', 'native_robot'), ('door_xml_path', 'native_door'), ('door_usd_path', 'door_usd')):
        path = assets[key]
        if Path(args[arg]).resolve() != path or bound.get(str(path)) != digest(path):
            raise ValueError('Exact original recorded panel assets required')
        _track(hashes, path)
    declaration = trial/'grasp-profile-definition.json'
    original = Path(args['grasp_profile_definition']).resolve()
    spec = json.loads(declaration.read_text())
    if (spec.get('profile') != profile or spec.get('robot_xml_sha256') != digest(assets['robot_path'])
            or digest(declaration) != digest(original) or bound.get(str(original)) != digest(declaration)):
        raise ValueError('Original run-bound prospective contact declaration required')
    _track(hashes, declaration); _track(hashes, original)
    continuation = trial/'standing-continuation-steps.json.gz'
    extra_evidence = None
    if continuation.exists():
        extra_evidence = dict(path=str(continuation), sha256=_track(hashes, continuation),
            independently_admitted=False,
            scope='Additional captured continuation input bound only; later live-stage admission remains separate')
    with gzip.open(paths[5], 'rt') as stream:
        rows = list(deque(iter_json_object_array(stream), maxlen=251))
    window = validate_released_window(rows, terminal=terminal)
    with np.load(paths[4], allow_pickle=False) as physics:
        times = physics['time_s']
        if (times.shape != (round(terminal/DT),) or not np.isfinite(times).all()
                or not np.allclose(times, (np.arange(len(times))+1)*DT, rtol=0, atol=1e-8)):
            raise ValueError('Complete original physical episode clock required')
        binding = extract_attained_state(configuration=configuration, motor_contract=motors,
            provenance=provenance, physics=physics, time_s=terminal)
        core = extract_standing_body_poses(configuration, physics, time_s=terminal)
        leaf = np.asarray(physics['standing_leaf_pose'])
        if leaf.shape != (len(times), 7) or not np.array_equal(leaf[-1], np.asarray(rows[-1]['leaf_pose'])):
            raise ValueError('Exact actual terminal leaf-body archive required')
        handoff = actual_motor_handoff(physics, motors, terminal=terminal)
    if (binding['robot_source_sha256'] != digest(assets['robot_path'])
            or binding['door_source_sha256'] != digest(assets['door_usd_path'])):
        raise ValueError('Actual endpoint asset binding differs')
    scene = LandedLeftScene(assets['robot_path'], assets['door_xml_path'])
    required = set(core['body_names']) | {'leaf'}
    for g, h in clearance_pairs(scene.m):
        for body in (int(scene.m.geom_bodyid[g]), int(scene.m.geom_bodyid[h])):
            if body: required.add(scene.m.body(body).name)
    complete = complete_measured_bodies(core, leaf[-1], rows[-1]['geometry']['body_poses_xyz_wxyz'], required_names=required)
    extracted = dict(binding=binding, measured_bodies=complete)
    measured, coordinates = admit_destination_planner(scene.m, extracted,
        motor_contract=motors, door_source_sha256=digest(assets['door_usd_path']))
    for name in ('isaac_panel_source.py', 'isaac_attained_state.py', 'standing_body_record.py',
            'destination_planner_admission.py', 'destination_planning_coordinates.py',
            'destination_return_kinematics.py', 'destination_state_binding.py',
            'motor_contract_identity.py', 'landed_left_planner.py', 'robot_design_identity.py',
            'robot_identity.py', 'standing_withdrawal_audit.py', 'isaac_withdrawal_audit.py',
            'json_record_stream.py', 'qualified_isaac_grasp.py', 'isaac_prefix_witness.py',
            'grasp_verification.py'):
        _track(hashes, Path(__file__).with_name(name))
    for path, expected in hashes.items():
        if digest(path) != expected: raise ValueError('Released source changed during admission')
    return dict(schema=SCHEMA, source_engine='isaac-physx', source_run=str(source),
        source_time_s=terminal, grasp_profile=profile, initial_qpos=measured.qpos.tolist(),
        initial_qvel=measured.qvel.tolist(), extracted_state=extracted,
        source_qualification=dict(passed=True, kind='actual-completed-isaac-withdrawal',
            time_s=terminal, state_sha256=binding['sha256'], withdrawal_audit=recorded,
            local_process_completion=completion),
        coordinate_admission=coordinates, motor_contract=copy.deepcopy(motors), motor_handoff=handoff,
        measured_released_window=window, actual_body_poses_xyz_wxyz=dict(zip(
            complete['body_names'], complete['body_poses_xyz_wxyz'])),
        predecessor_controller_diagnostics=copy.deepcopy(rows[-1]['teacher']),
        additional_continuation_evidence=extra_evidence,
        historical_source_captures=captures, input_sha256=hashes,
        **{key: str(path) for key, path in assets.items()},
        physics_steps=0, source_sample_playback=0, authorized_stages=0,
        scope='Actual released endpoint and original unstepped coordinate admission only; no panel plan, runtime authority, opening or traversal qualification')
