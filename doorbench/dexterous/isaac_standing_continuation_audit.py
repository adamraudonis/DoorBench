"""Independent, bounded-memory accounting of actual continuation observations.

This is an observation admission, never a physical-task or stage admission.
The legacy ``actual_joint_effort`` field is submitted generalized input; its
motor-equivalent reconstruction is not independently measured delivered torque.
"""
import copy
import gzip
import json
from pathlib import Path

import numpy as np

from .isaac_joint_passive import passive_profile
from .isaac_opening_measurements import pose_parts
from .isaac_prefix_witness import _historical_inputs
from .continuation_record_stream import iter_continuation_records, _finite_float
from .motor_contract_identity import motor_contract_fingerprint
from .npz_record_stream import iter_npz_records
from .qualified_isaac_grasp import digest
from .standing_body_record import PLANNER_BODIES, POSE_CONVENTION

SCHEMA = 'doorbench.isaac-standing-continuation-audit.v1'
OBSERVATION_SCHEMA = 'doorbench.isaac-standing-continuation-observation.v1'
BODY_NAMES = ('left_ankle_link', 'right_ankle_link', 'lh_palm', 'rh_palm', 'leaf', 'leaf_handle')
DT = .002
SHAPES = dict(time_s=(), root=(13,), joints=(69,), joint_velocity=(69,),
    motor_forces=(61,), door=(3,), door_velocity=(3,), standing_body_poses=(6, 7),
    standing_leaf_pose=(7,), actual_motor_forces=(61,), actual_joint_effort=(69,),
    continuation_body_poses=(6, 7), actual_foot_loads=(2,), pre_step_joint_velocity=(69,),
    legacy_root_state_w=(13,))
CAPTURE_SOURCES = ('isaac_opening.py', 'isaac_standing_continuation_measurements.py',
    'isaac_post_opening_measurements.py', 'isaac_opening_measurements.py',
    'isaac_joint_passive.py', 'bounded_evidence.py')
FILES = ('configuration.json', 'provenance.json', 'operation-report.json',
    'acquisition-physics.npz', 'standing-continuation-contract.json',
    'standing-continuation-steps.json.gz', 'standing-transfer-steps.json.gz',
    'motor-contract.json', 'motor-readback-contract.json', 'joint-passive-profile.json', 'scene.usda')
SUBMISSION_SEMANTICS = ('Submitted generalized actuation-input readback from the backend; '
                      'not independently measured joint torque')


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('Duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_float=_finite_float,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def _array(value, shape, label):
    original = np.asarray(value)
    if original.dtype.kind not in 'fiu': raise ValueError('Numeric ' + label + ' required')
    a = original.astype(float)
    if a.shape != tuple(shape) or not np.isfinite(a).all():
        raise ValueError('Finite exact-shape ' + label + ' required')
    return a


def _same(actual, expected, label, *, atol=1e-8):
    """Compare recomputed scalar reductions without inventing a physical tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('Different fields in ' + label)
        for key in expected: _same(actual[key], expected[key], label + '.' + key, atol=atol)
    elif isinstance(expected, list) and any(isinstance(v, str) for v in expected):
        if actual != expected: raise ValueError('Different ' + label)
    elif expected is None or isinstance(expected, (str, bool, int)):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError('Different ' + label)
    else:
        raw = np.asarray(actual)
        if raw.dtype.kind not in 'fiu': raise ValueError('Numeric ' + label + ' required')
        a, b = raw.astype(float), np.asarray(expected, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.allclose(a, b, atol=atol, rtol=0):
            raise ValueError('Recomputed ' + label + ' differs')


def _saved_scene_inventory(path):
    """Read authored collision participants, including instance proxies; no plant."""
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(path))
    if stage is None: raise ValueError('Saved scene cannot be opened')
    if stage.GetCompositionErrors(): raise ValueError('Unresolved saved scene composition')
    paths = set()
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI): continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(): continue
        body, current = prim, prim
        while current and not current.IsPseudoRoot():
            if current.HasAPI(UsdPhysics.RigidBodyAPI):
                body = current
                break
            current = current.GetParent()
        paths.add(str(body.GetPath()))
    layers = {}
    for layer in stage.GetUsedLayers():
        if layer.anonymous: continue
        layer_path = Path(layer.realPath).resolve()
        if not layer.realPath or not layer_path.is_file():
            raise ValueError('Unresolved saved scene layer')
        layers[str(layer_path)] = digest(layer_path)
    if not paths: raise ValueError('Saved authored collision inventory is empty')
    return paths, layers


def validate_capture_contract(contract, scene_paths):
    """Require complete saved-scene participants, never a hand-only subset."""
    if (contract.get('schema') != 'doorbench.isaac-standing-continuation-capture.v1'
            or contract.get('dt_s') != DT or type(contract.get('capacity')) is not int or contract.get('capacity') != 16384
            or contract.get('body_pose_order') != list(BODY_NAMES)
            or contract.get('clock') != 'completed PhysX interval, global episode time'
            or contract.get('normal_and_friction_slots_are_independent') is not True
            or type(contract.get('authorized_stages')) is not int or contract['authorized_stages'] != 0):
        raise ValueError('Original complete continuation capture contract required')
    paths = contract.get('sensor_paths')
    if (not isinstance(paths, list) or not all(type(p) is str for p in paths)
            or len(set(paths)) != len(paths)
            or set(paths) != {p for p in scene_paths if p.startswith('/World/H1/')}):
        raise ValueError('Omitted or duplicated authored robot contact participant')
    filters = contract.get('filter_paths')
    if (not isinstance(filters, list) or len(filters) != len(paths)
            or any(not isinstance(row, list) or not all(type(p) is str for p in row)
                   or len(row) != len(scene_paths) or set(row) != set(scene_paths) for row in filters)):
        raise ValueError('Every sensor needs the complete unique saved-scene filter inventory')
    names = [p.rsplit('/', 1)[-1] for p in paths]
    if any(names.count(name) != 1 for name in BODY_NAMES[:4]):
        raise ValueError('Unambiguous actual feet and palms required')
    return paths, filters


def decode_sparse_buffer(raw, shape, capacity, fields):
    """Decode occupied slots independently of the live packer.

    Occupied pairs retain their exact row-major order. Normal and friction
    inventories are decoded separately and may legitimately reuse slot numbers.
    """
    if not isinstance(raw, dict) or set(raw) != {'shape', 'pairs', 'slots', *fields} or raw['shape'] != list(shape):
        raise ValueError('Complete sparse buffer fields and pair shape required')
    pairs, slots = raw['pairs'], raw['slots']
    if not isinstance(pairs, list) or not isinstance(slots, list):
        raise ValueError('Explicit sparse slot inventory required')
    occupied, expected_slots, previous = set(), [], (-1, -1)
    spans = []
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 4 or any(type(v) is not int for v in pair):
            raise ValueError('Integer contact pair slices required')
        i, j, first, count = pair
        if (not 0 <= i < shape[0] or not 0 <= j < shape[1] or (i, j) <= previous
                or first < 0 or count < 1 or first + count > capacity):
            raise ValueError('Canonical in-bounds positive contact slices required')
        selected = list(range(first, first + count))
        if occupied.intersection(selected): raise ValueError('Overlapping contact slots')
        spans.append((i, j, len(expected_slots), count))
        expected_slots.extend(selected); occupied.update(selected); previous = (i, j)
    if (len(expected_slots) >= capacity or any(type(v) is not int for v in slots)
            or slots != expected_slots):
        raise ValueError('Truncated or misaligned sparse occupied inventory')
    values = {}
    for name, width in fields.items():
        value = raw[name]
        # JSON [] is the only canonical empty (0,width) representation.
        values[name] = (np.empty((0, width)) if value == []
                        else _array(value, (len(slots), width), name))
        if values[name].shape[0] != len(slots): raise ValueError('Missing sparse values')
    return spans, values


def decode_continuation_row(row, contract, *, time_s, submission_valid):
    """Independently reconstruct reductions from exact raw pair/slot records."""
    paths, filters, capacity = contract['sensor_paths'], contract['filter_paths'], contract['capacity']
    shape = (len(paths), len(filters[0]))
    if (row.get('schema') != OBSERVATION_SCHEMA or type(row.get('authorized_stages')) is not int
            or row['authorized_stages'] != 0 or type(submission_valid) is not bool):
        raise ValueError('Unprivileged continuation observation required')
    _same(row.get('time_s'), float(time_s), 'interval end', atol=1e-10)
    _same(row.get('pose_time_s'), float(time_s), 'pose epoch', atol=1e-10)
    _same(row.get('contact_interval_s'), [time_s-DT, time_s], 'contact interval', atol=1e-10)
    poses = row.get('body_poses')
    if not isinstance(poses, dict) or set(poses) != set(BODY_NAMES):
        raise ValueError('Every actual continuation body pose required')
    for pose in poses.values(): pose_parts(_array(pose, (7,), 'actual body pose'))
    raw = row.get('raw', {})
    if (set(raw) != {'capacity', 'pair_shape', 'normal_force_pairs', 'normal', 'friction'}
            or type(raw['capacity']) is not int or raw['capacity'] != capacity
            or raw['pair_shape'] != list(shape)):
        raise ValueError('Complete same-layout raw contact record required')
    normal_spans, normals = decode_sparse_buffer(raw['normal'], shape, capacity,
        {'force_N': 1, 'point_world': 3, 'normal_world': 3, 'distance_m': 1})
    friction_spans, friction = decode_sparse_buffer(raw['friction'], shape, capacity,
        {'force_N': 3, 'point_world': 3})
    matrix = np.zeros((*shape, 3)); previous = (-1, -1)
    for pair in raw['normal_force_pairs']:
        if (not isinstance(pair, list) or len(pair) != 5 or any(type(v) is not int for v in pair[:2])):
            raise ValueError('Explicit indexed normal-force pairs required')
        i, j = pair[:2]
        vector = _array(pair[2:], (3,), 'normal pair force')
        if not 0 <= i < shape[0] or not 0 <= j < shape[1] or (i, j) <= previous or not np.any(vector):
            raise ValueError('Canonical nonzero normal-force pair inventory required')
        matrix[i, j] = vector; previous = (i, j)
    reconstructed = np.zeros_like(matrix)
    feet = np.zeros(2); left_count = right_count = 0; left_load = 0.; release = []
    for i, j, first, count in normal_spans:
        name, other = paths[i].rsplit('/', 1)[-1], filters[i][j]
        for k in range(first, first + count):
            load = float(normals['force_N'][k, 0]); normal = normals['normal_world'][k]
            gap = float(normals['distance_m'][k, 0])
            if load < 0 or not np.isclose(np.linalg.norm(normal), 1., atol=1e-5, rtol=0):
                raise ValueError('Nonnegative force and original unit-normal tolerance required')
            reconstructed[i, j] += load * normal
            if name in BODY_NAMES[:2] and other.rsplit('/', 1)[-1] == 'floor':
                feet[BODY_NAMES.index(name)] += load * normal[2]
            contact = gap <= 0 or load > 1e-8
            if name.startswith('lh_') and not other.startswith('/World/H1/'):
                left_count += int(contact); left_load += load
                if contact and other == '/World/Door/Articulation/leaf': release.append(normal)
            if name.startswith('rh_') and not other.startswith('/World/H1/'):
                right_count += int(contact)
    normal_error = float(np.max(np.linalg.norm(matrix-reconstructed, axis=-1)))
    if normal_error > 1e-3: raise ValueError('Normal patches disagree with actual force matrix by >1e-3 N')
    for i, j, first, count in friction_spans:
        matrix[i, j] += friction['force_N'][first:first+count].sum(axis=0)
    hands = {p: matrix[i].sum(axis=0).tolist() for i, p in enumerate(paths)
             if p.rsplit('/', 1)[-1].startswith(('rh_', 'lh_'))}
    release_normal = None
    if release:
        mean = np.mean(release, axis=0)
        if np.linalg.norm(mean) < 1e-6: raise ValueError('Ambiguous actual panel release normal')
        release_normal = (mean/np.linalg.norm(mean)).tolist()
    evidence = dict(physics_qualified=submission_valid, left_hand_contacts=left_count,
        left_hand_load_N=left_load, right_environment_contacts=right_count)
    _same(row.get('foot_loads_N'), feet.tolist(), 'foot normal loads')
    _same(row.get('hand_forces_world_N'), hands, 'hand pair forces')
    _same(row.get('evidence'), evidence, 'submission flag and contact accounting')
    _same(row.get('release_normal_world'), release_normal, 'release normal')
    _, rotation = pose_parts(poses['leaf']); axis = rotation[:, 1]
    loads, vectors = {}, {}
    for i, path in enumerate(paths):
        name = path.rsplit('/', 1)[-1]
        if not name.startswith('lh_'): continue
        indices = [j for j, value in enumerate(filters[i]) if value == '/World/Door/Articulation/leaf']
        if len(indices) != 1: raise ValueError('Exact panel counterpart required')
        vector = matrix[i, indices[0]]
        loads[name] = max(0., float(-axis @ vector)); vectors[name] = vector.tolist()
    surface = dict(total_normal_load_N=sum(loads.values()), palm_normal_load_N=loads['lh_palm'],
        body_normal_loads_N=loads, body_panel_forces_world_N=vectors,
        scope='Privileged actual PhysX per-body projected normal-plus-friction surface load')
    return dict(foot_loads_N=feet, hand_forces_world_N=hands, evidence=evidence,
        release_normal_world=release_normal, surface=surface,
        maximum_normal_matrix_error_N=normal_error, normal_patch_count=len(raw['normal']['slots']),
        friction_patch_count=len(raw['friction']['slots']))


def _motor_model(motors, names, passive_receipt, readback):
    if len(names) != 69 or len(set(names)) != 69 or len(motors.get('actuators', [])) != 61:
        raise ValueError('Original 69-coordinate and 61-motor inventories required')
    matrix = np.zeros((61, 69)); indices = {name: i for i, name in enumerate(names)}
    seen = set()
    for i, motor in enumerate(motors['actuators']):
        if motor['name'] in seen or not motor['terms']: raise ValueError('Unique nonempty motor transmission required')
        seen.add(motor['name'])
        for name, coefficient in motor['terms'].items(): matrix[i, indices[name]] = coefficient
    caps = _array([motor['force_range'] for motor in motors['actuators']], (61, 2), 'motor caps')
    if not np.isfinite(matrix).all() or np.any(caps[:, 0] > caps[:, 1]): raise ValueError('Invalid motor transmission or caps')
    inverse = np.linalg.pinv(matrix.T)
    if np.linalg.matrix_rank(matrix.T) != 61 or not np.allclose(inverse@matrix.T, np.eye(61), atol=1e-12, rtol=0):
        raise ValueError('Original full-rank motor transmission required')
    declaration = passive_profile(motors, names, passive_receipt['profile'])
    if passive_receipt['profile'] == 'backend-dry-v2':
        _same(passive_receipt['joint_names'], names, 'passive joint order')
        for key in ('explicit_damping', 'explicit_friction'):
            _same(passive_receipt[key], declaration[key], key, atol=0)
        _same(passive_receipt['backend_friction_properties'],
            declaration['backend_friction_properties'].astype(np.float32), 'backend passive readback', atol=0)
        _same(passive_receipt['native_armature_readback_kg_m2'],
            declaration['native_armature'].astype(np.float32), 'backend armature readback', atol=0)
    if (readback.get('api') != 'ArticulationView.get_dof_actuation_forces'
            or readback.get('semantics') != SUBMISSION_SEMANTICS
            or readback.get('root_controller_field') != 'root_link_state_w'
            or readback.get('legacy_diagnostic_field') != 'legacy_root_state_w'
            or readback.get('initial_interval_s') != [0., 0.]
            or readback.get('archive_fields') != dict(
                actual_joint_effort='Legacy field name: backend submitted generalized input',
                actual_motor_forces='Legacy field name: motor-equivalent reconstruction of submitted input')):
        raise ValueError('Honest submitted-input and actor-origin readback semantics required')
    _array(readback['initial_motor_input'], (61,), 'unstepped initial input')
    _array(readback['root_com_offset_in_actor_m'], (3,), 'root COM offset')
    return matrix, inverse, caps, declaration


def _source(trial):
    trial = Path(trial).resolve()
    if not (trial/'configuration.json').is_file() and (trial/'trial').is_dir(): trial /= 'trial'
    hashes = {str(trial/name): digest(trial/name) for name in FILES}
    configuration = _json(trial/'configuration.json'); report = _json(trial/'operation-report.json')
    historical, captured = _historical_inputs(trial, _json(trial/'provenance.json'))
    if not set(CAPTURE_SOURCES) <= {Path(path).name for path in captured}:
        raise ValueError('Complete original continuation producer/helper captures required')
    hashes.update(historical)
    for module in (__file__, Path(__file__).with_name('npz_record_stream.py'),
            Path(__file__).with_name('continuation_record_stream.py'), Path(__file__).with_name('isaac_joint_passive.py'),
            Path(__file__).with_name('isaac_prefix_witness.py'), Path(__file__).with_name('isaac_opening_measurements.py'),
            Path(__file__).with_name('motor_contract_identity.py'), Path(__file__).with_name('qualified_isaac_grasp.py'),
            Path(__file__).with_name('standing_body_record.py')):
        hashes[str(Path(module).resolve())] = digest(module)
    args = configuration.get('args', {})
    if (configuration.get('dt') != DT or report.get('physics_dt_s') != DT
            or configuration.get('runtime_pose_writes') != 0 or configuration.get('direct_door_commands') is not False
            or args.get('acquisition') is not True or args.get('operate_after_acquisition') is not True
            or args.get('acquisition_stance_profile') != 'landed-foot-v1'
            or not args.get('standing_withdrawal_route')
            or configuration.get('standing_continuation_body_names') != list(BODY_NAMES)
            or configuration.get('standing_planner_body_names') != list(PLANNER_BODIES)
            or configuration.get('standing_planner_body_pose_convention') != POSE_CONVENTION
            or configuration.get('standing_leaf_pose_convention') != POSE_CONVENTION
            or configuration.get('root_state_convention') != 'actor-origin pose and world actor-origin linear/angular velocity'):
        raise ValueError('Original initialized-standing motor-only continuation capture required')
    duration = report.get('duration_s')
    if type(duration) not in (int, float) or not np.isfinite(duration) or duration <= 0 or abs(duration/DT-round(duration/DT)) > 1e-7:
        raise ValueError('Completed finite 500 Hz source duration required')
    count = round(duration/DT)
    capture = report.get('standing_continuation_capture', {})
    if (capture.get('observations') != count or capture.get('body_pose_order') != list(BODY_NAMES)
            or capture.get('archive') != 'standing-continuation-steps.json.gz'
            or capture.get('authorized_stages') != 0):
        raise ValueError('Finalized complete observation export required; checkpoints cannot be admitted')
    contract = _json(trial/'standing-continuation-contract.json')
    inventory, layers = _saved_scene_inventory(trial/'scene.usda'); hashes.update(layers)
    validate_capture_contract(contract, inventory)
    motors = _json(trial/'motor-contract.json')
    for key in ('motors', 'native_robot', 'robot_usd', 'door_usd'):
        path = str(Path(args[key]).resolve())
        if path not in historical or digest(path) != historical[path]:
            raise ValueError('Original motor and physical assets must be in captured provenance')
    if digest(trial/'motor-contract.json') != historical[str(Path(args['motors']).resolve())]:
        raise ValueError('Copied motor contract differs from captured input')
    if motors.get('source_xml_sha256') != historical[str(Path(args['native_robot']).resolve())]:
        raise ValueError('Motor contract original robot binding differs')
    names = configuration['robot_joint_names']
    door_names = configuration['door_joint_names']
    if len(door_names) != 3 or len(set(door_names)) != 3: raise ValueError('Exact three measured door coordinates required')
    passive = _json(trial/'joint-passive-profile.json')
    if passive.get('profile') != args.get('joint_passive_profile'):
        raise ValueError('Recorded passive declaration differs from selected profile')
    model = _motor_model(motors, names, passive, _json(trial/'motor-readback-contract.json'))
    return trial, hashes, configuration, report, count, contract, motors, model


def audit_standing_continuation(trial):
    """Replay finite saved evidence in bounded memory; malformed evidence raises.

    A false ``passed`` preserves correctly recorded invalid motor submissions.
    Neither value establishes task success, contact stability or traversal.
    """
    trial, hashes, config, report, count, contract, motors, model = _source(trial)
    matrix, inverse, caps, passive = model
    damp, friction = passive['explicit_damping'], passive['explicit_friction']
    maxima = dict(normal_matrix_error_N=0., motor_reconstruction_residual_Nm=0.,
        submitted_generalized_input_error_Nm=0., archived_motor_reconstruction_error_Nm=0.)
    invalid = commands_invalid = 0; normal_count = friction_count = 0
    previous_velocity = None; sticky_submission_valid = True; first_invalid = None
    physics = iter_npz_records(trial/'acquisition-physics.npz', SHAPES, expected_rows=count)
    with gzip.open(trial/'standing-continuation-steps.json.gz', 'rt') as stream, gzip.open(trial/'standing-transfer-steps.json.gz', 'rt') as surface_stream:
        records = iter_continuation_records(stream); surfaces = iter_continuation_records(surface_stream)
        for index, sample in enumerate(physics):
            row, surface = next(records, None), next(surfaces, None)
            if row is None or surface is None: raise ValueError('Missing same-epoch continuation or transfer record')
            t = (index+1)*DT
            if abs(float(sample['time_s'])-t) > 1e-10: raise ValueError('Missing, duplicated or shifted 500 Hz physical interval')
            velocity = sample['pre_step_joint_velocity']
            if previous_velocity is not None and not np.array_equal(velocity, previous_velocity):
                raise ValueError('Pre-step velocity differs from preceding actual readback')
            previous_velocity = sample['joint_velocity'].copy()
            v64 = velocity.astype(float)
            active = sample['actual_joint_effort'].astype(float)+damp*v64+friction*np.tanh(v64/.001)
            motor = inverse@active
            residual = float(np.max(abs(matrix.T@motor-active)))
            valid = residual <= 1e-5 and bool(np.all(motor >= caps[:, 0]-1e-5) and np.all(motor <= caps[:, 1]+1e-5))
            # The producer preserves failed inputs with its original fallback
            # expression, whose tanh uses the archived float32 velocity dtype.
            archived_motor = motor if valid else inverse@(sample['actual_joint_effort']+damp*velocity+friction*np.tanh(velocity/.001))
            reconstructed_error = float(np.max(abs(archived_motor-sample['actual_motor_forces'])))
            if reconstructed_error > 1e-8: raise ValueError('Archived motor-equivalent reconstruction differs')
            sticky_submission_valid = sticky_submission_valid and valid
            if not valid:
                invalid += 1
                if first_invalid is None: first_invalid = t
            commanded = matrix.T@sample['motor_forces']-damp*velocity-friction*np.tanh(velocity/.001)
            error = float(np.max(abs(commanded-sample['actual_joint_effort'])))
            if error >= 1e-4: commands_invalid += 1
            decoded = decode_continuation_row(row, contract, time_s=t, submission_valid=sticky_submission_valid)
            poses = np.asarray([row['body_poses'][name] for name in BODY_NAMES])
            if not np.array_equal(poses, sample['continuation_body_poses']): raise ValueError('Actual body poses differ between archives')
            for new_i, old_i in ((0, 0), (1, 1), (2, 4), (3, 3), (5, 5)):
                if not np.array_equal(poses[new_i], sample['standing_body_poses'][old_i]):
                    raise ValueError('Continuation body differs from same-epoch source body record')
            if not np.array_equal(poses[4], sample['standing_leaf_pose']): raise ValueError('Actual leaf pose differs between streams')
            _same(sample['actual_foot_loads'], decoded['foot_loads_N'], 'archived foot loads')
            _same(surface['time_s'], t, 'surface epoch', atol=1e-10)
            _same(surface['leaf_pose'], poses[4], 'surface leaf pose', atol=0)
            _same(surface['surface'], decoded['surface'], 'projected panel surface')
            pose_parts(sample['root'][:7])
            if not np.array_equal(sample['root'][:7], sample['legacy_root_state_w'][:7]) or not np.array_equal(sample['root'][10:], sample['legacy_root_state_w'][10:]):
                raise ValueError('Actor-origin pose/angular velocity differs from backend root diagnostic')
            maxima['normal_matrix_error_N'] = max(maxima['normal_matrix_error_N'], decoded['maximum_normal_matrix_error_N'])
            maxima['motor_reconstruction_residual_Nm'] = max(maxima['motor_reconstruction_residual_Nm'], residual)
            maxima['submitted_generalized_input_error_Nm'] = max(maxima['submitted_generalized_input_error_Nm'], error)
            maxima['archived_motor_reconstruction_error_Nm'] = max(maxima['archived_motor_reconstruction_error_Nm'], reconstructed_error)
            normal_count += decoded['normal_patch_count']; friction_count += decoded['friction_patch_count']
            endpoint = dict(time_s=t, contact_interval_s=row['contact_interval_s'],
                root13_actor_origin=sample['root'].tolist(), joint_positions=sample['joints'].tolist(),
                joint_velocities=sample['joint_velocity'].tolist(), door_positions=sample['door'].tolist(),
                door_velocities=sample['door_velocity'].tolist(), body_poses=copy.deepcopy(row['body_poses']),
                foot_loads_N=decoded['foot_loads_N'].tolist(), hand_forces_world_N=decoded['hand_forces_world_N'],
                evidence=decoded['evidence'], release_normal_world=decoded['release_normal_world'],
                command_time_s=t-DT, commanded_motor_input=sample['motor_forces'].tolist(),
                submitted_generalized_input=sample['actual_joint_effort'].tolist(),
                motor_equivalent_input=motor.tolist())
        if next(records, None) is not None or next(surfaces, None) is not None: raise ValueError('Extra observation records beyond physical archive')
    if invalid == 0:
        _same(report['standing_continuation_capture']['maximum_motor_input_reconstruction_residual_Nm'],
            maxima['motor_reconstruction_residual_Nm'], 'reported reconstruction maximum', atol=1e-8)
    for path, expected in hashes.items():
        if digest(path) != expected: raise ValueError('Evidence changed during streaming audit: '+path)
    return dict(schema=SCHEMA, passed=invalid == 0 and commands_invalid == 0,
        accounting_passed=True, source_run=str(trial.parent), source_trial=str(trial),
        source_report_passed=report.get('passed') is True, source_terminal_time_s=count*DT,
        physical_intervals=count, observation_intervals=count, physics_dt_s=DT,
        source_physics_sha256=hashes[str(trial/'acquisition-physics.npz')],
        motor_contract_sha256=motor_contract_fingerprint(motors), motor_semantics=SUBMISSION_SEMANTICS,
        robot_joint_names=config['robot_joint_names'], door_joint_names=config['door_joint_names'],
        body_pose_order=list(BODY_NAMES), invalid_submission_intervals=invalid,
        commanded_input_mismatch_intervals=commands_invalid, first_invalid_submission_time_s=first_invalid,
        normal_patches_checked=normal_count, friction_patches_checked=friction_count,
        maxima=maxima, endpoint=endpoint, input_sha256=hashes,
        first_pre_step_velocity_independently_crosschecked=False,
        first_velocity_scope='First measured pre-step velocity enters the recorded equation; no separate t=0 velocity archive exists. Subsequent velocities equal the preceding actual readback exactly.',
        row_physics_qualified_semantics='Sticky submitted-input reconstruction/cap validity only; not full physical task qualification.',
        physical_task_qualification=False, authorized_stages=0,
        scope='Read-only same-epoch observation accounting; separate actual physical source qualification and live exact prefix required before any new controller stage.')


def admit_standing_continuation_observations(trial, audit_path, expected_epoch_s, expected_physics_sha256):
    """Fresh accounting bound to a caller's separately qualified state identity."""
    audit_path = Path(audit_path).resolve(); saved_sha = digest(audit_path); saved = _json(audit_path)
    actual = audit_standing_continuation(trial)
    if saved != actual or actual['passed'] is not True or actual['accounting_passed'] is not True:
        raise ValueError('Current complete passing independent observation audit required')
    if (type(expected_epoch_s) not in (int, float) or not np.isfinite(expected_epoch_s)
            or expected_epoch_s != actual['source_terminal_time_s']
            or expected_physics_sha256 != actual['source_physics_sha256']):
        raise ValueError('Observation admission must bind separately qualified exact source epoch and NPZ')
    if digest(audit_path) != saved_sha: raise ValueError('Saved observation receipt changed during admission')
    result = copy.deepcopy(actual)
    result['input_sha256'][str(audit_path)] = saved_sha
    result['schema'] = 'doorbench.isaac-standing-continuation-admission.v1'
    result['requires_separate_physical_source_qualification'] = True
    return result
