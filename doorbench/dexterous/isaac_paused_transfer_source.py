"""Detached inspection of an explicitly paused, still-live Isaac transfer.

This scaffold validates immutable evidence/endpoint identity only. It does not
admit a completed run, infer contact success, or authorize a live resume. A
separate fresh phase auditor and original kinematic/material admission are
required before a planning context may be constructed.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .isaac_attained_state import extract_attained_state
from .isaac_prefix_witness import _historical_inputs
from .motor_contract_identity import motor_contract_fingerprint
from .npz_record_stream import iter_npz_records
from .qualified_isaac_grasp import digest
from .standing_body_record import extract_standing_body_poses


SCHEMA = 'doorbench.paused-isaac-transfer-snapshot.v1'
SOURCE_KIND = 'paused-live-isaac-transfer-v1'
INSPECTION_SCHEMA = 'doorbench.paused-isaac-transfer-inspection.v1'
DT = .002
REQUIRED_FILES = frozenset(('configuration', 'motor_contract', 'provenance',
    'physics', 'pad_steps', 'transfer_steps', 'rest_stop', 'phase_report',
    'grasp_profile_definition', 'mechanical_audit', 'reset_state', 'transfer_route',
    'source_prefix_witness', 'observer_state'))
OPTIONAL_FILES = frozenset(('continuation_steps', 'reference_tail',
    'submitted_input_contract', 'actual_materials'))
ANCHOR_FIELDS = frozenset(('pause_token', 'episode_id', 'controller_identity',
    'step_index', 'epoch_s', 'physics_dt_s', 'measurement_fingerprint',
    'controller_fingerprint'))


def _unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('Duplicate JSON key: ' + name)
        result[name] = value
    return result


def _finite_json(value):
    if type(value) is float and not np.isfinite(value):
        raise ValueError('Finite JSON scalars required')
    if type(value) is dict:
        for item in value.values(): _finite_json(item)
    if type(value) is list:
        for item in value: _finite_json(item)
    return value


def _read(path):
    return _finite_json(json.loads(Path(path).read_text(), object_pairs_hook=_unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value))))


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Explicit lowercase SHA256 or pause token required')
    return value


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_live_pause_anchor(anchor):
    """Check types/clock only; this cannot prove that a process is paused."""
    if type(anchor) is not dict or set(anchor) != ANCHOR_FIELDS:
        raise ValueError('Complete explicit live pause anchor required')
    for key in ('pause_token', 'controller_identity', 'measurement_fingerprint', 'controller_fingerprint'):
        _sha(anchor[key])
    if type(anchor['episode_id']) is not str or not anchor['episode_id'].strip():
        raise ValueError('Stable live episode identity required')
    count = anchor['step_index']
    if type(count) is not int or count < 251:
        raise ValueError('Completed prefix must contain the whole 251-sample rest window')
    if (type(anchor['epoch_s']) not in (int, float) or not np.isfinite(anchor['epoch_s'])
            or anchor['physics_dt_s'] != DT or type(anchor['physics_dt_s']) is not float
            or anchor['epoch_s'] != count*DT):
        raise ValueError('Exact original 500 Hz completed-step epoch required')
    return json.loads(_encode(anchor))


def _files(snapshot, directory):
    files = snapshot.get('files')
    if (type(files) is not dict or not REQUIRED_FILES <= set(files)
            or set(files) - REQUIRED_FILES - OPTIONAL_FILES):
        raise ValueError('Complete explicitly named paused evidence inventory required')
    paths = {}; hashes = {}; identities = set()
    for role, entry in files.items():
        if type(entry) is not dict or set(entry) != {'path', 'sha256'}:
            raise ValueError('Exact paused evidence path/digest declaration required')
        original = Path(entry['path'])
        path = original.resolve()
        if (not original.is_absolute() or original.is_symlink() or not path.is_file()
                or not path.is_relative_to(directory) or path in identities):
            raise ValueError('Distinct snapshot-local immutable evidence files required')
        expected = _sha(entry['sha256'])
        if digest(path) != expected:
            raise ValueError('Paused evidence changed: ' + role)
        paths[role] = path; hashes[str(path)] = expected; identities.add(path)
    # The new phase declaration cannot masquerade as a completed-run report.
    if paths['phase_report'].name in ('operation-report.json', 'result.json', 'coordinator-result.json'):
        raise ValueError('Distinct paused phase declaration required; no completed-run report')
    return paths, hashes


def _verify(hashes):
    for name, expected in hashes.items():
        if digest(name) != expected:
            raise ValueError('Paused source changed during inspection: ' + name)


@dataclass(frozen=True)
class PausedIsaacTransferSnapshot:
    """Detached copies only. Inspection is deliberately not qualification."""
    _inspection_json: str

    @property
    def inspection(self): return json.loads(self._inspection_json)

    @property
    def sha256(self): return hashlib.sha256(self._inspection_json.encode()).hexdigest()

    @property
    def terminal_time_s(self): return self.inspection['live_pause']['epoch_s']

    @property
    def state_archive_path(self): return Path(self.inspection['evidence_paths']['physics'])

    def verify_inputs(self): _verify(self.inspection['input_sha256'])


def inspect_paused_isaac_transfer_snapshot(snapshot_path):
    """Validate all core rows and exact endpoint, without trusting phase labels.

Contact streams are bound but not reduced here. The full phase auditor must
independently replay them, original physical gates and the first rest detector.
No serialized passed flag can turn this return value into a planning context.
"""
    snapshot_path = Path(snapshot_path).resolve()
    before = digest(snapshot_path); snapshot = _read(snapshot_path)
    if (type(snapshot) is not dict or set(snapshot) != {'schema', 'source_kind',
            'source_engine', 'live_pause', 'closed_prefix', 'episode_complete',
            'files', 'terminal_command'}
            or snapshot['schema'] != SCHEMA or snapshot['source_kind'] != SOURCE_KIND
            or snapshot['source_engine'] != 'isaac-physx'
            or snapshot['closed_prefix'] is not True or snapshot['episode_complete'] is not False):
        raise ValueError('Explicit closed-prefix, unfinished live-episode snapshot required')
    anchor = validate_live_pause_anchor(snapshot['live_pause'])
    paths, hashes = _files(snapshot, snapshot_path.parent)
    hashes[str(snapshot_path)] = before
    configuration, motors, provenance = [_read(paths[key]) for key in ('configuration', 'motor_contract', 'provenance')]
    if (configuration.get('dt') != DT or configuration.get('runtime_pose_writes') != 0
            or configuration.get('direct_door_commands') is not False):
        raise ValueError('Original motor-only 500 Hz physical observation contract required')
    historical, captures = _historical_inputs(snapshot_path.parent, provenance)
    for name, expected in historical.items():
        if name in hashes and hashes[name] != expected:
            raise ValueError('Conflicting captured runtime or asset binding')
        hashes[name] = expected
    args = configuration.get('args', {})
    if (args.get('acquisition') is not True or args.get('operate_after_acquisition') is not True
            or args.get('acquisition_stance_profile') != 'landed-foot-v1'
            or args.get('grasp_profile') != 'volar-phalange-v1'
            or not args.get('standing_transfer_route') or args.get('standing_withdrawal_route')):
        raise ValueError('Explicit original standing transfer before withdrawal required')
    robot_path = Path(args['native_robot']).resolve()
    if hashes.get(str(robot_path)) != motors.get('source_xml_sha256'):
        raise ValueError('Original robot XML and full motor contract must agree')
    declaration = _read(paths['grasp_profile_definition'])
    definition_path = Path(args['grasp_profile_definition']).resolve()
    if (declaration.get('profile') != args['grasp_profile']
            or declaration.get('robot_xml_sha256') != motors['source_xml_sha256']
            or hashes.get(str(definition_path)) != digest(paths['grasp_profile_definition'])):
        raise ValueError('Original prospective source-bound volar declaration required')
    joint_names = configuration.get('robot_joint_names')
    door_names = configuration.get('door_joint_names')
    actuators = motors.get('actuators', [])
    if (not isinstance(joint_names, list) or len(joint_names) != 69 or len(set(joint_names)) != 69
            or not isinstance(door_names, list) or len(door_names) != 3
            or set(door_names) != {'leaf_hinge', 'leaf_handle_hinge', 'leaf_latch_bolt_slide'}
            or not isinstance(actuators, list) or len(actuators) != 61):
        raise ValueError('Original complete robot/door/motor inventories required')
    motor_names = [motor['name'] for motor in actuators]
    ranges = np.asarray([motor['force_range'] for motor in actuators], float)
    if (len(set(motor_names)) != 61 or ranges.shape != (61, 2)
            or not np.isfinite(ranges).all() or np.any(ranges[:, 0] > ranges[:, 1])):
        raise ValueError('Original finite ordered motor caps required')
    shapes = dict(time_s=(), root=(13,), joints=(69,), joint_velocity=(69,),
        motor_forces=(61,), door=(3,), door_velocity=(3,), standing_body_poses=(6, 7),torso_tilt_deg=())
    count = 0; last = None
    for count, row in enumerate(iter_npz_records(paths['physics'], shapes,
            expected_rows=anchor['step_index']), start=1):
        if row['time_s'].dtype != np.dtype('float64') or float(row['time_s']) != count*DT:
            raise ValueError('Every recorded original 500 Hz post-step epoch required')
        if (np.any(row['motor_forces'] < ranges[:, 0]-1e-5)
                or np.any(row['motor_forces'] > ranges[:, 1]+1e-5)):
            raise ValueError('Recorded commanded input exceeds original motor caps')
        last = row
    command = snapshot['terminal_command']
    fields = {'command_time_s', 'post_step_time_s', 'motor_names', 'dtype', 'shape', 'bytes_sha256', 'values'}
    if (type(command) is not dict or set(command) != fields
            or command['command_time_s'] != (anchor['step_index']-1)*DT
            or command['post_step_time_s'] != anchor['epoch_s']
            or command['motor_names'] != motor_names or command['shape'] != [61]
            or command['dtype'] != last['motor_forces'].dtype.str):
        raise ValueError('Exact same-interval prior returned motor command required')
    raw_values = command['values']
    if (type(raw_values) is not list or len(raw_values) != 61
            or any(type(v) not in (int, float) for v in raw_values)):
        raise ValueError('Explicit finite numeric motor values required')
    values = np.asarray(raw_values, dtype=last['motor_forces'].dtype)
    if (values.shape != (61,) or not np.isfinite(values).all()
            or not np.array_equal(values.astype(float), np.asarray(raw_values, float))
            or values.tobytes() != last['motor_forces'].tobytes()
            or hashlib.sha256(values.tobytes()).hexdigest() != _sha(command['bytes_sha256'])):
        raise ValueError('Declared prior returned command differs from actual archived bytes')
    endpoint = {key:np.stack([value]) for key,value in last.items()}
    binding = extract_attained_state(configuration=configuration, motor_contract=motors,
        provenance=provenance, physics=endpoint, time_s=anchor['epoch_s'])
    bodies = extract_standing_body_poses(configuration, endpoint, time_s=anchor['epoch_s'])
    package = Path(__file__).resolve().parent
    for name in ('isaac_paused_transfer_source.py', 'isaac_attained_state.py',
                 'isaac_prefix_witness.py', 'motor_contract_identity.py',
                 'npz_record_stream.py', 'qualified_isaac_grasp.py',
                 'standing_body_record.py', 'destination_state_binding.py'):
        path = package/name
        hashes[str(path)] = digest(path)
    _verify(hashes)
    result = dict(schema=INSPECTION_SCHEMA, source_kind=SOURCE_KIND, source_engine='isaac-physx',
        snapshot_path=str(snapshot_path), snapshot_sha256=before, live_pause=anchor,
        closed_prefix=True, episode_complete=False, core_intervals=count,
        source_state_sha256=binding['sha256'], measured_state_binding=binding,
        measured_bodies=bodies, terminal_command=json.loads(_encode(command)),
        motor_contract_sha256=motor_contract_fingerprint(motors),
        evidence_paths={key:str(value) for key,value in paths.items()},
        historical_source_copies=captures, input_sha256=hashes,
        core_identity_checked=True, original_recorded_command_caps_checked=True,
        phase_qualified=False, raw_contacts_audited=False, rest_window_audited=False,
        material_frames_admitted=False, geometry_admitted=False,
        live_process_state_verified=False, resume_authorized=False, authorized_stages=0,
        physical_steps=0, active_state_writes=0,
        scope='Immutable complete-prefix core identity only; original physical/contact/rest/body/material gates and exact live-process resume handshake remain mandatory')
    return PausedIsaacTransferSnapshot(_encode(result))
