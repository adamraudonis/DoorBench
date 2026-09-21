"""Package a closed observed transfer prefix without ending its live episode.

Only supplied copies and files are read. The caller owns the no-update pause,
original measured checks and before/after live anchor comparison. A distinct
fresh phase audit is mandatory; assembly grants no planning or resume authority.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .bounded_evidence import BoundedEvidence
from .isaac_live_evidence_snapshot import snapshot_live_evidence
from .isaac_live_planning_pause import ANCHOR_SCHEMA, DT, _digest_text, _hex, _sha
from .isaac_paused_transfer_audit import CHECKS, PHASE_SCHEMA, REST_SCHEMA
from .isaac_paused_transfer_source import (
    SCHEMA, SOURCE_KIND, _read, validate_live_pause_anchor,
)


FILE_NAMES = dict(configuration='configuration.json', motor_contract='motor-contract.json',
    provenance='provenance.json', reset_state='reset-state.json',
    transfer_route='standing-transfer-route.json', grasp_profile_definition='grasp-profile-definition.json',
    submitted_input_contract='submitted-input-contract.json', actual_materials='actual-materials.json',
    reference_tail='reference-tail.json')
REQUIRED_SOURCE_FILES = frozenset(('configuration', 'motor_contract', 'provenance',
    'reset_state', 'transfer_route', 'grasp_profile_definition'))
CORE_SHAPES = dict(time_s=(), root=(13,), joints=(69,), joint_velocity=(69,),
    motor_forces=(61,), door=(3,), door_velocity=(3,), standing_body_poses=(6, 7), torso_tilt_deg=())


def _plain(value):
    """Detach exact finite JSON metadata; never discover/call object methods."""
    return json.loads(json.dumps(value, allow_nan=False))


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _entry(path):
    return dict(path=str(path), sha256=_sha(path))


def _copy_file(source, target, expected):
    source, target = Path(source), Path(target)
    if source.is_symlink() or not source.is_file() or _sha(source) != expected:
        raise ValueError('Original snapshot input changed: ' + str(source))
    with source.open('rb') as inp, target.open('xb') as out:
        for block in iter(lambda:inp.read(1024*1024), b''):
            out.write(block)
    if _sha(source) != expected or _sha(target) != expected:
        raise ValueError('Input changed during immutable snapshot copy: ' + str(source))


def _array_identity(value):
    return dict(dtype=value.dtype.str, shape=list(value.shape),
        sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def assemble_paused_transfer_snapshot(directory, *, pause_token, anchor,
        acquisition_states, writers, source_files, source_capture_directory, source_run,
        phase_checks, standing_transfer, operation_reference, max_motor_delivery_error_Nm,
        mechanical_audit, rest_detector, observer_state, source_prefix_witness,
        compression_level=3):
    """Write a fresh snapshot using actual completed arrays and copied metadata.

    ``source_files`` names six existing immutable declarations plus optional
    FILE_NAMES roles. ``writers`` holds pad_steps, transfer_steps and optionally
    continuation_steps. The original configuration/provenance bytes and maximum
    duration are retained; original .py identities resolve to existing captures,
    never today's working source. Other original assets remain externally bound.

    Checks are explicit producer declarations, including false checks. This
    routine does not compute or replace them and does not run phase admission.
    The supplied observer document must contain the actual previous returned
    command, not a reconstructed or newly evaluated controller command.

    Failure leaves partial new files and every live writer recoverable. No
    destination overwrite, writer export/finalization, controller call, plant
    getter, FK, physics step or completed-run result is performed.
    """
    if not _hex(pause_token):
        raise ValueError('Explicit unique pause token required')
    anchor = _plain(anchor)
    if (anchor.get('schema') != ANCHOR_SCHEMA or anchor.get('pending_unaccepted_command') is not False
            or any(_digest_text(anchor['inventory'][key]) != anchor[name] for key, name in
                (('measurement', 'measurement_fingerprint'), ('controller', 'controller_fingerprint'),
                 ('retained_objects', 'controller_identity')))):
        raise ValueError('Original complete live pause anchor required')
    live_pause = validate_live_pause_anchor(dict(pause_token=pause_token, **{
        key:anchor[key] for key in ('episode_id', 'controller_identity', 'step_index', 'epoch_s',
            'physics_dt_s', 'measurement_fingerprint', 'controller_fingerprint')}))
    count, epoch = live_pause['step_index'], live_pause['epoch_s']
    if type(compression_level) is not int or not 1 <= compression_level <= 9:
        raise ValueError('Explicit gzip level from1 through9 required')
    if (type(source_files) is not dict or not REQUIRED_SOURCE_FILES <= set(source_files)
            or set(source_files) - set(FILE_NAMES)):
        raise ValueError('Explicit original snapshot source-file roles required')
    if (type(writers) is not dict or not {'pad_steps', 'transfer_steps'} <= set(writers)
            or set(writers) - {'pad_steps', 'transfer_steps', 'continuation_steps'}
            or len({id(value) for value in writers.values()}) != len(writers)
            or any(not isinstance(value, BoundedEvidence) or value.exported_path is not None
                for value in writers.values())):
        raise ValueError('Distinct active pad/transfer evidence writers required')
    expected_counts = {key:count + (key == 'pad_steps') for key in writers}
    if any(len(writers[key]) != expected for key, expected in expected_counts.items()):
        raise ValueError('Complete same-epoch pad/transfer/continuation evidence counts required')
    recorded_counts = anchor['inventory']['evidence_counts']
    if any(recorded_counts.get(key) != expected for key, expected in
            dict(physical=count, pad=count+1, transfer=count).items()):
        raise ValueError('Anchor evidence counts differ from the actual copied prefix')
    if 'continuation_steps' in writers and recorded_counts.get('continuation') != count:
        raise ValueError('Anchor must bind all included continuation records')
    if (type(acquisition_states) is not dict or not set(CORE_SHAPES) <= set(acquisition_states)
            or any(type(key) is not str or not key for key in acquisition_states)):
        raise ValueError('Original complete physical array inventory required')
    arrays = {}
    for name, values in acquisition_states.items():
        value = np.array(values, copy=True, order='C')
        if (value.dtype.kind not in 'biuf' or not value.ndim or len(value) != count
                or not np.isfinite(value).all()
                or (name in CORE_SHAPES and value.shape != (count, *CORE_SHAPES[name]))):
            raise ValueError('Complete finite actual physical array required: ' + name)
        arrays[name] = value
    if (arrays['time_s'].dtype != np.dtype('float64')
            or not np.array_equal(arrays['time_s'], np.arange(1, count+1)*DT)):
        raise ValueError('Exact original float64 post-step clock required')
    if (type(phase_checks) is not dict or set(phase_checks) != CHECKS
            or any(type(value) is not bool for value in phase_checks.values())):
        raise ValueError('All original explicitly measured boolean phase checks required')
    source_run = Path(source_run)
    if not source_run.is_absolute():
        raise ValueError('Absolute actual live source run required')
    source_capture_directory = Path(source_capture_directory).resolve()
    sources = {key:Path(value) for key, value in source_files.items()}
    if (any(not p.is_absolute() or p.is_symlink() or not p.is_file() for p in sources.values())
            or len({p.resolve() for p in sources.values()}) != len(sources)):
        raise ValueError('Distinct original absolute source files required')
    sources = {key:p.resolve() for key, p in sources.items()}
    input_hashes = {str(p):_sha(p) for p in sources.values()}
    config, motors, provenance = [_read(sources[key]) for key in ('configuration', 'motor_contract', 'provenance')]
    if config.get('dt') != DT or config.get('runtime_pose_writes') != 0 or config.get('direct_door_commands') is not False:
        raise ValueError('Original motor-only500Hz configuration required')
    standing_transfer, operation_reference = _plain(standing_transfer), _plain(operation_reference)
    if (standing_transfer.get('route') != config['args'].get('standing_transfer_route')
            or standing_transfer.get('started_s') != config['args'].get('standing_transfer_start_seconds')):
        raise ValueError('Actual transfer declaration must match unchanged original recipe')
    if (type(max_motor_delivery_error_Nm) not in (int, float)
            or not np.isfinite(max_motor_delivery_error_Nm) or max_motor_delivery_error_Nm < 0):
        raise ValueError('Finite actual motor delivery error required')
    documents = {key:_plain(value) for key, value in dict(mechanical_audit=mechanical_audit,
        rest_detector=rest_detector, observer_state=observer_state, source_prefix_witness=source_prefix_witness).items()}
    if (documents['rest_detector'].get('terminal_time_s') != epoch
            or documents['rest_detector'].get('observed_samples') != count
            or documents['rest_detector'].get('triggered') is not True):
        raise ValueError('Actual first rest-detector endpoint must match the snapshot')
    command = arrays['motor_forces'][-1]
    motor_names = [motor['name'] for motor in motors['actuators']]
    observer = documents['observer_state']
    capture = observer.get('motor_capture', {})
    observed = np.asarray(capture.get('previous_command'))
    if (len(motor_names) != 61 or len(set(motor_names)) != 61 or command.dtype.kind != 'f'
            or observer.get('schema') != 'doorbench.paused-transfer-observers.v1'
            or capture.get('previous_time') != (count-1)*DT or capture.get('step_seconds') != DT
            or observed.shape != (61,) or not np.isfinite(observed).all()
            or not np.array_equal(observed, command)
            or observed.astype(command.dtype).tobytes() != command.tobytes()):
        raise ValueError('Actual retained previous outer command must match archived bytes')
    terminal_command = dict(command_time_s=(count-1)*DT, post_step_time_s=epoch,
        motor_names=motor_names, dtype=command.dtype.str, shape=[61],
        bytes_sha256=hashlib.sha256(command.tobytes()).hexdigest(), values=command.tolist())
    original_files = provenance.get('files')
    if type(original_files) is not dict or not original_files:
        raise ValueError('Original captured source/asset provenance required')
    captures = {}; basenames = set()
    for original, expected in original_files.items():
        p = Path(original)
        if not p.is_absolute() or not _hex(expected):
            raise ValueError('Absolute original source/asset identity required')
        if p.suffix.lower() == '.py':
            if p.name.lower() in basenames:
                raise ValueError('Ambiguous original source-capture basename')
            basenames.add(p.name.lower())
            p = source_capture_directory/('source-' + p.name)
            captures[p.name] = dict(original=original, path=str(p), sha256=expected)
        if p.is_symlink() or not p.is_file() or _sha(p) != expected:
            raise ValueError('Original captured input changed: ' + str(p))
        if str(p) in input_hashes and input_hashes[str(p)] != expected:
            raise ValueError('Conflicting original input hash')
        input_hashes[str(p)] = expected
    if 'isaac_opening.py' not in basenames:
        raise ValueError('Original captured Isaac producer required')
    for role, argument in (('transfer_route', 'standing_transfer_route'),
                           ('grasp_profile_definition', 'grasp_profile_definition')):
        original = str(Path(config['args'][argument]).resolve())
        if original_files.get(original) != input_hashes[str(sources[role])]:
            raise ValueError('Copied role must match its originally declared input: ' + role)
    folder = Path(directory).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    files = {}
    for role, p in sources.items():
        target = folder/FILE_NAMES[role]
        _copy_file(p, target, input_hashes[str(p)])
        files[role] = _entry(target)
    for name, item in captures.items():
        _copy_file(item['path'], folder/name, item['sha256'])
    physics_path = folder/'acquisition-physics.npz'
    with physics_path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    files['physics'] = _entry(physics_path)
    evidence = snapshot_live_evidence(writers, folder/'evidence', pause_token=pause_token,
        compression_level=compression_level)
    for role, item in evidence['streams'].items():
        files[role] = dict(path=item['rows_path'], sha256=item['rows_sha256'])
    phase = dict(schema=PHASE_SCHEMA, source_run=str(source_run.resolve()), episode_complete=False,
        duration_s=epoch, physics_dt_s=DT, checks=_plain(phase_checks), passed=all(phase_checks.values()),
        standing_transfer=standing_transfer, operation_reference=operation_reference,
        max_motor_delivery_error_Nm=float(max_motor_delivery_error_Nm),
        scope='Unmodified actual producer check declarations over this closed prefix; independent phase audit required',
        phase_qualified=False, authorized_stages=0)
    generated = dict(phase_report=('paused-transfer-phase.json', phase),
        rest_stop=('paused-transfer-rest-stop.json', dict(schema=REST_SCHEMA,
            episode_complete=False, detector=documents['rest_detector'])),
        mechanical_audit=('mechanical-audit.json', documents['mechanical_audit']),
        observer_state=('paused-transfer-observers.json', documents['observer_state']),
        source_prefix_witness=('source-prefix-witness.json', documents['source_prefix_witness']))
    for role, (name, value) in generated.items():
        target = folder/name
        _write_json(target, value)
        files[role] = _entry(target)
    _write_json(folder/'live-anchor.json', anchor)
    _write_json(folder/'evidence-snapshot.json', evidence)
    for name, expected in input_hashes.items():
        if _sha(name) != expected:
            raise ValueError('Original input changed during snapshot assembly: ' + name)
    for name, value in acquisition_states.items():
        if _array_identity(np.asarray(value)) != _array_identity(arrays[name]):
            raise ValueError('Supplied physical arrays changed during assembly: ' + name)
    if any(len(writers[key]) != expected or writers[key].exported_path is not None
            for key, expected in expected_counts.items()):
        raise ValueError('Live writers changed during assembly')
    snapshot = dict(schema=SCHEMA, source_kind=SOURCE_KIND, source_engine='isaac-physx',
        live_pause=live_pause, closed_prefix=True, episode_complete=False,
        files=files, terminal_command=terminal_command)
    snapshot_path = folder/'snapshot.json'
    _write_json(snapshot_path, snapshot)
    receipt = dict(schema='doorbench.paused-isaac-snapshot-assembly.v1',
        snapshot_path=str(snapshot_path), snapshot_sha256=_sha(snapshot_path),
        pause_token=pause_token, epoch_s=epoch, core_intervals=count,
        array_identities={key:_array_identity(value) for key, value in arrays.items()},
        input_sha256=input_hashes, output_sha256={str(p):_sha(p) for p in folder.rglob('*') if p.is_file()},
        original_source_captures=captures, producer_declared_checks_passed=phase['passed'],
        phase_qualified=False, episode_complete=False, live_process_state_verified=False,
        live_writers_finalized=False, authorized_stages=0, physical_steps=0, active_state_writes=0)
    _write_json(folder/'assembly-receipt.json', receipt)
    return receipt
