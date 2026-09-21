"""Detached source data for a later withdrawal-controller constructor hook.

Native v1 keeps its original admission and file layout. The explicit Isaac mode
uses actual qualified PhysX evidence, the new candidate and its independent
dense audit. Nothing here exposes a scene, articulation, motor interface, state
setter or stage authorization. The live prefix witness remains a separate gate.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .motor_contract_identity import motor_contract_fingerprint
from .qualified_isaac_grasp import digest


ISAAC_DENSE_CHECKS = frozenset((
    'exact_initial_normalized_qpos', 'complete_2001_samples', 'fixed_source_mechanism',
    'finite_joint_and_tendon_limits', 'foot_and_palm_position', 'foot_and_palm_orientation',
    'upright_torso', 'environment_penetration', 'selected_right_anatomy',
    'whole_handle_geometry', 'final_hand_environment_clearance', 'joint_reference_speed'))


def _bound_digest(bindings, path):
    # Same alias handling as the original native constructor: relative and
    # absolute spellings may identify one file, but conflicting hashes cannot.
    resolved = Path(path).resolve()
    values = {value for name,value in bindings.items() if Path(name).resolve() == resolved}
    if len(values) > 1:
        raise ValueError('Conflicting digest aliases for the same withdrawal input')
    return next(iter(values), None)


def _verified_hashes(bindings):
    result = {}
    for name, expected in bindings.items():
        path = Path(name).resolve()
        if digest(path) != expected:
            raise ValueError('Withdrawal input bytes changed: ' + str(path))
        if str(path) in result and result[str(path)] != expected:
            raise ValueError('Conflicting withdrawal input hash aliases')
        result[str(path)] = expected
    return result


@dataclass(frozen=True)
class WithdrawalSourceContext:
    """Only detached data. Obtain through load_withdrawal_source_context."""
    _payload_json: str

    @property
    def data(self):
        return json.loads(self._payload_json)

    @property
    def initial_qpos(self):
        return np.asarray(self.data['initial_qpos'], dtype=float)


def _native(config, motors, measured_rest, source, screen_path, audit_path, screen, audit):
    """The original native constructor's source admission, without a controller."""
    admission = None
    if measured_rest:
        from .release_source_admission import admit_release_source
        admission = admit_release_source(source, profile=config['grasp_profile'],
            contact_audit_name=config['contact_audit_name'], measured_rest=True)
        if (screen.get('grasp_profile') != config['grasp_profile']
                or audit.get('grasp_profile') != config['grasp_profile']):
            raise ValueError('Withdrawal route and dense audit must retain the selected source profile')
        if screen.get('source_admission') != admission:
            raise ValueError('Withdrawal screen must bind this exact measured-rest source')
    from .release_motion_admission import validate_motion_screen
    validate_motion_screen(config, screen_path, source)
    if audit.get('passed') is not True or audit.get('samples') != 2001 or audit.get('physics_steps') != 0:
        raise ValueError('Independent dense withdrawal admission required')
    archive = source/'trajectory.npz'
    if (_bound_digest(audit['input_sha256'], screen_path) != digest(screen_path)
            or _bound_digest(audit['input_sha256'], archive) != digest(archive)):
        raise ValueError('Withdrawal audit belongs to another state or route')
    hashes = _verified_hashes(audit['input_sha256'])
    manifest_path = source/'manifest.json'
    manifest_sha = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    cfg = manifest['configuration']
    if manifest['inputs']['robot']['sha256'] != motors['source_xml_sha256']:
        raise ValueError('Withdrawal requires its original robot motor contract')
    with np.load(archive, allow_pickle=False) as z:
        actual, start = z['terminal_qpos'].copy(), float(z['terminal_time_s'])
    hashes[str(manifest_path.resolve())] = manifest_sha
    if admission is not None:
        hashes.update(_verified_hashes(admission['input_sha256']))
    return dict(source_engine='native-mujoco', source_admission=admission,
        initial_qpos=actual.tolist(), start_time_s=start, duration_s=float(audit['duration_s']),
        robot_path=str(Path(cfg['robot']).resolve()),
        door_xml_path=str((Path(cfg['door'])/'door.xml').resolve()), door_usd_path=None,
        source_archive_path=str(archive.resolve()), source_state_sha256=None,
        input_sha256=hashes, live_exact_prefix_required=False,
        motor_binding_scope='Original native v1 robot XML identity check, unchanged')


def _isaac(config, motors, measured_rest, source, screen_path, audit_path, screen, audit):
    from .isaac_release_planning import admit_isaac_release_context
    from .isaac_release_geometry_audit import validate_candidate_binding, ORIGINAL_LIMITS
    if measured_rest is not True or config.get('measured_rest_transfer') is not True:
        raise ValueError('Isaac withdrawal requires explicit measured-rest transfer admission')
    unsupported = [name for name in config if name.startswith(('coupled_', 'leaf_motion_'))
                   or name in ('panel_plan_path', 'panel_plan_sha256', 'panel_continuations')]
    if unsupported:
        raise ValueError('Native motion/coupled/panel proofs are not admitted for Isaac: ' + ', '.join(sorted(unsupported)))
    if config.get('contact_audit_name') != 'independent-contact-audit.json':
        raise ValueError('Original actual Isaac contact audit required')
    context = admit_isaac_release_context(source, robot=config['robot_path'],
        door_xml=config['door_xml_path'], door_usd=config['door_usd_path'], profile=config['grasp_profile'])
    from .isaac_release_context_reconciliation import reconcile_isaac_release_context
    context, reconciliation = reconcile_isaac_release_context(screen, context)
    if reconciliation['difference_count']:
        # Diagnostic output is captured in the launch log, outside the immutable
        # source identity used by all downstream candidate/audit comparisons.
        print('ISAAC_SOURCE_RECONCILIATION '+json.dumps(reconciliation,sort_keys=True),flush=True)
    admission = context.admission
    validate_candidate_binding(screen, context)
    if (screen.get('runtime_route_exported') is not False
            or screen.get('physical_contact_qualification') is not False):
        raise ValueError('Preserve the original unpromoted Isaac candidate')
    if (config.get('source_state_sha256') != admission['source_qualification']['state_sha256']
            or config.get('start_time_s') != context.terminal_time_s):
        raise ValueError('Explicit Isaac source state and exact stage epoch required')
    if (audit.get('schema') != 'doorbench.isaac-release-dense-geometry-audit.v1'
            or audit.get('source_engine') != 'isaac-physx' or audit.get('passed') is not True
            or audit.get('samples') != 2001 or audit.get('physics_steps') != 0
            or audit.get('active_state_writes') != 0 or audit.get('source_sample_playback') != 0
            or audit.get('source_admission') != admission
            or audit.get('source_context_sha256') != context.sha256
            or audit.get('initial_episode_time_s') != context.terminal_time_s
            or audit.get('grasp_profile') != config['grasp_profile']
            or audit.get('original_thresholds') != ORIGINAL_LIMITS
            or set(audit.get('checks', {})) != ISAAC_DENSE_CHECKS
            or any(value is not True for value in audit['checks'].values())
            or audit.get('failures') != [] or audit.get('motor_contract_identity_bound') is not True
            or audit.get('runtime_route_exported') is not False
            or audit.get('physical_contact_qualification') is not False
            or audit.get('delivered_motor_force_checked') is not False
            or audit.get('motor_force_feasibility_inferred') is not False):
        raise ValueError('Complete original Isaac dense geometry audit required')
    archive = context.state_archive_path.resolve()
    if (Path(audit.get('source_physics_archive', '')).resolve() != archive
            or _bound_digest(audit.get('input_sha256', {}), archive) != digest(archive)
            or _bound_digest(audit['input_sha256'], screen_path) != digest(screen_path)):
        raise ValueError('Isaac audit must bind the actual NPZ and this exact candidate')
    for name, expected in screen['input_sha256'].items():
        if _bound_digest(audit['input_sha256'], name) != expected:
            raise ValueError('Isaac audit omits candidate/source evidence: ' + name)
    hashes = _verified_hashes(audit['input_sha256'])
    helper = Path(__file__).with_name('isaac_release_context_reconciliation.py').resolve()
    hashes[str(helper)] = digest(helper)
    motor_path = source/'trial/motor-contract.json'
    if admission['input_sha256'].get(str(motor_path.resolve())) != digest(motor_path):
        raise ValueError('Original Isaac motor contract bytes must be source-bound')
    recorded_motors = json.loads(motor_path.read_text())
    if motor_contract_fingerprint(motors) != motor_contract_fingerprint(recorded_motors):
        raise ValueError('Complete actual Isaac motor contract differs, including order/caps/transmission')
    if motors.get('source_xml_sha256') != digest(admission['robot_path']):
        raise ValueError('Isaac controller motor contract uses another robot asset')
    duration = audit['duration_s']
    if (type(duration) not in (int, float) or not np.isfinite(duration) or duration <= 0
            or type(config.get('duration_s')) not in (int, float) or config['duration_s'] != duration):
        raise ValueError('Isaac runtime duration must equal the independently audited clock')
    trials = screen.get('trials')
    if not isinstance(trials, list) or len(trials) != 1:
        raise ValueError('Exactly one independently audited Isaac route required')
    rows = trials[0].get('rows', [])
    if not rows or not np.array_equal(np.asarray(rows[0].get('qpos')), context.qpos):
        raise ValueError('Isaac first route row differs from exact normalized source coordinates')
    context.verify_inputs()
    return dict(source_engine='isaac-physx', source_admission=admission,
        initial_qpos=context.qpos.tolist(), start_time_s=context.terminal_time_s,
        duration_s=float(duration), robot_path=admission['robot_path'],
        door_xml_path=admission['door_xml_path'], door_usd_path=admission['door_usd_path'],
        source_archive_path=str(archive), source_state_sha256=admission['source_qualification']['state_sha256'],
        source_context_sha256=context.sha256, input_sha256=hashes,
        live_exact_prefix_required=True,
        motor_contract_sha256=motor_contract_fingerprint(recorded_motors),
        motor_binding_scope='Complete static original contract; no delivered-force feasibility inferred')


def load_withdrawal_source_context(config, motors, *, measured_rest):
    """Validate source data for a future constructor; does not wire a runtime.

    Existing native v1 documents omit ``source_engine``. Isaac requires the
    explicit value ``isaac-physx`` and the additional fields documented above.
    Fresh admission is mandatory; a serialized context is not an authorization.
    """
    if type(config) is not dict or config.get('schema') != 'doorbench.standing-withdrawal.v1':
        raise ValueError('Explicit standing withdrawal v1 configuration required')
    if type(measured_rest) is not bool or config.get('measured_rest_transfer', False) != measured_rest:
        raise ValueError('Withdrawal must explicitly identify its measured-rest transfer bridge')
    # Snapshot caller data so subsequent mutation cannot change this admission.
    config = json.loads(json.dumps(config, allow_nan=False))
    motors = json.loads(json.dumps(motors, allow_nan=False))
    engine = config.get('source_engine', 'native-mujoco')
    if engine not in ('native-mujoco', 'isaac-physx'):
        raise ValueError('Unknown explicit withdrawal source engine')
    source = Path(config['source_run']).resolve()
    screen_path, audit_path = [Path(config[name]).resolve() for name in ('screen_path','audit_path')]
    before = {str(screen_path):digest(screen_path), str(audit_path):digest(audit_path)}
    if before[str(screen_path)] != config['screen_sha256'] or before[str(audit_path)] != config['audit_sha256']:
        raise ValueError('Withdrawal evidence changed')
    screen, audit = [json.loads(path.read_text()) for path in (screen_path,audit_path)]
    load = _native if engine == 'native-mujoco' else _isaac
    result = load(config,motors,measured_rest,source,screen_path,audit_path,screen,audit)
    if (not np.isfinite([result['start_time_s'],result['duration_s']]).all()
            or result['duration_s'] <= 0):
        raise ValueError('Finite positive withdrawal duration required')
    result['input_sha256'].update(before)
    _verified_hashes(result['input_sha256'])
    result.update(schema='doorbench.withdrawal-source-context.v1', source_run=str(source),
        screen_path=str(screen_path), audit_path=str(audit_path), screen=screen, audit=audit,
        authorized_stages=0, physical_release_qualification=False,
        scope='Detached admission/state/assets only; no plant or motor interface. Original constructor reference checks and live-stage gates remain required.')
    return WithdrawalSourceContext(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False))
