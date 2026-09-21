"""Read-only, fail-closed witness of an initialized-standing Isaac prefix.

Call ``observe`` with each *newly recorded* post-step 500 Hz sample, then call
``require_stage_entry`` before submitting the first command for a new stage.
The source archive supplies comparison evidence only: this module exposes no
reference states, simulator interface, state setters, or motor commands.

Source qualification and old captured code remain immutable. Current code may
differ, but the new run must capture its own actual provenance and independently
pass its physical/contact audits. A matching prefix is not a continuation-task
success, proof of equal contacts, or a measurement of delivered actuator torque.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .motor_contract_identity import motor_contract_fingerprint
from .qualified_isaac_grasp import digest


# Keep these fields aligned with scripts/isaac/run_local_operation.py's offline
# audit_transfer_prefix. Byte/dtype equality here is stricter than array_equal.
PREFIX_FIELDS = ('time_s', 'root', 'joints', 'joint_velocity', 'motor_forces',
                 'door', 'door_velocity', 'standing_body_poses')
PHYSICS_DT_S = .002
COORDINATE_FIELDS = ('robot_joint_names', 'door_joint_names',
                     'root_state_convention', 'standing_planner_body_names',
                     'standing_planner_body_pose_convention')
COMMAND_SEMANTICS = ('motor_forces contains the commanded motor-force vector '
                     'submitted for this interval, not measured delivered torque')


class PrefixDivergenceError(ValueError):
    """Sticky failure; the attached receipt may be saved with failed evidence."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def _qualified_source(run, audit):
    # Reuse the original local admission, including independent transfer gates
    # when the source report declares a standing transfer. No planner is run.
    from scripts.dexterous.plan_local_isaac_transfer import load_local_source
    return load_local_source(run, audit)


def _sha256(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Explicit source endpoint SHA-256 required')
    return value


def _clock(value):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isfinite(value)):
        raise ValueError('Finite scalar simulation epoch required')
    return float(value)


def _check_hashes(hashes):
    for name, expected in hashes.items():
        if digest(name) != expected:
            raise ValueError('Source evidence hash mismatch: ' + name)


def _historical_inputs(trial, provenance):
    """Resolve original .py hashes to captured copies, never current .py files."""
    files = provenance.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Complete original source provenance required')
    tracked, captured, names = {}, {}, set()
    for original, expected in files.items():
        _sha256(expected)
        path = Path(original)
        if not path.is_absolute():
            raise ValueError('Original provenance paths must be absolute')
        if path.suffix.lower() == '.py':
            if path.name.lower() in names:
                raise ValueError('Ambiguous captured source basename: ' + path.name)
            names.add(path.name.lower())
            archived = trial / ('source-' + path.name)
            # A source-*.py symlink to today's working tree is not a capture.
            if archived.is_symlink() or archived.resolve().parent != trial:
                raise ValueError('Historical source must be a local captured file')
            captured[original] = dict(captured_path=str(archived), sha256=expected)
            path = archived
        tracked[str(path)] = expected
    if 'isaac_opening.py' not in names:
        raise ValueError('Captured original Isaac producer required')
    _check_hashes(tracked)
    return tracked, captured


def historical_source_hashes(source_run):
    """Verify original captures and assets without requiring today's code bytes.

    This admits historical evidence only. A live witness must still compare all
    newly executed samples before any continuation command may be submitted.
    """
    trial = Path(source_run).resolve() / 'trial'
    provenance = json.loads((trial / 'provenance.json').read_text())
    hashes, _ = _historical_inputs(trial, provenance)
    return hashes


def _coordinates(configuration):
    result = {key: configuration.get(key) for key in COORDINATE_FIELDS}
    if any(value is None for value in result.values()):
        raise ValueError('Complete recorded coordinate semantics required')
    args = configuration.get('args', {})
    if (configuration.get('dt') != PHYSICS_DT_S
            or configuration.get('runtime_pose_writes') != 0
            or configuration.get('direct_door_commands') is not False
            or args.get('acquisition') is not True
            or args.get('operate_after_acquisition') is not True
            or args.get('acquisition_stance_profile') != 'landed-foot-v1'):
        raise ValueError('500 Hz initialized-standing motor-only operation required')
    return copy.deepcopy(result)


class LiveIsaacPrefixWitness:
    """Qualify once, compare each recorded sample, and authorize only at its end.

    ``expected_source_state_sha256`` must come from the new stage's admitted
    source binding, not from an arbitrary archive selected by the runtime.
    ``runtime_configuration`` must describe the arrays actually recorded by the
    new run; names/order are checked before any numeric comparison is accepted.
    Preserve ``receipt()`` in the new run's evidence on success *and* failure.
    """

    def __init__(self, source_run, *, expected_source_state_sha256,
                 stage_start_s, runtime_configuration, runtime_motor_contract,
                 independent_audit=None):
        run = Path(source_run).resolve()
        trial = run / 'trial'
        audit = (Path(independent_audit).resolve() if independent_audit is not None
                 else run / 'independent-contact-audit.json')
        expected = _sha256(expected_source_state_sha256)
        stage_start = _clock(stage_start_s)
        extracted, motors, qualification = _qualified_source(run, audit)
        if (qualification.get('passed') is not True
                or qualification.get('state_sha256') != expected
                or extracted['binding']['sha256'] != expected
                or qualification['time_s'] != stage_start):
            raise ValueError('Stage must bind the exact qualified source endpoint')
        hashes = dict(qualification['input_sha256'])
        _check_hashes(hashes)
        configuration = json.loads((trial / 'configuration.json').read_text())
        provenance = json.loads((trial / 'provenance.json').read_text())
        coordinates = _coordinates(configuration)
        if _coordinates(runtime_configuration) != coordinates:
            raise ValueError('Runtime coordinate order or convention differs from source')
        motor_sha = motor_contract_fingerprint(motors)
        if motor_contract_fingerprint(runtime_motor_contract) != motor_sha:
            raise ValueError('Runtime motor contract or commanded-motor order differs')
        if len(motors.get('actuators', [])) != 61:
            raise ValueError('Complete 61 commanded-motor inventory required')
        historical, captured = _historical_inputs(trial, provenance)
        for name, sha in historical.items():
            if name in hashes and hashes[name] != sha:
                raise ValueError('Conflicting source evidence bindings: ' + name)
            hashes[name] = sha
        with np.load(trial / 'acquisition-physics.npz', allow_pickle=False) as archive:
            if any(key not in archive.files for key in PREFIX_FIELDS):
                raise ValueError('Complete physical and commanded-motor archive required')
            arrays = {key: np.array(archive[key], copy=True, order='C') for key in PREFIX_FIELDS}
        clock = arrays['time_s']
        n = len(clock) if clock.ndim == 1 else 0
        if (not n or clock.dtype != np.dtype('float64')
                or not np.array_equal(clock, np.arange(1, n + 1) * PHYSICS_DT_S)
                or float(clock[-1]) != stage_start):
            raise ValueError('Complete exact 500 Hz source clock through stage epoch required')
        widths = dict(time_s=(), root=(13,), joints=(69,), joint_velocity=(69,),
                      motor_forces=(61,), door=(len(coordinates['door_joint_names']),),
                      door_velocity=(len(coordinates['door_joint_names']),),
                      standing_body_poses=(6, 7))
        for key, array in arrays.items():
            if (array.shape != (n, *widths[key]) or array.dtype.kind != 'f'
                    or not np.isfinite(array).all()):
                raise ValueError('Incomplete or nonfinite source prefix field: ' + key)
            array.setflags(write=False)
        _check_hashes(hashes)
        self._arrays = arrays
        self._hashes = hashes
        self._qualification = copy.deepcopy(qualification)
        self._provenance = copy.deepcopy(provenance)
        self._captured_sources = captured
        self._coordinates = coordinates
        self._motor_sha256 = motor_sha
        self._source_run = str(run)
        self._stage_start_s = stage_start
        self._count = 0
        self._total = n
        self._failure = None
        self._authorized = False
        self._live_digest = hashlib.sha256()
        self._live_digest.update(b'doorbench.isaac-exact-prefix-samples.v1\0')

    @property
    def intervals_verified(self):
        return self._count

    @property
    def complete(self):
        return self._failure is None and self._count == self._total

    @property
    def failed(self):
        return self._failure is not None

    def receipt(self):
        """Return a detached evidence record; it cannot mutate the witness."""
        return copy.deepcopy(dict(
            schema='doorbench.live-isaac-prefix-witness.v1',
            passed=self._authorized and self._failure is None,
            prefix_complete=self.complete, stage_entry_authorized=self._authorized and not self.failed,
            intervals_verified=self._count, intervals_required=self._total,
            source_terminal_time_s=self._stage_start_s,
            last_verified_time_s=float(self._arrays['time_s'][self._count - 1]) if self._count else None,
            fields=list(PREFIX_FIELDS), comparison='Exact dtype, shape and bytes; no tolerance or interpolation',
            motor_forces_semantics=COMMAND_SEMANTICS, measured_motor_torque_checked=False,
            coordinate_semantics=self._coordinates, runtime_motor_contract_sha256=self._motor_sha256,
            source_run=self._source_run, source_qualification=self._qualification,
            source_provenance=self._provenance, historical_source_copies=self._captured_sources,
            input_sha256=self._hashes, matched_sample_sha256=self._live_digest.hexdigest(),
            failure=self._failure,
            scope='Read-only witness of independently supplied live records. Does not set plant state, replay commands, compare raw contacts, qualify a new stage, or substitute historical code for new-run provenance.'))

    def _fail(self, message, **details):
        if self._failure is None:
            self._failure = dict(message=message, next_interval_index=self._count, **details)
        self._authorized = False
        raise PrefixDivergenceError(self._failure['message'], self.receipt())

    def observe(self, recorded_sample):
        """Check one newly recorded post-step sample; no coercion or tolerances.

        Extra fields are allowed but unverified. Do not call after completion;
        the continuation's new samples are outside the prerequisite prefix.
        """
        if self.failed:
            self._fail('Previously rejected prefix cannot resume')
        if self._count >= self._total:
            self._fail('Source prefix already complete; extra observation rejected')
        if not isinstance(recorded_sample, dict):
            self._fail('A complete recorded sample mapping is required')
        checked = []
        for key in PREFIX_FIELDS:
            if key not in recorded_sample:
                self._fail('Missing recorded prefix field', field=key)
            try:
                actual = np.asarray(recorded_sample[key])
                expected = np.asarray(self._arrays[key][self._count])
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    self._fail('Recorded prefix shape or dtype differs', field=key,
                               expected_shape=list(expected.shape), actual_shape=list(actual.shape),
                               expected_dtype=expected.dtype.str, actual_dtype=actual.dtype.str)
                if actual.tobytes(order='C') != expected.tobytes(order='C'):
                    self._fail('Recorded physical/commanded-motor prefix diverged', field=key,
                               expected_time_s=float(self._arrays['time_s'][self._count]))
                checked.append((key, actual.tobytes(order='C')))
            except PrefixDivergenceError:
                raise
            except Exception as error:
                self._fail('Malformed recorded prefix field', field=key, error_type=type(error).__name__)
        # Commit this interval only after every field matches; partial rows never
        # advance the witness or its digest.
        for key, data in checked:
            self._live_digest.update(key.encode() + b'\0' + data)
        self._count += 1
        return self.complete

    def require_stage_entry(self, simulation_time_s):
        """Authorize only before the first new-stage command, at the source end.

        A premature, stale or repeated entry attempt is a sticky failure. Hashes
        are rechecked once here, never in the per-interval 500 Hz comparison.
        """
        if self.failed:
            self._fail('Previously rejected prefix cannot authorize a new stage')
        try:
            now = _clock(simulation_time_s)
        except (TypeError, ValueError):
            self._fail('Invalid stage-entry simulation epoch')
        if not self.complete or now != self._stage_start_s or self._authorized:
            self._fail('New stage requires the complete prefix at its exact terminal epoch',
                       attempted_time_s=now)
        try:
            _check_hashes(self._hashes)
        except (OSError, ValueError) as error:
            self._fail('Source evidence changed before stage entry', error=str(error))
        self._authorized = True
        return self.receipt()
