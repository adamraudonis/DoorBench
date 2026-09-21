"""Streaming exact witness for a newly executed, actually released Isaac source.

This never installs state, replays a command, or admits native panel evidence.
Fresh released-source and continuation-input audits precede comparison. The
comparison retains fixed-size NPZ blocks and one JSON record, not an episode.
The existing upstream released-source admission may use larger temporary data.
"""
import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .isaac_panel_source import admit_isaac_panel_source
from .isaac_prefix_witness import (PREFIX_FIELDS, COMMAND_SEMANTICS,
    PrefixDivergenceError, _check_hashes, _clock, _coordinates, _sha256)
from .isaac_standing_continuation_measurements import BODY_NAMES
from .continuation_record_stream import iter_continuation_records
from .motor_contract_identity import motor_contract_fingerprint
from .npz_record_stream import iter_npz_records
from .qualified_isaac_grasp import digest
from .standing_body_record import POSE_CONVENTION


SCHEMA = 'doorbench.live-isaac-panel-prefix-witness.v1'
DT = .002
CONTINUATION_FIELDS = ('standing_leaf_pose', 'actual_motor_forces',
    'actual_joint_effort', 'pre_step_joint_velocity', 'continuation_body_poses',
    'actual_foot_loads', 'legacy_root_state_w')
FIELDS = PREFIX_FIELDS + CONTINUATION_FIELDS


def _json_bytes(value):
    """Canonical object-key order only; arrays, scalar types and values exact."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _admit_observations(trial, audit, epoch, physics_sha):
    from .isaac_standing_continuation_audit import admit_standing_continuation_observations
    return admit_standing_continuation_observations(trial, audit,
        expected_epoch_s=epoch, expected_physics_sha256=physics_sha)


def _json_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        yield from iter_continuation_records(stream)


def _merge_hashes(target, incoming):
    if not isinstance(incoming, dict) or not incoming:
        raise ValueError('Complete actual input hashes required')
    for name, expected in incoming.items():
        path = Path(name)
        _sha256(expected)
        if not path.is_absolute():
            raise ValueError('Absolute actual input identities required')
        canonical = str(path.resolve())
        if canonical in target and target[canonical] != expected:
            raise ValueError('Conflicting actual input identity: ' + canonical)
        target[canonical] = expected


class LiveIsaacPanelPrefixWitness:
    """Check the entire live predecessor; authorize its terminal epoch once.

    The caller must feed actual completed-interval records, never source values.
    ``runtime_*_contract`` describe the live sensors/backend, not copied proof.
    ``observe`` is read-only and must precede any new-stage command at T.
    """
    def __init__(self, source_run, *, robot, door_xml, door_usd,
                 expected_source_state_sha256, stage_start_s,
                 runtime_configuration, runtime_motor_contract,
                 runtime_continuation_contract, runtime_motor_readback_contract,
                 continuation_audit):
        run = Path(source_run).resolve(); trial = run/'trial'
        state_sha = _sha256(expected_source_state_sha256)
        terminal = _clock(stage_start_s)
        if terminal <= 0 or abs(terminal/DT-round(terminal/DT)) > 1e-7:
            raise ValueError('Exact positive 500 Hz released-source epoch required')
        source = admit_isaac_panel_source(run, robot=robot, door_xml=door_xml, door_usd=door_usd)
        qualification = source['source_qualification']
        if (source.get('schema') != 'doorbench.isaac-released-panel-source.v1'
                or source.get('source_engine') != 'isaac-physx'
                or Path(source['source_run']).resolve() != run
                or source.get('source_time_s') != terminal
                or qualification.get('kind') != 'actual-completed-isaac-withdrawal'
                or qualification.get('passed') is not True
                or qualification.get('time_s') != terminal
                or qualification.get('state_sha256') != state_sha
                or source['extracted_state']['binding']['sha256'] != state_sha
                or source['coordinate_admission'].get('passed') is not True
                or any(type(source.get(k)) is not int or source[k] != 0
                       for k in ('authorized_stages', 'physics_steps', 'source_sample_playback'))):
            raise ValueError('Exact independently qualified actual released source required')
        configuration = json.loads((trial/'configuration.json').read_text())
        coordinates = _coordinates(configuration)
        if _coordinates(runtime_configuration) != coordinates:
            raise ValueError('Runtime physical coordinate semantics differ from source')
        for key, expected in (('standing_leaf_pose_convention', POSE_CONVENTION),
                              ('standing_continuation_body_names', list(BODY_NAMES))):
            if configuration.get(key) != expected or runtime_configuration.get(key) != expected:
                raise ValueError('Explicit identical continuation body order/convention required: ' + key)
        motor_sha = motor_contract_fingerprint(source['motor_contract'])
        if (motor_contract_fingerprint(runtime_motor_contract) != motor_sha
                or len(runtime_motor_contract.get('actuators', [])) != 61):
            raise ValueError('Exact complete original 61-motor contract required')
        paths = dict(physics=trial/'acquisition-physics.npz',
            observations=trial/'standing-continuation-steps.json.gz',
            contract=trial/'standing-continuation-contract.json',
            readback=trial/'motor-readback-contract.json')
        hashes = {}; _merge_hashes(hashes, source['input_sha256'])
        for key, path in paths.items():
            actual = digest(path)
            if key in ('physics', 'observations') and hashes.get(str(path)) != actual:
                raise ValueError('Released source must already bind its actual ' + key)
            _merge_hashes(hashes, {str(path): actual})
        for key, live in (('contract', runtime_continuation_contract),
                          ('readback', runtime_motor_readback_contract)):
            recorded = json.loads(paths[key].read_text())
            if type(live) is not dict or _json_bytes(live) != _json_bytes(recorded):
                raise ValueError('Live measurement contract differs from actual source: ' + key)
        audit = Path(continuation_audit).resolve()
        observation_admission = _admit_observations(trial, audit, terminal, hashes[str(paths['physics'])])
        n = round(terminal/DT)
        if (observation_admission.get('schema') != 'doorbench.isaac-standing-continuation-admission.v1'
                or observation_admission.get('passed') is not True
                or observation_admission.get('accounting_passed') is not True
                or observation_admission.get('source_report_passed') is not True
                or Path(observation_admission.get('source_trial', '')).resolve() != trial
                or Path(observation_admission.get('source_run', '')).resolve() != run
                or observation_admission.get('source_terminal_time_s') != terminal
                or observation_admission.get('physics_dt_s') != DT
                or any(type(observation_admission.get(k)) is not int or observation_admission[k] != n
                       for k in ('physical_intervals', 'observation_intervals'))
                or observation_admission.get('source_physics_sha256') != hashes[str(paths['physics'])]
                or observation_admission.get('motor_contract_sha256') != motor_sha
                or observation_admission.get('body_pose_order') != list(BODY_NAMES)
                or observation_admission.get('robot_joint_names') != coordinates['robot_joint_names']
                or observation_admission.get('door_joint_names') != coordinates['door_joint_names']
                or type(observation_admission.get('authorized_stages')) is not int
                or observation_admission['authorized_stages'] != 0
                or observation_admission.get('physical_task_qualification') is not False):
            raise ValueError('Independent observations must bind the complete exact released-source epoch and inputs')
        # The independent observation adapter replays the whole stream. It must
        # return the bindings for the exact audit and recording just consumed.
        _merge_hashes(hashes, observation_admission['input_sha256'])
        if hashes.get(str(audit)) != digest(audit):
            raise ValueError('Independent continuation admission must bind its saved audit')
        for path in (Path(__file__).resolve(), Path(__file__).with_name('npz_record_stream.py'),
                     Path(__file__).with_name('continuation_record_stream.py')):
            _merge_hashes(hashes, {str(path): digest(path)})
        _check_hashes(hashes)
        widths = dict(time_s=(), root=(13,), joints=(69,), joint_velocity=(69,),
            motor_forces=(61,), door=(len(coordinates['door_joint_names']),),
            door_velocity=(len(coordinates['door_joint_names']),), standing_body_poses=(6, 7),
            standing_leaf_pose=(7,), actual_motor_forces=(61,), actual_joint_effort=(69,),
            pre_step_joint_velocity=(69,), continuation_body_poses=(6, 7), actual_foot_loads=(2,),
            legacy_root_state_w=(13,))
        self._core = iter_npz_records(paths['physics'], widths, expected_rows=n, block_rows=128)
        self._observations = _json_rows(paths['observations'])
        self._hashes = hashes; self._source = copy.deepcopy(source)
        self._observation_admission = copy.deepcopy(observation_admission)
        self._coordinates = coordinates; self._motor_sha = motor_sha
        self._terminal = terminal; self._total = n; self._count = 0
        self._failure = None; self._authorized = False; self._closed = False
        self._core_digest = hashlib.sha256(b'doorbench.isaac-panel-core-prefix.v1\0')
        self._observation_digest = hashlib.sha256(b'doorbench.isaac-panel-ordered-inputs.v1\0')

    @property
    def complete(self):
        return self._failure is None and self._count == self._total

    @property
    def failed(self):
        return self._failure is not None

    @property
    def intervals_verified(self):
        return self._count

    def receipt(self):
        authorized = self._authorized and not self.failed
        return copy.deepcopy(dict(schema=SCHEMA, passed=authorized,
            prefix_complete=self.complete, stage_entry_authorized=authorized,
            intervals_verified=self._count, intervals_required=self._total,
            source_terminal_time_s=self._terminal,
            last_verified_time_s=self._count*DT if self._count else None,
            source_run=self._source['source_run'],
            source_qualification=self._source['source_qualification'],
            source_state_sha256=self._source['extracted_state']['binding']['sha256'],
            runtime_motor_contract_sha256=self._motor_sha, coordinate_semantics=self._coordinates,
            core=dict(fields=list(FIELDS), comparison='Exact dtype, shape and bytes; no tolerance or interpolation',
                matched_sample_sha256=self._core_digest.hexdigest()),
            continuation=dict(comparison='Exact scalar JSON representations and ordered arrays; dictionary keys canonicalized',
                contact_order='Recorded sparse pairs, slots and occupied values retained in order; no multiset comparison',
                historical_tensor_dtype_compared=False,
                matched_sample_sha256=self._observation_digest.hexdigest(),
                independently_admitted=True, admission=self._observation_admission),
            motor_forces_semantics=COMMAND_SEMANTICS,
            backend_effort_semantics='Submitted backend joint actuation and transmission reconstruction; not measured joint reaction torque',
            input_sha256=self._hashes, failure=self._failure,
            comparison_memory='At most 128 NPZ rows per field and one bounded JSON record; upstream source admission is separate',
            scope='Read-only exact live prerequisite witness only; no plant writes, command playback, panel geometry admission or physical panel success'))

    def close(self):
        """Release source readers. Closing an incomplete witness cannot authorize."""
        if not self._closed:
            self._core.close(); self._observations.close(); self._closed = True

    def _fail(self, message, **details):
        if self._failure is None:
            self._failure = dict(message=message, next_interval_index=self._count, **details)
        self._authorized = False
        self.close()
        raise PrefixDivergenceError(self._failure['message'], self.receipt())

    def observe(self, recorded_sample, *, continuation_observation):
        """Compare one actual post-step record without advancing any controller."""
        if self.failed:
            self._fail('Previously rejected panel prefix cannot resume')
        if self._closed or self._count >= self._total:
            self._fail('Closed or complete source prefix cannot accept another observation')
        try:
            expected = next(self._core)
            prior = next(self._observations)
            epoch = (self._count+1)*DT
            if (np.asarray(expected['time_s']).dtype != np.dtype('float64')
                    or expected['time_s'] != epoch
                    or _clock(prior.get('time_s')) != epoch
                    or _clock(prior.get('pose_time_s')) != epoch):
                self._fail('Complete exact source physical and observation clocks required')
            if type(recorded_sample) is not dict or type(continuation_observation) is not dict:
                self._fail('Complete actual physical and observation mappings required')
            checked = []
            for key in FIELDS:
                actual = np.asarray(recorded_sample[key]); wanted = np.asarray(expected[key])
                if actual.dtype != wanted.dtype or actual.shape != wanted.shape:
                    self._fail('Actual physical prefix dtype or shape differs', field=key,
                        expected_dtype=wanted.dtype.str, actual_dtype=actual.dtype.str,
                        expected_shape=list(wanted.shape), actual_shape=list(actual.shape))
                data = actual.tobytes(order='C')
                if data != wanted.tobytes(order='C'):
                    self._fail('Actual physical or commanded-motor prefix diverged', field=key, time_s=epoch)
                checked.append((key, data))
            actual_bytes = _json_bytes(continuation_observation)
            if actual_bytes != _json_bytes(prior):
                keys = sorted(set(prior) | set(continuation_observation))
                difference = next(key for key in keys if key not in prior or key not in continuation_observation
                    or _json_bytes(prior[key]) != _json_bytes(continuation_observation[key]))
                self._fail('Actual ordered continuation measurement differs', field=difference, time_s=epoch)
            if self._count+1 == self._total:
                # Fully exhaust both formats, checking CRC/trailing/missing or
                # extra rows before committing the final accepted interval.
                for iterator in (self._core, self._observations):
                    try: next(iterator)
                    except StopIteration: pass
                    else: self._fail('Source has extra records after its terminal epoch')
            for key, data in checked:
                self._core_digest.update(key.encode()+b'\0'+data)
            self._observation_digest.update(len(actual_bytes).to_bytes(8, 'little')+actual_bytes)
            self._count += 1
            if self.complete: self.close()
            return self.complete
        except PrefixDivergenceError:
            raise
        except Exception as error:
            self._fail('Missing, malformed or truncated actual/source prefix record',
                       error_type=type(error).__name__)

    def require_stage_entry(self, simulation_time_s):
        """Authorize once after all comparisons, before the first panel command."""
        if self.failed:
            self._fail('Previously rejected panel prefix cannot authorize entry')
        try: now = _clock(simulation_time_s)
        except (TypeError, ValueError): self._fail('Invalid panel-entry epoch')
        if not self.complete or now != self._terminal or self._authorized:
            self._fail('Panel entry needs the complete prefix at its exact source epoch', attempted_time_s=now)
        try: _check_hashes(self._hashes)
        except (OSError, ValueError) as error:
            self._fail('Source evidence changed before panel entry', error_type=type(error).__name__)
        self._authorized = True
        return self.receipt()
