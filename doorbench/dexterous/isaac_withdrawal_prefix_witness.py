"""Read-only exact source-prefix witness with synchronized measured leaf poses.

The historical core archive stays unchanged. Historical transfer JSON numbers
and live leaf samples are canonicalized to float64; their original tensor dtype
is unknown and is deliberately not claimed to have been compared.
"""
import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .isaac_prefix_witness import (LiveIsaacPrefixWitness, PrefixDivergenceError,
                                   PHYSICS_DT_S, _check_hashes, _clock)
from .json_record_stream import iter_json_object_array
from .qualified_isaac_grasp import digest
from .standing_body_record import POSE_CONVENTION

SCHEMA = 'doorbench.live-isaac-withdrawal-prefix-witness.v1'
LEAF_POSE_CONVENTION = POSE_CONVENTION


def _pose(value):
    pose = np.asarray(value)
    if (pose.shape != (7,) or pose.dtype.kind != 'f'
            or not np.isfinite(pose).all()
            or abs(np.linalg.norm(pose[3:]) - 1.) > 2e-6):
        raise ValueError('Finite measured leaf XYZ/WXYZ pose with unit quaternion required')
    return np.array(pose, dtype=np.float64, copy=True, order='C')


class LiveIsaacWithdrawalPrefixWitness:
    """Compare every newly recorded core row and its same-epoch leaf sample."""

    def __init__(self, source_run, *, expected_source_state_sha256, stage_start_s,
                 runtime_configuration, runtime_motor_contract, independent_audit=None):
        if runtime_configuration.get('standing_leaf_pose_convention') != LEAF_POSE_CONVENTION:
            raise ValueError('Explicit synchronized leaf-pose convention required')
        self.core = LiveIsaacPrefixWitness(source_run,
            expected_source_state_sha256=expected_source_state_sha256,
            stage_start_s=stage_start_s, runtime_configuration=runtime_configuration,
            runtime_motor_contract=runtime_motor_contract, independent_audit=independent_audit)
        core = self.core.receipt()
        run = Path(core['source_run'])
        trial = run / 'trial'
        report_path = trial / 'operation-report.json'
        stream_path = trial / 'standing-transfer-steps.json.gz'
        audit_path = run / 'isaac-transfer-audit.json'
        paths = (report_path, stream_path, audit_path, trial / 'configuration.json')
        hashes = {str(path): digest(path) for path in paths}
        report = json.loads(report_path.read_text())
        configuration = json.loads((trial / 'configuration.json').read_text())
        if ('standing_transfer' not in report
                or configuration.get('standing_planner_body_pose_convention') != POSE_CONVENTION):
            raise ValueError('Qualified measured standing-transfer source required')
        # Core admission has independently checked the transfer audit; ensure it
        # bound these exact bytes, rather than a parallel unrelated leaf stream.
        for path in (report_path, stream_path, audit_path):
            if core['input_sha256'].get(str(path)) != hashes[str(path)]:
                raise ValueError('Leaf stream must be bound by core source qualification')
        poses = []
        with gzip.open(stream_path, 'rt') as stream:
            for index, row in enumerate(iter_json_object_array(stream)):
                if _clock(row.get('time_s')) != (index + 1) * PHYSICS_DT_S:
                    raise ValueError('Complete exact 500 Hz leaf clock required')
                poses.append(_pose(row.get('leaf_pose')))
        if len(poses) != core['intervals_required']:
            raise ValueError('Leaf stream must cover every source interval')
        _check_hashes(hashes)
        self._poses = np.asarray(poses, dtype=np.float64)
        self._poses.setflags(write=False)
        self._hashes = hashes
        self._count = 0
        self._failure = None
        self._authorized = False
        self._digest = hashlib.sha256(b'doorbench.isaac-leaf-prefix-float64.v1\0')

    @property
    def complete(self):
        return not self.failed and self.core.complete and self._count == len(self._poses)

    @property
    def failed(self):
        return self._failure is not None or self.core.failed

    def receipt(self):
        core = self.core.receipt()
        permitted = self._authorized and not self.failed
        leaf = dict(passed=permitted, prefix_complete=not self.failed and self._count == len(self._poses),
            stage_entry_authorized=permitted, intervals_verified=self._count,
            intervals_required=len(self._poses),
            last_verified_time_s=self._count * PHYSICS_DT_S if self._count else None,
            pose_convention=LEAF_POSE_CONVENTION,
            comparison='Exact bytes after explicit float64 canonicalization; historical tensor dtype unrecorded',
            historical_tensor_dtype_compared=False, input_sha256=self._hashes,
            matched_sample_sha256=self._digest.hexdigest())
        return copy.deepcopy(dict(schema=SCHEMA, passed=permitted,
            stage_entry_authorized=permitted, prefix_complete=self.complete,
            source_run=core['source_run'], source_terminal_time_s=core['source_terminal_time_s'],
            source_qualification=core['source_qualification'], core=core, leaf_pose=leaf,
            input_sha256={**core['input_sha256'], **self._hashes},
            failure=self._failure or core['failure'],
            scope='Read-only actual core and leaf prefix comparison; no plant writes, command replay or contact qualification'))

    def _fail(self, message, **details):
        if self._failure is None:
            self._failure = dict(message=message, next_interval_index=self._count, **details)
        self._authorized = False
        raise PrefixDivergenceError(self._failure['message'], self.receipt())

    def observe(self, recorded_sample, *, leaf_pose, leaf_time_s):
        if self.failed:
            self._fail('Previously rejected withdrawal prefix cannot resume')
        if self._count >= len(self._poses):
            self._fail('Withdrawal prefix already complete; extra observation rejected')
        try:
            expected_time = (self._count + 1) * PHYSICS_DT_S
            if _clock(leaf_time_s) != expected_time or _clock(recorded_sample.get('time_s')) != expected_time:
                self._fail('Leaf and core must share the exact next post-step epoch')
            pose = _pose(leaf_pose)
            if pose.tobytes() != self._poses[self._count].tobytes():
                self._fail('Recorded measured leaf prefix diverged', expected_time_s=expected_time)
            self.core.observe(recorded_sample)
        except PrefixDivergenceError as error:
            self._fail(str(error))
        except (ValueError, TypeError, AttributeError) as error:
            self._fail('Malformed synchronized leaf/core record', error=str(error))
        self._digest.update(np.float64(expected_time).tobytes() + pose.tobytes())
        self._count += 1
        return self.complete

    def require_stage_entry(self, simulation_time_s):
        if self.failed:
            self._fail('Previously rejected withdrawal prefix cannot authorize entry')
        try:
            now = _clock(simulation_time_s)
            if not self.complete or self._authorized or now != self.core.receipt()['source_terminal_time_s']:
                self._fail('Withdrawal requires its complete prefix at the exact terminal epoch')
            _check_hashes(self._hashes)
            self.core.require_stage_entry(now)
        except PrefixDivergenceError as error:
            self._fail(str(error))
        except (OSError, ValueError, TypeError) as error:
            self._fail('Source leaf evidence changed or entry epoch invalid', error=str(error))
        self._authorized = True
        return self.receipt()
