"""Detached selected-window IK/servo observation; no controller interface.

All references, gains, terms and pipeline outputs are supplied by the caller
after the existing calculations. This module does not run IK/FK, evaluate PD,
update filters, infer targets from commands, or authorize a physical stage.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np


SCHEMA = 'doorbench.ik-command-observation.v1'
DT = .002
COMPONENT_FIELDS = frozenset(('ik_joint_names', 'held_ik_position',
    'last_refresh_time_s', 'servo_motor_indices', 'servo_target',
    'servo_reference_velocity', 'measured_position', 'measured_velocity',
    'kp', 'affine_bias', 'impedance_gain', 'extra_damping',
    'reference_velocity_coefficient', 'model_bias_contribution',
    'other_feedforward', 'servo_unclipped', 'goal_position_world',
    'goal_orientation_kind', 'goal_orientation_world'))


def _names(values, label):
    if (type(values) not in (list, tuple) or not values
            or any(type(v) is not str or not v for v in values)
            or len(set(values)) != len(values)):
        raise ValueError('Complete unique ordered '+label+' required')
    return tuple(values)


def _array(value, shape, label):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError('Complete finite '+label+' required')
    return array.copy()


def _time(value, label):
    if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative '+label+' required')
    return float(value)


def _tick(value):
    time = _time(value, 'command clock')
    tick = round(time/DT)
    if abs(time-tick*DT) > 1e-10:
        raise ValueError('Original 500 Hz command clock required')
    return tick


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _motor(value, count, label):
    array = _array(value, (count,), label)
    if array.dtype.kind != 'f' or array.dtype.itemsize not in (4, 8):
        raise ValueError('Actual float32/float64 '+label+' required')
    return dict(value=array.tolist(), dtype=array.dtype.str,
        sha256=hashlib.sha256(array.tobytes()).hexdigest())


class SelectedWindowIkObserver:
    """One data-only snapshot call per selected pre-step command.

    Windows are inclusive command epochs, each at 2 ms spacing. Full vectors
    are kept for every selected tick; capacity is checked before collection.
    The caller must supply every selected command, including window endpoints.
    Unselected calls store nothing. A failure is sticky and cannot be promoted
    into a complete observation. No returned snapshot is a motor command.
    """
    def __init__(self, *, windows, joint_names, motor_names, motor_caps,
                 component_names, source_binding, max_records=2000):
        self.joint_names = _names(joint_names, 'robot joint inventory')
        self.motor_names = _names(motor_names, 'motor inventory')
        self.component_names = _names(component_names, 'component inventory')
        self._caps = _array(motor_caps, (len(self.motor_names), 2), 'original motor caps')
        if np.any(self._caps[:, 0] > self._caps[:, 1]):
            raise ValueError('Ordered original motor caps required')
        if type(max_records) is not int or not 1 <= max_records <= 10000:
            raise ValueError('Explicit bounded selected-window capacity required')
        if type(windows) not in (list, tuple) or not windows:
            raise ValueError('Prospective observation windows required')
        ticks = []
        for pair in windows:
            if type(pair) not in (list, tuple) or len(pair) != 2:
                raise ValueError('Explicit start/end window pairs required')
            start, end = map(_tick, pair)
            if start > end or (ticks and start <= ticks[-1][1]):
                raise ValueError('Ordered disjoint inclusive observation windows required')
            ticks.append((start, end))
        count = sum(end-start+1 for start, end in ticks)
        if count > max_records:
            raise ValueError('Selected windows exceed explicit record capacity')
        if (type(source_binding) is not dict or set(source_binding) != {'episode_id', 'input_sha256'}
                or type(source_binding['episode_id']) is not str or not source_binding['episode_id']
                or type(source_binding['input_sha256']) is not dict or not source_binding['input_sha256']
                or any(type(k) is not str or not k or type(v) is not str
                       or re.fullmatch('[0-9a-f]{64}', v) is None
                       for k, v in source_binding['input_sha256'].items())):
            raise ValueError('Explicit caller-bound episode and input digests required')
        self._binding_json = _encoded(source_binding)
        self._windows = tuple(ticks)
        self._expected = count
        self._capacity = max_records
        self._window = 0
        self._next = ticks[0][0]
        self._last_call = None
        self._rows = []  # Serialized detached copies, never caller arrays.
        self.failure = None

    def capture(self, *, command_time_s, measured_time_s, phase,
                root13_actor_origin, joint_position, joint_velocity,
                components, pipeline_stages, final_motor_command):
        """Copy one already computed command and its same-epoch inputs.

        ``components`` contains every declared component; inactive ones are
        explicitly None. A component supplies the complete held IK solve
        vector, its refresh clock, motor-coordinate servo inputs and existing
        pre-clipping output. It does not substitute producer ``ctrl`` for IK.
        ``pipeline_stages`` is an ordered list of (name, full motor vector)
        copies from actual return/override boundaries. Its last vector must
        equal the final returned command in both dtype and bytes.

        Records describe commands at t, not an observed interval at t+.002.
        A later archive audit must separately cross-bind the physical interval.
        """
        if self.failure is not None:
            raise ValueError('IK observation remains incomplete after failure')
        try:
            return self._capture(command_time_s, measured_time_s, phase,
                root13_actor_origin, joint_position, joint_velocity,
                components, pipeline_stages, final_motor_command)
        except Exception as error:
            self.failure = dict(reason=str(error), recorded_samples=len(self._rows))
            raise

    def _capture(self, time, measured, phase, root, q, v, components, stages, final):
        tick = _tick(time)
        if self._last_call is not None and tick <= self._last_call:
            raise ValueError('Strictly increasing command observations required')
        if self._next is not None and tick > self._next:
            raise ValueError('A selected 500 Hz command was omitted')
        if self._next is None or tick < self._next:
            self._last_call = tick
            return False
        if _time(measured, 'measured input clock') != float(time):
            raise ValueError('Pre-PD position and velocity must share the command epoch')
        if type(phase) is not str or not phase:
            raise ValueError('Actual controller phase required')
        count = len(self.motor_names)
        root = _array(root, (13,), 'actor-origin root state')
        q = _array(q, (len(self.joint_names),), 'complete measured joint position')
        v = _array(v, q.shape, 'complete measured joint velocity')
        if type(components) is not dict or set(components) != set(self.component_names):
            raise ValueError('All declared active/inactive components required')
        copied = {}
        for name in self.component_names:
            component = components[name]
            if component is None:
                copied[name] = None
                continue
            if type(component) is not dict or set(component) != COMPONENT_FIELDS:
                raise ValueError('Complete explicit pre-PD component fields required: '+name)
            names = _names(component['ik_joint_names'], 'held IK solve coordinates')
            if not set(names) <= set(self.joint_names):
                raise ValueError('IK coordinates must belong to the actual robot')
            refresh = _time(component['last_refresh_time_s'], 'actual IK refresh clock')
            if refresh > float(time) or _tick(refresh) > tick:
                raise ValueError('Future IK endpoint cannot be observed')
            indices = component['servo_motor_indices']
            if (type(indices) not in (list, tuple) or not indices or any(type(i) is not int for i in indices)
                    or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= count):
                raise ValueError('Complete ordered component motor indices required')
            width = len(indices)
            item = dict(ik_joint_names=list(names), servo_motor_indices=list(indices), last_refresh_time_s=refresh,
                held_ik_position=_array(component['held_ik_position'], (len(names),), 'held IK position').tolist())
            orientation = component['goal_orientation_kind']
            if orientation not in ('rotation-matrix', 'axis-z'):
                raise ValueError('Record the actual IK orientation constraint without inventing full rotation')
            item['goal_orientation_kind'] = orientation
            for key in COMPONENT_FIELDS - set(item):
                value = component[key]
                if key == 'servo_reference_velocity' and value is None:
                    item[key] = None
                    continue
                shape = ((width, 3) if key == 'affine_bias' else (3,)
                    if key == 'goal_position_world' else ((3, 3) if orientation == 'rotation-matrix' else (3,))
                    if key == 'goal_orientation_world' else (width,))
                item[key] = _array(value, shape, key).tolist()
            if item['servo_reference_velocity'] is None and any(item['reference_velocity_coefficient']):
                raise ValueError('Absent reference velocity must have zero active coefficient')
            copied[name] = item
        if type(stages) not in (list, tuple) or not stages:
            raise ValueError('Actual ordered complete motor pipeline outputs required')
        stage_names = _names([row[0] for row in stages], 'motor pipeline stages')
        pipeline = []
        for name, row in zip(stage_names, stages):
            if type(row) not in (list, tuple) or len(row) != 2:
                raise ValueError('Named already computed motor stage vectors required')
            pipeline.append(dict(name=name, **_motor(row[1], count, 'pipeline motor command')))
        returned = _motor(final, count, 'final returned motor command')
        if any(returned[k] != pipeline[-1][k] for k in ('dtype', 'sha256')):
            raise ValueError('Last pipeline output must exactly equal final returned motor command')
        values = np.asarray(final)
        if np.any(values < self._caps[:, 0]) or np.any(values > self._caps[:, 1]):
            raise ValueError('Final command exceeds original motor caps')
        row = dict(command_time_s=float(time), measured_time_s=float(measured),
            intended_interval_s=[float(time), (tick+1)*DT], completed_interval_observed=False,
            window_index=self._window, phase=phase, root13_actor_origin=root.tolist(),
            joint_position=q.tolist(), joint_velocity=v.tolist(), components=copied,
            motor_pipeline=pipeline, final_motor_command=returned)
        self._rows.append(_encoded(row))
        self._last_call = tick
        end = self._windows[self._window][1]
        if tick == end:
            self._window += 1
            self._next = self._windows[self._window][0] if self._window < len(self._windows) else None
        else:
            self._next = tick+1
        return True

    def receipt(self, *, include_rows=False):
        """A complete window means complete observations, never task success."""
        result = dict(schema=SCHEMA, selected_windows_complete=self.failure is None and len(self._rows) == self._expected,
            failure=self.failure, dt_s=DT, windows_s=[[a*DT,b*DT] for a,b in self._windows],
            expected_records=self._expected, captured_records=len(self._rows), capacity_records=self._capacity,
            source_binding=json.loads(self._binding_json), source_binding_independently_verified=False,
            joint_names=list(self.joint_names), motor_names=list(self.motor_names), motor_caps=self._caps.tolist(),
            component_names=list(self.component_names), physical_qualification=False,
            motor_delivery_measured=False, physical_intervals_cross_bound=False, authorized_stages=0,
            scope='Copies of supplied IK/servo inputs and final returned commands only; no target inference, interpolation, control update or physical admission')
        if include_rows:
            result['rows'] = [json.loads(value) for value in self._rows]
        return json.loads(_encoded(result))

    def export(self, path):
        """Write a new file; incomplete captures remain explicitly incomplete."""
        with Path(path).open('x', encoding='utf-8') as stream:
            json.dump(self.receipt(include_rows=True), stream, allow_nan=False, indent=2)
            stream.write('\n')


def window_finite_differences(document):
    """Detached 2 ms differences; never bridge gaps or different IK coordinates.

    Coefficients are diagnostic algebra from the captured pre-PD primitives.
    This function does not reconstruct or produce a motor command.
    """
    if document.get('schema') != SCHEMA or not document.get('rows'):
        raise ValueError('Recorded selected-window document required')
    output = []
    previous = {}
    for row in document['rows']:
        for name in document['component_names']:
            current = row['components'][name]
            old = previous.get(name)
            if current is None:
                previous.pop(name, None)
                continue
            same = (old is not None and old[0]['window_index'] == row['window_index']
                and abs(row['command_time_s']-old[0]['command_time_s']-DT) < 1e-10
                and current['ik_joint_names'] == old[1]['ik_joint_names']
                and current['servo_motor_indices'] == old[1]['servo_motor_indices'])
            try:
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    coefficient = (np.asarray(current['affine_bias'], dtype=float)[:,2]
                        - np.asarray(current['extra_damping'], dtype=float))
                if not np.isfinite(coefficient).all():
                    raise FloatingPointError('nonfinite result')
            except (FloatingPointError, OverflowError) as error:
                raise ValueError('Nonfinite derived measured velocity coefficient for '+name) from error
            item = dict(command_time_s=row['command_time_s'], component=name,
                ik_joint_names=current['ik_joint_names'], consecutive_sample_pair=same,
                held_ik_step=None, held_ik_interval_velocity=None,
                measured_velocity_coefficient=coefficient.tolist(),
                reference_velocity_coefficient=current['reference_velocity_coefficient'])
            if same:
                try:
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        delta = (np.asarray(current['held_ik_position'], dtype=float)
                            - np.asarray(old[1]['held_ik_position'], dtype=float))
                        velocity = delta/DT
                    if not np.isfinite(delta).all() or not np.isfinite(velocity).all():
                        raise FloatingPointError('nonfinite result')
                except (FloatingPointError, OverflowError) as error:
                    raise ValueError('Nonfinite derived held IK difference or interval velocity for '+name) from error
                item.update(held_ik_step=delta.tolist(), held_ik_interval_velocity=velocity.tolist(),
                    actual_refresh_clock_changed=current['last_refresh_time_s'] != old[1]['last_refresh_time_s'])
            output.append(item)
            previous[name] = (row, current)
    return dict(schema='doorbench.ik-window-finite-differences.v1', rows=output,
        observation_sha256=hashlib.sha256(_encoded(document).encode()).hexdigest(),
        source_binding=json.loads(_encoded(document['source_binding'])),
        source_binding_independently_verified=False,
        scope='Backward differences of recorded held targets; not instantaneous velocity, admissible interpolation duration or tested smoothing',
        interpolation_executed=False, physical_qualification=False, authorized_stages=0)
