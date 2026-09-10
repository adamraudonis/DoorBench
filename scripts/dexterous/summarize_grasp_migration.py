#!/usr/bin/env python3
"""Stream recorded pad contacts into compact migration diagnostics, not task gates."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from doorbench.dexterous.json_record_stream import iter_json_object_array


def summarize(rows, *, loaded_threshold_N=.1, bin_seconds=1.):
    if not math.isfinite(loaded_threshold_N) or loaded_threshold_N <= 0:
        raise ValueError('Positive finite diagnostic load threshold required')
    if not math.isfinite(bin_seconds) or bin_seconds <= 0:
        raise ValueError('Positive finite bin duration required')
    bins = {}
    first = None
    previous = -math.inf
    count = 0
    for row in rows:
        t = float(row['sim_time_s'])
        if not math.isfinite(t) or t <= previous:
            raise ValueError('Finite strictly increasing measured epochs required')
        previous = t
        count += 1
        index = math.floor(t / bin_seconds)
        bucket = bins.setdefault(index, {'samples': 0, 'valid_grasp_samples': 0, 'bodies': {}})
        bucket['samples'] += 1
        bucket['valid_grasp_samples'] += bool(row['valid_pad_grasp'])
        for contact in row['contacts']:
            force = float(contact['normal_force_N'])
            if not math.isfinite(force) or force < 0:
                raise ValueError('Finite nonnegative contact normal force required')
            if force <= loaded_threshold_N:
                continue
            body = contact['body'].rsplit('/', 1)[-1]
            position = contact['body_position_m']
            if len(position) != 3 or not all(math.isfinite(x) for x in position):
                raise ValueError('Finite body-local contact position required')
            if not contact['pad_qualified'] and first is None:
                first = {'time_s': t, 'contact': contact}
            entry = bucket['bodies'].setdefault(body, {
                'patches': 0, 'unqualified_patches': 0, 'normal_force_sum_N': 0.,
                'maximum_normal_force_N': 0., 'weighted_position_sum': [0., 0., 0.]})
            entry['patches'] += 1
            entry['unqualified_patches'] += not contact['pad_qualified']
            entry['normal_force_sum_N'] += force
            entry['maximum_normal_force_N'] = max(entry['maximum_normal_force_N'], force)
            for axis in range(3):
                entry['weighted_position_sum'][axis] += force * position[axis]
    result = []
    for index, bucket in sorted(bins.items()):
        for entry in bucket['bodies'].values():
            entry['force_weighted_body_position_m'] = [
                x / entry['normal_force_sum_N'] for x in entry.pop('weighted_position_sum')]
        result.append({'start_s': index * bin_seconds, **bucket})
    return {'scope': 'Diagnostic aggregation only. Force-weighted positions are in each body frame; '
            'positions from different bodies cannot be subtracted. No task qualification or causal inference.',
            'loaded_threshold_N': loaded_threshold_N, 'bin_seconds': bin_seconds,
            'recorded_samples': count, 'first_unqualified_loaded_patch': first, 'bins': result}


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pads = args.trial / 'acquisition-pad-steps.json.gz'
    with gzip.open(pads, 'rt') as stream:
        report = summarize(iter_json_object_array(stream))
    report['input_sha256'] = {pads.name: file_hash(pads)}
    event = report['first_unqualified_loaded_patch']
    trace = args.trial / 'trace.json'
    if event and trace.exists():
        nearest = None
        with trace.open() as stream:
            for row in iter_json_object_array(stream):
                distance = abs(row['time_s'] - event['time_s'])
                if nearest is None or distance < nearest[0]:
                    teacher = row.get('teacher') or {}
                    nearest = (distance, {'time_s': row['time_s'], 'offset_from_contact_s':
                        row['time_s'] - event['time_s'], 'teacher': {k: teacher[k] for k in (
                        'actual_handle_rad', 'actual_leaf_rad', 'commanded_operator_reference_rad',
                        'tracking_error_m', 'hub_gap_m') if k in teacher}})
        report['nearest_trace_to_first_unqualified_contact'] = nearest[1] if nearest else None
        report['input_sha256'][trace.name] = file_hash(trace)
    report['script_sha256'] = file_hash(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'recorded_samples': report['recorded_samples'],
        'first_unqualified_time_s': event['time_s'] if event else None, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
