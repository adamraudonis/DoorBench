#!/usr/bin/env python3
"""Audit a saved finite sensor packet archive independently of teacher state."""
import argparse
import json
from pathlib import Path

import numpy as np

from doorbench.dexterous.sensor_contract import ACTOR_KEYS, SENSOR_KEYS


def audit_recording(directory):
    directory = Path(directory)
    layout = json.loads((directory/'layout.json').read_text())
    expected_numeric = (ACTOR_KEYS-{'rgb_left','rgb_right'}) | {'time_s'}
    with np.load(directory/'actor-sensors.npz', allow_pickle=False) as data, np.load(directory/'actor-rgb.npz', allow_pickle=False) as rgb:
        numeric = {key:data[key] for key in data.files}
        frames = {key:rgb[key] for key in rgb.files}
    time = numeric['time_s']
    n = len(time)
    shapes = {'joint_position':(n,len(layout['joint_order'])), 'joint_velocity':(n,len(layout['joint_order'])),
              'imu_gyro':(n,3), 'imu_accelerometer':(n,3), 'tactile':(n,layout['tactile_dimension']),
              'previous_action':(n,len(layout['action_order'])), 'sensor_time_s':(n,len(SENSOR_KEYS)),
              'sensor_valid':(n,len(SENSOR_KEYS)), 'time_s':(n,)}
    checks = {
        'strict_numeric_keys':set(numeric) == expected_numeric,
        'strict_rgb_keys':set(frames) == {'rgb_left','rgb_right','time_s'},
        'all_shapes':all(key in numeric and numeric[key].shape == shape for key,shape in shapes.items()),
        'finite_arrays':all(array.dtype.kind in 'fbu' and np.isfinite(array).all() for array in numeric.values()),
        'ordered_sample_clock':bool(n > 1 and np.all(np.diff(time) > 0)),
        'normalized_motor_action':bool(np.max(np.abs(numeric['previous_action'])) <= 1),
        'tactile_clipping':bool(np.max(np.abs(numeric['tactile'])) <= 100),
        'fresh_streams':bool(numeric['sensor_valid'].all()),
    }
    for key in ('rgb_left','rgb_right'):
        checks[key+'_shape_type'] = bool(frames[key].dtype == np.uint8 and frames[key].shape == (len(frames['time_s']),128,128,3))
        index = SENSOR_KEYS.index(key)
        capture = numeric['sensor_time_s'][:, index]
        checks[key+'_causal'] = bool(np.all(capture <= time+1e-10) and np.all(time-capture < .041))
        checks[key+'_true_capture_clock'] = bool(np.isin(capture,frames['time_s']).all())
    checks['rgb_acquisition_ordered'] = bool(np.all(np.diff(frames['time_s']) > 0))
    offsets = np.cumsum([0]+[row['dimension'] for row in layout['sensors']])
    hand_indices = np.concatenate([np.arange(offsets[i],offsets[i+1]) for i,row in enumerate(layout['sensors'])
                                   if row['name'].startswith('rh_')])
    checks['right_hand_touch_present'] = bool(np.max(np.abs(numeric['tactile'][:,hand_indices])) > .5)
    return dict(scope='Saved sensor-stream contract, not policy/task capability', passed=all(checks.values()),
                checks=checks, samples=n, frames_per_eye=len(frames['time_s']), duration_s=float(time[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sensors', type=Path, required=True)
    args = parser.parse_args()
    report = audit_recording(args.sensors)
    (args.sensors/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
