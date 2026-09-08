#!/usr/bin/env python3
"""Build an acquisition candidate by reversing a measured native release.

This preserves the measured root poses as world-space reference information,
not runtime robot pose writes. Small recorded soft-limit excursions are clipped
only in command targets; the original states and motor controls are retained.
A static screen is not evidence of successful motor-driven acquisition.
"""
import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.provenance import capture
from scripts.dexterous.plan_acquisition import collision_failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'robot', 'door', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--start-seconds', type=float, default=.9,
                   help='Beginning of measured release interval, while still grasping')
    p.add_argument('--end-seconds', type=float, default=7.5,
                   help='End of measured interval, after contact-free retreat')
    args = p.parse_args()
    if not all(np.isfinite(v) for v in (args.start_seconds, args.end_seconds)) or not 0 <= args.start_seconds < args.end_seconds:
        p.error('Use finite increasing nonnegative source times')
    if args.output.exists():
        raise SystemExit('Use a new output directory to preserve earlier candidates')
    release_audit = json.loads((args.release / 'release-audit.json').read_text())
    if not release_audit.get('passed'):
        raise SystemExit('Source initialized release did not pass its physical audit')
    ref = json.loads((args.release / 'reference.json').read_text())
    trace = json.loads((args.release / 'trace.json').read_text())
    trajectory = np.load(args.release / 'trajectory.npz', allow_pickle=False)
    if any(len(trajectory[key]) != len(trace) for key in ('qpos', 'qvel', 'ctrl')):
        raise ValueError('Measured state, control and timestamp lengths disagree')
    indices = [i for i, row in enumerate(trace)
               if args.start_seconds <= row['sim_time_s'] <= args.end_seconds][::-1]
    if len(indices) < 2:
        raise ValueError('Selected source interval must contain at least two samples')
    capture(Path(__file__).resolve().parents[2], args.output,
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            timing='before_reverse_native_trace')
    sim = DexterousDoorEnv(args.door, args.robot,
                          json.loads(args.robot.with_suffix('.audit.json').read_text()))
    try:
        sim.reset(images=False, randomize=False)
        m, d = sim.m, sim.d
        ids = [m.joint('robot/' + n).id for n in ref['acquisition']['joint_names']]
        qa = m.jnt_qposadr[ids]
        raw = trajectory['qpos'][indices].copy()
        if raw.shape[1] != m.nq or not np.isfinite(raw).all():
            raise ValueError('Recorded state is nonfinite or incompatible with this plant')
        recorded = raw.copy()
        path = np.clip(raw[:, qa], m.jnt_range[ids, 0], m.jnt_range[ids, 1])
        clipping = float(np.max(np.abs(path - raw[:, qa])))
        if clipping > .02:
            raise ValueError('Recorded joint excursion exceeds the source physical gate')
        recorded[:, qa] = path
        root_path = recorded[:, sim.root_qadr:sim.root_qadr + 7]
        d.qpos[:] = recorded[0]
        mujoco.mj_forward(m, d)
        hand_geoms = [g for g in range(m.ngeom) if m.geom_contype[g]
                      and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
        lever = m.geom('leaf_handle_lever_col_n').id
        initial_gap = min(float(mujoco.mj_geomDistance(m, d, g, lever, 1., None)) for g in hand_geoms)
        failures = []
        if initial_gap < .02 or trace[indices[0]]['contacts']['contacts']:
            failures.append(dict(reason='selected interval does not end contact-free', gap_m=initial_gap))
        if not trace[indices[-1]]['contacts']['opposed']:
            failures.append(dict(reason='selected interval does not begin with loaded opposition'))
        motor_targets = []
        for i in range(len(recorded)):
            for fraction in ([0., .2, .4, .6, .8] if i < len(recorded) - 1 else [0.]):
                d.qpos[:] = recorded[i] if fraction == 0. else (
                    (1. - fraction) * recorded[i] + fraction * recorded[i + 1])
                quaternion = d.qpos[sim.root_qadr + 3:sim.root_qadr + 7]
                quaternion /= np.linalg.norm(quaternion)
                mujoco.mj_forward(m, d)
                failures.extend(dict(sample=i, fraction=fraction, **f)
                                for f in collision_failures(m, d))
                if fraction == 0.:
                    motor_targets.append(d.actuator_length[sim.actuators].copy())
        report = dict(scope=__doc__, passed=not failures, failure_count=len(failures),
                      failures=failures, recorded_source=str(args.release),
                      source_frames=indices, command_target_clipping_max_rad=clipping,
                      source_interval_s=[trace[indices[-1]]['sim_time_s'], trace[indices[0]]['sim_time_s']],
                      initial_hand_gap_m=initial_gap, runtime_robot_pose_writes=0)
        result = copy.deepcopy(ref)
        result['initial_root'] = root_path[0].tolist()
        result['initial_joints'] = dict(zip(ref['acquisition']['joint_names'], path[0].tolist()))
        result['acquisition'].update(
            scope=__doc__, path_qpos=path.tolist(), recorded_root_path=root_path.tolist(),
            initial_plant_qpos=recorded[0].tolist(), source_release=str(args.release),
            source_frames=indices, command_target_clipping_max_rad=clipping)
        (args.output / 'geometry-audit.json').write_text(json.dumps(report, indent=2) + '\n')
        (args.output / 'reference.json').write_text(json.dumps(result, indent=2) + '\n')
        np.savez_compressed(args.output / 'recorded-source.npz',
                            qpos=raw, qvel=trajectory['qvel'][indices], ctrl=trajectory['ctrl'][indices],
                            command_motor_length=np.asarray(motor_targets))
        print(json.dumps({k: v for k, v in report.items() if k not in ('failures', 'source_frames')}))
    finally:
        sim.close()
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
