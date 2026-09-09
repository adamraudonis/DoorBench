#!/usr/bin/env python3
"""Audit complete native episode transitions independently of teacher reports.

Replays kinematics only, never physics or controller execution. Checks every
state/velocity boundary, command and warning interval, with sampled body FK and
every terminal-second collision bound. Contact anatomy has a separate audit.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
from doorbench.dexterous.passage import RobotBounds
from doorbench.dexterous.native_warning_audit import warning_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run/'manifest.json').read_text())['configuration']
    robot, door = Path(config['robot']), Path(config['door'])
    for path, archived in ((robot, run/'robot-input.xml'), (door/'door.xml', run/'door-input.xml')):
        if path.read_bytes() != archived.read_bytes():
            raise ValueError('Physical input differs from captured run')
    sim = DexterousDoorEnv(door, robot, json.loads(robot.with_suffix('.audit.json').read_text()))
    m = sim.m
    data = mujoco.MjData(m)
    bounds = RobotBounds(m)
    ids = np.array([i for i in range(m.nu) if m.actuator(i).name.startswith('robot/')])
    door_ids = np.array([i for i in range(m.nu) if i not in ids], dtype=int)
    caps = m.actuator_forcerange[ids]
    checks = dict(clock=True, state_continuity=True, velocity_continuity=True,
                  finite=True, warning_continuity=True, no_warning_increments=True,
                  all_warning_intervals_present=True)
    previous_q = previous_v = previous_warnings = None
    end = 0.
    count = 0
    geometry_error = force_error = cap_excess = door_command = 0.
    geometry_samples = 0
    tail = deque(maxlen=501)
    warning_sidecar = None
    if (run/'actual-warning-intervals.npz').exists():
        with np.load(run/'actual-warning-intervals.npz',allow_pickle=False) as z:
            warning_sidecar = {k:z[k].copy() for k in ('times','before','after')}
    for row in NativeTransitionArchive.read(run/'raw-transitions'):
        q, v, after_q, after_v, ctrl, force = [np.asarray(row[k]) for k in
            ('qpos_before','qvel_before','qpos_after','qvel_after','controls','actuator_force')]
        start, current_end = row['interval_start_s'], row['interval_end_s']
        checks['clock'] &= abs(start-end)<1e-8 and abs(current_end-start-m.opt.timestep)<1e-8 and row['geometry_time_s']==start
        checks['finite'] &= bool(np.isfinite(np.r_[q,v,after_q,after_v,ctrl,force]).all())
        if previous_q is not None:
            checks['state_continuity'] &= np.array_equal(q, previous_q)
            checks['velocity_continuity'] &= np.array_equal(v, previous_v)
        warning = row.get('mujoco_warning_interval')
        if warning is None and warning_sidecar is not None:
            if not np.array_equal(warning_sidecar['times'][count],[start,current_end]):
                raise ValueError('Warning sidecar does not match actual interval clock')
            warning = warning_interval(warning_sidecar['before'][count],warning_sidecar['after'][count])
        checks['all_warning_intervals_present'] &= warning is not None
        if warning is not None:
            if previous_warnings is not None:
                checks['warning_continuity'] &= warning['before']==previous_warnings
            checks['no_warning_increments'] &= warning['passed']
            previous_warnings = warning['after']
        force_error = max(force_error, float(np.max(abs(ctrl[ids]-force[ids]))))
        cap_excess = max(cap_excess, float(np.max(np.maximum(force[ids]-caps[:,1], caps[:,0]-force[ids]))))
        if len(door_ids):
            door_command = max(door_command, float(np.max(abs(ctrl[door_ids]))))
        if count % 100 == 0 and row['body_ids']:
            data.qpos[:] = q
            mujoco.mj_kinematics(m, data)
            bodies = row['body_ids']
            geometry_samples += 1
            geometry_error = max(geometry_error,
                float(np.max(abs(data.xpos[bodies]-row['body_positions_world_m']))),
                float(np.max(abs(data.xmat[bodies].reshape(-1,3,3)-row['body_rotations_world']))))
        tail.append((current_end, after_q.copy(), after_v.copy()))
        previous_q, previous_v, end = after_q, after_v, current_end
        count += 1
    with np.load(run/'trajectory.npz', allow_pickle=False) as z:
        checks['terminal_state'] = np.array_equal(previous_q,z['terminal_qpos']) and np.array_equal(previous_v,z['terminal_qvel']) and abs(end-float(z['terminal_time_s']))<1e-8
    report = json.loads((run/'report.json').read_text())
    if warning_sidecar is not None:
        checks['complete_warning_sidecar'] = all(len(a)==count for a in warning_sidecar.values())
    checks['complete_duration'] = count == round(report['expected_duration_s']/m.opt.timestep)
    speed = excursion = 0.
    minimum_y = float('inf')
    origin = tail[0][1][sim.root_qadr:sim.root_qadr+2]
    for _, q, v in tail:
        data.qpos[:] = q
        mujoco.mj_kinematics(m, data)
        minimum_y = min(minimum_y, float(bounds(data)[0][1]))
        speed = max(speed, float(np.linalg.norm(v[sim.root_vadr:sim.root_vadr+2])))
        excursion = max(excursion, float(np.linalg.norm(q[sim.root_qadr:sim.root_qadr+2]-origin)))
    checks.update(original_61_motors=len(ids)==61, actual_motor_delivery=force_error<1e-5,
        original_motor_caps=cap_excess<1e-5, no_direct_door_commands=door_command==0.,
        sampled_actual_body_frames=geometry_samples>0 and geometry_error<1e-9,
        quiet_second_beyond_doorway=len(tail)==501 and minimum_y>.2 and speed<.03 and excursion<.03)
    result = dict(passed=all(checks.values()), checks=checks, actual_intervals=count,
        final_time_s=end, maximum_motor_delivery_error_Nm=force_error,
        maximum_motor_cap_excess_Nm=cap_excess, maximum_body_frame_error=geometry_error,
        final_second_minimum_body_y_m=minimum_y, final_second_max_horizontal_speed_m_s=speed,
        final_second_excursion_m=excursion, body_frame_sample_stride=100, geometry_samples=geometry_samples,
        scope=__doc__, raw_manifest_sha256=hashlib.sha256((run/'raw-transitions/manifest.json').read_bytes()).hexdigest())
    (run/'independent-continuous-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    sim.close()
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
