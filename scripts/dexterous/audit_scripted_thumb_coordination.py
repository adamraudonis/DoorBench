#!/usr/bin/env python3
"""Audit an attained-state thumb goal experiment without stepping a plant.

Preserves both changed-command and pre-coordination tracking diagnostics.
Successful replay/prefix checks are not physical task qualification.
"""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from doorbench.dexterous.grasp_verification import scalar_transmission_matrix
from doorbench.dexterous.robot_thumb_flexion_force import RobotThumbFlexionForce

THUMB = ['rh_THJ' + str(k) for k in (5, 4, 3, 2, 1)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trial', 'baseline', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh independent receipt required')
    trial, baseline = args.trial, args.baseline
    protocol = json.loads((trial / 'thumb-coordination-protocol.json').read_text())
    provenance = json.loads((trial / 'provenance.json').read_text())
    robot = Path(provenance['parameters']['robot'])
    if sha(robot) != protocol['robot_xml_sha256']:
        raise ValueError('Authored robot identity changed')
    if sha(baseline / 'provenance.json') != protocol['source_trial_provenance_sha256']:
        raise ValueError('Wrong source physical trial')
    model = mujoco.MjModel.from_xml_path(str(robot))
    names = [model.joint(i).name for i in range(1, model.njnt)]
    actions = [model.actuator(i).name for i in range(model.nu)]
    transmission = scalar_transmission_matrix(model, np.arange(61), np.arange(1, model.njnt))
    calculator = RobotThumbFlexionForce(model, names, actions, transmission)
    with np.load(trial / 'trajectory.npz') as source:
        trajectory = {key: source[key].copy() for key in source.files}
    with np.load(baseline / 'trajectory.npz') as source:
        original = {key: source[key].copy() for key in source.files}
    info, old_info = rows(trial / 'controller.jsonl.gz'), rows(baseline / 'controller.jsonl.gz')
    physical = rows(trial / 'physics.jsonl.gz')
    count = len(info)
    if (len(physical) != count or len(trajectory['force']) != count
            or trajectory['qpos'].shape != (count + 1, 79)
            or list(trajectory['joint_names']) != names
            or list(trajectory['action_names']) != actions):
        raise ValueError('Unmatched actual state/control epochs or original robot order')
    # The trajectory's time column is a nominal integer-tick axis. The frozen
    # producer calls the controller with d.time, copied into the actual interval
    # start, so retain its floating accumulation when replaying the quintic.
    times = np.array([row['contact_interval_start_s'] for row in physical])
    if not np.allclose(times, np.arange(count) * .002, atol=1e-8, rtol=0):
        raise ValueError('Require every consecutive original 2ms decision')
    prefix_steps = round(protocol['start_s'] / .002)
    prefix = {key: bool(np.array_equal(trajectory[key][:prefix_steps + (key != 'force')],
                                       original[key][:prefix_steps + (key != 'force')]))
              for key in ('qpos', 'qvel', 'force', 'time')}
    prefix['joint_goals'] = all(info[i]['goal_joint_names'] == old_info[i]['goal_joint_names']
                               and info[i]['goal_joint_position_rad'] == old_info[i]['goal_joint_position_rad']
                               for i in range(prefix_steps))
    thumb_joint = np.array([names.index(name) for name in THUMB])
    thumb_motor = np.array([actions.index('rh_A_THJ' + str(k)) for k in (5, 4, 3, 2, 1)])
    actual = trajectory['qpos'][1:, 10 + thumb_joint]
    changed = np.array([[dict(zip(row['goal_joint_names'], row['goal_joint_position_rad']))[n]
                         for n in THUMB] for row in info])
    before = np.array([[row['thumb_goals_before_coordination'][n] for n in THUMB] for row in info])
    terminal = np.array([protocol['terminal_goal'][n] for n in THUMB])
    u = np.clip((times - protocol['start_s']) / protocol['ramp_s'], 0, 1)
    blend = u**3 * (10 + u * (-15 + 6*u))
    replay = before * (1 - blend[:, None]) + terminal * blend[:, None]
    replay_error = float(np.max(abs(changed - replay)))
    blend_error = float(np.max(abs(blend - [row['scripted_thumb_coordination_blend'] for row in info])))
    old_error, new_error = actual - before, actual - changed
    bounds = model.jnt_range[[model.joint(n).id for n in THUMB]]
    actual_margin = np.minimum(actual - bounds[:, 0], bounds[:, 1] - actual)
    nominal_margin = np.minimum(changed - bounds[:, 0], bounds[:, 1] - changed)
    start_goal_error = float(np.max(abs(before[prefix_steps] -
                                  [protocol['source_goal_at_start'][n] for n in THUMB])))
    speed = np.diff(changed, axis=0)/.002
    acceleration = np.diff(speed, axis=0)/.002
    pressure = calculator.groups['th'][1]
    removed, retained, modifications = [], [], []
    for i in range(prefix_steps, count):
        unit, _ = calculator.motor_bias(trajectory['qpos'][i, 10:], np.ones(5))
        direction = unit[pressure]
        projection = np.outer(direction, direction) / (direction @ direction)
        gain = info[i]['effective_finger_position_gain_multiplier'] * model.actuator_gainprm[:, 0]
        delta_effort = gain[thumb_motor[-2:]] * (changed[i, -2:] - before[i, -2:])
        removed.append(projection @ delta_effort)
        retained.append(delta_effort - removed[-1])
        modifications.append(delta_effort)
    removed, retained, modifications = map(np.asarray, (removed, retained, modifications))

    def first(mask):
        indices = np.flatnonzero(mask)
        return None if not len(indices) else float(trajectory['time'][indices[0]+1])

    def snapshot(index):
        return dict(endpoint_s=float(trajectory['time'][index+1]),
                    actual_thumb_rad=actual[index].tolist(), goals_rad=changed[index].tolist(),
                    pre_coordination_goals_rad=before[index].tolist(),
                    actual_joint_margin_rad=actual_margin[index].tolist(),
                    qualified_pad_load_N=physical[index]['pad_grasp']['qualified_pad_forces_N'],
                    opposed_grasp=physical[index]['pad_grasp']['valid_pad_grasp'],
                    handle_angle_rad=physical[index]['handle_angle_rad'])

    last = slice(max(prefix_steps, count-500), count)
    checks = dict(exact_complete_23s_prefix=all(prefix.values()),
                  recorded_thumb_goals_replayed=replay_error < 1e-12,
                  recorded_blend_replayed=blend_error < 1e-12,
                  frozen_source_start_goal=start_goal_error < 1e-8,
                  nominal_goals_inside_original_stops=bool(np.min(nominal_margin) >= 0),
                  declared_post23_thumb_speed=bool(np.max(abs(speed[prefix_steps:])) < .08),
                  declared_post23_thumb_acceleration=bool(np.max(abs(acceleration[prefix_steps:])) < .3),
                  calculator_never_stepped=calculator.d.time == 0.)
    result = dict(scope=__doc__, passed=all(checks.values()), checks=checks,
                  actual_task_passed=json.loads((trial/'report.json').read_text())['passed'],
                  decision_count=count, prefix_s=23., first23_seconds_bitwise=prefix,
                  replay_clock='actual pre-integration contact_interval_start_s, identical to the frozen controller d.time input',
                  maximum_nominal_vs_actual_clock_difference_s=float(np.max(abs(times-trajectory['time'][:-1]))),
                  maximum_replayed_goal_error_rad=replay_error, maximum_replayed_blend_error=blend_error,
                  maximum_post23_thumb_goal_speed_rad_s=float(np.max(abs(speed[prefix_steps:]))),
                  maximum_post23_thumb_goal_acceleration_rad_s2=float(np.max(abs(acceleration[prefix_steps:]))),
                  thumb_joint_order=THUMB,
                  max_tracking_changed_goal_rad=np.max(abs(new_error[9500:]), axis=0).tolist(),
                  max_tracking_pre_coordination_goal_rad=np.max(abs(old_error[9500:]), axis=0).tolist(),
                  first_changed_goal_40mrad_failure_s=first(np.max(abs(new_error),axis=1)>=.04),
                  first_pre_coordination_40mrad_failure_s=first(np.max(abs(old_error),axis=1)>=.04),
                  first_actual_thumb_authored_stop_crossing_s=first(np.min(actual_margin,axis=1)<0),
                  minimum_actual_thumb_joint_margin_rad=np.min(actual_margin[9500:],axis=0).tolist(),
                  final_second_mean_actual_thumb_rad=np.mean(actual[last],axis=0).tolist(),
                  final_second_mean_changed_goal_tracking_rad=np.mean(new_error[last],axis=0).tolist(),
                  final_second_mean_pre_coordination_tracking_rad=np.mean(old_error[last],axis=0).tolist(),
                  final_second_pressure_goal_change_effort_Nm=np.mean(modifications[-500:],axis=0).tolist(),
                  final_second_removed_pressure_goal_change_effort_Nm=np.mean(removed[-500:],axis=0).tolist(),
                  final_second_retained_pressure_goal_change_effort_Nm=np.mean(retained[-500:],axis=0).tolist(),
                  effort_scope='Same-state algebraic position-effort change from declared goals; not additional delivered force or physical force response',
                  snapshots=[snapshot(min(count-1, round(t/.002)-1)) for t in (23.,24.,24.3,25.872,27.938,36.)],
                  source_sha256=sha(__file__), robot_xml_sha256=sha(robot),
                  input_sha256={str(path):sha(path) for path in
                      [trial/name for name in ('trajectory.npz','controller.jsonl.gz','physics.jsonl.gz','report.json','provenance.json','thumb-coordination-protocol.json')]
                      + [baseline/name for name in ('trajectory.npz','controller.jsonl.gz','provenance.json')]})
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='snapshots'}, indent=2))


if __name__ == '__main__':
    main()
