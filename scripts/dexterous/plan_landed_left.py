#!/usr/bin/env python3
"""Plan left contact from a named measured state or a hash-bound native archive."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import time

import numpy as np

from doorbench.dexterous import landed_left_audit, landed_left_planner
from doorbench.dexterous.landed_left_planner import LandedLeftScene, make_landed_left_plan


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_state(scene, directory, requested_time):
    """Use recorded timestamps, never an inferred NPZ row-rate convention."""
    directory = Path(directory)
    for name, actual in [('robot-input.xml', scene.robot_xml), ('door-input.xml', scene.door_xml)]:
        if sha(directory/name) != sha(actual):
            raise ValueError('Trajectory model input bytes do not match: '+name)
    trace = json.loads((directory/'trace.json').read_text())
    with np.load(directory/'trajectory.npz', allow_pickle=False) as trajectory:
        if len(trace) != len(trajectory['qpos']):
            raise ValueError('Trace timestamps and recorded states have different lengths')
        times = np.array([row['sim_time_s'] for row in trace])
        if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
            raise ValueError('Recorded pose timestamps must be finite and strictly increasing')
        index = int(np.argmin(abs(times-requested_time)))
        if abs(times[index]-requested_time) > 1e-7:
            raise ValueError('Requested time must identify an actual saved pose exactly')
        state = scene.state_from_qpos(trajectory['qpos'][index], pose_time_s=times[index])
        leaf_error = abs(trace[index]['door_q']-state['door_positions']['leaf_hinge'])
        height_error = abs(trace[index]['root_height_m']-state['root'][2])
        if max(leaf_error, height_error) > 1e-9:
            raise ValueError('Recorded qpos and trace endpoint coordinates do not match')
    files = {str(directory/name): sha(directory/name) for name in
             ('trajectory.npz', 'trace.json', 'report.json', 'robot-input.xml', 'door-input.xml')}
    return state, dict(files=files, sample_index=index, actual_pose_time_s=float(times[index]),
                      source_run_passed=json.loads((directory/'report.json').read_text())['passed'],
                      scope='Actual archived state only; source task qualification is not inherited')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('robot', 'door', 'targets', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--state', type=Path, help='Complete actual root/joints/door_positions/pose_time_s JSON')
    source.add_argument('--trial', type=Path, help='Original native archive with trace.json and trajectory.npz')
    parser.add_argument('--time', type=float, help='Exact archived pose timestamp, required with --trial')
    parser.add_argument('--subdivisions', type=int, default=10)
    parser.add_argument('--audit-targets', type=Path, help='Audit an existing candidate instead of fitting one')
    args = parser.parse_args()
    if args.trial and (args.time is None or not np.isfinite(args.time)):
        parser.error('--trial needs a finite exact --time')
    if args.output.exists():
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    frozen = {}
    for src in [Path(__file__), Path(inspect.getfile(landed_left_planner)), Path(inspect.getfile(landed_left_audit))]:
        dst = args.output/src.name
        shutil.copy2(src, dst)
        frozen[dst.name] = sha(dst)
    try:
        return execute(args, frozen)
    except Exception as exc:
        report = dict(passed=False, complete=False, error=repr(exc), sources=frozen,
                      scope='Retained planning/preflight failure; no physical or geometric pass',
                      arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
        return 1


def execute(args, frozen):
    started = time.time()
    scene = LandedLeftScene(args.robot, args.door)
    if args.trial:
        state, evidence = archive_state(scene, args.trial, args.time)
    else:
        state = json.loads(args.state.read_text())
        scene.freeze(state)
        evidence = dict(files={str(args.state): sha(args.state)}, scope='Caller-supplied actual numeric state')
    original = json.loads(args.targets.read_text())
    (args.output/'frozen-state.json').write_text(json.dumps(state, indent=2)+'\n')
    if args.audit_targets:
        config = json.loads(args.audit_targets.read_text())
        landed_left_planner.verify_robot_design_identity(args.robot, config['source_design_identity'])
        report = landed_left_audit.audit_landed_left_path(scene, config, state, subdivisions=args.subdivisions)
    else:
        config, report = make_landed_left_plan(args.robot, args.door, original, state,
                                             subdivisions=args.subdivisions)
        (args.output/'target-config.json').write_text(json.dumps(config, indent=2)+'\n')
    files = {str(p): sha(p) for p in [args.robot, args.door, args.targets]}
    if args.audit_targets:
        files[str(args.audit_targets)] = sha(args.audit_targets)
    report.update(inputs=files, sources=frozen, source_state=evidence,
                  destination_compiled_robot_sha256=landed_left_planner.robot_file_identity(args.robot)['sha256'],
                  destination_door_design_sha256=landed_left_planner.robot_design_identity(args.door)['sha256'],
                  wall_seconds=time.time()-started,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k in ('passed', 'checks', 'fit', 'wall_seconds')}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
