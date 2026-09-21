"""Independently check RH release geometry against a recorded moving door.

The observed door path is a diagnostic replay, never a forecast for a changed
controller. No forces, physical release, support or motor tracking are inferred.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from doorbench.dexterous.landed_left_planner import LandedLeftScene
from doorbench.dexterous.grasp_verification import shadow_surface_qualified


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def interpolate(rows, initial, root, elapsed, duration):
    times = np.r_[0., [r['time_s'] + .5 for r in rows]]
    qs = np.vstack([initial, [r['qpos'] for r in rows]])
    if np.any(np.diff(times) <= 0) or not np.isfinite(qs).all():
        raise ValueError('Finite monotonic route required')
    u = np.clip(elapsed / duration, 0., 1.)
    clock = times[-1] * u**3 * (10 + u * (-15 + 6 * u))
    values = np.array([np.interp(clock, times, qs[:, i]) for i in range(qs.shape[1])]).T
    rotation = Slerp(times, Rotation.from_quat(qs[:, root:root+7][:, [4, 5, 6, 3]]))
    values[:, root+3:root+7] = rotation(clock).as_quat()[:, [3, 0, 1, 2]]
    return values, clock


def right_contacts(m, d, lever):
    mujoco.mj_kinematics(m, d)
    mujoco.mj_collision(m, d)
    invalid = []
    selected = []
    for c in d.contact[:d.ncon]:
        if c.dist >= 0:
            continue
        bodies = [m.body(m.geom_bodyid[g]).name or '' for g in c.geom]
        if not any(b.startswith('robot/rh_') for b in bodies):
            continue
        if 'leaf_handle' not in bodies:
            # Preserve the existing 3 mm nonfoot self/environment penetration gate.
            if c.dist < -.003:
                invalid.append(dict(reason='non-handle nonfoot penetration', bodies=bodies,
                                    penetration_m=-float(c.dist)))
            continue
        side = next(i for i, b in enumerate(bodies) if b.startswith('robot/rh_'))
        name = bodies[side]
        if lever not in c.geom:
            invalid.append(dict(reason='outside grasped lever', body=name, penetration_m=-float(c.dist)))
            continue
        b = int(m.geom_bodyid[c.geom[side]])
        matrix = d.xmat[b].reshape(3, 3)
        local = matrix.T @ (c.pos - d.xpos[b])
        normal = matrix.T @ (c.frame[:3] * (1 if side == 0 else -1))
        axis = d.geom_xmat[lever].reshape(3, 3)[:, 2]
        relative = c.pos - d.geom_xpos[lever]
        axial = float(relative @ axis)
        radial = relative - axial * axis
        margin = float(m.geom_size[lever, 1] - abs(axial))
        alignment = float((matrix @ normal) @ (-radial / max(np.linalg.norm(radial), 1e-12)))
        match = re.fullmatch(r'robot/rh_(ff|mf|rf|lf|th)(distal|middle|proximal)', name)
        anatomy = bool(match and shadow_surface_qualified(*match.groups(), local, normal,
                                                         profile='volar-phalange-v1'))
        row = dict(body=name, body_position_m=local.tolist(), axial_clearance_m=margin,
                   inward_radial_normal_alignment=alignment, penetration_m=-float(c.dist))
        if not anatomy or margin < .001 or alignment <= .8 or c.dist < -.003:
            invalid.append(dict(row, reason='original anatomy/axial/normal/penetration gate'))
        else:
            selected.append(row)
    return invalid, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--diagnostic-run', type=Path, required=True)
    parser.add_argument('--screen', type=Path, required=True)
    parser.add_argument('--previous-screen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh geometry evidence required')
    source = args.source_run.resolve()
    diagnostic = args.diagnostic_run.resolve()
    candidate = json.loads(args.screen.read_text())
    previous = json.loads(args.previous_screen.read_text())
    if candidate['source_admission'] != previous['source_admission']:
        raise ValueError('Routes must share one exact admitted source')
    for report in (candidate, previous):
        if report['source_trajectory_sha256'] != sha(source/'trajectory.npz'):
            raise ValueError('Route source trajectory changed')
    manifest = json.loads((source/'manifest.json').read_text())
    cfg = manifest['configuration']
    robot, door = Path(cfg['robot']), Path(cfg['door'])/'door.xml'
    if sha(robot) != manifest['inputs']['robot']['sha256'] or sha(door) != manifest['inputs']['door']['door.xml']:
        raise ValueError('Original source model required')
    source_raw = json.loads((source/'raw-transitions/manifest.json').read_text())
    raw = json.loads((diagnostic/'raw-transitions/manifest.json').read_text())
    if not source_raw['complete'] or not raw['complete']:
        raise ValueError('Complete actual archives required')
    prefix = len(source_raw['chunks'])
    if [r['sha256'] for r in source_raw['chunks']] != [r['sha256'] for r in raw['chunks'][:prefix]]:
        raise ValueError('Diagnostic episode does not preserve actual source prefix')
    with np.load(source/'trajectory.npz') as z:
        initial = z['terminal_qpos'].copy()
        start = float(z['terminal_time_s'])
    scene = LandedLeftScene(robot, door)
    m, d = scene.m, scene.d
    times, actual = [start], [initial]
    chunks = {}
    for row in raw['chunks'][prefix:]:
        path = diagnostic/'raw-transitions'/row['file']
        if sha(path) != row['sha256']:
            raise ValueError('Measured diagnostic chunk changed')
        chunks[str(path)] = row['sha256']
        with np.load(path) as z:
            times.extend(z['interval_end_s'].tolist())
            actual.extend(z['qpos_after'].copy())
    times, actual = np.array(times), np.array(actual)
    # Every eighth millisecond is an actual recorded state, including both ends.
    indices = np.arange(0, len(times), 4)
    if indices[-1] != len(times)-1:
        indices = np.r_[indices, len(times)-1]
    times, actual = times[indices], actual[indices]
    elapsed = times-start
    if len(times) != 2001 or abs(elapsed[-1]-16.) > 1e-7:
        raise ValueError('This bounded screen expects the complete original 16-second release')
    door_qa = [m.jnt_qposadr[m.joint(n).id] for n in scene.door_names]
    lever = m.geom('leaf_handle_lever_col_n').id
    hand = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
    environment = [g for g in range(m.ngeom) if m.geom_contype[g] and not m.body(m.geom_bodyid[g]).name.startswith('robot/')]
    results = {}
    for label, screen in [('previous', previous), ('candidate', candidate)]:
        reference, clocks = interpolate(screen['trials'][0]['rows'], initial, scene.root, elapsed, 16.)
        for mode in ('nominal_frozen_door', 'recorded_door_replay'):
            failures, touched, surface_samples = [], [], []
            for index, (t, q) in enumerate(zip(times, reference)):
                d.qpos[:] = q
                if mode == 'recorded_door_replay':
                    d.qpos[door_qa] = actual[index, door_qa]
                invalid, selected = right_contacts(m, d, lever)
                if invalid:
                    failures.append(dict(time_s=float(t), route_clock_s=float(clocks[index]), patches=invalid))
                if selected:
                    touched.append(float(t))
                    surface_samples.extend(selected)
            clearance = min(float(mujoco.mj_geomDistance(m, d, g, h, .5, None)) for g in hand for h in environment)
            results[label+'_'+mode] = dict(passed=not failures and clearance >= .04,
                sampled_configurations=len(times), invalid_samples=len(failures),
                first_invalid=failures[0] if failures else None,
                last_invalid=failures[-1] if failures else None,
                invalid_patches=sum(len(row['patches']) for row in failures),
                last_selected_lever_contact_s=max(touched) if touched else None,
                final_rh_environment_clearance_m=clearance,
                minimum_selected_axial_clearance_m=min((r['axial_clearance_m'] for r in surface_samples), default=None),
                failures=failures)
    inputs = [args.screen, args.previous_screen, source/'trajectory.npz', source/'manifest.json',
              source/'raw-transitions/manifest.json', diagnostic/'raw-transitions/manifest.json', robot, door, Path(__file__)]
    output = dict(schema='doorbench.recorded-door-rh-release-geometry.v1', scope=__doc__,
        physics_steps=0, API_calls=0, source_prefix_chunks_equal=prefix,
        measured_episode_interval_s=[float(times[0]), float(times[-1])],
        physical_contact_qualification=False, support_qualification=False,
        unchanged_gates=dict(profile='volar-phalange-v1', anatomical_middle_upper_m=.025,
            axial_margin_m=.001, radial_alignment_strictly_greater_than=.8,
            nonfoot_penetration_limit_m=.003, final_clearance_m=.04),
        results=results, input_sha256={**{str(p.resolve()):sha(p) for p in inputs}, **chunks})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in r.items() if n not in ('failures','first_invalid','last_invalid')}
                      for k,r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
