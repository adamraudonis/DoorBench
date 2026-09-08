#!/usr/bin/env python3
"""Compare actual transition palm support during two frozen hand withdrawals.

Only archived dynamics wrenches are used. Fresh kinematics check their frames;
collision distance queries are geometric measurements, not new force solves.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--from-time', type=float, default=67.)
    parser.add_argument('--until-time', type=float, default=69.804)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.from_time < args.until_time:
        raise ValueError('Require a finite increasing interval')
    sys.path.insert(0, str(args.run[0].resolve().with_name(args.run[0].resolve().name+'-source')))
    from doorbench.dexterous.environment import DexterousDoorEnv
    from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
    columns = ['time_s', 'leaf_rad', 'leaf_velocity_rad_s', 'palm_normal_N',
               'other_lh_normal_N', 'palm_leaf_distance_m', 'palm_alignment_deg',
               'palm_leaf_x_m', 'palm_leaf_y_m', 'palm_leaf_z_m']
    results = []
    args.output.mkdir(parents=True, exist_ok=False)
    for index, run_arg in enumerate(args.run):
        run = run_arg.resolve()
        cfg = json.loads((run/'manifest.json').read_text())['configuration']
        robot = Path(cfg['robot'])
        sim = DexterousDoorEnv(cfg['door'], robot, json.loads(robot.with_suffix('.audit.json').read_text()))
        model, data = sim.m, sim.d
        palm = model.body('robot/lh_palm').id
        site = model.site('robot/lh_palm_touch').id
        leaf = model.body('leaf').id
        joint = model.joint('leaf_hinge').id
        enabled = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        palm_geoms = np.flatnonzero(enabled & (model.geom_bodyid == palm))
        leaf_geoms = np.flatnonzero(enabled & (model.geom_bodyid == leaf))
        rows = []
        frame_error = 0.
        for raw in NativeTransitionArchive.read(run/'raw-transitions'):
            t = float(raw['interval_start_s'])
            if t < args.from_time-1e-8:
                continue
            if t >= args.until_time-1e-8:
                break
            data.qpos[:] = raw['qpos_before']
            mujoco.mj_kinematics(model, data)
            ids = raw['body_ids']
            error = max(float(np.max(abs(data.xpos[ids]-raw['body_positions_world_m']))),
                        float(np.max(abs(data.xmat[ids].reshape(-1,3,3)-raw['body_rotations_world']))))
            frame_error = max(frame_error, error)
            if error > 1e-9:
                raise ValueError('Actual contact and current FK frames differ')
            load = [0., 0.]
            for contact in raw['contacts']:
                bodies = [model.body(int(b)).name for b in contact['body']]
                sides = [i for i, name in enumerate(bodies) if name.startswith('robot/lh_')]
                if len(sides) != 1 or 'leaf' not in bodies:
                    continue
                side = sides[0]
                load[0 if int(contact['body'][side]) == palm else 1] += max(0., float(contact['wrench_contact_frame'][0]))
            distance = min(float(mujoco.mj_geomDistance(model, data, int(a), int(b), .1, None))
                           for a in palm_geoms for b in leaf_geoms)
            rotation = data.xmat[leaf].reshape(3,3)
            palm_rotation = data.site_xmat[site].reshape(3,3)
            local = rotation.T @ (data.site_xpos[site]-data.xpos[leaf])
            alignment = float(np.degrees(np.arccos(np.clip((-palm_rotation[:,2]) @ rotation[:,1], -1., 1.))))
            rows.append([t, data.qpos[model.jnt_qposadr[joint]], raw['qvel_before'][model.jnt_dofadr[joint]],
                         *load, distance, alignment, *local])
        values = np.asarray(rows)
        if not len(values) or not np.isfinite(values).all() or not np.allclose(np.diff(values[:,0]), .002, atol=1e-9, rtol=0):
            raise ValueError('Require complete finite consecutive actual intervals')
        under = values[:,3] < 2.
        padded = np.r_[False, under, False]
        starts = np.flatnonzero(padded[1:] & ~padded[:-1])
        ends = np.flatnonzero(~padded[1:] & padded[:-1])
        gaps = [{'start_s': float(values[a,0]), 'end_s': float(values[b-1,0]+.002),
                 'duration_s': float((b-a)*.002), 'maximum_palm_gap_m': float(max(values[a:b,5]))}
                for a,b in zip(starts, ends)]
        np.savez_compressed(args.output/f'run-{index+1}.npz', values=values)
        result = dict(run=str(run), samples=len(values), maximum_actual_body_frame_error=frame_error,
                      final=dict(zip(columns, values[-1].tolist())),
                      actual_palm_normal_impulse_Ns=float(np.sum(values[:,3])*.002),
                      under_2N_intervals=gaps, maximum_palm_gap_m=float(max(values[:,5])),
                      source_raw_manifest_sha256=hashlib.file_digest((run/'raw-transitions/manifest.json').open('rb'), 'sha256').hexdigest(),
                      original_trial_passed=json.loads((run/'report.json').read_text())['passed'])
        results.append(result)
        sim.close()
    report = dict(scope='Actual dynamics load comparison; geometric gaps do not replace the original frozen support gates.',
                  columns=columns, from_time_s=args.from_time, until_time_s=args.until_time, runs=results)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.output/'audit-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
