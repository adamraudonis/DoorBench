#!/usr/bin/env python3
"""Replay one measured tracking error across a candidate path without stepping physics.

This is a diagnostic, not a physical qualification or a bound on future errors.
It tests elbow/leaf clearance only; the full dense and physical audits still apply.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

# A shared virtualenv may be editable-installed against another worktree.
# Bind the diagnostic to the source beside this script, not that installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.json_record_stream import iter_json_object_array
from doorbench.dexterous.screened_panel_path import ScreenedPanelPath


def pose(plan, progress):
    path = ScreenedPanelPath(plan['progress'], plan['coordinates'], plan['duration_s'])
    x = path.spline(progress)
    q = np.array(plan['initial_qpos'], dtype=float)
    r = plan['root_qpos_address']
    rotation = Rotation.from_quat(q[r:r+7][[4, 5, 6, 3]])
    q[r:r+3] += x[:3]
    q[r+3:r+7] = (Rotation.from_rotvec(x[3:6]) * rotation).as_quat()[[3, 0, 1, 2]]
    q[np.array(plan['joint_qpos_addresses'])] = x[6:]
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-trial', type=Path, required=True)
    ap.add_argument('--source-plan', type=Path, required=True)
    ap.add_argument('--candidate-plan', type=Path, required=True)
    ap.add_argument('--time-s', type=float, required=True)
    ap.add_argument('--samples', type=int, default=1001)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.samples < 1001 or not np.isfinite(a.time_s):
        raise ValueError('Require finite measurement time and at least1001 samples')
    source = json.loads(a.source_plan.read_text())
    candidate = json.loads(a.candidate_plan.read_text())
    for key in ('robot_xml_sha256', 'joint_names', 'joint_qpos_addresses', 'root_qpos_address', 'initial_qpos'):
        if source[key] != candidate[key]:
            raise ValueError('Plans must share the exact source pose and model indexing')
    trace = a.source_trial / 'controller-steps.json.gz'
    with gzip.open(trace, 'rt') as f:
        row = next(r for r in iter_json_object_array(f) if r['time_s'] >= a.time_s-1e-8)
    if abs(row['time_s']-a.time_s) > 1e-8:
        raise ValueError('Requested exact controller epoch absent')
    raw = a.source_trial / 'raw-transitions'
    manifest = json.loads((raw / 'manifest.json').read_text())
    chunk = next(c for c in manifest['chunks'] if c['interval_start_s'] <= a.time_s < c['interval_end_s'])
    chunk_path = raw / chunk['file']
    if hashlib.sha256(chunk_path.read_bytes()).hexdigest() != chunk['sha256']:
        raise ValueError('Raw chunk hash mismatch')
    with np.load(chunk_path) as z:
        i = int(np.argmin(abs(z['interval_start_s']-a.time_s)))
        if abs(float(z['interval_start_s'][i])-a.time_s) > 1e-8:
            raise ValueError('Raw interval epoch mismatch')
        actual = z['qpos_before'][i].copy()
    planned = pose(source, row['panel_progress'])
    r = source['root_qpos_address']
    dr = Rotation.from_quat(actual[r:r+7][[4, 5, 6, 3]]) * Rotation.from_quat(planned[r:r+7][[4, 5, 6, 3]]).inv()
    dp = actual[r:r+3] - planned[r:r+3]
    qa = np.array(source['joint_qpos_addresses'])
    dq = actual[qa] - planned[qa]
    cfg = json.loads((a.source_trial / 'manifest.json').read_text())['configuration']
    robot = Path(cfg['robot'])
    if hashlib.sha256(robot.read_bytes()).hexdigest() != source['robot_xml_sha256']:
        raise ValueError('Source robot XML changed')
    sim = DexterousDoorEnv(cfg['door'], robot, json.loads(robot.with_suffix('.audit.json').read_text()))
    m, d = sim.m, sim.d
    leafq = m.joint('leaf_hinge').qposadr[0]
    elbow = m.body('robot/left_elbow_link').id
    leaf = m.body('leaf').id
    ge = [g for g in range(m.ngeom) if m.geom_bodyid[g] == elbow and m.geom_contype[g]]
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == leaf and m.geom_contype[g]]
    if not ge or not gs:
        raise ValueError('Original elbow/leaf colliders required')
    # The source controller records the reference aperture separately from root/joints.
    da = float(actual[leafq] - row['panel_reference_aperture_rad'])
    angles = [candidate['initial_leaf_angle_rad'], candidate['final_leaf_angle_rad']]
    results = {}
    for name in ('nominal', 'measured_root_error', 'measured_root_joint_leaf_error'):
        worst = {'clearance_m': float('inf')}
        count = 0
        for s in np.linspace(0., 1., a.samples):
            q = pose(candidate, s)
            q[leafq] = float(np.interp(s, [0., 1.], angles))
            if name != 'nominal':
                q[r:r+3] += dp
                q[r+3:r+7] = (dr * Rotation.from_quat(q[r:r+7][[4, 5, 6, 3]])).as_quat()[[3, 0, 1, 2]]
            if name == 'measured_root_joint_leaf_error':
                q[qa] += dq
                q[leafq] += da
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            gap = min(float(mujoco.mj_geomDistance(m, d, g, h, .1, None)) for g in ge for h in gs)
            count += gap < .003
            if gap < worst['clearance_m']:
                worst = {'clearance_m': gap, 'progress': float(s), 'leaf_angle_rad': float(q[leafq])}
        results[name] = {'worst': worst, 'samples_below_3mm': count}
    sim.close()
    report = {'scope': __doc__, 'time_s': a.time_s, 'samples': a.samples,
              'source_raw_chunk_sha256': chunk['sha256'],
              'source_plan_sha256': hashlib.sha256(a.source_plan.read_bytes()).hexdigest(),
              'candidate_plan_sha256': hashlib.sha256(a.candidate_plan.read_bytes()).hexdigest(),
              'root_error_rotvec_rad': dr.as_rotvec().tolist(), 'root_error_translation_m': dp.tolist(),
              'joint_names': source['joint_names'],
              'joint_errors_rad': dq.tolist(), 'leaf_error_rad': da, 'results': results}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
