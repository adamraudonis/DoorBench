#!/usr/bin/env python3
"""Read-only, frame-complete feasibility audit of DoorBench reference-motion v1.

This measures the frozen recording, never reruns its generator. Collision queries
are exact MuJoCo primitive/convex distances for the published stylized avatar;
that avatar is an approximation of a human, and a sampled check is not continuous
collision detection or a dynamics/contact/balance certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

JOINTS = ['pelvis', 'chest', 'neck', 'head', 'shoulder_l', 'elbow_l', 'wrist_l',
          'shoulder_r', 'elbow_r', 'wrist_r', 'hip_l', 'knee_l', 'ankle_l',
          'hip_r', 'knee_r', 'ankle_r']
BONES = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (1, 7), (7, 8),
         (8, 9), (0, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15)]
LIMBS = [(4, 5, .30), (5, 6, .28), (7, 8, .30), (8, 9, .28),
         (10, 11, .43), (11, 12, .43), (13, 14, .43), (14, 15, .43)]
CRITERIA = dict(limb_error_m=.00001, active_wrist_error_m=.02,
                planted_foot_drift_m=.005, planted_foot_height_error_m=.015,
                body_penetration_m=.003, point_speed_m_s=3.,
                point_step_m=.15, point_acceleration_m_s2=15.,
                angular_speed_deg_s=720., elbow_flexion_deg=150.,
                knee_flexion_deg=155., hip_extension_deg=30.,
                hip_flexion_deg=130., hip_abduction_deg=50.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def flexion(p, a, b, c):
    cos = np.sum(unit(p[:, a]-p[:, b])*unit(p[:, c]-p[:, b]), axis=-1)
    return 180-np.degrees(np.arccos(np.clip(cos, -1, 1)))


def support_metrics(p, contact):
    """Use native declared support states, not an inferred low-height contact."""
    drift = np.zeros(contact.shape); height = np.zeros(contact.shape)
    segments = 0
    for foot, ankle in enumerate((12, 15)):
        anchor = None
        for i, planted in enumerate(contact[:, foot]):
            if not planted:
                anchor = None
                continue
            if anchor is None:
                anchor = p[i, ankle, :2].copy(); segments += 1
            drift[i, foot] = np.linalg.norm(p[i, ankle, :2]-anchor)
            # Viewer foot center is ankle -.024, height .065; support is z=0.
            height[i, foot] = p[i, ankle, 2]-.024-.065/2
    worst_frame, worst_foot = np.unravel_index(np.argmax(drift), drift.shape)
    return {'declared_stance_segments': segments,
            'declared_stance_foot_frames': int(contact.sum()),
            'max_stance_anchor_drift_m': float(drift.max()),
            'worst_slide_frame': int(worst_frame), 'worst_slide_foot': 'left' if worst_foot == 0 else 'right',
            'stance_foot_frames_drift_gt_5mm': int((drift > CRITERIA['planted_foot_drift_m']).sum()),
            'max_stance_foot_height_error_m': float(abs(height).max()),
            'stance_foot_frames_height_error_gt_15mm': int((abs(height) > CRITERIA['planted_foot_height_error_m']).sum()),
            'frames_without_declared_support': int((~contact.any(axis=1)).sum())}


def skeleton_metrics(p, times, targets, active, contact):
    dt = np.diff(times)
    assert np.all(dt > 0)
    errors = np.array([abs(np.linalg.norm(p[:, a]-p[:, b], axis=1)-length)
                       for a, b, length in LIMBS])
    residual = np.linalg.norm(p[:, 9]-targets, axis=1)
    step = np.linalg.norm(np.diff(p, axis=0), axis=2)
    velocity = np.diff(p, axis=0)/dt[:, None, None]
    acceleration = np.diff(velocity, axis=0)/((dt[1:]+dt[:-1])/2)[:, None, None]
    angles = {f'elbow_{side}': flexion(p, *ids) for side, ids in
              [('l', (4, 5, 6)), ('r', (7, 8, 9))]}
    angles.update({f'knee_{side}': flexion(p, *ids) for side, ids in
                   [('l', (10, 11, 12)), ('r', (13, 14, 15))]})
    left = unit(p[:, 4]-p[:, 7]); forward = np.c_[left[:, 1], -left[:, 0], np.zeros(len(p))]
    bad_rom = np.zeros(len(p), bool)
    for name, values in angles.items():
        bad_rom |= values > CRITERIA['elbow_flexion_deg' if name.startswith('elbow') else 'knee_flexion_deg']
    for side, h, k in [('l', 10, 11), ('r', 13, 14)]:
        v = unit(p[:, k]-p[:, h]); f = np.sum(v*forward, axis=1)
        hip = np.degrees(np.arctan2(f, -v[:, 2]))
        abduction = np.degrees(np.arctan2(abs(np.sum(v*left, axis=1)), np.sqrt(f*f+v[:, 2]**2)))
        angles[f'hip_flexion_{side}'] = hip; angles[f'hip_abduction_{side}'] = abduction
        bad_rom |= (hip < -30) | (hip > 130) | (abduction > 50)
    angular_speed = max(float(np.max(abs(np.diff(v))/dt)) for v in angles.values())
    speed = np.linalg.norm(velocity, axis=2)
    worst_frame, worst_joint = np.unravel_index(np.argmax(speed), speed.shape)
    active_indices = np.flatnonzero(active)
    worst_hand = int(active_indices[np.argmax(residual[active])]) if active.any() else None
    result = {'max_native_limb_length_error_m': float(errors.max()),
              'limb_length_frames_failing': int((errors.max(axis=0) > CRITERIA['limb_error_m']).sum()),
              'active_hand_frames': int(active.sum()),
              'max_active_wrist_error_m': float(residual[active].max()) if active.any() else 0.,
              'active_hand_frames_error_gt_20mm': int(((residual > .02) & active).sum()),
              'active_hand_frames_error_gt_80mm': int(((residual > .08) & active).sum()),
              'max_joint_step_m': float(step.max()), 'max_joint_speed_m_s': float(np.linalg.norm(velocity, axis=2).max()),
              'max_joint_acceleration_m_s2': float(np.linalg.norm(acceleration, axis=2).max()) if len(acceleration) else 0.,
              'joint_frames_speed_gt_3m_s': int((np.linalg.norm(velocity, axis=2) > 3).sum()),
              'joint_frames_step_gt_150mm': int((step > .15).sum()),
              'max_inferred_angular_speed_deg_s': angular_speed,
              'worst_joint_speed': {'joint': JOINTS[worst_joint], 'from_frame': int(worst_frame),
                                    'from_time_s': float(times[worst_frame]), 'to_time_s': float(times[worst_frame+1]),
                                    'displacement_m': float(step[worst_frame, worst_joint]),
                                    'speed_m_s': float(speed[worst_frame, worst_joint])},
              'worst_active_wrist_frame': worst_hand,
              'frames_outside_screening_rom': int(bad_rom.sum()),
              'inferred_angles_deg': {k: {'min': float(v.min()), 'max': float(v.max())} for k, v in angles.items()},
              'support': support_metrics(p, contact)}
    return result


def collision_model(xml):
    """Append visual-avatar primitives to a private spec, preserving native DOFs."""
    spec = mujoco.MjSpec.from_file(str(xml)); proxies = []
    for a, b in BONES:
        radius = .14 if (a, b) == (0, 1) else .07 if (a, b) == (1, 2) else .055 if a >= 10 else .037
        name = f'audit_bone_{a}_{b}'
        spec.worldbody.add_body(name=name+'_body', mocap=True).add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                size=[radius, .2, 0], contype=0, conaffinity=0)
        proxies.append((name, 'bone', a, b, radius))
    for a in range(16):
        radius = .108 if a == 3 else .11 if a == 0 else .049
        name = f'audit_joint_{a}'
        spec.worldbody.add_body(name=name+'_body', mocap=True).add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=[radius, 0, 0], contype=0, conaffinity=0)
        proxies.append((name, 'joint', a, a, radius))
    for a in (12, 15):
        name = f'audit_foot_{a}'
        spec.worldbody.add_body(name=name+'_body', mocap=True).add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[.105/2, .22/2, .065/2], contype=0, conaffinity=0)
        proxies.append((name, 'foot', a, a, 0.))
    model = spec.compile(); data = mujoco.MjData(model)
    gids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, p[0]) for p in proxies])
    native = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    return model, data, proxies, gids, native


def collision_metrics(xml, ir, poses, qpos, times, active, targets, lead_frames):
    model, data, proxies, gids, native = collision_model(xml)
    meta = {g['name']: g for b in ir['bodies'] for g in b['geoms']}
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in native]
    semantics = [meta.get(n, {}).get('semantic', 'unknown') for n in names]
    categories = Counter(); geom_hits = Counter(); proxy_hits = Counter(); examples = []
    frames = set(); body_frames = set(); head_frames = set(); feet_frames = set(); core_frames = set()
    minimum = 0.; exact_calls = 0; intended_contacts = 0; worst = None
    point = np.zeros(6); quaternion = np.zeros(4)
    local_center = model.geom_aabb[native, :3].copy()
    local_half = model.geom_aabb[native, 3:].copy()
    for i, p in enumerate(poses):
        data.qpos[:] = qpos[max(0, i-lead_frames)]
        left = unit(p[4]-p[7]); forward = np.array([left[1], -left[0], 0.])
        mins = []; maxs = []
        for g, (_, kind, a, b, radius) in zip(gids, proxies):
            mocap = int(model.body_mocapid[model.geom_bodyid[g]])
            if kind == 'bone':
                delta = p[b]-p[a]
                data.mocap_pos[mocap] = (p[a]+p[b])/2
                mujoco.mju_quatZ2Vec(quaternion, delta)
                data.mocap_quat[mocap] = quaternion
                model.geom_size[g, 1] = np.linalg.norm(delta)/2
                # A capsule box safely encloses the shorter flat-ended cylinder.
                mins.append(np.minimum(p[a], p[b])-radius); maxs.append(np.maximum(p[a], p[b])+radius)
            elif kind == 'joint':
                data.mocap_pos[mocap] = p[a]; mins.append(p[a]-radius); maxs.append(p[a]+radius)
            else:
                center = p[a]+forward*.045+[0, 0, -.024]
                theta = np.arctan2(forward[1], forward[0])-np.pi/2
                data.mocap_pos[mocap] = center; data.mocap_quat[mocap] = [np.cos(theta/2), 0, 0, np.sin(theta/2)]
                half = np.array([abs(np.cos(theta))*.0525+abs(np.sin(theta))*.11,
                                 abs(np.sin(theta))*.0525+abs(np.cos(theta))*.11, .0325])
                mins.append(center-half); maxs.append(center+half)
        mujoco.mj_kinematics(model, data)
        matrices = data.geom_xmat[native].reshape(-1, 3, 3)
        centers = data.geom_xpos[native]+np.einsum('nij,nj->ni', matrices, local_center)
        half = np.einsum('nij,nj->ni', abs(matrices), local_half)
        overlap = np.all(np.array(maxs)[:, None, :] >= (centers-half)[None, :, :], axis=2)
        overlap &= np.all(np.array(mins)[:, None, :] <= (centers+half)[None, :, :], axis=2)
        for human, env in zip(*np.nonzero(overlap)):
            g = int(gids[human]); other = int(native[env])
            name, kind, a, b, _ = proxies[human]
            # A foot intentionally stands on the floor; deep penetration still fails.
            distance = mujoco.mj_geomDistance(model, data, g, other, .01, point); exact_calls += 1
            if distance >= -CRITERIA['body_penetration_m']:
                continue
            semantic = semantics[env]
            if kind == 'foot' and semantic == 'floor' and distance >= -.005:
                continue
            if distance < minimum:
                minimum = float(distance)
                worst = {'frame': i, 'time_s': round(float(times[i]), 4), 'avatar_part': name,
                         'source_geom': names[env], 'semantic': semantic, 'signed_distance_m': float(distance)}
            frames.add(i)
            # Only local right-hand contact with hardware is classified as intended.
            intended = (active[i] and a == b == 9 and semantic in ('operator', 'lock', 'latch', 'sensor')
                        and np.linalg.norm(point[3:]-targets[i]) < .09)
            if intended:
                intended_contacts += 1
            else:
                body_frames.add(i)
                if kind == 'joint' and a == 3: head_frames.add(i)
                if kind == 'foot': feet_frames.add(i)
                if a not in range(4, 10) and b not in range(4, 10): core_frames.add(i)
            categories[semantic] += 1; geom_hits[names[env]] += 1; proxy_hits[name] += 1
            if len(examples) < 12 and not intended:
                examples.append({'frame': i, 'time_s': round(float(times[i]), 4), 'avatar_part': name,
                                 'source_geom': names[env], 'semantic': semantic, 'signed_distance_m': round(float(distance), 6)})
    return {'evaluated_frames': len(poses), 'native_collision_geoms': len(native),
            'visual_only_geoms_not_queried': int(model.ngeom-len(native)-len(proxies)),
            'avatar_primitives': len(proxies), 'signed_distance_queries': exact_calls,
            'frames_with_any_penetration_gt_3mm': len(frames),
            'frames_with_unintended_penetration_gt_3mm': len(body_frames),
            'frames_with_core_body_penetration_gt_3mm': len(core_frames),
            'frames_with_head_penetration_gt_3mm': len(head_frames),
            'frames_with_foot_penetration_gt_5mm_floor_or_3mm_obstacle': len(feet_frames),
            'minimum_signed_distance_m': minimum, 'local_intended_wrist_hardware_intersections': intended_contacts,
            'worst_penetration': worst,
            'penetrating_pair_samples_by_semantic': dict(categories),
            'most_intersected_geoms': dict(geom_hits.most_common(8)),
            'most_intersected_avatar_parts': dict(proxy_hits.most_common(8)), 'examples': examples}


def capability(spec):
    opening = spec['opening']; family = spec['family']; width = opening['width']; height = opening['height']
    notes = []
    if family == 'pet_door':
        notes.append('Pet aperture cannot support this upright or mild-crouch reference gait; do not label synthetic-base traversal as human traversal.')
        if min(width, height) < .216:
            notes.append('Intrinsic obstruction for this rigid avatar: aperture dimension is smaller than the 0.216 m head diameter, even before clearance.')
        elif min(width, height) < .28:
            notes.append('Torso diameter screen fails (0.28 m); this is a conservative orientation-dependent proxy, not proof that every crawling posture is impossible.')
        else:
            notes.append('A crawl/sideways planner would require separate validation; this audit does not assert that all human postures are impossible.')
    if opening.get('horizontal'):
        notes.append('Horizontal opening requires climbing/descent support and vertical traversal; planar walking-through-y is not a validated traversal.')
    if family == 'hatch_ceiling':
        notes.append('Elevated work requires reachable hardware or an explicitly modeled support/ladder; none is certified by this motion.')
    return {'opening_width_m': width, 'opening_height_m': height,
            'intrinsic_rigid_head_obstruction': family == 'pet_door' and min(width, height) < .216,
            'requires_nonwalking_traversal': family in ('pet_door', 'hatch_floor', 'hatch_ceiling'), 'notes': notes}


def audit_one(args):
    row, root, assets = args; door_id = row['door_id']; directory = Path(assets)/'doors'/door_id
    root = Path(root); c = json.loads((root/row['clip']).read_text())
    assert sha(root/row['clip']) == row['clip_sha256']
    assert sha(root/row['trajectory']) == row['trajectory_sha256']
    for name, digest in row['source_sha256'].items():
        assert sha(directory/name) == digest, (door_id, name)
    assert c['avatar_joint_names'] == JOINTS
    with np.load(root/row['trajectory'], allow_pickle=False) as z:
        p = z['actor_joints'].astype(float); times = z['actor_time'].astype(float)
        qpos = z['qpos'].astype(float); contact = z['foot_contact'].astype(bool)
    active = np.asarray(c['hand_active'], bool); targets = np.asarray(c['targets'])
    lead = len(p)-len(qpos)
    assert p.shape == (len(times), 16, 3) and contact.shape == (len(times), 2)
    numeric = skeleton_metrics(p, times, targets, active, contact)
    numeric['max_web_quantized_limb_length_error_m'] = max(float(abs(np.linalg.norm(
        np.asarray(c['avatar']).reshape(-1, 16, 3)[:, a]-np.asarray(c['avatar']).reshape(-1, 16, 3)[:, b], axis=1)-length).max())
        for a, b, length in LIMBS)
    collisions = collision_metrics(directory/'door.xml', json.loads((directory/'model.json').read_text()),
                                   p, qpos, times, active, targets, lead)
    caps = capability(json.loads((directory/'spec.json').read_text()))
    failures = []
    for key, failed in [('fixed_limb_lengths', numeric['limb_length_frames_failing'] > 0),
                        ('wrist_reach', numeric['active_hand_frames_error_gt_20mm'] > 0),
                        ('planted_foot_slide', numeric['support']['stance_foot_frames_drift_gt_5mm'] > 0),
                        ('planted_foot_height', numeric['support']['stance_foot_frames_height_error_gt_15mm'] > 0),
                        ('joint_range_screen', numeric['frames_outside_screening_rom'] > 0),
                        ('point_velocity_screen', numeric['max_joint_speed_m_s'] > 3),
                        ('point_acceleration_screen', numeric['max_joint_acceleration_m_s2'] > 15),
                        ('joint_angular_velocity_screen', numeric['max_inferred_angular_speed_deg_s'] > 720),
                        ('body_obstruction', collisions['frames_with_unintended_penetration_gt_3mm'] > 0),
                        ('intrinsic_rigid_head_obstruction', caps['intrinsic_rigid_head_obstruction'])]:
        if failed: failures.append(key)
    return {'door_id': door_id, 'family': row['family'], 'native_benchmark_outcome': row['outcome'],
            'frames': len(p), 'source_sha256': row['source_sha256'], 'numeric': numeric,
            'collision': collisions, 'capability': caps, 'failed_screening_criteria': failures,
            'screening_pass': not failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('out/reference-motions'))
    parser.add_argument('--assets', type=Path, default=Path('assets'))
    parser.add_argument('--out', type=Path, default=Path('out/reference-feasibility-v1'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--doors', default='all', help='all, families (first per family), or comma-separated IDs')
    args = parser.parse_args(); start = time.monotonic()
    index_path = args.root/'index.json'; index_hash = sha(index_path); index = json.loads(index_path.read_text())
    assert sha(args.assets/'manifest.json') == index['manifest_sha256']
    rows = index['clips']
    if args.doors == 'families':
        seen = set(); rows = [r for r in rows if r['family'] not in seen and not seen.add(r['family'])]
    elif args.doors != 'all':
        wanted = set(args.doors.split(',')); rows = [r for r in rows if r['door_id'] in wanted]
        assert {r['door_id'] for r in rows} == wanted
    args.out.mkdir(parents=True, exist_ok=True); (args.out/'doors').mkdir(exist_ok=True)
    reports = []
    jobs = [(row, str(args.root.resolve()), str(args.assets.resolve())) for row in rows]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for report in pool.map(audit_one, jobs):
            reports.append(report)
            (args.out/'doors'/f"{report['door_id']}.json").write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            if len(reports) % 25 == 0 or len(reports) == len(rows):
                print(f'{len(reports)}/{len(rows)} audited in {time.monotonic()-start:.1f}s', flush=True)
    assert sha(index_path) == index_hash, 'Input index changed during audit'
    counts = Counter(f for r in reports for f in r['failed_screening_criteria'])
    families = {}
    for family in sorted({r['family'] for r in reports}):
        selected = [r for r in reports if r['family'] == family]
        families[family] = {'doors': len(selected), 'screening_passes': sum(r['screening_pass'] for r in selected),
                            'failures': dict(Counter(f for r in selected for f in r['failed_screening_criteria'])),
                            'representative': selected[0]['door_id']}
    report = {'schema': 'doorbench.reference-feasibility-audit.v1', 'source_index_sha256': index_hash,
              'source_generator_commit': index['generator_commit'], 'audit_script_sha256': sha(__file__),
              'runtime': {'mujoco': mujoco.__version__, 'numpy': np.__version__},
              'elapsed_s': round(time.monotonic()-start, 2), 'doors': len(reports), 'families': len(families),
              'actor_frames': sum(r['frames'] for r in reports), 'criteria': CRITERIA,
              'screening_passes': sum(r['screening_pass'] for r in reports), 'doors_failing': dict(counts),
              'native_outcomes': dict(Counter(r['native_benchmark_outcome'] for r in reports)),
              'signed_distance_queries': sum(r['collision']['signed_distance_queries'] for r in reports),
              'head_collision_doors': sum(r['collision']['frames_with_head_penetration_gt_3mm'] > 0 for r in reports),
              'core_body_collision_doors': sum(r['collision']['frames_with_core_body_penetration_gt_3mm'] > 0 for r in reports),
              'limitations': ['Frame-complete at recorded 20 Hz, not continuous collision detection.',
                  'MuJoCo signed distances are exact engine queries on native colliders versus mathematical viewer primitives; meshes use engine convex geometry.',
                  'Human shape is stylized, not anthropometric or deformable. Visual-only door parts, self-collision, balance, support friction, force closure and actuator limits are not certified.',
                  'Declared stance comes from native NPZ and is not exported in web clips. Stance drift is measured from each continuous declared stance anchor.',
                  'ROM and velocity thresholds are explicit engineering screening criteria, not medical or robot-specific joint/actuator limits. Axial rotations cannot be recovered from joint positions.',
                  'Intended contact exemption is limited to active right wrist within 9 cm of its target on operator/lock/latch/sensor geometry. This does not certify a usable grasp.',
                  'Native benchmark success describes scripted generalized forces and synthetic base, independently of human motion feasibility.'],
              'by_family': families,
              'door_reports': [{'door_id': r['door_id'], 'path': f"doors/{r['door_id']}.json", 'failed_screening_criteria': r['failed_screening_criteria']} for r in reports]}
    (args.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('by_family', 'door_reports', 'limitations')}, indent=2))


if __name__ == '__main__':
    main()
