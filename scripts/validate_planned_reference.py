#!/usr/bin/env python3
"""Independently accept/reject saved planned-reference v1 kinematics.

Recompiles the saved original actor MJCF and source door; never imports the IK
solver or trusts its cached diagnostics. Acceptance is scoped to this rigid rig,
contact schedule, explicit tolerances and sampled interpolation resolution.
It is not a dynamics, balance, force-closure or continuous-collision certificate.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

SCHEMA = 'doorbench.planned-reference.v1'
RIG_FINGERPRINT = '7aaa9dc6d1679b4775f9c79dbcb9a077f7c8e8d56a52d0ecf7fe97af071a9dd1'
LANDMARKS = ['pelvis', 'chest', 'neck', 'head', 'shoulder_l', 'elbow_l', 'wrist_l',
             'shoulder_r', 'elbow_r', 'wrist_r', 'hip_l', 'knee_l', 'ankle_l',
             'hip_r', 'knee_r', 'ankle_r']
DEFAULTS = dict(clearance_m=.003, collision_tolerance_m=.00001, ground_penetration_m=.0001,
                fk_position_m=.00001, fk_orientation_rad=.0001, joint_limit_rad=.000001,
                foot_position_m=.001, foot_orientation_rad=math.radians(.5),
                hand_position_m=.01, root_speed_m_s=.8, root_angular_speed_rad_s=1.5,
                hand_surface_gap_m=.005, hand_contact_penetration_m=.003,
                joint_speed_rad_s=2.5, root_acceleration_m_s2=3.,
                root_angular_acceleration_rad_s2=8., joint_acceleration_rad_s2=15.,
                max_sample_dt_s=.025, max_sample_translation_m=.01, max_sample_rotation_rad=.05)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InvalidInput(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise InvalidInput(message)


def normalized_rig_fingerprint(xml):
    root = ET.fromstring(xml)
    pelvis = root.find("./worldbody/body[@name='actor_pelvis']")
    require(pelvis is not None, 'Missing original actor pelvis')
    pelvis.attrib.pop('pos', None); pelvis.attrib.pop('quat', None)
    return hashlib.sha256(ET.tostring(root, encoding='utf-8')).hexdigest()


def angle(a, b):
    """Quaternion sign invariant angular distance; inputs must already be unit."""
    return 2*np.arccos(np.clip(abs(np.sum(a*b, axis=-1)), 0, 1))


def qwidth(model, joint):
    return 7 if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE else 4 if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_BALL else 1


def joint_map(source, combined, names, addresses, width, label):
    expected = [source.joint(i).name for i in range(source.njnt)]
    require(len(names) == len(expected) and set(names) == set(expected), f'{label} joint names must cover the exact model')
    require(len(addresses) == len(names) and all(type(x) is int for x in addresses), f'{label} qpos addresses invalid')
    mapping = []; covered = []
    for name, address in zip(names, addresses):
        original = int(source.joint(name).id); target = int(combined.joint(name).id)
        size = qwidth(source, original)
        require(address >= 0 and address+size <= width, f'{label} qpos address out of bounds')
        covered.extend(range(address, address+size))
        mapping.extend((address+k, int(combined.jnt_qposadr[target])+k) for k in range(size))
    require(sorted(covered) == list(range(width)), f'{label} qpos addresses overlap or leave holes')
    return mapping


class Checks:
    def __init__(self):
        self.counts = Counter(); self.examples = defaultdict(list); self.maxima = {}; self.worst = {}

    def fail(self, code, frame=None, detail=None):
        self.counts[code] += 1
        if len(self.examples[code]) < 6:
            self.examples[code].append({'frame': frame, 'detail': detail})

    def maximum(self, key, value):
        self.maxima[key] = max(self.maxima.get(key, 0.), float(value))

    def bound(self, key, value, limit, frame, detail=None):
        if value > self.maxima.get(key, -math.inf):
            self.worst[key] = {'frame': frame, 'measured': float(value), 'limit': float(limit), 'context': detail}
        self.maximum(key, value)
        if value > limit:
            self.fail(key, frame, {'measured': float(value), 'limit': float(limit), 'context': detail})


def _array(z, name, shape):
    require(name in z, f'Missing trajectory array {name}')
    value = np.asarray(z[name])
    require(value.shape == shape, f'{name} shape {value.shape}, expected {shape}')
    numeric = np.issubdtype(value.dtype, np.number) or (name in ('foot_contact', 'hand_contact') and value.dtype == np.bool_)
    require(numeric and np.isrealobj(value) and np.isfinite(value).all(), f'{name} must be finite real numeric')
    return value.astype(float)


def _quaternions(value, name):
    require(np.all(abs(np.linalg.norm(value, axis=-1)-1) < .00001), f'{name} quaternions must be unit WXYZ')


def _compile(clip, directory):
    actor_xml = clip['actor']['mjcf_xml']
    require(normalized_rig_fingerprint(actor_xml) == RIG_FINGERPRINT, 'Actor XML differs from the original fixed rig')
    actor = mujoco.MjModel.from_xml_string(actor_xml)
    require(actor.nq == 38 and actor.nv == 37, 'Original actor coordinate dimensions changed')
    source = directory/'door.xml'; native = mujoco.MjModel.from_xml_path(str(source))
    spec = mujoco.MjSpec.from_file(str(source)); child = mujoco.MjSpec.from_string(actor_xml)
    spec.attach(child, frame=spec.worldbody.add_frame(name='validator_actor_attach'), prefix='', suffix='')
    model = spec.compile()
    return actor, native, model


def _geometry_metadata(clip, actor):
    rows = clip['actor']['geometries']
    expected = {actor.geom(i).name: i for i in range(actor.ngeom)}
    require(isinstance(rows, list) and len(rows) == len(expected), 'Actor geometry metadata inventory mismatch')
    require({g.get('name') for g in rows} == set(expected), 'Actor geometry metadata names mismatch')
    for row in rows:
        i = expected[row['name']]
        require(row['type'] == mujoco.mjtGeom(actor.geom_type[i]).name.removeprefix('mjGEOM_').lower(), 'Actor geometry type mismatch')
        require(row['body_name'] == actor.body(int(actor.geom_bodyid[i])).name, 'Actor geometry parent mismatch')
        for key, value in [('size', actor.geom_size[i]), ('pos', actor.geom_pos[i])]:
            got = np.asarray(row[key], float)
            require(got.shape == value.shape and np.allclose(got, value, atol=1e-8, rtol=0), f'Actor geometry {key} mismatch')
        quat = np.asarray(row['quat_wxyz'], float)
        require(quat.shape == (4,) and np.isfinite(quat).all(), 'Actor geometry quaternion invalid')
        _quaternions(quat, 'actor geometry')
        require(angle(quat, actor.geom_quat[i]) < 1e-7, 'Actor geometry orientation mismatch')


def _body_names(metadata, expected, label):
    names = metadata['body_names']
    require(isinstance(names, list) and len(names) == len(expected) and set(names) == set(expected), f'{label} body names mismatch')
    return names


def _minimum(model, data, pairs):
    """Exact queried minimum; bounding spheres only prune provably farther pairs."""
    if not pairs: return None, None, 0
    ids = np.asarray(pairs, int); a, b = ids[:, 0], ids[:, 1]
    lower = np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b], axis=1)-model.geom_rbound[a]-model.geom_rbound[b]
    lower[(model.geom_type[a] == mujoco.mjtGeom.mjGEOM_PLANE) | (model.geom_type[b] == mujoco.mjtGeom.mjGEOM_PLANE)] = -np.inf
    best = math.inf; pair = None; queries = 0
    for k in np.argsort(lower):
        if lower[k] > best: break
        d = float(mujoco.mj_geomDistance(model, data, int(a[k]), int(b[k]), 1e6, None)); queries += 1
        if d < best: best = d; pair = [model.geom(int(a[k])).name, model.geom(int(b[k])).name]
    return best, pair, queries


def _support_gaps(model, data, foot, floors):
    """Exact vertical sole-corner gaps to upward horizontal native boxes/planes.

    All four sole corners must have an actual supporting face. Unsupported floor
    geometry/orientations fail closed; side contact alone cannot count as support.
    """
    half = model.geom_size[foot]
    corners = np.array([[x, y, -half[2]] for x in (-half[0], half[0]) for y in (-half[1], half[1])])
    corners = corners @ data.geom_xmat[foot].reshape(3, 3).T+data.geom_xpos[foot]
    gaps = []
    for corner in corners:
        candidates = []
        for g in floors:
            matrix = data.geom_xmat[g].reshape(3, 3); normal = matrix[:, 2]
            if normal[2] < .999999: continue
            local = matrix.T@(corner-data.geom_xpos[g]); typ = model.geom_type[g]
            if typ == mujoco.mjtGeom.mjGEOM_BOX:
                if np.any(abs(local[:2]) > model.geom_size[g, :2]+1e-8): continue
                candidates.append(float(local[2]-model.geom_size[g, 2]))
            elif typ == mujoco.mjtGeom.mjGEOM_PLANE:
                candidates.append(float(local[2]))
        gaps.append(min(candidates, key=abs) if candidates else None)
    return gaps


def task_completion(clip, spec, pelvis, times):
    """Separate task evidence from geometric/derivative acceptance.

    Pelvis crossing comes from fresh FK. Source success and recognition remain
    declared source evidence, never a new mechanism/contact semantics certificate.
    """
    check = Checks(); proposal = clip.get('proposal', {})
    name = proposal.get('scenario', spec.get('benchmark', {}).get('primary_scenario'))
    candidates = [s for s in spec.get('benchmark', {}).get('scenarios', []) if s.get('name') == name]
    require(len(candidates) == 1, 'Proposed scenario is not uniquely declared in the verified spec')
    scenario = candidates[0]; source = proposal.get('source_outcome', {})
    require(proposal.get('door_id') == clip['door_id'] and source.get('door_id') == clip['door_id'], 'Proposal/source door identity mismatch')
    require(source.get('scenario') == name, 'Source outcome scenario mismatch')
    require(proposal.get('source_sha256') == clip['source_sha256'], 'Proposal source hashes mismatch')
    if not clip['complete_proposal']: check.fail('incomplete_proposal')
    if source.get('success') is not True or source.get('outcome') != 'success' or source.get('error'):
        check.fail('task_source_outcome_failed', detail={'success': source.get('success'), 'outcome': source.get('outcome')})
    result = {'scenario': name, 'source_success_declared': source.get('success') is True,
              'proposal_traversal': proposal.get('traversal'), 'complete_proposal': clip['complete_proposal'],
              'completion_scope': 'Observed actor route plus declared source outcome; contact, lock recognition and mechanism semantics are not independently certified.'}
    if name in ('open_and_traverse', 'unlock_and_traverse'):
        if proposal.get('traversal') != 'proposed':
            check.fail('task_traversal_unresolved', detail={'state': proposal.get('traversal'), 'reason': proposal.get('traversal_reason')})
        plane = scenario['pass_plane']; center = np.asarray(plane['center'], float)
        normal = np.asarray(plane['normal'], float); direction = np.asarray(plane['traverse_direction'], float)
        require(center.shape == normal.shape == direction.shape == (3,) and np.isfinite([center, normal, direction]).all(), 'Invalid source pass plane')
        require(abs(np.linalg.norm(normal)-1) < 1e-6 and abs(np.dot(normal, direction)) > .99, 'Unsupported source pass-plane orientation')
        normal *= np.sign(np.dot(normal, direction))
        horizontal = abs(normal[2]) > .99
        height_axis = np.array([0., 1., 0.]) if horizontal else np.array([0., 0., 1.])
        width_axis = np.cross(normal, height_axis); width_axis /= np.linalg.norm(width_axis)
        height_axis = np.cross(width_axis, normal); height_axis /= np.linalg.norm(height_axis)
        width = float(plane['width']); height = float(plane['height'])
        require(math.isfinite(width) and math.isfinite(height) and min(width, height) > 0, 'Invalid pass-plane extents')
        signed = (pelvis-center)@normal; crossings = []; inside_forward = 0
        for i in range(len(pelvis)-1):
            before, after = signed[i:i+2]
            forward = before < 0 <= after; reverse = before > 0 >= after
            if not (forward or reverse): continue
            fraction = float(-before/(after-before)); point = pelvis[i]+fraction*(pelvis[i+1]-pelvis[i])
            offset = point-center
            inside = abs(float(offset@width_axis)) <= width/2+1e-6 and abs(float(offset@height_axis)) <= height/2+1e-6
            record = {'from_frame': i, 'time_s': float(times[i]+fraction*(times[i+1]-times[i])),
                      'position': point.tolist(), 'inside_aperture': inside, 'forward': bool(forward)}
            crossings.append(record)
            if inside and forward: inside_forward += 1
            if not inside: check.fail('task_outside_aperture_crossing', i, record)
        if inside_forward == 0: check.fail('task_no_inside_forward_crossing')
        if signed[0] >= -.001: check.fail('task_did_not_start_before_plane')
        if signed[-1] <= .001: check.fail('task_did_not_finish_beyond_plane')
        goal = scenario.get('goal'); require(isinstance(goal, dict), 'Traversing scenario requires a goal region')
        goal_center = np.asarray(goal['center'], float); radius = float(goal['radius'])
        require(goal_center.shape == (3,) and np.isfinite(goal_center).all() and math.isfinite(radius) and radius > 0, 'Invalid source goal')
        # Ordinary goals describe ground XY discs. Horizontal hatch goals carry Z.
        error = float(np.linalg.norm(pelvis[-1]-goal_center) if horizontal else np.linalg.norm(pelvis[-1, :2]-goal_center[:2]))
        if error > radius+1e-6: check.fail('task_goal_region_not_reached', len(pelvis)-1, {'distance_m': error, 'radius_m': radius})
        result.update(inside_forward_crossings=inside_forward, crossings=crossings[:12],
                      initial_signed_distance_m=float(signed[0]), final_signed_distance_m=float(signed[-1]),
                      final_pelvis=pelvis[-1].tolist(), goal_distance_m=error, goal_radius_m=radius,
                      goal_metric='3D pelvis distance for horizontal hatch' if horizontal else 'XY pelvis projection into ground goal disc')
    elif name == 'locked_recognize':
        result['recognition_scope'] = 'Nontraversing motion permitted. Recognition/locked mechanism evidence comes only from the declared source outcome.'
    else:
        check.fail('task_scenario_completion_not_implemented', detail=name)
    result.update(evidence_pass=not bool(check.counts), failure_counts=dict(check.counts), failure_examples=dict(check.examples))
    return result


def _validate(clip_path, trajectory_path, assets, settings):
    clip_hash = sha(clip_path); trajectory_hash = sha(trajectory_path)
    clip = json.loads(Path(clip_path).read_text()); require(clip.get('schema') == SCHEMA, 'Wrong planned-reference schema')
    require(isinstance(clip.get('door_id'), str) and clip['door_id'] not in ('.', '..') and Path(clip['door_id']).name == clip['door_id'], 'Invalid door ID')
    directory = Path(assets)/'doors'/clip['door_id']
    require(clip.get('units') == 'metres/radians/seconds' and clip.get('up_axis') == 'Z', 'Unsupported units/axis convention')
    require(type(clip.get('complete_proposal')) is bool, 'complete_proposal must be explicit')
    require(clip.get('trajectory_sha256') == trajectory_hash, 'Trajectory checksum mismatch')
    require(clip['actor'].get('landmark_names') == LANDMARKS, 'Actor landmark order mismatch')
    require(set(clip['source_sha256']) == {'door.xml', 'model.json', 'spec.json'}, 'Source hash inventory mismatch')
    for name, digest in clip['source_sha256'].items(): require(sha(directory/name) == digest, f'Source hash mismatch: {name}')
    actor, native, model = _compile(clip, directory); _geometry_metadata(clip, actor)
    actor_names = _body_names(clip['actor'], [actor.body(i).name for i in range(1, actor.nbody)], 'Actor')
    native_names = _body_names(clip['native'], [native.body(i).name for i in range(native.nbody)], 'Native')
    amap = joint_map(actor, model, clip['actor']['joint_names'], clip['actor']['qpos_addresses'], actor.nq, 'Actor')
    nnames = clip['native'].get('joint_names', [native.joint(i).name for i in range(native.njnt)])
    naddr = clip['native'].get('qpos_addresses', native.jnt_qposadr.tolist())
    nmap = joint_map(native, model, nnames, naddr, native.nq, 'Native')
    with np.load(trajectory_path, allow_pickle=False) as z:
        require('actor_time' in z and z['actor_time'].ndim == 1, 'actor_time must be a vector')
        n = len(z['actor_time']); require(n >= 2, 'At least two actor frames are required')
        arrays = {key: _array(z, key, shape) for key, shape in {
            'actor_time': (n,), 'native_time': (n,), 'qpos': (n, native.nq), 'actor_qpos': (n, actor.nq),
            'actor_joints': (n, 16, 3), 'body_pos': (n, len(native_names), 3), 'body_quat': (n, len(native_names), 4),
            'actor_body_pos': (n, len(actor_names), 3), 'actor_body_quat': (n, len(actor_names), 4),
            'foot_pos': (n, 2, 3), 'foot_quat': (n, 2, 4), 'foot_contact': (n, 2),
            'hand_target': (n, 2, 3), 'hand_contact': (n, 2)}.items()}
        for prefix in ['foot_target']:
            if prefix+'_pos' in z or prefix+'_quat' in z:
                arrays[prefix+'_pos'] = _array(z, prefix+'_pos', (n, 2, 3))
                arrays[prefix+'_quat'] = _array(z, prefix+'_quat', (n, 2, 4))
                _quaternions(arrays[prefix+'_quat'], prefix)
    t = arrays['actor_time']; native_time = arrays['native_time']
    require(clip.get('frames') == n and abs(float(clip['duration'])-float(t[-1])) < 1e-6, 'Clip frame count/duration mismatch')
    require(np.all(np.diff(t) > 0) and t[0] >= 0, 'Actor time must be strictly increasing and nonnegative')
    require(np.all(np.diff(native_time) >= 0) and native_time[0] >= 0, 'Native timewarp must be nondecreasing and nonnegative')
    for key in ['foot_contact', 'hand_contact']:
        require(np.isin(arrays[key], [0, 1]).all(), f'{key} must contain binary states')
    for key in ['body_quat', 'actor_body_quat', 'foot_quat']:
        _quaternions(arrays[key], key)
    excluded = clip.get('contact_exclusions')
    require(isinstance(excluded, list) and len(excluded) == n, 'contact_exclusions must have one list per actor frame')
    q = np.repeat(model.qpos0[None], n, axis=0)
    for source, target in amap: q[:, target] = arrays['actor_qpos'][:, source]
    for source, target in nmap: q[:, target] = arrays['qpos'][:, source]
    for j in range(model.njnt):
        address = int(model.jnt_qposadr[j]); typ = model.jnt_type[j]
        if typ == mujoco.mjtJoint.mjJNT_FREE: _quaternions(q[:, address+3:address+7], model.joint(j).name)
        elif typ == mujoco.mjtJoint.mjJNT_BALL: _quaternions(q[:, address:address+4], model.joint(j).name)
    actor_geoms = [int(model.geom(actor.geom(i).name).id) for i in range(actor.ngeom)]
    native_geoms = [int(model.geom(native.geom(i).name).id) for i in range(native.ngeom)
                    if native.geom_contype[i] or native.geom_conaffinity[i]]
    ir = json.loads((directory/'model.json').read_text())
    floor_names = {g['name'] for b in ir['bodies'] for g in b['geoms'] if g.get('semantic') == 'floor'}
    floors = [g for g in native_geoms if model.geom(g).name in floor_names]
    feet = [int(model.geom('actor_geom_foot_'+side).id) for side in ('l', 'r')]
    hands = [int(model.geom('actor_geom_hand_'+side).id) for side in ('l', 'r')]
    hand_names = {model.geom(g).name: side for side, g in enumerate(hands)}
    excluded_ids = []
    for i, rows in enumerate(excluded):
        require(isinstance(rows, list), 'Every frame needs a contact-exclusion list')
        pairs = set()
        for pair in rows:
            require(isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair), 'Exclusion must be [specific actor hand, specific native geom]')
            a, b = pair; require(a in hand_names, 'Only a specific hand geom can be excluded')
            side = hand_names[a]; require(arrays['hand_contact'][i, side] == 1, 'Inactive hand cannot have a collision exclusion')
            try: gid = int(model.geom(b).id)
            except KeyError as exc: raise InvalidInput('Unknown contact-exclusion geometry') from exc
            require(gid in native_geoms and gid not in floors, 'Grip exclusion must name one collidable non-floor native geom')
            require((hands[side], gid) not in pairs, 'Duplicate contact-exclusion pair')
            pairs.add((hands[side], gid))
        excluded_ids.append(pairs)
        for side in range(2):
            require(not arrays['hand_contact'][i, side] or any(a == hands[side] for a, _ in pairs), 'Active hand contact must name its specific native contact geometry')
    base_pairs = []
    for a, b in itertools.combinations(actor_geoms, 2):
        ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
        if ba != bb and model.body_parentid[ba] != bb and model.body_parentid[bb] != ba:
            base_pairs.append((a, b))
    ground_pairs = [(a, b) for a in feet for b in floors]
    base_pairs += [(a, b) for a in actor_geoms for b in native_geoms if (a, b) not in ground_pairs]
    body_ids = [int(model.body(name).id) for name in native_names]
    actor_body_ids = [int(model.body(name).id) for name in actor_names]
    landmarks = [int(model.site('actor_site_'+name).id) for name in LANDMARKS]
    foot_sites = [int(model.site('actor_site_ankle_'+s).id) for s in ('l', 'r')]
    hand_sites = [int(model.site('actor_site_wrist_'+s).id) for s in ('l', 'r')]
    actor_joints = [int(model.joint(actor.joint(j).name).id) for j in range(actor.njnt)]
    actor_dofs = np.array([k for j in actor_joints for k in range(int(model.jnt_dofadr[j]),
        int(model.jnt_dofadr[j])+(6 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 3 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1))])
    root_dof = int(model.joint('actor_root').dofadr[0])
    root_address = int(model.joint('actor_root').qposadr[0])
    dof_labels = []
    for k in actor_dofs:
        joint = int(model.dof_jntid[k]); label = model.joint(joint).name
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            label += ':'+['x', 'y', 'z', 'rx', 'ry', 'rz'][k-model.jnt_dofadr[joint]]
        dof_labels.append(label)
    interpolation_limits = np.full(model.nv, settings['max_sample_rotation_rad'])
    for joint in range(model.njnt):
        adr = int(model.jnt_dofadr[joint])
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE:
            interpolation_limits[adr] = settings['max_sample_translation_m']
        elif model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            interpolation_limits[adr:adr+3] = settings['max_sample_translation_m']
    velocity_limits = np.array([settings['root_speed_m_s'] if root_dof <= k < root_dof+3 else
        settings['root_angular_speed_rad_s'] if root_dof+3 <= k < root_dof+6 else settings['joint_speed_rad_s'] for k in actor_dofs])
    acceleration_limits = np.array([settings['root_acceleration_m_s2'] if root_dof <= k < root_dof+3 else
        settings['root_angular_acceleration_rad_s2'] if root_dof+3 <= k < root_dof+6 else settings['joint_acceleration_rad_s2'] for k in actor_dofs])
    check = Checks(); data = mujoco.MjData(model); anchors = [None, None]; quats = np.zeros((2, 4))
    stance_anchors = []
    pelvis_positions = []
    query_count = 0; collision_samples = 0; min_clearance = math.inf; min_ground = math.inf

    def collisions(state, exceptions, frame):
        nonlocal query_count, collision_samples, min_clearance, min_ground
        data.qpos[:] = state; mujoco.mj_kinematics(model, data)
        distance, pair, queries = _minimum(model, data, [p for p in base_pairs if p not in exceptions]); query_count += queries
        collision_samples += 1
        if distance is not None:
            min_clearance = min(min_clearance, distance)
            if distance < settings['clearance_m']-settings['collision_tolerance_m']:
                check.fail('noncontact_clearance', frame, {'signed_distance_m': distance, 'pair': pair})
        distance, pair, queries = _minimum(model, data, ground_pairs); query_count += queries
        if distance is not None:
            min_ground = min(min_ground, distance)
            if distance < -settings['ground_penetration_m']:
                check.fail('foot_ground_penetration', frame, {'signed_distance_m': distance, 'pair': pair})

    for i in range(n):
        collisions(q[i], excluded_ids[i], i)
        pelvis_positions.append(data.site_xpos[landmarks[0]].copy())
        for field, actual in [('body_pos', data.xpos[body_ids]), ('actor_body_pos', data.xpos[actor_body_ids]),
                              ('actor_joints', data.site_xpos[landmarks]), ('foot_pos', data.site_xpos[foot_sites])]:
            check.bound('fk_'+field, float(np.linalg.norm(actual-arrays[field][i], axis=-1).max()), settings['fk_position_m'], i)
        for k, sid in enumerate(foot_sites): mujoco.mju_mat2Quat(quats[k], data.site_xmat[sid])
        for field, actual in [('body_quat', data.xquat[body_ids]), ('actor_body_quat', data.xquat[actor_body_ids]), ('foot_quat', quats)]:
            check.bound('fk_'+field, float(angle(actual, arrays[field][i]).max()), settings['fk_orientation_rad'], i)
        for j in actor_joints:
            if model.jnt_limited[j]:
                value = q[i, model.jnt_qposadr[j]]; lo, hi = model.jnt_range[j]
                check.bound('joint_limit_violation_rad', max(0., lo-value, value-hi), settings['joint_limit_rad'], i, model.joint(j).name)
        for k in range(2):
            position = data.site_xpos[foot_sites[k]].copy(); quat = quats[k].copy()
            if arrays['foot_contact'][i, k]:
                if anchors[k] is None: anchors[k] = (position.copy(), quat.copy())
                check.bound('stance_position_drift_m', np.linalg.norm(position-anchors[k][0]), settings['foot_position_m'], i, k)
                check.bound('stance_orientation_drift_rad', angle(quat, anchors[k][1]), settings['foot_orientation_rad'], i, k)
                gaps = _support_gaps(model, data, feet[k], floors)
                if any(x is None for x in gaps): check.fail('unsupported_stance_sole', i, {'foot': k, 'corner_gaps_m': gaps})
                else: check.bound('stance_sole_height_error_m', max(abs(x) for x in gaps), settings['foot_position_m'], i, k)
                if 'foot_target_pos' in arrays:
                    check.bound('foot_target_position_error_m', np.linalg.norm(position-arrays['foot_target_pos'][i, k]), settings['foot_position_m'], i, k)
                    check.bound('foot_target_orientation_error_rad', angle(quat, arrays['foot_target_quat'][i, k]), settings['foot_orientation_rad'], i, k)
            else: anchors[k] = None
            if arrays['hand_contact'][i, k]:
                check.bound('active_hand_target_error_m', np.linalg.norm(data.site_xpos[hand_sites[k]]-arrays['hand_target'][i, k]), settings['hand_position_m'], i, k)
                distances = [float(mujoco.mj_geomDistance(model, data, a, b, 1e6, None))
                             for a, b in excluded_ids[i] if a == hands[k]]
                query_count += len(distances)
                check.bound('hand_contact_surface_gap_m', max(0., min(distances)), settings['hand_surface_gap_m'], i, k)
                check.bound('hand_contact_penetration_m', max(0., -min(distances)), settings['hand_contact_penetration_m'], i, k)
        if not arrays['foot_contact'][i].any(): check.fail('no_declared_support', i)
        stance_anchors.append([None if a is None else (a[0].copy(), a[1].copy()) for a in anchors])
    velocities = np.zeros((n-1, model.nv)); dt = np.diff(t)
    for i in range(n-1):
        mujoco.mj_differentiatePos(model, velocities[i], float(dt[i]), q[i], q[i+1])
        ratio = abs(velocities[i, actor_dofs])/velocity_limits
        k = int(np.argmax(ratio))
        check.bound('actor_velocity_limit_ratio', float(ratio[k]), 1.0001, i,
                    {'dof': dof_labels[k], 'velocity': float(velocities[i, actor_dofs[k]]), 'limit': float(velocity_limits[k])})
        displacement = velocities[i]*dt[i]
        steps = max(2, math.ceil(dt[i]/settings['max_sample_dt_s']),
                    math.ceil(np.linalg.norm(displacement[root_dof:root_dof+3])/settings['max_sample_translation_m']),
                    math.ceil(np.max(abs(displacement)/interpolation_limits)))
        require(steps <= 1000, 'Excessive interval displacement; refuse unbounded subdivision')
        for sub in range(1, steps):
            state = q[i].copy(); mujoco.mj_integratePos(model, state, displacement, sub/steps)
            label = f'{i}+{sub}/{steps}'
            collisions(state, excluded_ids[i] & excluded_ids[i+1], label)
            for side, sid in enumerate(foot_sites):
                if arrays['foot_contact'][i, side] and arrays['foot_contact'][i+1, side]:
                    origin, orientation = stance_anchors[i][side]
                    quat = np.zeros(4); mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
                    check.bound('interpolated_stance_position_drift_m', np.linalg.norm(data.site_xpos[sid]-origin), settings['foot_position_m'], label, side)
                    check.bound('interpolated_stance_orientation_drift_rad', angle(quat, orientation), settings['foot_orientation_rad'], label, side)
    if n > 2:
        # Free-root rotation differences are tangent vectors at each start pose.
        # Transport them to world axes before taking an angular acceleration.
        world_velocity = velocities.copy(); rotation = np.zeros(9)
        for i in range(n-1):
            mujoco.mju_quat2Mat(rotation, q[i, root_address+3:root_address+7])
            world_velocity[i, root_dof+3:root_dof+6] = rotation.reshape(3, 3)@velocities[i, root_dof+3:root_dof+6]
        acceleration = np.diff(world_velocity[:, actor_dofs], axis=0)/((dt[1:]+dt[:-1])/2)[:, None]
        for i, values in enumerate(acceleration):
            ratios = abs(values)/acceleration_limits; k = int(np.argmax(ratios))
            check.bound('actor_acceleration_limit_ratio', float(ratios[k]), 1.0001, i+1,
                        {'dof': dof_labels[k], 'acceleration': float(values[k]), 'limit': float(acceleration_limits[k]),
                         'root_rotation_basis': 'world'})
    require(sha(clip_path) == clip_hash and sha(trajectory_path) == trajectory_hash, 'Saved motion changed during validation')
    for name, digest in clip['source_sha256'].items(): require(sha(directory/name) == digest, f'Source changed during validation: {name}')
    completion = task_completion(clip, json.loads((directory/'spec.json').read_text()), np.asarray(pelvis_positions), t)
    all_counts = dict(check.counts); all_counts.update(completion['failure_counts'])
    all_examples = dict(check.examples); all_examples.update(completion['failure_examples'])
    accepted = not bool(check.counts) and completion['evidence_pass']
    return {'schema': 'doorbench.planned-reference-validation.v1', 'door_id': clip['door_id'],
            'status': 'accepted_kinematic' if accepted else 'rejected', 'accepted': accepted,
            'kinematic_accepted': not bool(check.counts), 'task_completion': completion,
            'frames': n, 'collision_samples': collision_samples, 'signed_distance_queries': query_count,
            'minimum_noncontact_distance_m': None if math.isinf(min_clearance) else min_clearance,
            'minimum_foot_ground_distance_m': None if math.isinf(min_ground) else min_ground,
            'failure_counts': all_counts, 'failure_examples': all_examples, 'maxima': check.maxima, 'worst_samples': check.worst,
            'settings': settings, 'rig_fingerprint': RIG_FINGERPRINT, 'actor_geometry_count': len(actor_geoms),
            'native_collision_geometry_count': len(native_geoms), 'actor_self_pairs_exclude': 'same-body and directly adjacent bodies only',
            'contact_exclusion_pair_frames': sum(len(x) for x in excluded_ids),
            'clip_sha256': clip_hash, 'trajectory_sha256': trajectory_hash, 'source_sha256': clip['source_sha256'],
            'native_timewarp': {'start_s': float(native_time[0]), 'end_s': float(native_time[-1]), 'nondecreasing': True},
            'scope': ['Recomputed FK from saved qpos on original fixed actor MJCF and source door.',
                      'Exact engine distances for declared rigid collision geometry, including nonadjacent actor self-collision.',
                      'Frame and adaptively subdivided geodesic joint interpolation checks, not continuous-collision certification.',
                      'Stance requires fixed ankle pose and four sole corners supported by horizontal native floor boxes/planes.',
                      'Joint-range/derivative acceptance applies to the actor. Native source poses/FK and monotonic timewarp are verified; source dynamics and soft joint-limit compliance are not re-certified.',
                      'No dynamics, balance, friction, force closure, actuator torque or retimed-native-physics certification.'],
            'runtime': {'mujoco': mujoco.__version__, 'numpy': np.__version__}}


def validate(clip_path, trajectory_path, assets='assets', **settings):
    unknown = set(settings)-set(DEFAULTS)
    if unknown: raise ValueError(f'Unknown settings: {sorted(unknown)}')
    settings = {**DEFAULTS, **settings}
    require(all(math.isfinite(v) and v > 0 for v in settings.values()), 'Validation settings must be finite and positive')
    try:
        return _validate(clip_path, trajectory_path, assets, settings)
    except (InvalidInput, KeyError, ValueError, OSError, TypeError, ET.ParseError) as exc:
        return {'schema': 'doorbench.planned-reference-validation.v1', 'status': 'invalid_input', 'accepted': False,
                'failure_counts': {'input_contract': 1}, 'failure_examples': {'input_contract': [str(exc)]}, 'settings': settings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', type=Path, required=True); parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=Path('assets')); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); report = validate(args.clip, args.trajectory, args.assets)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'accepted': report['accepted'], 'failure_counts': report['failure_counts']}))
    if not report['accepted']: raise SystemExit(1)


if __name__ == '__main__': main()
