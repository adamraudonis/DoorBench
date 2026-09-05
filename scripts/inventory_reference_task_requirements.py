#!/usr/bin/env python3
"""Read-only inventory of manipulation evidence missing from native pose references.

This does not label human feasibility or certify mechanism operation. It binds
every examined clip/trajectory to the reference index and all three source files,
then inventories authored hardware and observed oracle generalized efforts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'doorbench.reference-task-requirements.v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effort_evidence(joints, native, arrays):
    """Concurrent efforts are evidence of oracle control, not a hand-count proof."""
    names = native['joint_names']
    addresses = native['qvel_addresses']
    if len(names) != len(addresses) or len(names) != len(set(names)):
        raise ValueError('Native joint names/addresses are invalid')
    tau = arrays['tau']; velocity = arrays['qvel']; times = arrays['time']
    if (tau.ndim != 2 or tau.shape != velocity.shape or tau.shape[0] != len(times)
            or len(times) < 2 or np.any(np.diff(times) <= 0)
            or not all(np.isfinite(a).all() for a in (tau, velocity, times))):
        raise ValueError('Native time/effort/velocity arrays are invalid')
    by_name = dict(zip(names, addresses))
    hardware = [j for j in joints if j.get('role') in ('operator', 'lock')]
    if any(j['name'] not in by_name for j in hardware):
        raise ValueError('Source hardware is missing from native joint metadata')
    cols = [by_name[j['name']] for j in hardware]
    if any(type(i) is not int or i < 0 or i >= tau.shape[1] for i in cols):
        raise ValueError('Hardware DOF address is invalid')
    active = np.abs(tau[:, cols]) > 1e-3
    moving = active & (np.abs(velocity[:, cols]) > .01)
    return {
        'frames': len(times),
        'hardware_joint_count': len(hardware),
        'commanded_hardware_joints': [j['name'] for j, used in zip(hardware, active.any(axis=0)) if used],
        'maximum_simultaneous_hardware_efforts': int(active.sum(axis=1).max()),
        'multiple_hardware_effort_frames': int((active.sum(axis=1) > 1).sum()),
        'maximum_simultaneously_moving_effort_driven_hardware': int(moving.sum(axis=1).max()),
        'multiple_moving_hardware_frames': int((moving.sum(axis=1) > 1).sum()),
    }


def requirements(spec, joints, events, effort):
    """Candidate requirements from authored features; never an operation schedule."""
    tags = set()
    family = spec['family']; lock = spec['lock']; operator = spec['operator']['model']
    if any(j.get('role') in ('operator', 'lock') and j.get('type') == 'hinge' for j in joints):
        tags.add('rotational_hardware_grasp_frame')
    if any(j.get('role') in ('operator', 'lock') and j.get('type') == 'slide' for j in joints):
        tags.add('linear_hardware_contact_direction')
    if effort['maximum_simultaneous_hardware_efforts'] > 1:
        tags.add('audit_concurrent_oracle_hardware_commands')
    if 'keypad' in lock['model'] or 'keypad' in operator:
        tags.add('ordered_key_press_and_release_evidence')
    if 'badge' in events:
        tags.add('credential_or_release_api_event_needs_actor_evidence')
    if lock['engaged'] and lock['model'] != 'none':
        tags.add('lock_release_state_and_permitted_side')
    if lock['model'] in ('dogs', 'multipoint'):
        tags.add('all_retaining_parts_release_schedule')
    if 'wheel' in operator or lock['model'] == 'vault_wheel':
        tags.add('wheel_orientation_and_regrasp_schedule')
    if family in ('automatic_sliding', 'automatic_swing', 'elevator'):
        tags.add('powered_trigger_and_safety_sensor_timing')
    if spec['closer']['model'] != 'none' or family in ('automatic_sliding', 'automatic_swing', 'elevator'):
        tags.add('support_or_power_while_holding_open')
    if family in ('swing_double', 'sliding_bypass', 'bifold', 'dutch', 'saloon'):
        tags.add('moving_leaf_coordination')
    if family in ('revolving', 'turnstile_fullheight', 'turnstile_tripod'):
        tags.add('rotation_phase_and_traversal_coordination')
    if family in ('hatch_floor', 'hatch_ceiling'):
        tags.add('horizontal_aperture_access_and_body_support')
    if family == 'pet_door':
        tags.add('fixed_adult_aperture_feasibility_before_planning')
    if family == 'strip_curtain':
        tags.add('distributed_body_contact_with_flexible_barrier')
    if spec['benchmark']['primary_scenario'] == 'locked_recognize':
        tags.add('probe_observation_and_locked_declaration')
    return sorted(tags)


def checked_member(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f'Recording path escapes or is absent: {relative}')
    return path


def inventory(assets, recordings):
    assets = Path(assets).resolve(); recordings = Path(recordings).resolve()
    manifest_path = assets/'manifest.json'; index_path = recordings/'index.json'
    bound = {str(p): sha(p) for p in (manifest_path, index_path)}
    manifest = json.loads(manifest_path.read_text()); index = json.loads(index_path.read_text())
    if index.get('schema') != 'doorbench.reference-motion.v1':
        raise ValueError('Unsupported reference index schema')
    if index.get('manifest_sha256') != bound[str(manifest_path)]:
        raise ValueError('Reference index is bound to a different asset manifest')
    rows = index['clips']; ids = [r['door_id'] for r in rows]
    expected = [r['id'] for r in manifest['doors']]
    if len(set(ids)) != len(ids) or len(set(expected)) != len(expected) or set(ids) != set(expected):
        raise ValueError('Manifest and reference index must contain the same unique IDs')
    results = []
    for row in sorted(rows, key=lambda r: r['door_id']):
        door_id = row['door_id']; directory = assets/'doors'/door_id
        source = {name: sha(directory/name) for name in ('door.xml', 'model.json', 'spec.json')}
        if source != row['source_sha256']:
            raise ValueError(f'{door_id}: source bytes differ from reference index')
        paths = {name: checked_member(recordings, row[name]) for name in ('clip', 'trajectory')}
        for name, path in paths.items():
            if sha(path) != row[name+'_sha256']:
                raise ValueError(f'{door_id}: {name} differs from reference index')
        clip = json.loads(paths['clip'].read_text())
        if (clip.get('schema') != 'doorbench.reference-motion.v1' or clip.get('door_id') != door_id
                or clip.get('source_sha256') != source):
            raise ValueError(f'{door_id}: clip identity/source mismatch')
        spec = json.loads((directory/'spec.json').read_text())
        model = json.loads((directory/'model.json').read_text())
        if spec['id'] != door_id or clip['scenario'] != spec['benchmark']['primary_scenario']:
            raise ValueError(f'{door_id}: spec/scenario mismatch')
        joints = [b['joint'] for b in model['bodies'] if b.get('joint')]
        with np.load(paths['trajectory'], allow_pickle=False) as arrays:
            effort = effort_evidence(joints, clip['native'], arrays)
        events = sorted({event[0] for event in clip['outcome']['events']})
        results.append({'door_id': door_id, 'family': spec['family'], 'operator': spec['operator']['model'],
            'latch': spec['latch']['model'], 'lock': spec['lock']['model'],
            'lock_engaged': spec['lock']['engaged'], 'robot_side_release': spec['lock']['robot_side_release'],
            'scenario': clip['scenario'], 'source_outcome': clip['outcome']['outcome'],
            'source_sha256': source, 'recording_sha256': {name: row[name+'_sha256'] for name in paths},
            'rotational_hardware_joints': [j['name'] for j in joints if j.get('role') in ('operator', 'lock') and j.get('type') == 'hinge'],
            'source_api_events': [e for e in events if e in ('badge', 'declare_locked')],
            'oracle_effort_evidence': effort, 'requirements': requirements(spec, joints, events, effort)})
        bound.update({str(directory/name): value for name, value in source.items()})
        bound.update({str(path): row[name+'_sha256'] for name, path in paths.items()})
    # Detect source/index replacement while this read-only inventory was running.
    if any(sha(path) != value for path, value in bound.items()):
        raise ValueError('An input changed during the inventory')
    groups = {}
    for key in ('family', 'lock'):
        groups[key] = {}
        for value in sorted({r[key] for r in results}):
            subset = [r for r in results if r[key] == value]
            groups[key][value] = {'doors': len(subset),
                'source_outcomes': dict(Counter(r['source_outcome'] for r in subset)),
                'requirements': dict(Counter(tag for r in subset for tag in r['requirements'])),
                'examples': [r['door_id'] for r in subset[:3]]}
    return {'schema': SCHEMA, 'source_verified': True,
        'scope': 'Authored requirements and oracle command inventory; no human feasibility, grasp, mechanism or causal-success certification.',
        'source': {'manifest_sha256': bound[str(manifest_path)], 'reference_index_sha256': bound[str(index_path)],
            'reference_generator_commit': index['generator_commit'], 'inventory_script_sha256': sha(__file__)},
        'evidence_limits': ['Native target is one world XYZ point per frame; grasp orientation and selected contact identity are not recorded.',
            'Native tau is direct generalized effort, not measured humanoid contact wrench.',
            'Concurrent efforts do not establish how many hands are required; stationary holds and coupled parts need a separate operation schedule.',
            'A feature requirement does not imply the source succeeded, that every mechanism was manually operated, or that a goal is intrinsically infeasible.'],
        'counts': {'doors': len(results), 'families': len(groups['family']),
            'source_outcomes': dict(Counter(r['source_outcome'] for r in results)),
            'requirements': dict(Counter(tag for r in results for tag in r['requirements'])),
            'source_api_events': dict(Counter(e for r in results for e in r['source_api_events']))},
        'by_family': groups['family'], 'by_lock': groups['lock'], 'doors': results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', default='assets'); p.add_argument('--recordings', default='out/reference-motions')
    p.add_argument('--out', default='out/planned-reference-scope/requirements.json')
    args = p.parse_args(); result = inventory(args.assets, args.recordings)
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(result['counts'], sort_keys=True))


if __name__ == '__main__':
    main()
