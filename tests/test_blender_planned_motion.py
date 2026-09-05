"""Validate planned replay provenance and frame conventions without Blender."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from doorbench.appearance.pipeline import digest
from scripts import blender_planned_motion as replay


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    for name, value in [('model.json', {'bodies': [{'name': 'world_env'}, {'name': 'leaf'}]}),
                        ('spec.json', {'id': 'tiny'}), ('door.xml', {})]:
        (source/name).write_text(json.dumps(value))
    hashes = {name: replay.sha256(source/name) for name in ('spec.json', 'model.json', 'door.xml')}
    job = {'door_id': 'tiny', 'door_dir': str(source), 'hardware_dir': str(tmp_path),
           'source_sha256': hashes, 'renderer_sha256': {}, 'mesh_sha256': {},
           'reference_state': {'body_aliases': {'world_env': 'world'}}}
    job['job_sha256'] = digest(job)
    arrays = {
        'actor_time': np.array([0., .5, 1., 1.5]),
        'native_time': np.array([0., 0., .6, .6]),
        'qpos': np.zeros((4, 1)), 'body_pos': np.zeros((4, 2, 3)),
        'body_quat': np.tile([1., 0., 0., 0.], (4, 2, 1)),
        'actor_qpos': np.zeros((4, 7)), 'actor_joints': np.zeros((4, 3, 3)),
        'actor_body_pos': np.zeros((4, 2, 3)),
        'actor_body_quat': np.tile([1., 0., 0., 0.], (4, 2, 1)),
        'foot_pos': np.zeros((4, 2, 3)), 'foot_quat': np.tile([1., 0., 0., 0.], (4, 2, 1)),
        'foot_contact': np.ones((4, 2), dtype=bool),
    }
    clip = {
        'schema': 'doorbench.planned-reference.v1', 'door_id': 'tiny', 'source_sha256': hashes,
        'up_axis': 'Z', 'units': 'metres/radians/seconds', 'fps': 30, 'duration': 1.5,
        'status': 'candidate', 'qa': {'passed': False},
        'native': {'body_names': ['world', 'leaf'], 'nq': 1},
        'actor': {'body_names': ['pelvis', 'foot_left'], 'nq': 7, 'geometries': [
            {'name': 'torso', 'body_name': 'pelvis', 'type': 'capsule', 'size': [.08, .15, 0.], 'pos': [0., 0., .2], 'quat_wxyz': [1., 0., 0., 0.]},
            {'name': 'foot', 'body_name': 'foot_left', 'type': 'box', 'size': [.05, .12, .025], 'pos': [0., .03, -.025], 'quat_wxyz': [1., 0., 0., 0.]},
        ]},
    }
    return tmp_path, job, clip, arrays


def load(fixture, *, bind=True):
    root, job, clip, arrays = fixture
    np.savez(root/'trajectory.npz', **arrays)
    if bind:
        clip['trajectory_sha256'] = replay.sha256(root/'trajectory.npz')
    (root/'job.json').write_text(json.dumps(job))
    (root/'clip.json').write_text(json.dumps(clip))
    return replay.load_inputs(root/'job.json', root/'clip.json', root/'trajectory.npz')


def test_actor_timeline_binds_all_native_poses_without_lead_subtraction(inputs):
    result = load(inputs)
    assert result[-1] == {'world_env': 0, 'leaf': 1}
    np.testing.assert_array_equal(result[2]['actor_time'], [0., .5, 1., 1.5])
    np.testing.assert_array_equal(result[2]['native_time'], [0., 0., .6, .6])
    assert result[2]['body_pos'].shape[0] == result[2]['actor_body_pos'].shape[0] == 4


def test_rejects_unbound_or_other_archive(inputs):
    with pytest.raises(ValueError, match='trajectory checksum'):
        load(inputs, bind=False)
    load(inputs)
    inputs[3]['qpos'][1, 0] = .1
    with pytest.raises(ValueError, match='trajectory checksum'):
        load(inputs, bind=False)


@pytest.mark.parametrize('damage,match', [
    ('job', 'job checksum'), ('source', 'Source changed'), ('sourcebinding', 'source hashes'),
    ('schema', 'schema or door_id'), ('units', 'Z up'), ('fps', 'fps'),
    ('time', 'strictly increase'), ('warp', 'native_time'), ('shape', 'body_pos'),
    ('actorpose', 'actor_body_pos'), ('qpos', 'qpos width'), ('duration', 'duration'),
    ('jointshape', 'actor_joints'), ('quat', 'unit WXYZ'), ('footquat', 'unit WXYZ'),
    ('localquat', 'unit WXYZ'), ('localpos', 'finite vector'), ('size', 'dimensions'),
    ('geomtype', 'Unsupported actor geometry'), ('duplicate', 'unique'),
    ('actorbody', 'unknown body'), ('nativebody', 'No native body pose'),
    ('extra_native_body', 'without source geometry'), ('finite', 'finite numeric'),
    ('pickle', 'Object arrays'), ('contact', 'binary'), ('qa', 'qa must be an object'),
])
def test_invalid_plans_fail_before_scene_creation(inputs, damage, match):
    root, job, clip, arrays = inputs
    geom = clip['actor']['geometries'][0]
    if damage == 'job': job['door_id'] = 'changed'
    if damage == 'source': (Path(job['door_dir'])/'door.xml').write_text('changed')
    if damage == 'sourcebinding': clip['source_sha256'] = {}
    if damage == 'schema': clip['schema'] = 'doorbench.reference-motion.v1'
    if damage == 'units': clip['up_axis'] = 'Y'
    if damage == 'fps': clip['fps'] = True
    if damage == 'time': arrays['actor_time'][1] = 0.
    if damage == 'warp': arrays['native_time'][-1] = .1
    if damage == 'shape': arrays['body_pos'] = arrays['body_pos'][:3]
    if damage == 'actorpose': arrays['actor_body_pos'] = arrays['actor_body_pos'][:, :1]
    if damage == 'qpos': clip['native']['nq'] = 2
    if damage == 'duration': clip['duration'] = 7
    if damage == 'jointshape': arrays['actor_joints'] = np.zeros((4, 3))
    if damage == 'quat': arrays['actor_body_quat'][1, 0] = 0.
    if damage == 'footquat': arrays['foot_quat'][1, 0] = 0.
    if damage == 'localquat': geom['quat_wxyz'] = [0., 0., 0., 0.]
    if damage == 'localpos': geom['pos'] = [0, float('nan'), 0]
    if damage == 'size': geom['size'] = [-1., .2]
    if damage == 'geomtype': geom['type'] = 'mesh'
    if damage == 'duplicate': clip['actor']['geometries'].append(dict(geom))
    if damage == 'actorbody': geom['body_name'] = 'absent'
    if damage == 'nativebody': clip['native']['body_names'][1] = 'absent'
    if damage == 'extra_native_body':
        clip['native']['body_names'].append('unrendered')
        arrays['body_pos'] = np.zeros((4, 3, 3))
        arrays['body_quat'] = np.tile([1., 0., 0., 0.], (4, 3, 1))
    if damage == 'finite': arrays['actor_qpos'][0, 0] = float('inf')
    if damage == 'pickle': arrays['extra'] = np.array([{'payload': 1}], dtype=object)
    if damage == 'contact': arrays['foot_contact'] = np.full((4, 2), 3.)
    if damage == 'qa': clip['qa'] = []
    with pytest.raises(ValueError, match=match):
        load(inputs)


def test_local_capsule_rotation_and_world_body_pose_are_both_applied():
    quarter = math.sqrt(.5)
    geom = {'type': 'capsule', 'size': [.1, .3], 'pos': [.2, 0., 0.],
            'quat_wxyz': [quarter, 0., quarter, 0.]}
    corners = replay.geometry_world_corners(geom, np.array([[1., 2., 3.]]),
                                            np.array([[quarter, 0., 0., quarter]]))
    # Local capsule +Z becomes body +X, then world +Y. Its center offset rotates too.
    np.testing.assert_allclose(corners.min(axis=(0,1)), [.9, 1.8, 2.9], atol=1e-14)
    np.testing.assert_allclose(corners.max(axis=(0,1)), [1.1, 2.6, 3.1], atol=1e-14)


@pytest.mark.parametrize('clip,expected', [
    ({}, 'Unvalidated proposal'),
    ({'status': 'accepted', 'qa': {'passed': False, 'independent': True}}, 'QA failed'),
    ({'status': 'accepted', 'qa': {'passed': True}}, 'acceptance pending'),
    ({'status': 'accepted', 'qa': {'passed': True, 'independent': True}}, 'External validation report required'),
    ({'status': 'accepted', 'qa': {'passed': 'true', 'independent': True}}, 'Unvalidated proposal'),
])
def test_acceptance_wording_requires_explicit_independent_pass(clip, expected):
    title, detail = replay.validation_label(clip)
    assert expected in detail
    assert 'dynamically feasible' not in (title+detail).lower()


def external_report(inputs):
    root, job, clip, arrays = inputs
    clip.update(complete_proposal=True, frames=len(arrays['actor_time']))
    load(inputs)
    report = {'schema': 'doorbench.planned-reference-validation.v1', 'door_id': clip['door_id'],
              'clip_sha256': replay.sha256(root/'clip.json'), 'trajectory_sha256': replay.sha256(root/'trajectory.npz'),
              'source_sha256': job['source_sha256'], 'accepted': True, 'kinematic_accepted': True,
              'status': 'accepted_kinematic', 'failure_counts': {}, 'frames': clip['frames'],
              'task_completion': {'complete_proposal': True, 'evidence_pass': True, 'failure_counts': {}, 'source_success_declared': True},
              'scope': ['Sampled kinematics; not balance or force closure.']}
    return report


def test_external_validation_is_hash_bound_and_does_not_mutate_original_clip(inputs):
    report = external_report(inputs); root, job, clip, _ = inputs
    (root/'validation.json').write_text(json.dumps(report))
    original = (root/'clip.json').read_bytes()
    verification = replay.load_validation_report(root/'validation.json', root/'clip.json', root/'trajectory.npz', job)
    assert verification['report_sha256'] == replay.sha256(root/'validation.json')
    assert (root/'clip.json').read_bytes() == original
    assert clip['qa']['passed'] is False, 'fresh external checks are separate from old clip declarations'
    title, detail = replay.validation_label(clip, verification)
    assert title == 'Sampled kinematic checks passed'
    assert 'Forces and balance not certified' in detail


@pytest.mark.parametrize('field,value', [
    ('schema', 'unknown'), ('door_id', 'other'), ('clip_sha256', 'changed'),
    ('trajectory_sha256', 'changed'), ('source_sha256', {}), ('accepted', False),
    ('accepted', 'true'), ('kinematic_accepted', False), ('status', 'rejected'),
    ('failure_counts', {'collision': 1}), ('frames', 99),
    ('task_completion', {'complete_proposal': False, 'evidence_pass': True}),
    ('task_completion', {'complete_proposal': True, 'evidence_pass': False}),
])
def test_external_validation_rejects_incomplete_failed_or_unbound_reports(inputs, field, value):
    report = external_report(inputs); root, job, _, _ = inputs
    report[field] = value
    (root/'validation.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        replay.load_validation_report(root/'validation.json', root/'clip.json', root/'trajectory.npz', job)


def test_external_validation_rechecks_actual_source_bytes(inputs):
    report = external_report(inputs); root, job, _, _ = inputs
    (root/'validation.json').write_text(json.dumps(report))
    (Path(job['door_dir'])/'door.xml').write_text('modified after validation')
    with pytest.raises(ValueError, match='Validation source changed'):
        replay.load_validation_report(root/'validation.json', root/'clip.json', root/'trajectory.npz', job)
