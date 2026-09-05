"""The semantic inventory must remain bound to its claimed frozen inputs."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1]/'scripts/inventory_reference_task_requirements.py'
spec = importlib.util.spec_from_file_location('task_requirements', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_simultaneous_efforts_are_distinct_from_moving_parts():
    joints = [{'name': 'knob', 'role': 'operator'}, {'name': 'bolt', 'role': 'lock'}]
    native = {'joint_names': ['knob', 'bolt'], 'qvel_addresses': [1, 0]}
    arrays = {'time': np.array([0., .1]), 'tau': np.array([[3., 4.], [0., 4.]]),
              'qvel': np.array([[0., .2], [0., .2]])}
    result = module.effort_evidence(joints, native, arrays)
    assert result['maximum_simultaneous_hardware_efforts'] == 2
    assert result['maximum_simultaneously_moving_effort_driven_hardware'] == 1
    assert result['commanded_hardware_joints'] == ['knob', 'bolt']


@pytest.fixture
def dataset(tmp_path):
    assets = tmp_path/'assets'; recordings = tmp_path/'recordings'; door_id = 'fixture_swing'
    door = assets/'doors'/door_id; door.mkdir(parents=True)
    (recordings/'clips').mkdir(parents=True); (recordings/'trajectories').mkdir()
    spec = {'id': door_id, 'family': 'swing_single', 'operator': {'model': 'knob_round'},
            'latch': {'model': 'tubular'}, 'lock': {'model': 'none', 'engaged': False, 'robot_side_release': True},
            'closer': {'model': 'none'}, 'benchmark': {'primary_scenario': 'open_and_traverse'}}
    (door/'spec.json').write_text(json.dumps(spec))
    (door/'model.json').write_text(json.dumps({'bodies': [{'joint': {'name': 'knob', 'role': 'operator', 'type': 'hinge'}}]}))
    (door/'door.xml').write_text('<mujoco/>')
    source = {name: module.sha(door/name) for name in ('door.xml', 'model.json', 'spec.json')}
    clip = {'schema': 'doorbench.reference-motion.v1', 'door_id': door_id, 'source_sha256': source,
            'scenario': 'open_and_traverse', 'native': {'joint_names': ['knob'], 'qvel_addresses': [0]},
            'outcome': {'outcome': 'success', 'events': []}}
    (recordings/'clips'/f'{door_id}.json').write_text(json.dumps(clip))
    np.savez(recordings/'trajectories'/f'{door_id}.npz', time=[0., .1], tau=[[0.], [1.]], qvel=[[0.], [.2]])
    row = {'door_id': door_id, 'source_sha256': source, 'clip': f'clips/{door_id}.json',
           'trajectory': f'trajectories/{door_id}.npz'}
    row.update({key+'_sha256': module.sha(recordings/row[key]) for key in ('clip', 'trajectory')})
    (assets/'manifest.json').write_text(json.dumps({'doors': [{'id': door_id}]}))
    (recordings/'index.json').write_text(json.dumps({'schema': 'doorbench.reference-motion.v1',
        'generator_commit': 'frozen-fixture', 'manifest_sha256': module.sha(assets/'manifest.json'), 'clips': [row]}))
    return assets, recordings, door_id


def test_verified_inventory_reports_requirements_without_certifying_success(dataset):
    assets, recordings, _ = dataset
    result = module.inventory(assets, recordings)
    assert result['source_verified'] is True
    assert result['counts']['doors'] == 1
    assert result['by_family']['swing_single']['requirements']['rotational_hardware_grasp_frame'] == 1
    assert 'accepted' not in result['doors'][0]


@pytest.mark.parametrize('target', ['clip', 'trajectory', 'spec'])
def test_replaced_bytes_cannot_keep_frozen_provenance(dataset, target):
    assets, recordings, door_id = dataset
    path = {'clip': recordings/'clips'/f'{door_id}.json',
            'trajectory': recordings/'trajectories'/f'{door_id}.npz',
            'spec': assets/'doors'/door_id/'spec.json'}[target]
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='differ'):
        module.inventory(assets, recordings)


def test_manifest_omission_is_not_silently_counted_as_a_complete_inventory(dataset):
    assets, recordings, _ = dataset
    (assets/'manifest.json').write_text('{"doors":[]}')
    with pytest.raises(ValueError, match='different asset manifest'):
        module.inventory(assets, recordings)
