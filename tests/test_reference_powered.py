"""Conservative powered schedule eligibility and immutable native replay mapping."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from doorbench.reference.powered import PoweredIneligible, make_powered_guide, powered_eligibility

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'assets'; RECORDINGS = ROOT/'out/reference-motions'
POSITIVE = ['db0153_automatic_sliding', 'db0203_automatic_sliding']
NEGATIVE = {'db0011_automatic_swing': 'nonzero_native_manual_effort',
            'db0193_sliding_single': 'not_powered_family',
            'db0053_elevator': 'source_failed',
            'db0102_automatic_swing': 'unsupported_scenario'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def write(path, value): Path(path).write_text(json.dumps(value))


def real_inputs(door_id):
    if not (RECORDINGS/'index.json').exists() or not (ASSETS/'doors'/door_id/'spec.json').exists():
        pytest.skip('Opt-in generated dataset and frozen native recordings are not available')
    index = read(RECORDINGS/'index.json'); row = next(r for r in index['clips'] if r['door_id'] == door_id)
    clip = read(RECORDINGS/row['clip']); spec = read(ASSETS/'doors'/door_id/'spec.json')
    with np.load(RECORDINGS/row['trajectory'], allow_pickle=False) as arrays:
        source = {k: arrays[k].astype(float) for k in ['time', 'qpos', 'tau', 'base', 'target']}
    return clip, spec, source, row


@pytest.fixture
def example():
    outcome = {'success': True, 'outcome': 'success', 'error': None, 'damage': False, 'env_damage': False,
               'labels': {'touched_door': False, 'touched_operator': False, 'operator_actuated': False,
                          'lock_released': False, 'door_damaged': False, 'robot_passed_through': True, 'door_open_clear': True}}
    clip = {'scenario': 'open_and_traverse', 'outcome': outcome}
    spec = {'family': 'automatic_sliding', 'kinematics': {'actuator': {'powered': True}},
            'benchmark': {'scenarios': [{'name': 'open_and_traverse', 'goal': {'center': [0, 1.5, 0]},
                                        'pass_plane': {'center': [0, 0, 1], 'normal': [0, 1, 0]}}]}}
    source = {'time': np.array([0., .5, 1.]), 'qpos': np.array([[0.], [.4], [.8]]), 'tau': np.zeros((3, 1)),
              'base': np.array([[0., -1., .5], [0., -.1, .5], [0., .5, .5]]), 'target': np.zeros((3, 3))}
    return clip, spec, source


def test_pure_guard_accepts_explicit_nonmanual_powered_traversal_without_mutation(example):
    clip, spec, source = example; before = copy.deepcopy(example)
    assert powered_eligibility(*example) == ()
    assert (clip, spec) == before[:2]
    for key in source: np.testing.assert_array_equal(source[key], before[2][key])


@pytest.mark.parametrize('field,expected', [
    ('family', 'not_powered_family'), ('powered', 'actuator_not_explicitly_powered'),
    ('success', 'source_failed'), ('outcome', 'source_failed'), ('error', 'source_failed'),
    ('scenario', 'unsupported_scenario'), ('damage', 'source_damage_or_missing_damage_evidence'),
    ('tau', 'nonzero_native_manual_effort'), ('touched_door', 'source_manual_door_touch'),
    ('touched_operator', 'source_manual_operator_touch'), ('operator_actuated', 'source_operator_actuated'),
    ('lock_released', 'source_lock_released'), ('missing_touch', 'missing_touched_door_evidence'),
    ('traversal', 'source_traversal_evidence_missing'), ('crossing', 'unsupported_or_missing_source_crossing'),
    ('normal', 'unsupported_or_missing_source_crossing'), ('time', 'invalid_native_arrays'),
    ('nan', 'invalid_native_arrays'), ('empty_tau', 'invalid_native_arrays')])
def test_each_guard_rejects_ambiguous_manual_failed_or_invalid_input(example, field, expected):
    clip, spec, source = example
    if field == 'family': spec['family'] = 'sliding_single'
    elif field == 'powered': spec['kinematics']['actuator']['powered'] = False
    elif field in ('success', 'damage'): clip['outcome'][field] = field == 'damage'
    elif field == 'outcome': clip['outcome']['outcome'] = 'fail'
    elif field == 'error': clip['outcome']['error'] = 'recording error'
    elif field == 'scenario': clip['scenario'] = 'locked_recognize'
    elif field == 'tau': source['tau'][1, 0] = np.nextafter(0., 1.)  # No arbitrary residual-effort tolerance.
    elif field in ('touched_door', 'touched_operator', 'operator_actuated', 'lock_released'): clip['outcome']['labels'][field] = True
    elif field == 'missing_touch': del clip['outcome']['labels']['touched_door']
    elif field == 'traversal': clip['outcome']['labels']['robot_passed_through'] = False
    elif field == 'crossing': source['base'][:, 1] = -1.
    elif field == 'normal': spec['benchmark']['scenarios'][0]['pass_plane']['normal'] = [0, 0, 1]
    elif field == 'time': source['time'][2] = source['time'][1]
    elif field == 'nan': source['qpos'][1, 0] = np.nan
    else: source['tau'] = np.zeros((0, 1))
    assert expected in powered_eligibility(*example)


@pytest.mark.parametrize('door_id', POSITIVE)
def test_real_positive_original_clock_poses_contacts_and_sources_are_preserved(door_id):
    clip, spec, source, row = real_inputs(door_id)
    paths = [RECORDINGS/'index.json', RECORDINGS/row['clip'], RECORDINGS/row['trajectory'], ASSETS/'manifest.json',
             *(ASSETS/'doors'/door_id/name for name in clip['source_sha256'])]
    before = {path: sha(path) for path in paths}
    assert powered_eligibility(clip, spec, source) == ()
    guide = make_powered_guide(ASSETS/'doors'/door_id, RECORDINGS, fps=60, gait_profile='smooth')
    assert guide.metadata['traversal'] == 'proposed' and guide.metadata['hand'] is None
    assert guide.metadata['powered_schedule']['power_and_trigger_causality'] == 'unverified'
    assert not guide.hand_contact.any() and not guide.hand_weight.any() and guide.foot_contact.any(axis=1).all()
    assert set(guide.phases) == {'powered_wait', 'traverse', 'settle'}
    expected = np.stack([np.interp(guide.native_time, source['time'], source['qpos'][:, k]) for k in range(source['qpos'].shape[1])], axis=1)
    np.testing.assert_array_equal(guide.native_qpos, expected)
    stop = np.flatnonzero(source['base'][:, 1] > .05)[0]
    assert guide.native_time[0] == 0 and guide.native_time[-1] == source['time'][stop]
    assert np.all(np.diff(guide.native_time) >= 0) and np.all(np.diff(guide.time) > 0)
    active_walk = np.array(guide.phases) != 'powered_wait'
    assert np.all(guide.native_time[active_walk] == source['time'][stop])
    np.testing.assert_array_equal(guide.native_qpos[-1], source['qpos'][stop])
    np.testing.assert_allclose(guide.pelvis[-1, :2], spec['benchmark']['scenarios'][0]['goal']['center'][:2], atol=.002)
    for foot in range(2):
        consecutive_contact = guide.foot_contact[:-1, foot] & guide.foot_contact[1:, foot]
        assert np.max(np.abs(np.diff(guide.foot_pos[:, foot], axis=0)[consecutive_contact])) == 0.
    assert {path: sha(path) for path in paths} == before
    assert '/Users/' not in json.dumps(guide.metadata) and '/private/' not in json.dumps(guide.metadata)


@pytest.mark.parametrize('door_id,code', NEGATIVE.items())
def test_real_manual_failed_and_recognition_cases_stop_before_navigation(door_id, code, monkeypatch):
    clip, spec, source, _ = real_inputs(door_id)
    assert code in powered_eligibility(clip, spec, source)
    import doorbench.reference.planning as planning
    monkeypatch.setattr(planning, 'SceneNavigator', lambda *args: pytest.fail('An ineligible input reached native geometry compilation'))
    with pytest.raises(PoweredIneligible) as error:
        make_powered_guide(ASSETS/'doors'/door_id, RECORDINGS)
    assert code in error.value.reasons


@pytest.fixture
def copied_inputs(tmp_path):
    door_id = POSITIVE[0]; clip, spec, source, row = real_inputs(door_id)
    assets = tmp_path/'assets'; door = assets/'doors'/door_id
    shutil.copytree(ASSETS/'doors'/door_id, door)
    shutil.copyfile(ASSETS/'manifest.json', assets/'manifest.json')
    model = read(door/'model.json')
    for body in model['bodies']:
        for geom in body['geoms']:
            if geom.get('type') == 'mesh':
                name = 'hardware/'+geom['mesh_name']+'.obj'; (assets/name).parent.mkdir(exist_ok=True)
                shutil.copyfile(ASSETS/name, assets/name)
    recordings = tmp_path/'recordings'; recordings.mkdir(); shutil.copyfile(RECORDINGS/'index.json', recordings/'index.json')
    for key in ('clip', 'trajectory'):
        (recordings/row[key]).parent.mkdir(exist_ok=True); shutil.copyfile(RECORDINGS/row[key], recordings/row[key])
    return door, recordings, row


@pytest.mark.parametrize('change', ['clip', 'trajectory', 'source', 'index_outcome', 'index_duplicate', 'manifest', 'symlink'])
def test_replaced_recordings_or_source_fail_hash_and_identity_guards(copied_inputs, change):
    door, recordings, row = copied_inputs
    if change in ('clip', 'trajectory'):
        path = recordings/row[change]; path.write_bytes(path.read_bytes()+b' ')
    elif change == 'source': (door/'spec.json').write_bytes((door/'spec.json').read_bytes()+b' ')
    elif change == 'manifest': (door.parent.parent/'manifest.json').write_bytes(b'{}')
    elif change == 'symlink':
        path = recordings/row['trajectory']; outside = recordings.parent/'outside.npz'; path.rename(outside); path.symlink_to(outside)
    else:
        path = recordings/'index.json'; index = read(path); record = next(r for r in index['clips'] if r['door_id'] == door.name)
        if change == 'index_outcome': record['success'] = False
        else: index['clips'].append(record)
        write(path, index)
    with pytest.raises(ValueError): make_powered_guide(door, recordings)


def test_inputs_changed_during_planning_cannot_return_guide(copied_inputs, monkeypatch):
    door, recordings, _ = copied_inputs
    import doorbench.reference.guidance as guidance
    original = guidance.smooth_body_guidance
    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        path = door/'spec.json'; path.write_bytes(path.read_bytes()+b' ')
        return result
    monkeypatch.setattr(guidance, 'smooth_body_guidance', changing)
    with pytest.raises(ValueError, match='changed during'):
        make_powered_guide(door, recordings)


@pytest.mark.parametrize('fps,profile', [(0, 'smooth'), (float('nan'), 'smooth'), (True, 'smooth'), (60, 'unsupported')])
def test_invalid_options_fail_before_input_loading(fps, profile):
    with pytest.raises(ValueError): make_powered_guide('missing', 'missing', fps=fps, gait_profile=profile)
