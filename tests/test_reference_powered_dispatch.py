"""Guide dispatch preserves manual behavior and propagates powered integrity errors."""
import json

import numpy as np
import pytest

from doorbench.reference import guidance, powered
from tests.test_reference_powered import example, real_inputs, POSITIVE, ASSETS, RECORDINGS


@pytest.fixture
def inputs(tmp_path, example):
    clip, spec, source = example
    directory = tmp_path/'assets/doors/fixture'; directory.mkdir(parents=True)
    recordings = tmp_path/'recordings'; (recordings/'clips').mkdir(parents=True); (recordings/'trajectories').mkdir()
    (directory/'spec.json').write_text(json.dumps(spec))
    (recordings/'clips/fixture.json').write_text(json.dumps(clip))
    np.savez_compressed(recordings/'trajectories/fixture.npz', **source)
    return directory, recordings, example


def test_eligible_dispatch_preserves_options_and_explicit_schedule_metadata(inputs, monkeypatch):
    directory, recordings, _ = inputs; calls = []
    sentinel = object()
    def selected(*args, **kwargs): calls.append((args, kwargs)); return sentinel
    monkeypatch.setattr(powered, 'make_powered_guide', selected)
    monkeypatch.setattr(guidance, 'SceneNavigator', lambda *args: pytest.fail('Manual planner used for eligible powered input'))
    assert guidance.make_guide(directory, recordings, fps=60, gait_profile='controlled') is sentinel
    assert calls == [((directory, recordings), {'fps': 60, 'gait_profile': 'controlled'})]


@pytest.mark.parametrize('reason', ['manual_family', 'manual_effort', 'manual_touch', 'failed_source', 'recognition'])
def test_ineligible_dispatch_preserves_existing_baseline_entry(inputs, monkeypatch, reason):
    directory, recordings, (clip, spec, source) = inputs
    if reason == 'manual_family': spec['family'] = 'sliding_single'
    elif reason == 'manual_effort': source['tau'][1, 0] = .1
    elif reason == 'manual_touch': clip['outcome']['labels']['touched_door'] = True
    elif reason == 'failed_source': clip['outcome']['success'] = False
    else:
        clip['scenario'] = 'locked_recognize'; spec['benchmark']['scenarios'][0]['name'] = 'locked_recognize'
    (directory/'spec.json').write_text(json.dumps(spec)); (recordings/'clips/fixture.json').write_text(json.dumps(clip))
    np.savez_compressed(recordings/'trajectories/fixture.npz', **source)
    class BaselineReached(Exception): pass
    def original_path(path):
        assert path == directory
        raise BaselineReached
    monkeypatch.setattr(guidance, 'SceneNavigator', original_path)
    monkeypatch.setattr(powered, 'make_powered_guide', lambda *a, **k: pytest.fail('Ineligible source dispatched to powered planner'))
    with pytest.raises(BaselineReached): guidance.make_guide(directory, recordings)


def test_selected_powered_integrity_error_is_never_swallowed_as_fallback(inputs, monkeypatch):
    directory, recordings, _ = inputs
    def corrupted(*args, **kwargs): raise ValueError('Powered input hash mismatch: recording_trajectory')
    monkeypatch.setattr(powered, 'make_powered_guide', corrupted)
    monkeypatch.setattr(guidance, 'SceneNavigator', lambda *args: pytest.fail('Integrity error entered manual fallback'))
    with pytest.raises(ValueError, match='input hash mismatch'): guidance.make_guide(directory, recordings)


@pytest.mark.parametrize('door_id', POSITIVE)
def test_real_dispatch_is_the_same_immutable_powered_guide(door_id):
    real_inputs(door_id)
    selected = guidance.make_guide(ASSETS/'doors'/door_id, RECORDINGS, fps=60)
    direct = powered.make_powered_guide(ASSETS/'doors'/door_id, RECORDINGS, fps=60)
    assert selected.metadata == direct.metadata
    assert selected.metadata['powered_schedule']['power_and_trigger_causality'] == 'unverified'
    for name in ('time', 'native_time', 'native_qpos', 'pelvis', 'yaw', 'foot_pos', 'foot_quat', 'foot_contact',
                 'hands', 'hand_contact', 'hand_weight'):
        np.testing.assert_array_equal(getattr(selected, name), getattr(direct, name))
