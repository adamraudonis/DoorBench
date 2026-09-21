"""CPU-only source binding and planning-boundary tests, not physical trials."""
import copy
import hashlib
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous import isaac_release_planning as planner
from doorbench.dexterous import isaac_release_source as source_loader


@pytest.fixture
def source(tmp_path, monkeypatch):
    path = tmp_path/'actual-source-fixture.npz'
    path.write_bytes(b'Explicit synthetic admission fixture, not a physics episode')
    receipt = dict(schema='doorbench.isaac-release-source.v1', source_engine='isaac-physx',
        source_run=str(tmp_path), grasp_profile='volar-phalange-v1',
        source_qualification=dict(passed=True, state_sha256='a'*64),
        coordinate_admission=dict(passed=True, note='synthetic normalized endpoint'),
        measured_rest=dict(terminal_time_s=42., endpoint_contacts=[]),
        initial_qpos=[.11, .22, .33], robot_path='original.xml', door_xml_path='door.xml',
        input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()},
        physics_steps=0, source_sample_playback=0)
    calls = []
    def admit(*args, **kwargs):
        calls.append((args, kwargs))
        return copy.deepcopy(receipt)
    monkeypatch.setattr(source_loader, 'admit_isaac_release_source', admit)
    return receipt, path, calls


def context(source):
    return planner.admit_isaac_release_context('actual-run', robot='robot.xml',
        door_xml='door.xml', door_usd='door.usda')


def test_calls_actual_admission_and_keeps_detached_original_receipt(source):
    expected, _, calls = source
    value = context(source)
    assert calls == [(('actual-run',), dict(robot='robot.xml', door_xml='door.xml',
                                          door_usd='door.usda', profile='volar-phalange-v1'))]
    assert value.admission == expected and value.terminal_time_s == 42.
    value.qpos[:] = 999
    changed = value.admission
    changed['initial_qpos'][0] = 999
    changed['source_qualification']['passed'] = False
    assert value.admission == expected
    np.testing.assert_array_equal(value.qpos, expected['initial_qpos'])
    assert value.state_archive_path.name == 'acquisition-physics.npz'
    assert 'manifest.json' not in json.dumps(value.admission)


@pytest.mark.parametrize('change', ['engine', 'physical', 'coordinates', 'steps', 'playback'])
def test_source_cannot_relabel_native_data_or_skip_actual_admission(source, change):
    receipt = source[0]
    if change == 'engine': receipt['source_engine'] = 'native-mujoco'
    if change == 'physical': receipt['source_qualification']['passed'] = False
    if change == 'coordinates': receipt['coordinate_admission']['passed'] = False
    if change == 'steps': receipt['physics_steps'] = 1
    if change == 'playback': receipt['source_sample_playback'] = 1
    with pytest.raises(ValueError, match='actual Isaac'): context(source)


def test_source_mutation_after_admission_is_rejected(source):
    value = context(source)
    source[1].write_bytes(b'changed')
    with pytest.raises(ValueError, match='source changed'): value.verify_inputs()


def test_old_report_contributes_no_coordinates_or_qualification():
    old = dict(passed=True, source_qualification={'passed':True}, initial_qpos=[999],
        trials=[{'rows':[{'qpos':[999]}]}], endpoint_contacts=[{'body':'fake'}],
        configuration=dict(radial_clearance_m=.02, source_run='old native episode'))
    before = copy.deepcopy(old)
    result = planner.numeric_release_preferences(old)
    assert set(result) == set(planner.DEFAULT_PREFERENCES)
    assert result['radial_clearance_m'] == .02
    assert old == before


@pytest.mark.parametrize('key,value', [('radial_clearance_m', .1),
    ('radial_clearance_m', True), ('separation_reserve_m', 0.),
    ('thumb_j3_margin_rad', float('nan')), ('early_palm_clearance_m', -.001),
    ('whole_body', 1), ('finger_profile', 'invented'), ('retreat_profile', 'unknown')])
def test_preference_bounds_and_types_are_original_and_explicit(key, value):
    with pytest.raises(ValueError): planner.numeric_release_preferences({key:value})


def test_generated_candidate_is_fresh_and_cannot_claim_runtime_admission(source, monkeypatch):
    value = context(source)
    received = []
    def solve(actual, preferences):
        received.append(actual.qpos)
        return ([dict(time_s=0., qpos=actual.qpos.tolist()),
                 dict(time_s=8., qpos=(actual.qpos+.01).tolist())], [], [], .0001)
    monkeypatch.setattr(planner, '_generate_profiled_rows', solve)
    old = dict(configuration={}, initial_qpos=[999], trials=[dict(rows=[dict(qpos=[999])])])
    result = planner.generate_release_candidate(value, old)
    np.testing.assert_array_equal(received[0], value.qpos)
    assert result['source_admission'] == value.admission
    assert result['source_context_sha256'] == value.sha256
    assert result['initial_qpos'] == result['trials'][0]['rows'][0]['qpos']
    assert not result['geometric_admission'] and not result['physical_contact_qualification']
    assert not result['runtime_route_exported'] and result['physics_steps'] == 0


def test_candidate_cannot_shift_even_one_bit_of_initial_source_coordinate(source, monkeypatch):
    value = context(source)
    moved = value.qpos
    moved[0] = np.nextafter(moved[0], np.inf)
    monkeypatch.setattr(planner, '_generate_profiled_rows',
                        lambda *args: ([dict(qpos=moved.tolist())], [], [], 0.))
    with pytest.raises(ValueError, match='exact normalized source endpoint'):
        planner.generate_release_candidate(value, {})


def test_source_changed_during_numerical_solve_cannot_export_candidate(source, monkeypatch):
    value = context(source)
    def solve(*args):
        source[1].write_bytes(b'changed during planning')
        return ([dict(qpos=value.qpos.tolist())], [], [], 0.)
    monkeypatch.setattr(planner, '_generate_profiled_rows', solve)
    with pytest.raises(ValueError, match='source changed'):
        planner.generate_release_candidate(value, {})


def test_omitted_thumb_preference_preserves_legacy_payload_exactly():
    assert planner.numeric_release_preferences({}) == planner.DEFAULT_PREFERENCES
    assert 'thumb_posture_continuity_weight' not in planner.numeric_release_preferences({})


@pytest.mark.parametrize('value', [True, None, '.1', 0., .0099999, .1000001,
                                 float('nan'), float('inf')])
def test_thumb_preference_rejects_unbounded_or_non_numeric_values(value):
    with pytest.raises(ValueError, match='thumb posture/continuity'):
        planner.numeric_release_preferences({'thumb_posture_continuity_weight': value})


@pytest.mark.parametrize('weight', [.01, .1])
def test_thumb_preference_is_explicit_numeric_only_without_mutation(weight):
    old = {'configuration': {'thumb_posture_continuity_weight': weight},
           'initial_qpos': [999], 'source_qualification': {'passed': True}}
    before = copy.deepcopy(old)
    actual = planner.numeric_release_preferences(old)
    assert actual == {**planner.DEFAULT_PREFERENCES, 'thumb_posture_continuity_weight': weight}
    assert old == before


@pytest.mark.parametrize('digit,extending', [('TH', False), ('FF', False),
    ('MF', False), ('RF', False), ('LF', False), ('FF', True), ('MF', True),
    ('RF', True), ('LF', True)])
def test_default_regularization_is_byte_identical_to_historical_arithmetic(digit, extending):
    v = np.array([.001, -.00499999, .17, -.3999], dtype=np.float64)
    preference = np.array([.2, .03, -.18, .005])
    previous = np.nextafter(preference, -np.inf)
    expected = ((.25 if extending else .01)*(v-preference), .01*(v-previous))
    actual = planner._digit_regularization(v, preference, previous,
        thumb=digit == 'TH', extending=extending)
    assert [a.tobytes() for a in actual] == [a.tobytes() for a in expected]


@pytest.mark.parametrize('digit,extending', [('FF', False), ('MF', False),
    ('RF', False), ('LF', False), ('FF', True), ('MF', True), ('RF', True), ('LF', True)])
def test_thumb_opt_in_cannot_change_non_thumb_residuals(digit, extending):
    v, preference, previous = np.array([.03, -.17]), np.array([.01, .07]), np.array([.025, .06])
    expected = planner._digit_regularization(v, preference, previous, thumb=False, extending=extending)
    actual = planner._digit_regularization(v, preference, previous, thumb=False,
        extending=extending, thumb_posture_continuity_weight=.1)
    assert [a.tobytes() for a in actual] == [a.tobytes() for a in expected]


def test_thumb_opt_in_changes_both_soft_terms_only_and_copies_inputs():
    v, preference, previous = np.array([.03, -.17]), np.array([.01, .07]), np.array([.025, .06])
    before = [x.copy() for x in (v, preference, previous)]
    actual = planner._digit_regularization(v, preference, previous, thumb=True,
        extending=False, thumb_posture_continuity_weight=.1)
    np.testing.assert_array_equal(actual[0], .1*(v-preference))
    np.testing.assert_array_equal(actual[1], .1*(v-previous))
    for current, original in zip((v, preference, previous), before):
        np.testing.assert_array_equal(current, original)


def test_candidate_records_opt_in_and_still_has_no_geometry_or_runtime_authority(source, monkeypatch):
    value = context(source)
    received = []
    def solve(actual, preferences):
        received.append(preferences)
        return ([dict(qpos=actual.qpos.tolist())], [], [], .0002)
    monkeypatch.setattr(planner, '_generate_profiled_rows', solve)
    result = planner.generate_release_candidate(value, {'thumb_posture_continuity_weight': .1})
    assert result['configuration']['thumb_posture_continuity_weight'] == .1
    assert received == [result['configuration']]
    assert result['initial_qpos'] == value.qpos.tolist()
    assert result['source_admission'] == value.admission
    assert result['maximum_pad_goal_residual_m'] == .0002
    for key in ('geometric_admission', 'physical_contact_qualification', 'runtime_route_exported'):
        assert result[key] is False


def _planning_cli():
    path = Path(__file__).resolve().parents[1]/'scripts/dexterous/plan_local_isaac_release.py'
    spec = importlib.util.spec_from_file_location('thumb_planning_cli', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('weight', [None, .1])
def test_cli_preserves_omitted_weight_or_forwards_explicit_value(source, tmp_path, monkeypatch, weight):
    cli = _planning_cli()
    value = context(source)
    received = []
    monkeypatch.setattr(cli, 'admit_isaac_release_context', lambda *args, **kwargs: value)
    def generate(actual, preferences):
        received.append(preferences)
        return dict(input_sha256=actual.admission['input_sha256'], trials=[{'rows':[{}]}])
    monkeypatch.setattr(cli, 'generate_release_candidate', generate)
    args = SimpleNamespace(output=tmp_path/'prospective-candidate', preferences=None,
        isaac_source=tmp_path, robot=tmp_path/'robot.xml', door_xml=tmp_path/'door.xml',
        door_usd=tmp_path/'door.usda', grasp_profile='volar-phalange-v1',
        thumb_posture_continuity_weight=weight)
    assert cli.run(args) == 0
    expected = planner.DEFAULT_PREFERENCES.copy()
    if weight is not None: expected['thumb_posture_continuity_weight'] = weight
    assert received == [expected]
    exported = json.loads((args.output/'candidate.json').read_text())
    assert str(Path(planner.__file__).resolve()) in exported['input_sha256']
    assert (args.output/'source-isaac_release_planning.py').read_bytes() == Path(planner.__file__).read_bytes()
