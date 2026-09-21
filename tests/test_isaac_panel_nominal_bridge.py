"""Detached math with explicit synthetic upstream admission seams; no rollout."""
import copy
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from doorbench.dexterous import isaac_panel_nominal_bridge as bridge
from doorbench.dexterous import isaac_panel_nominal_bridge_audit as audit
from doorbench.dexterous import isaac_panel_bridge_seed as seed_module
from doorbench.dexterous.qualified_isaac_grasp import digest
from test_isaac_panel_bridge_seed import source, prepare


@pytest.fixture
def seed(source, monkeypatch):
    for name in ('mj_step', 'mj_step1', 'mj_step2'):
        monkeypatch.setattr(mujoco, name, lambda *a, **k:pytest.fail('Numerical bridge stepped physics'))
    return prepare(source)


def change_reference(seed, *, positions=None, velocities=None):
    """Modify a synthetic numerical boundary and all its redundant copies."""
    seed = copy.deepcopy(seed)
    previous = seed['previous_reference']; names = previous['joint_names']
    if positions is not None:
        previous['value'] = np.asarray(positions).tolist()
        seed['previous_three_references'][-1]['coupled']['value'] = previous['value'].copy()
        seed['preserved_nominal_joint_values'] = [previous['value'][6+names.index(n)] for n in seed['preserved_nominal_joint_names']]
        mapped = np.asarray(seed['panel_first_target_in_old_chart']).copy()
        for n in seed['preserved_nominal_joint_names']:mapped[6+names.index(n)] = previous['value'][6+names.index(n)]
        seed['panel_first_target_in_old_chart'] = mapped.tolist()
    if velocities is not None:
        previous['velocity'] = np.asarray(velocities).tolist()
        seed['previous_three_references'][-1]['coupled']['velocity'] = previous['velocity'].copy()
    return seed


def rest_seed(seed):
    return change_reference(seed, positions=seed['panel_first_target_in_old_chart'], velocities=np.zeros(75))


def test_exact_boundary_first_new_command_and_complete_audit(seed):
    before = copy.deepcopy(seed)
    result = bridge.build_panel_nominal_bridge(seed, duration_s=4.)
    start = bridge.sample_panel_nominal_bridge(result, 0.)
    first = bridge.sample_panel_nominal_bridge(result, .002)
    end = bridge.sample_panel_nominal_bridge(result, 4.)
    p0 = np.array(seed['previous_reference']['value']); v0 = np.array(seed['previous_reference']['velocity'])
    active = result['active_coordinate_indices']; held = result['preserved_coordinate_indices']
    np.testing.assert_array_equal(start['position'], p0)
    np.testing.assert_array_equal(start['stored_prior_velocity'], v0)
    np.testing.assert_array_equal(start['polynomial_velocity'][active], v0[active])
    np.testing.assert_array_equal(start['polynomial_velocity'][held], np.zeros(44))
    assert np.max(abs(v0[held])) > 0
    assert first['command_time_s'] == pytest.approx(seed['source_time_s'], abs=1e-15)
    assert not np.array_equal(first['position'][active], p0[active])
    np.testing.assert_array_equal(first['position'][held], p0[held])
    np.testing.assert_array_equal(end['position'], seed['panel_first_target_in_old_chart'])
    np.testing.assert_array_equal(end['polynomial_velocity'], np.zeros(75))
    np.testing.assert_array_equal(end['polynomial_acceleration'], np.zeros(75))
    checked = audit.audit_panel_nominal_bridge(result)
    assert checked['passed']
    assert checked['checked_new_command_samples'] == 2000
    assert checked['total_checked_samples'] == 2002
    assert checked['first_new_command']['joint_acceleration_rad_s2'] == pytest.approx(((first['position'][6:]-p0[6:])/.002-v0[6:])/.002, abs=1e-10)
    assert checked['geometry_samples'] == 0
    assert checked['continuous_derivative_bound_proved'] is False
    assert seed == before
    first['position'][:] = 99.
    assert bridge.sample_panel_nominal_bridge(result, .002)['position'][0] != 99.


def test_name_overlay_and_noncommuting_original_chart(seed):
    result = bridge.build_panel_nominal_bridge(seed, duration_s=4.)
    oldp, oldr = seed_module._pose(seed['previous_reference']['root_coordinate_origin_xyz_wxyz'])
    panelp, panelr = seed_module._pose(seed['panel_root_coordinate_origin_xyz_wxyz'])
    end = np.array(result['final_position'])
    np.testing.assert_allclose(oldp+end[:3], panelp, atol=1e-15)
    np.testing.assert_allclose(Rotation.from_rotvec(end[3:6]).as_matrix()@oldr, panelr, atol=1e-14)
    assert result['joint_names'] != seed_module.JOINT_NAMES+seed['preserved_nominal_joint_names']
    assert len(result['active_coordinate_indices']) == 31
    assert len(result['preserved_coordinate_indices']) == 44
    for name in seed['preserved_nominal_joint_names']:
        i = 6+result['joint_names'].index(name)
        assert result['final_position'][i] == seed['previous_reference']['value'][i]
    assert result['root_coordinate_origin_xyz_wxyz'] == seed['previous_reference']['root_coordinate_origin_xyz_wxyz']


@pytest.mark.parametrize('sign', [-1., 1.])
def test_preserved_velocity_original_boundary_and_incompatible_stop(seed, sign):
    seed = rest_seed(seed); index = 6+seed['previous_reference']['joint_names'].index(seed['preserved_nominal_joint_names'][0])
    v = np.zeros(75); v[index] = sign*.006
    compatible = change_reference(seed, velocities=v)
    candidate = bridge.build_panel_nominal_bridge(compatible, duration_s=.01)
    report = audit.audit_panel_nominal_bridge(candidate)
    assert report['passed']
    assert report['first_new_command']['joint_acceleration_rad_s2'][index-6] == pytest.approx(-sign*3.)
    assert candidate['initial_polynomial_velocity'][index] == 0.
    assert candidate['stored_prior_velocity'][index] == sign*.006
    v[index] = sign*(.006+2e-12)
    with pytest.raises(ValueError, match='cannot stop'):
        bridge.build_panel_nominal_bridge(change_reference(seed, velocities=v), duration_s=.01)


def test_same_endpoint_nonzero_velocity_has_interior_overshoot_and_full_failure_count(seed):
    seed = rest_seed(seed); index = 6+seed['previous_reference']['joint_names'].index('torso')
    v = np.zeros(75); v[index] = 1.
    candidate = bridge.build_panel_nominal_bridge(change_reference(seed, velocities=v), duration_s=.1)
    # The endpoints alone look stationary and have identical positions.
    assert candidate['initial_position'] == candidate['final_position']
    assert bridge.sample_panel_nominal_bridge(candidate, .04)['position'][index] > candidate['initial_position'][index]+.01
    progress = []; report = audit.audit_panel_nominal_bridge(candidate, progress=progress.append)
    assert not report['passed'] and report['complete']
    assert report['checked_new_command_samples'] == 50
    assert report['total_checked_samples'] == 52
    assert report['failed_samples'] == len(report['failures']) > 1
    assert report['failures'][-1]['sample_index'] > report['failures'][0]['sample_index']
    assert report['maximum_discrete_rates']['joint_acceleration_rad_s2'] > 3
    assert progress[-1]['new_command_samples_checked'] == 50
    assert candidate['rate_audit_performed'] is False


def test_short_candidate_survives_failed_rate_audit_without_truncated_denominator(seed):
    candidate = bridge.build_panel_nominal_bridge(seed, duration_s=.002)
    before = copy.deepcopy(candidate); report = audit.audit_panel_nominal_bridge(candidate)
    assert report['complete'] and not report['passed']
    assert report['checked_new_command_samples'] == 1
    assert report['total_checked_samples'] == 3
    assert report['prospective_terminal_hold_checks'] == 1
    assert candidate == before


def test_auditor_does_not_use_builder_or_sampler(seed, monkeypatch):
    candidate = bridge.build_panel_nominal_bridge(seed, duration_s=4.)
    monkeypatch.setattr(bridge, 'sample_panel_nominal_bridge', lambda *a, **k:pytest.fail('Sampler used by independent audit'))
    monkeypatch.setattr(bridge, 'build_panel_nominal_bridge', lambda *a, **k:pytest.fail('Builder used by independent audit'))
    assert audit.audit_panel_nominal_bridge(candidate)['passed']


def test_exact_motor_grid_and_audit_same_normalized_clock(seed, monkeypatch):
    candidate = bridge.build_panel_nominal_bridge(rest_seed(seed), duration_s=.7)
    assert candidate['requested_duration_s'] == .7
    assert candidate['duration_s'] == 350*.002
    assert bridge.sample_panel_nominal_bridge(candidate, 350*.002)['elapsed_s'] == candidate['duration_s']
    visited = []; original = audit._polynomial
    def track(coefficients, u, duration):
        if duration == candidate['duration_s']:visited.append(u)
        return original(coefficients, u, duration)
    monkeypatch.setattr(audit, '_polynomial', track)
    report = audit.audit_panel_nominal_bridge(candidate)
    assert report['passed'] and report['checked_new_command_samples'] == 350
    assert visited == [(i*.002)/candidate['duration_s'] for i in range(1, 351)]
    assert report['limits']['arithmetic_tolerance'] == 1e-12
    assert report['limits']['joint_velocity_change_per_step_rad_s'] == 3*.002


def test_sampled_analytic_maxima_include_prior_polynomial_boundary(seed):
    seed = rest_seed(seed); v = np.zeros(75)
    index = 6+seed['previous_reference']['joint_names'].index('torso'); v[index] = .2
    candidate = bridge.build_panel_nominal_bridge(change_reference(seed, velocities=v), duration_s=.1)
    report = audit.audit_panel_nominal_bridge(candidate)
    assert report['sampled_analytic_derivative_maxima']['joint_speed_rad_s'] == .2


@pytest.mark.parametrize('duration', [0., -.1, .003, 60.002, float('nan'), float('inf'), True, '1'])
def test_invalid_duration_rejects(seed, duration):
    with pytest.raises(ValueError):bridge.build_panel_nominal_bridge(seed, duration_s=duration)


@pytest.mark.parametrize('time', [-1e-15, 1.000000000000001, float('nan'), True, '0'])
def test_sampler_never_extrapolates_or_clamps(seed, time):
    candidate = bridge.build_panel_nominal_bridge(seed, duration_s=1.)
    with pytest.raises(ValueError):bridge.sample_panel_nominal_bridge(candidate, time)


@pytest.mark.parametrize('change', ['position', 'stored_velocity', 'coefficients', 'preserved', 'root_chart', 'clock',
    'order', 'limit', 'authority', 'seed_digest', 'source', 'binding', 'helper_binding', 'nan'])
def test_corrupt_candidate_rejected(seed, change):
    c = bridge.build_panel_nominal_bridge(seed, duration_s=4.)
    if change == 'position':c['initial_position'][0] += .001
    elif change == 'stored_velocity':c['stored_prior_velocity'][0] = 0.
    elif change == 'coefficients':c['coefficients_normalized_time_ascending'][4][0] += .0001
    elif change == 'preserved':c['coefficients_normalized_time_ascending'][1][c['preserved_coordinate_indices'][0]] = .00001
    elif change == 'root_chart':c['root_coordinate_origin_xyz_wxyz'][0] += .001
    elif change == 'clock':c['first_new_command_time_s'] += .002
    elif change == 'order':c['joint_names'].reverse()
    elif change == 'limit':c['limits']['joint_acceleration_rad_s2'] = 4.
    elif change == 'authority':c['geometric_admission'] = True
    elif change == 'seed_digest':c['seed_sha256'] = 'f'*64
    elif change == 'source':c['source_time_s'] += .002
    elif change == 'binding':c['input_sha256'].pop(next(iter(c['seed']['input_sha256'])))
    elif change == 'helper_binding':c['input_sha256'].pop(str(Path(bridge.__file__).resolve()))
    elif change == 'nan':c['coefficients_normalized_time_ascending'][2][0] = float('nan')
    with pytest.raises(ValueError):audit.audit_panel_nominal_bridge(c)


@pytest.mark.parametrize('field', ['value', 'velocity'])
def test_seed_requires_exact_previous_reference_copy(seed, field):
    seed['previous_reference'][field][0] += .001
    with pytest.raises(ValueError, match='reference tail'):bridge.build_panel_nominal_bridge(seed, duration_s=1.)


def test_changed_bound_source_rejected(seed, source):
    source.archive.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):bridge.build_panel_nominal_bridge(seed, duration_s=1.)


def test_input_mutation_during_complete_audit_rejected(seed, source):
    candidate = bridge.build_panel_nominal_bridge(seed, duration_s=4.)
    def mutate(progress):source.archive.write_bytes(b'changed during detached audit')
    with pytest.raises(ValueError, match='changed'):audit.audit_panel_nominal_bridge(candidate, progress=mutate)


def test_candidate_mutation_during_complete_audit_rejected(seed):
    candidate = bridge.build_panel_nominal_bridge(seed, duration_s=4.)
    def mutate(progress):candidate['duration_s'] = 99.
    with pytest.raises(ValueError, match='candidate changed'):audit.audit_panel_nominal_bridge(candidate, progress=mutate)


def test_hashes_and_authority_are_explicit_on_success(seed):
    candidate = bridge.build_panel_nominal_bridge(seed, duration_s=4.)
    report = audit.audit_panel_nominal_bridge(candidate)
    assert report['candidate_content_sha256'] == bridge.content_sha256(candidate)
    for obj in (candidate, report):
        for key, value in bridge.AUTHORITY.items():assert obj[key] == value and type(obj[key]) is type(value)
        assert obj['input_sha256'][str(Path(bridge.__file__).resolve())] == digest(bridge.__file__)
    assert report['input_sha256'][str(Path(audit.__file__).resolve())] == digest(audit.__file__)


def test_interior_analytic_derivatives_match_independent_finite_differences(seed):
    candidate = bridge.build_panel_nominal_bridge(seed, duration_s=4.)
    t = 1.23; h = 1e-5
    row = bridge.sample_panel_nominal_bridge(candidate, t)
    left = bridge.sample_panel_nominal_bridge(candidate, t-h)
    right = bridge.sample_panel_nominal_bridge(candidate, t+h)
    np.testing.assert_allclose(row['polynomial_velocity'], (right['position']-left['position'])/(2*h), atol=1e-10, rtol=0)
    np.testing.assert_allclose(row['polynomial_acceleration'], (right['polynomial_velocity']-left['polynomial_velocity'])/(2*h), atol=1e-10, rtol=0)
