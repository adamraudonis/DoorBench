"""Unstepped synthetic point tests; no H1 trajectory or contact qualification."""
import copy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from test_isaac_panel_planning import actual_fixture, candidate_without_solves, toy_scene
from doorbench.dexterous import isaac_panel_geometry_audit as old_audit
from doorbench.dexterous import isaac_panel_geometry_probe as probe_module
from doorbench.dexterous import isaac_panel_planning as planner


@pytest.fixture
def case(actual_fixture, monkeypatch):
    candidate = candidate_without_solves(actual_fixture, monkeypatch, target=.09100001)
    probe = probe_module.IsaacPanelGeometryProbe(actual_fixture.context, candidate)
    return SimpleNamespace(fixture=actual_fixture, candidate=candidate, probe=probe,
        x=probe.initial_coordinates(), angles=dict(leaf=probe.start_angle, operator=0., latch=0.))


def evaluate(case, **kwargs):
    options = dict(reference_aperture_rad=case.probe.start_angle, measured_angles=case.angles)
    options.update(kwargs)
    return case.probe.evaluate(case.x, **options)


def test_initial_point_is_exact_detached_and_never_physics(case, monkeypatch):
    monkeypatch.setattr(mujoco, 'mj_step', lambda *a: pytest.fail('Probe stepped physics'))
    original = case.fixture.context.qpos.copy()
    x, angles, candidate = case.x.copy(), dict(case.angles), copy.deepcopy(case.candidate)
    result = evaluate(case)
    assert result['passed'] and max(result['values'].values()) < 1e-12
    assert result['authorized_stages'] == result['physics_steps'] == result['active_state_writes'] == 0
    assert not result['physical_admission'] and not result['runtime_route_exported']
    np.testing.assert_array_equal(case.fixture.context.qpos, original)
    np.testing.assert_array_equal(case.x, x)
    assert case.angles == angles and case.candidate == candidate
    result['limits']['left_position_m'] = 100.
    assert evaluate(case)['limits']['left_position_m'] == .0001


def test_geometry_matches_prior_dense_audit_at_zero_lag(case):
    args = SimpleNamespace(duration_s=40., samples=3, actual_leaf_lag_rad=0.,
        lag_start_angle_rad=None, aperture_speed_limit_rad_s=.149,
        aperture_acceleration_limit_rad_s2=.08)
    receipt, _, traces = old_audit._dense_geometry(case.candidate, toy_scene(), args)
    measured = dict(case.angles)
    maxima = {}
    for row in traces:
        measured['leaf'] = float(case.probe.start_angle+row[1]*(case.probe.final_angle-case.probe.start_angle))
        point = case.probe.evaluate(row[2:2+len(case.x)], reference_aperture_rad=measured['leaf'], measured_angles=measured)
        for key, value in point['values'].items(): maxima[key] = max(maxima.get(key, 0.), value)
    for key, value in maxima.items(): assert value == pytest.approx(receipt['maxima'][key], abs=1e-14)


@pytest.mark.parametrize('delta', [-.02, .02])
def test_reference_goal_and_actual_collision_leaf_are_distinct(case, delta):
    nominal = evaluate(case)
    measured = dict(case.angles, leaf=case.angles['leaf']+delta, operator=.005, latch=.0005)
    point = evaluate(case, measured_angles=measured)
    assert point['mechanism_qpos'] == measured
    assert point['reference_minus_measured_leaf_rad'] == pytest.approx(-delta)
    # The LH target remains at the same requested reference angle, while the
    # actual leaf stored in the isolated collision calculator really differs.
    np.testing.assert_allclose(point['requested_left_palm_position'], nominal['requested_left_palm_position'], atol=1e-14, rtol=0)
    np.testing.assert_allclose(point['requested_left_palm_rotation'], nominal['requested_left_palm_rotation'], atol=1e-14, rtol=0)
    address = case.probe.m.jnt_qposadr[case.probe.mechanism['leaf']]
    assert case.probe.d.qpos[address] == measured['leaf']


def test_arbitrary_full_bundle_is_checked_and_no_missing_joint_is_invented(case):
    bundle = evaluate(case)['robot_joint_targets']
    assert len(bundle) == 25
    point = evaluate(case, robot_joint_targets=bundle)
    assert point['passed'] and point['robot_target_scope'] == 'Complete explicit target bundle'
    incomplete = dict(bundle); incomplete.pop('torso')
    with pytest.raises(ValueError, match='Complete finite robot'): evaluate(case, robot_joint_targets=incomplete)
    different = dict(bundle, torso=.01)
    with pytest.raises(ValueError, match='agree'): evaluate(case, robot_joint_targets=different)


def test_joint_outside_panel_subset_is_explicit_and_geometry_checked(actual_fixture, monkeypatch, tmp_path):
    import xml.etree.ElementTree as ET
    scene = toy_scene(); xml_path = tmp_path/'extra-finger.xml'
    mujoco.mj_saveLastXML(str(xml_path), scene.m)
    tree = ET.parse(xml_path)
    base = tree.find(".//body[@name='robot/base']")
    extra = ET.SubElement(base, 'body', name='robot/rh_extra', pos='.3 -.3 0')
    ET.SubElement(extra, 'joint', name='robot/rh_FFJ1', range='-.8 .8')
    ET.SubElement(extra, 'geom', size='.005')
    tree.write(xml_path)
    def expanded(*a):
        m = mujoco.MjModel.from_xml_path(str(xml_path)); d = mujoco.MjData(m)
        d.qpos[m.joint('leaf_hinge').qposadr[0]] = .091
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        return SimpleNamespace(m=m, d=d)
    expanded_scene = expanded()
    actual_fixture.admitted['initial_qpos'] = expanded_scene.d.qpos.tolist()
    actual_fixture.admitted['initial_qvel'] = expanded_scene.d.qvel.tolist()
    monkeypatch.setattr(planner, 'LandedLeftScene', expanded)
    actual_fixture.context = planner.admit_isaac_panel_context('synthetic', robot='r', door_xml='d', door_usd='u')
    candidate = candidate_without_solves(actual_fixture, monkeypatch, target=.09100001)
    probe = probe_module.IsaacPanelGeometryProbe(actual_fixture.context, candidate)
    args = dict(reference_aperture_rad=probe.start_angle, measured_angles=dict(leaf=.091,operator=0.,latch=0.))
    initial = probe.evaluate(probe.initial_coordinates(), **args)
    assert initial['passed'] and len(initial['robot_joint_targets']) == 26
    full = initial['robot_joint_targets']; full['rh_FFJ1'] = .800002
    point = probe.evaluate(probe.initial_coordinates(), robot_joint_targets=full, **args)
    assert not point['passed'] and point['violations']['joint_violation_increase_rad'] > .000001
    assert point['robot_joint_targets']['rh_FFJ1'] == .800002
    del full['rh_FFJ1']
    with pytest.raises(ValueError, match='Complete finite robot'):
        probe.evaluate(probe.initial_coordinates(), robot_joint_targets=full, **args)


def test_bad_foot_and_root_targets_cannot_pass(case):
    case.x[0] = .031
    point = evaluate(case)
    assert not point['passed']
    assert point['violations']['foot_position_m'] == pytest.approx(.031)
    assert point['violations']['root_translation_m'] == pytest.approx(.031)
    assert point['limits'] == probe_module.STATIC_LIMITS


def test_joint_limit_increase_retains_original_tolerance(case):
    case.x[6+planner.JOINT_NAMES.index('torso')] = 2.0001
    point = evaluate(case)
    assert not point['passed'] and point['violations']['joint_violation_increase_rad'] > .000001


def test_tiny_measured_mechanism_excursion_retains_existing_joint_tolerance(case):
    measured = dict(case.angles, operator=-.1000005)
    point = evaluate(case, measured_angles=measured)
    assert point['mechanism_qpos']['operator'] == measured['operator']
    assert point['values']['joint_violation_increase_rad'] == pytest.approx(.0000005)
    assert 'joint_violation_increase_rad' not in point['violations']
    with pytest.raises(ValueError, match='source-relative joint envelope'):
        evaluate(case, measured_angles=dict(measured, operator=-.100002))


@pytest.mark.parametrize('options', [
    {'reference_aperture_rad': float('nan')}, {'reference_aperture_rad': True},
    {'reference_aperture_rad': .0909}, {'reference_aperture_rad': .092},
    {'measured_angles': {'leaf': .091, 'operator': 0.}},
    {'measured_angles': {'leaf': .091, 'operator': 0., 'latch': 0., 'extra': 0.}},
    {'measured_angles': {'leaf': .091, 'operator': float('nan'), 'latch': 0.}},
    {'measured_angles': {'leaf': .091, 'operator': True, 'latch': 0.}},
    {'measured_angles': {'leaf': .091, 'operator': 1.001, 'latch': 0.}},
])
def test_bad_or_undeclared_states_are_rejected(case, options):
    with pytest.raises(ValueError): evaluate(case, **options)


def test_nonfinite_reference_and_distance_fail(case, monkeypatch):
    case.x[0] = float('inf')
    with pytest.raises(ValueError): evaluate(case)
    case.x[0] = 0.
    # Force a close candidate, then ensure a nonfinite exact distance is never
    # hidden by min()/NaN ordering or treated as an empty broad phase.
    monkeypatch.setattr(probe_module, 'clearance_candidate_pairs', lambda *a: np.array([[0, 0]]))
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *a: float('nan'))
    with pytest.raises(ValueError, match='Nonfinite original geometry distance'): evaluate(case)


def test_signed_plane_broad_phase_preserves_below_floor_failure(case, monkeypatch):
    # Deliberate private model replacement is confined to this synthetic test.
    # A plane is one-sided; magnitude-only pruning could lose a shape below it.
    monkeypatch.setattr(planner, 'LandedLeftScene', lambda *a: toy_scene(floor_height=1.1))
    case.probe = probe_module.IsaacPanelGeometryProbe(case.fixture.context, case.candidate)
    point = evaluate(case)
    assert not point['passed'] and point['right_scene_clearance_capped_m'] < 0.
    assert point['violations']['right_scene_clearance_m'] < 0.


def test_stale_source_fails_explicit_batch_binding_recheck(case):
    case.fixture.file.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'): case.probe.verify_inputs()


def test_candidate_is_validated_before_probe_construction(case):
    candidate = copy.deepcopy(case.candidate)
    candidate['configuration']['right_hand_frame'] = 'root'
    with pytest.raises(ValueError): probe_module.IsaacPanelGeometryProbe(case.fixture.context, candidate)
