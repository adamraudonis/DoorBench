"""Explicit manual contact roles and held-native transfer, not a dynamics test."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from doorbench.reference.manual_contacts import (
    EFFORT_THRESHOLD, UnsupportedManualContact, build_manual_guide, plan_manual_contacts,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    spec = {'id': 'example_slide', 'family': 'sliding_single', 'operator': {'model': 'pull_d'},
            'lock': {'model': 'padlock', 'engaged': False}, 'latch': {'model': 'none'},
            'closer': {'model': 'none'}, 'kinematics': {}, 'benchmark': {'primary_scenario': 'open_and_traverse', 'scenarios': [
                {'name': 'open_and_traverse', 'goal': {'center': [0, 1.8, .5], 'radius': .2},
                 'pass_plane': {'center': [0, 0, 1], 'normal': [0, 1, 0]}}]}}
    def geom(name, kind, size, semantic):
        return {'name': name, 'type': kind, 'size': size, 'pos': [0, 0, 0], 'quat': [1, 0, 0, 0],
                'semantic': semantic, 'collision': True}
    model = {'bodies': [
        {'name': 'panel', 'parent': 'world', 'joint': {'name': 'panel_slide', 'role': 'primary', 'type': 'slide'},
         'sites': [{'name': 'panel_pull_grip_n', 'role': 'grip', 'pos': [0, -.1, 1]}],
         'geoms': [geom('panel_pull_col_n', 'capsule', [.0125, .13], 'operator')]},
        {'name': 'panel_hasp', 'parent': 'panel', 'joint': {'name': 'panel_hasp_hinge', 'role': 'lock', 'type': 'hinge'},
         'sites': [{'name': 'panel_hasp_grip', 'role': 'grip', 'pos': [0, 0, 0]}],
         'geoms': [geom('panel_hasp_strap', 'box', [.05, .02, .002], 'lock')]}]}
    clip = {'door_id': spec['id'], 'scenario': 'open_and_traverse', 'outcome': {
        'door_id': spec['id'], 'scenario': 'open_and_traverse', 'success': True, 'outcome': 'success',
        'error': None, 'damage': False, 'env_damage': False, 'events': [],
        'labels': {'door_damaged': False, 'lock_released': False}}, 'native': {
            'joint_names': ['panel_slide', 'panel_hasp_hinge'], 'qpos_addresses': [0, 1], 'qvel_addresses': [0, 1]}}
    source = {'time': np.arange(9)*.1, 'qpos': np.zeros((9, 2)), 'tau': np.zeros((9, 2)),
              'target': np.zeros((9, 3)), 'base': np.tile([0., -1., .5], (9, 1))}
    source['tau'][2, 1] = .1; source['tau'][3:, 1] = .0001; source['tau'][5:, 0] = 10
    source['qpos'][5:, 0] = [.1, .2, .3, .4]; source['base'][-1, 1] = .5
    return spec, model, clip, source


def test_named_roles_generalize_beyond_fixture_id_without_mutation(inputs):
    before = copy.deepcopy(inputs)
    plan = plan_manual_contacts(*inputs)
    assert plan.first_active_index == 2 and plan.first_contact_index == 1
    assert plan.transfer_index == 4 and plan.stop_index == 8
    assert {r.id: (r.body_name, r.site_name, r.geom_name) for r in plan.roles} == {
        'hasp': ('panel_hasp', 'panel_hasp_grip', 'panel_hasp_strap'),
        'pull': ('panel', 'panel_pull_grip_n', 'panel_pull_col_n')}
    assert plan.residual_hasp_effort_max == .0001
    assert any('mechanical necessity is unverified' in s for s in plan.assumptions)
    transfer = [s for s in plan.segments if s.phase in ('withdraw_hasp', 'transfer', 'reach_pull')]
    assert all(s.native_frozen and s.source_start_index == s.source_end_index == 4 for s in transfer)
    assert inputs[:3] == before[:3]
    for k in inputs[3]: np.testing.assert_array_equal(inputs[3][k], before[3][k])


@pytest.mark.parametrize('change,code', [
    ('locked', 'requires_already_unlocked_hasp'), ('family', 'unsupported_family'),
    ('operator', 'unsupported_pull_operator'), ('latch', 'unsupported_latch_or_closer'),
    ('closer', 'unsupported_latch_or_closer'), ('powered', 'powered_leaf_requires_other_schedule'),
    ('failed', 'unsupported_or_failed_source_scenario'), ('badge', 'unsupported_source_api_action'),
    ('released', 'damage_or_unmodeled_lock_release'), ('damage', 'damage_or_unmodeled_lock_release'),
    ('identity', 'source_identity_mismatch'), ('extra_joint', 'unsupported_extra_or_concurrent_controls'),
    ('parent', 'hasp_not_attached_to_primary_leaf'), ('visual_pull', 'pull_requires_exact_collision_primitive'),
    ('mesh_pull', 'pull_requires_exact_collision_primitive'), ('missing_hasp', 'hasp_geometry_missing_or_ambiguous'),
    ('ambiguous_site', 'hasp_site_missing_or_ambiguous'), ('concurrent', 'unsupported_concurrent_efforts'),
    ('boundary_concurrent', 'unsupported_concurrent_efforts'), ('reversed', 'unsupported_repeated_or_reversed_roles'),
    ('zero', 'missing_sequential_hasp_and_pull_effort'), ('mapping', 'native_joint_mapping_mismatch'),
    ('time', 'invalid_native_time'), ('nan', 'nonfinite_native_arrays'), ('crossing', 'source_did_not_cross_passage')])
def test_unsupported_semantics_and_concurrency_fail_closed(inputs, change, code):
    spec, model, clip, source = inputs
    if change == 'locked': spec['lock']['engaged'] = True
    elif change == 'family': spec['family'] = 'swing_single'
    elif change == 'operator': spec['operator']['model'] = 'pull_flush_recessed'
    elif change in ('latch', 'closer'): spec[change]['model'] = 'other'
    elif change == 'powered': spec['kinematics']['actuator'] = {'powered': True}
    elif change == 'failed': clip['outcome']['success'] = False
    elif change == 'badge': clip['outcome']['events'] = [['badge', .2]]
    elif change == 'released': clip['outcome']['labels']['lock_released'] = True
    elif change == 'damage': clip['outcome']['damage'] = True
    elif change == 'identity': clip['door_id'] = 'other'
    elif change == 'extra_joint': model['bodies'].append(copy.deepcopy(model['bodies'][1]))
    elif change == 'parent': model['bodies'][1]['parent'] = 'world'
    elif change == 'visual_pull': model['bodies'][0]['geoms'][0]['collision'] = False
    elif change == 'mesh_pull': model['bodies'][0]['geoms'][0]['type'] = 'mesh'
    elif change == 'missing_hasp': model['bodies'][1]['geoms'] = []
    elif change == 'ambiguous_site': model['bodies'][1]['sites'] *= 2
    elif change == 'concurrent': source['tau'][5, 1] = .1
    elif change == 'boundary_concurrent': source['tau'][5, 1] = EFFORT_THRESHOLD
    elif change == 'reversed': source['tau'][:, 0] = 0; source['tau'][1, 0] = 10
    elif change == 'zero': source['tau'][:] = 0
    elif change == 'mapping': clip['native']['qvel_addresses'] = [0, 0]
    elif change == 'time': source['time'][2] = source['time'][1]
    elif change == 'nan': source['target'][0, 0] = np.nan
    else: source['base'][:, 1] = -1
    with pytest.raises(UnsupportedManualContact) as error: plan_manual_contacts(*inputs)
    assert code in error.value.reasons


def real_inputs():
    door = ROOT/'assets/doors/db0193_sliding_single'; recordings = ROOT/'out/reference-motions'
    if not (recordings/'index.json').exists(): pytest.skip('Generated native corpus is optional')
    clip = json.loads((recordings/'clips'/f'{door.name}.json').read_text())
    with np.load(recordings/'trajectories'/f'{door.name}.npz', allow_pickle=False) as z:
        source = {key: z[key].astype(float) for key in ('time', 'qpos', 'tau', 'base', 'target')}
    return door, recordings, clip, source


def test_real_guide_freezes_native_state_and_releases_between_exact_roles():
    door, recordings, clip, source = real_inputs()
    before = {key: a.copy() for key, a in source.items()}
    result = build_manual_guide(door, recordings); guide = result.guide
    assert len(guide.time) == len(result.contact_role_ids) == 2606
    assert guide.hand_contact.sum(axis=1).max() == 1
    phase = np.asarray(guide.phases)
    transfer = np.flatnonzero(np.isin(phase, ['withdraw_hasp', 'transfer_reposition', 'transfer_face_pull', 'reach_pull']))
    assert len(transfer) == 142
    assert np.ptp(guide.native_time[transfer]) == 0 and np.ptp(guide.native_qpos[transfer], axis=0).max() == 0
    assert not guide.hand_contact[transfer[1:-1]].any()
    assert guide.hand_contact[transfer[-1], 1]
    for i in range(len(guide.time)):
        if guide.hand_contact[i].any(): assert result.contact_role_ids[i] in result.roles
    assert set(result.roles) == {'hasp', 'pull'}
    assert result.roles['hasp'].geom_name == 'leaf_hasp_strap' and result.roles['pull'].geom_name == 'leaf_pull_col_n'
    expected = np.stack([np.interp(guide.native_time, source['time'], source['qpos'][:, j]) for j in range(2)], axis=1)
    np.testing.assert_array_equal(guide.native_qpos, expected)
    assert np.all(np.diff(guide.native_time) >= 0)
    # Role switches happen only near the rest hand, not via a contact-target jump.
    switches = [i for i in range(1, len(guide.time)) if result.contact_role_ids[i] != result.contact_role_ids[i-1]]
    assert switches
    for i in switches:
        assert guide.hand_weight[i-1:i+1].max() < .0001
        assert np.linalg.norm(guide.hands[i]-guide.hands[i-1], axis=-1).max() < .003
    assert guide.metadata['source_sha256'] == clip['source_sha256']
    assert '/Users/' not in json.dumps(guide.metadata)
    for key in source: np.testing.assert_array_equal(source[key], before[key])


def test_real_guide_matches_passed_temporary_prototype_timing_and_native_path():
    door, recordings, _, _ = real_inputs()
    prototype = ROOT/'out/reference-contact-prototype-refined/db0193_sliding_single/trajectory.npz'
    if not prototype.exists(): pytest.skip('Optional independently accepted prototype artifact')
    guide = build_manual_guide(door, recordings).guide
    with np.load(prototype, allow_pickle=False) as p:
        np.testing.assert_array_equal(guide.time, p['proposal_time'])
        np.testing.assert_array_equal(guide.native_time, p['native_time'])
        np.testing.assert_array_equal(guide.native_qpos, p['qpos'])
        np.testing.assert_array_equal(guide.hand_contact, p['hand_contact'])


def test_ineligible_case_never_compiles_geometry(monkeypatch):
    import doorbench.reference.planning as planning
    monkeypatch.setattr(planning, 'SceneNavigator', lambda *a, **k: pytest.fail('Ineligible case compiled geometry'))
    if not (ROOT/'out/reference-motions/index.json').exists(): pytest.skip('Generated native corpus is optional')
    with pytest.raises(UnsupportedManualContact, match='unsupported_family'):
        build_manual_guide(ROOT/'assets/doors/db0002_swing_single', ROOT/'out/reference-motions')


@pytest.mark.parametrize('fps,profile', [(0, 'smooth'), (True, 'smooth'), (60.5, 'smooth'), (60, 'bad')])
def test_invalid_options_fail_before_input_reads(fps, profile):
    with pytest.raises(UnsupportedManualContact): build_manual_guide('missing', 'missing', fps=fps, gait_profile=profile)


def test_first_actual_effort_target_must_match_the_verified_hasp_site(monkeypatch):
    import doorbench.reference.manual_contacts as manual
    door, recordings, _, _ = real_inputs()
    loaded = list(manual._inputs(door, recordings))
    loaded[3]['target'] = loaded[3]['target'].copy()
    loaded[3]['target'][11] += [0, 0, .2]
    monkeypatch.setattr(manual, '_inputs', lambda *a: tuple(loaded))
    with pytest.raises(UnsupportedManualContact, match='first_active_target_does_not_match_hasp_site'):
        manual.build_manual_guide(door, recordings)


def test_second_real_geometry_has_its_own_roles_and_frozen_transfer():
    door = ROOT/'assets/doors/db0966_sliding_single'; recordings = ROOT/'out/reference-motions'
    if not (recordings/'index.json').exists(): pytest.skip('Generated native corpus is optional')
    result = build_manual_guide(door, recordings)
    guide = result.guide; phase = np.asarray(guide.phases)
    assert guide.metadata['door_id'] == 'db0966_sliding_single'
    assert len(guide.time) == 2677 and result.plan.residual_hasp_effort_max < EFFORT_THRESHOLD
    transfer = np.flatnonzero(np.isin(phase, ['withdraw_hasp', 'transfer_reposition', 'transfer_face_pull', 'reach_pull']))
    assert np.ptp(guide.native_time[transfer]) == 0 and np.ptp(guide.native_qpos[transfer], axis=0).max() == 0
    assert not guide.hand_contact[transfer[1:-1]].any()
    assert {r.joint_role for r in result.roles.values()} == {'lock', 'primary'}
