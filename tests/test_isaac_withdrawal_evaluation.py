import copy
import pytest

from doorbench.dexterous.isaac_withdrawal_evaluation import pack_withdrawal_row, withdrawal_checks


def episode():
    rows = []
    for i in range(1, 1001):
        t = i*.002
        geometry = None if t <= 1. else dict(time_s=t, geometry_time_s=t,
            native_mirror_steps=0, right_environment_clearance_m=.05,
            maximum_pose_position_error_m=0., maximum_pose_rotation_error=0.,
            body_poses_xyz_wxyz={'rh_palm': [0, 0, 0, 1, 0, 0, 0]})
        rows.append(pack_withdrawal_row(t,
            dict(leaf=.08 if t <= 1.5 else .15, operator=0., latch=0.),
            dict(sim_time_s=t, valid_pad_grasp=t <= 1.5, contacts=[]),
            dict(palm_normal_load_N=3., total_normal_load_N=9., body_panel_forces_world_N={'lh_palm': [0, -3, 0]}),
            dict(phase='standing_withdrawal'), geometry,
            leaf_pose=[0, 0, 0, 1, 0, 0, 0]))
    return rows


def score(rows):
    return withdrawal_checks(dict(sustained_pad_grasp=False, finite=True,
        partial_leaf_opening_held=False, opening_bounded_for_transfer=False), rows,
        dt=.002, duration=2., started=1., release_started=1.5, completed=True)


def test_intentional_release_preserves_qualified_windows_and_requires_clearance():
    result = score(episode())
    assert 'sustained_pad_grasp' not in result
    assert all(result.values())


@pytest.mark.parametrize('field', ['sim_time_s', 'leaf_pose', 'geometry'])
def test_missing_or_stale_actual_record_cannot_pass(field):
    rows = episode()
    if field == 'sim_time_s':
        rows[-1][field] -= .002
        assert not score(rows)['complete_withdrawal_clock']
    elif field == 'geometry':
        rows[-1][field] = None
        assert not score(rows)['synchronized_withdrawal_geometry']
    else:
        rows[-1][field] = [0, 0, 0, 0, 0, 0, 0]
        with pytest.raises(ValueError):score(rows)


def test_finger_force_cannot_replace_one_unloaded_palm_interval():
    rows = episode()
    rows[-1]['left_surface'].update(palm_normal_load_N=0., total_normal_load_N=100.,
        body_panel_forces_world_N={'lh_palm': [0, 0, 0]})
    result = score(rows)
    assert result['measured_palm_load_accounting']
    assert not result['final_left_palm_support']


def test_forged_palm_scalar_cannot_hide_zero_measured_force():
    rows = episode()
    rows[-1]['left_surface']['body_panel_forces_world_N']['lh_palm'] = [0, 0, 0]
    assert not score(rows)['measured_palm_load_accounting']


def test_original_anatomical_and_clearance_limits_remain_strict():
    rows = episode()
    rows[-1]['pad_grasp']['contacts'] = [dict(pad_qualified=False, normal_force_N=.001)]
    rows[-1]['right_environment_clearance_m'] = .039999
    result = score(rows)
    assert not result['no_invalid_loaded_right_surfaces']
    assert not result['final_hand_clear_of_environment']


def test_entry_requires_every_original_half_second_palm_sample():
    rows = episode()
    rows[300]['left_surface'].update(palm_normal_load_N=0., body_panel_forces_world_N={'lh_palm': [0, 0, 0]})
    assert not score(rows)['resting_palm_support_before_withdrawal']


def test_packer_copies_contacts_and_rejects_different_contact_clock():
    row = episode()[0]
    pad = dict(sim_time_s=.002, valid_pad_grasp=True, contacts=[])
    packed = pack_withdrawal_row(.002, dict(leaf=.08, operator=0., latch=0.), pad,
        row['left_surface'], {}, leaf_pose=row['leaf_pose'])
    pad['contacts'].append({'pad_qualified':False})
    assert packed['pad_grasp']['contacts'] == []
    with pytest.raises(ValueError, match='epochs'):
        pack_withdrawal_row(.004, dict(leaf=.08, operator=0., latch=0.), pad,
            row['left_surface'], {}, leaf_pose=row['leaf_pose'])
