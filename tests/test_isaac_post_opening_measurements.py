import numpy as np
import pytest

from doorbench.dexterous.isaac_post_opening_measurements import continuation_contact_summary


def fixture():
    paths = ['/World/H1/'+n for n in ('left_ankle_link', 'right_ankle_link',
                                     'lh_palm', 'rh_palm')]
    return dict(sensor_paths=paths, filter_paths=[['/World/floor'], ['/World/floor'],
        ['/World/Door/Articulation/leaf'], ['/World/Door/Articulation/leaf_handle']],
        normal_forces=np.array([[200.], [201.], [4.], [0.], [np.nan]]),
        normals=np.array([[0., 0, 1], [0, 0, 1], [0, -1, 0], [0, -1, 0], [np.nan]*3]),
        distances=np.array([[-.0001], [-.0001], [-.0001], [.002], [np.nan]]),
        counts=np.ones((4, 1), int), starts=np.arange(4).reshape(4, 1),
        capacity=5, physics_qualified=True)


def test_actual_contacts_keep_feet_panel_and_clear_right_hand_distinct():
    result = continuation_contact_summary(**fixture())
    np.testing.assert_array_equal(result['foot_loads'], [200, 201])
    assert result['evidence'] == dict(physics_qualified=True, left_hand_contacts=1,
                                    left_hand_load_N=4, right_environment_contacts=0)
    np.testing.assert_array_equal(result['release_normal_world'], [0, -1, 0])


@pytest.mark.parametrize('other', ['/World/H1/right_ankle_link', '/World/Door/Articulation/leaf'])
def test_robot_and_door_loads_do_not_count_as_foot_ground_support(other):
    args = fixture()
    args['filter_paths'][0] = [other]
    np.testing.assert_array_equal(continuation_contact_summary(**args)['foot_loads'], [0, 201])


def test_positive_gap_loaded_contact_still_blocks_release():
    args = fixture()
    args['normal_forces'][3] = .001
    assert continuation_contact_summary(**args)['evidence']['right_environment_contacts'] == 1
    args['normal_forces'][3] = 0
    args['distances'][3] = 0
    assert continuation_contact_summary(**args)['evidence']['right_environment_contacts'] == 1
    args['distances'][2] = .001
    assert continuation_contact_summary(**args)['evidence']['left_hand_contacts'] == 1


def test_missing_panel_contact_cannot_invent_release_direction():
    args = fixture()
    args['normal_forces'][2] = 0
    args['distances'][2] = .001
    assert continuation_contact_summary(**args)['release_normal_world'] is None


@pytest.mark.parametrize('change', ['overlap', 'truncated', 'nonfinite', 'normal', 'missing_palm'])
def test_corrupt_evidence_is_rejected(change):
    args = fixture()
    if change == 'overlap': args['starts'][1] = 0
    elif change == 'truncated': args['counts'][3] = 2
    elif change == 'nonfinite': args['normal_forces'][0] = np.nan
    elif change == 'normal': args['normals'][0] *= 2
    else: args['sensor_paths'][2] = '/World/H1/lh_ffdistal'
    with pytest.raises(ValueError): continuation_contact_summary(**args)


def test_left_self_contact_load_is_not_counted_twice():
    args = fixture()
    args['sensor_paths'] += ['/World/H1/lh_ffdistal']
    args['filter_paths'][2] = ['/World/H1/lh_ffdistal']
    args['filter_paths'] += [['/World/H1/lh_palm']]
    args['counts'] = np.ones((5, 1), int)
    args['starts'] = np.arange(5).reshape(5, 1)
    args['normal_forces'] = np.array([[200.], [201.], [4.], [0.], [4.], [np.nan]])
    args['normals'] = np.array([[0., 0, 1], [0, 0, 1], [0, -1, 0],
                               [0, -1, 0], [0, 1, 0], [np.nan]*3])
    args['distances'] = np.array([[-.0001], [-.0001], [-.0001], [.002], [-.0001], [np.nan]])
    args['capacity'] = 6
    result = continuation_contact_summary(**args)
    assert result['evidence']['left_hand_contacts'] == 1
    assert result['evidence']['left_hand_load_N'] == 4
