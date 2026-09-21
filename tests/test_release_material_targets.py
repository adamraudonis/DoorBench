import pytest
from doorbench.dexterous.release_material_targets import loaded_segment_targets


def patch(segment, force, z, *, qualified=True):
    return dict(digit='ff', body='robot/rh_ff' + segment,
                normal_force_N=force, body_position_m=[0., -.004, z],
                pad_qualified=qualified)


def test_each_loaded_segment_is_preserved_even_when_distal_dominates():
    result = loaded_segment_targets([patch('middle', 1., .020),
        patch('middle', 3., .024), patch('distal', 100., .010)], 'ff')
    assert len(result) == 2
    middle = next(row for row in result if row['body'].endswith('middle'))
    assert middle['body_position_m'][2] == pytest.approx(.023)
    assert middle['source_force_N'] == 4.


def test_no_contacts_or_loaded_invalid_source_fail_closed():
    with pytest.raises(ValueError):
        loaded_segment_targets([], 'ff')
    with pytest.raises(ValueError):
        loaded_segment_targets([patch('middle', .2, .026, qualified=False)], 'ff')


@pytest.mark.parametrize('force', [-1., float('nan'), float('inf')])
def test_corrupt_force_cannot_become_a_target(force):
    with pytest.raises(ValueError):
        loaded_segment_targets([patch('middle', force, .020)], 'ff')


def test_route_interpolation_keeps_exact_source_during_initial_hold():
    import numpy as np
    from scripts.dexterous.screen_profiled_radial_release import interpolate
    initial = np.array([1., 2., 3., 1., 0., 0., 0., .2])
    final = initial.copy()
    final[-1] = .5
    rows = [dict(time_s=0., qpos=initial.tolist()),
            dict(time_s=8., qpos=final.tolist())]
    values, clock = interpolate(rows, initial, 0, np.array([0., .194, 16.]), 16.)
    assert np.array_equal(values[0], initial)
    assert np.array_equal(values[1], initial)
    assert np.array_equal(values[-1], final)
    assert clock[1] < .0002


def test_bad_route_clock_cannot_silently_interpolate():
    import numpy as np
    from scripts.dexterous.screen_profiled_radial_release import interpolate
    initial = np.array([1., 2., 3., 1., 0., 0., 0., .2])
    rows = [dict(time_s=0., qpos=initial.tolist()), dict(time_s=0., qpos=initial.tolist())]
    with pytest.raises(ValueError):
        interpolate(rows, initial, 0, np.array([0., 16.]), 16.)
