import numpy as np
import pytest
from doorbench.dexterous.transfer_preload import transfer_preload


INITIAL = dict(ff=2., mf=3., rf=2., lf=2., th=6.)


def test_reference_change_is_continuous_bounded_and_does_not_mutate_initial():
    before = dict(INITIAL)
    assert transfer_preload(INITIAL, 0., 'index-6n') == before
    rows = [transfer_preload(INITIAL, t, 'index-6n') for t in np.linspace(0, 2, 2001)]
    for digit, end in dict(ff=6., mf=4., rf=4., lf=4., th=8.).items():
        values = np.array([row[digit] for row in rows])
        assert values.min() >= INITIAL[digit] and values.max() <= end
        assert np.all(np.diff(values) >= 0)
        assert abs(values[1]-values[0]) < 1e-6
        assert abs(values[1000]-values[999]) < 1e-6
        assert values[-1] == end
    assert INITIAL == before


def test_default_profile_preserves_existing_preloads():
    for t in (0., .5, 1., 100.):
        assert transfer_preload(INITIAL, t, 'maintain') == INITIAL


@pytest.mark.parametrize('initial,t,profile', [
    (INITIAL, -1., 'index-6n'), (INITIAL, float('nan'), 'maintain'),
    ({**INITIAL, 'ff': 9.}, 0., 'maintain'),
    ({k:v for k,v in INITIAL.items() if k != 'th'}, 0., 'maintain'),
    (INITIAL, 0., 'unrecorded-experiment'),
])
def test_invalid_experiments_fail_before_motor_changes(initial, t, profile):
    with pytest.raises(ValueError):
        transfer_preload(initial, t, profile)
