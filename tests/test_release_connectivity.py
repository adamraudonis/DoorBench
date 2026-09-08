import numpy as np
import pytest
from doorbench.dexterous.release_connectivity import monotone_release_path


def test_individually_feasible_frames_can_be_disconnected():
    free = np.array([[False, False, True], [False, False, True], [True, False, False]])
    assert free.any(axis=1).all()
    assert monotone_release_path(free) is None


def test_adjacent_safe_turn_is_returned_without_corner_jump():
    free = np.array([[False, False, True], [True, True, True], [True, False, False]])
    path = monotone_release_path(free)
    assert path[0] == (0, 2) and path[-1] == (2, 0)
    for a, b in zip(path, path[1:]):
        assert (b[0]-a[0], b[1]-a[1]) in ((1, 0), (0, -1))
        assert free[b]
    free[1, 1] = False
    assert monotone_release_path(free) is None


def test_missing_endpoints_and_backtracking_route_rejected():
    free = np.ones((3, 3), dtype=bool)
    free[0, 2] = False
    assert monotone_release_path(free) is None
    assert monotone_release_path(np.eye(3, dtype=bool)[::-1]) is None


@pytest.mark.parametrize('bad', [np.ones((3, 3)), [[True]], [True, False], [[float('nan')]*2]*2])
def test_non_boolean_or_degenerate_grid_rejected(bad):
    with pytest.raises(ValueError):
        monotone_release_path(bad)
