import numpy as np
import pytest
from doorbench.dexterous.handle_hub_avoidance import avoidance_force


def test_avoidance_is_outward_bounded_and_vanishes_with_clearance():
    np.testing.assert_allclose(avoidance_force(.01,[1,0,0],[0,0,0]),0)
    np.testing.assert_allclose(avoidance_force(.003,[1,0,0],[0,0,0]),[.8,0,0])
    np.testing.assert_allclose(avoidance_force(-.001,[1,0,0],[-1,0,0]),[3,0,0])
    np.testing.assert_allclose(avoidance_force(.003,[1,0,0],[1,0,0]),0)
    with pytest.raises(ValueError):avoidance_force(0,[2,0,0],[0,0,0])
