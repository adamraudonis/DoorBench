import numpy as np
import pytest
from doorbench.dexterous.pad_tracking import tangent_force


def test_normal_force_removed_and_tangent_bounded():
    np.testing.assert_allclose(tangent_force([50., 3., 4.], [1., 0., 0.], 1.), [0., .6, .8])
    np.testing.assert_allclose(tangent_force([0., .2, -.1], [1., 0., 0.], 1.), [0., .2, -.1])


def test_rotation_equivariance():
    from scipy.spatial.transform import Rotation
    r=Rotation.from_rotvec([.3, -.4, .5]).as_matrix()
    f=np.array([2., 3., -4.]);n=np.array([0., 1., 0.])
    np.testing.assert_allclose(tangent_force(r@f,r@n,1.),r@tangent_force(f,n,1.),atol=1e-12)


def test_bad_normal_rejected():
    with pytest.raises(ValueError):
        tangent_force([1., 2., 3.], [0., 0., 0.], 1.)
