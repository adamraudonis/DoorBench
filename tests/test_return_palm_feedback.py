import numpy as np
import pytest
from doorbench.dexterous.return_palm_feedback import integrate_correction


def test_correction_is_rate_limited_and_cannot_cross_original_joint_stop():
    q=integrate_correction(np.zeros(6),np.eye(6),np.ones(6),.01,
        np.full(6,-1.),np.array([.0002,1,1,1,1,1]))
    np.testing.assert_allclose(q,[.0002,.0006,.0006,.0006,.0006,.0006])
    q=integrate_correction(np.full(6,.06),np.eye(6),np.ones(6),.01,
        np.full(6,-1.),np.ones(6))
    np.testing.assert_allclose(q,.06)


def test_singular_arm_has_no_unbounded_correction_and_bad_clock_is_rejected():
    q=integrate_correction(np.zeros(7),np.zeros((6,7)),np.ones(6),.01,
        np.full(7,-1.),np.ones(7))
    np.testing.assert_array_equal(q,0.)
    for dt in [-.01,.1,float('nan')]:
        with pytest.raises(ValueError):
            integrate_correction(np.zeros(6),np.eye(6),np.ones(6),dt,-np.ones(6),np.ones(6))
