from types import SimpleNamespace
import numpy as np
import pytest
from doorbench.dexterous.standing_support_feedback import StandingSupportFeedback


def test_measured_surface_velocity_and_bounded_transition():
    c=StandingSupportFeedback.__new__(StandingSupportFeedback)
    c.previous=None;c.started=None;c.surface=np.array([[0.,.01,0.],[0.,.01,.01],[0.,0.,0.]])
    c.left=SimpleNamespace(palm=0,d=SimpleNamespace(site_xpos=np.array([[.2,0.,0.]]),site_xmat=np.eye(3).reshape(1,9)))
    c.update(0.,np.array([0.,0.,0.,1.,0.,0.,0.]),4.,2.25)
    assert c.left.hybrid_blend==0.
    for i in range(1,201):
        t=i*.01;c.update(t,np.array([0.,t*.003,0.,1.,0.,0.,0.]),2.25,2.25)
    np.testing.assert_allclose(c.left.surface_velocity_world,[0.,.003,0.],atol=1e-12)
    np.testing.assert_allclose(c.left.normal_contact_point_local,[0.,.01,.005])
    assert c.left.hybrid_blend==1.
    assert 2.25<=c.left.filtered_palm_load<2.251
    with pytest.raises(ValueError,match='clock'):c.update(2.,np.array([0.,0.,0.,1.,0.,0.,0.]),2.25,2.25)


def test_left_velocity_uses_ik_clock_not_repeated_physics_calls():
    c=StandingSupportFeedback.__new__(StandingSupportFeedback)
    c.left=SimpleNamespace(last_update=0.,target=np.zeros(7))
    c.update_target_velocity()
    for i in range(1,201):
        c.left.last_update=i*.01;c.left.target=np.full(7,i*.002)
        c.update_target_velocity();once=c.left.target_velocity.copy()
        for _ in range(4):c.update_target_velocity()
        np.testing.assert_array_equal(c.left.target_velocity,once)
    np.testing.assert_allclose(c.left.target_velocity,.2,atol=1e-8)
    c.left.last_update+=.01;c.left.target+=100.
    c.update_target_velocity()
    assert np.all(c.left.target_velocity<=2.)
    c.left.last_update-=.02
    with pytest.raises(ValueError,match='clock'):c.update_target_velocity()
