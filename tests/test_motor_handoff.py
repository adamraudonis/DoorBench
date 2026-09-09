import numpy as np
import pytest
from doorbench.dexterous.motor_handoff import MotorHandoff


def test_switch_matches_previous_and_recovers_live_controller_within_original_caps():
    previous=np.array([12.,-5.]);first=np.array([-13.,11.]);caps=np.array([[-30.,30.],[-20.,20.]])
    switch=MotorHandoff(previous,first,caps,1.)
    np.testing.assert_allclose(switch.force(first,0.),previous,atol=1e-14)
    # The new controller continues receiving feedback while its offset decays.
    live=np.array([29.,-19.]);mid=switch.force(live,.5)
    assert np.all(mid>=caps[:,0]) and np.all(mid<=caps[:,1])
    np.testing.assert_array_equal(switch.force(live,1.),live)
    np.testing.assert_array_equal(switch.force(first,2.),first)
    with pytest.raises(ValueError):switch.force(first,.5)


@pytest.mark.parametrize('previous,first,seconds', [
    ([31.,0.],[0.,0.],1.),([0.,0.],[float('nan'),0.],1.),
    ([0.,0.],[0.,0.],0.),([0.,0.],[0.,0.],3.),
])
def test_invalid_transition_rejected(previous,first,seconds):
    with pytest.raises(ValueError):MotorHandoff(previous,first,[[-30.,30.],[-20.,20.]],seconds)
