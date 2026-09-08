import numpy as np
import pytest
from doorbench.dexterous.screened_panel_path import ScreenedPanelPath


def fixture(duration=12.):
    s=np.linspace(0,1,11)
    return ScreenedPanelPath(s,np.column_stack([s,2*s*s,s**3]),duration)


def test_derivatives_match_independent_finite_differences():
    path=fixture();dt=1e-4
    for t in (1.,3.,5.,9.,11.):
        a=path.sample(t-dt);b=path.sample(t);c=path.sample(t+dt)
        np.testing.assert_allclose(b['velocity'],(c['position']-a['position'])/(2*dt),atol=1e-8)
        np.testing.assert_allclose(b['acceleration'],(c['velocity']-a['velocity'])/(2*dt),atol=1e-8)


def test_time_scaling_reduces_velocity_and_acceleration_without_changing_route():
    first=fixture(12.);second=fixture(24.)
    for t in np.linspace(0,12,51):
        a,b=first.sample(t),second.sample(2*t)
        np.testing.assert_array_equal(a['position'],b['position'])
        np.testing.assert_allclose(a['velocity'],2*b['velocity'],atol=1e-12)
        np.testing.assert_allclose(a['acceleration'],4*b['acceleration'],atol=1e-12)


def test_endpoint_position_and_zero_first_second_derivatives():
    path=fixture()
    for t,expected in ((-1.,[0,0,0]),(0.,[0,0,0]),(12.,[1,2,1]),(13.,[1,2,1])):
        sample=path.sample(t)
        np.testing.assert_allclose(sample['position'],expected,atol=1e-12)
        np.testing.assert_array_equal(sample['velocity'],np.zeros(3))
        np.testing.assert_array_equal(sample['acceleration'],np.zeros(3))


def test_invalid_path_and_clock_rejected():
    with pytest.raises(ValueError):ScreenedPanelPath([0,.5,.4,1],np.zeros((4,2)),12.)
    with pytest.raises(ValueError):ScreenedPanelPath([0,.3,.6,1],np.zeros((4,2)),2.)
    with pytest.raises(ValueError):fixture().sample(float('nan'))


def test_measured_phase_preserves_rate_bounds_even_when_measurement_jumps():
    from doorbench.dexterous.screened_panel_path import MeasuredAperturePhase
    phase=MeasuredAperturePhase(.28,1.2,.145)
    last_angle=.28;last_velocity=.145
    for t in np.arange(0,24,.002):
        angle,velocity,acceleration=phase.update(float(t),1.2)
        assert angle>=last_angle and angle<=1.2
        assert 0<=velocity<=.149+1e-12
        assert abs(acceleration)<=.08+1e-12
        if t and velocity>1e-14:np.testing.assert_allclose(angle-last_angle,.5*(velocity+last_velocity)*.002,atol=1e-8)
        last_angle,last_velocity=angle,velocity
    assert abs(angle-1.2)<1e-6
    assert velocity<1e-5


def test_phase_cannot_accept_missing_physics_or_unbounded_initial_velocity():
    from doorbench.dexterous.screened_panel_path import MeasuredAperturePhase
    with pytest.raises(ValueError):MeasuredAperturePhase(.2,1.2,.3)
    phase=MeasuredAperturePhase(.2,1.2,.1);phase.update(0,.2)
    with pytest.raises(ValueError):phase.update(.01,.2)
