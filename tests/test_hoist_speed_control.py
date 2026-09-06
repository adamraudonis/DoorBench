"""Load-error and saturation regressions; not a door-physics certificate."""
import pytest
from doorbench.hoist_speed_control import HoistSpeedState,speed_force


def plant(load,state):
    x=0.;v=0.;peak=0.;dt=.001
    for tick in range(45000):
        desired=-min(.45,max(0.,3*(1.-x)))
        force,_=speed_force(desired,v,tick*dt,120.,True,state)
        v+=(force+load-8*v)/20*dt
        x+=-.25*v*dt
        peak=max(peak,abs(force))
    return x,peak


@pytest.mark.parametrize('load',(15.8,43.8,70.))
def test_integral_removes_finite_load_endpoint_error_without_raising_cap(load):
    old,_=plant(load,None)
    state=HoistSpeedState();new,peak=plant(load,state)
    assert abs(1-old)>.02
    assert abs(1-new)<.01
    assert peak<=120
    assert state.integral_force_N==pytest.approx(-load,abs=1.)


def test_overload_cannot_be_hidden_by_windup():
    state=HoistSpeedState()
    for tick in range(20000):
        force,integral=speed_force(-.45,.10,tick*.001,120.,True,state)
        assert force==-120.
        assert integral==0.


def test_memory_is_per_run_and_resets_on_direction_or_time_restart():
    a=HoistSpeedState();b=HoistSpeedState()
    speed_force(-.1,0,0.,120.,True,a)
    speed_force(-.1,0,1.,120.,True,a)
    assert a.integral_force_N<0 and b.integral_force_N==0
    speed_force(0,0,1.,120.,True,a)
    before=a.integral_force_N
    speed_force(0,0,1.,120.,True,a)
    assert a.integral_force_N==before
    speed_force(0,0,2.,120.,False,a)
    assert a.integral_force_N==0
    speed_force(-.1,0,3.,120.,False,a)
    speed_force(0,0,0.,120.,False,a)
    assert a.integral_force_N==0
