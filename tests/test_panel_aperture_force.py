import pytest
from doorbench.dexterous.panel_aperture_force import PanelApertureForce


def test_static_tracking_error_builds_bounded_force_and_releases_windup():
    c=PanelApertureForce();first,_=c.update(0.,.75,.65,2.25)
    for i in range(1,10001):last,info=c.update(i*.002,.75,.65,2.25)
    assert last>first and last<=6 and info['panel_force_integral_N']<=3
    before=info['panel_force_integral_N']
    for i in range(10001,10501):last,info=c.update(i*.002,.65,.75,2.25)
    assert info['panel_force_integral_N']<before and 2.05<=last<=6


def test_invalid_measurements_and_clock_cannot_modify_the_target():
    c=PanelApertureForce()
    with pytest.raises(ValueError):c.update(0.,float('nan'),.5,2.25)
    c.update(0.,.5,.5,2.25)
    with pytest.raises(ValueError,match='clock'):c.update(0.,.5,.5,2.25)
    with pytest.raises(ValueError,match='clock'):c.update(.1,.5,.5,2.25)


def test_terminal_braking_releases_accumulated_force_smoothly_and_stays_latched():
    c=PanelApertureForce(terminal_aperture=.75)
    for i in range(10001):target,_=c.update(i*.002,.75,.65,2.25)
    assert target>4.
    before=target
    target,info=c.update(20.002,.75,.741,2.25)
    assert info['panel_braking_started_s']==20.002
    for i in range(10002,10201):target,_=c.update(i*.002,.75,.741,2.25)
    assert 2.05<=target<=2.25
    # A small backward fluctuation cannot restore the accumulated pushing load.
    target,info=c.update(20.402,.75,.738,2.25)
    assert target<=2.25 and info['panel_braking_started_s']==20.002
    with pytest.raises(ValueError):PanelApertureForce(terminal_aperture=float('nan'))
