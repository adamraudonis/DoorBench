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
