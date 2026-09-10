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


def test_braking_can_retain_explicit_bounded_support_margin():
    c=PanelApertureForce(terminal_aperture=.75,terminal_support_margin_N=.5)
    for i in range(1001):c.update(i*.002,.75,.65,2.25)
    for i in range(1001,1301):target,info=c.update(i*.002,.75,.741,2.25)
    assert target==2.75 and info['panel_force_profile']=='bounded-pi-stop-v2'
    with pytest.raises(ValueError):PanelApertureForce(terminal_aperture=.75,terminal_support_margin_N=.6)


def test_stiction_assistance_builds_bounded_force_only_for_stalled_progress():
    controller=PanelApertureForce(terminal_aperture=1.2,terminal_support_margin_N=.5,stiction_assist=True)
    ordinary=PanelApertureForce(terminal_aperture=1.2,terminal_support_margin_N=.5)
    for i in range(2501):
        force,info=controller.update(i*.002,.755,.75,2.25)
        baseline,_=ordinary.update(i*.002,.755,.75,2.25)
    assert info['panel_stiction_active']
    assert force>baseline+.4
    assert force<=controller.maximum_target_N
    for i in range(2501,2752):
        force,info=controller.update(i*.002,1.2,1.195,2.25)
    assert not info['panel_stiction_active']
    assert force<=2.75


def test_stiction_assistance_does_not_boost_moving_or_completed_reference():
    controller=PanelApertureForce(terminal_aperture=1.2,stiction_assist=True)
    for i in range(501):
        angle=.75+i*.002*.01
        _,info=controller.update(i*.002,angle+.005,angle,2.25)
    assert not info['panel_stiction_active']
    controller=PanelApertureForce(terminal_aperture=1.2,stiction_assist=True)
    _,info=controller.update(0.,.75,.75,2.25)
    assert not info['panel_stiction_active']


def test_explicit_seven_newton_profile_overcomes_integral_cap_and_still_brakes():
    c=PanelApertureForce(terminal_aperture=1.2,terminal_support_margin_N=.5,stiction_assist=True,load_profile='bounded-7N-v1')
    for i in range(30001):target,info=c.update(i*.002,1.12,1.10,2.25)
    assert 6<target<=7 and 3<info['panel_force_integral_N']<=4.5
    assert info['panel_load_profile']=='bounded-7N-v1'
    for i in range(30001,30201):target,info=c.update(i*.002,1.2,1.195,2.25)
    assert target==2.75
    with pytest.raises(ValueError):PanelApertureForce(load_profile='bounded-7N-v1')
    with pytest.raises(ValueError):PanelApertureForce(load_profile='unbounded')


def test_terminal_support_floor_preserves_prebrake_force_and_original_caps():
    baseline=PanelApertureForce(terminal_aperture=.75,terminal_support_margin_N=.5,stiction_assist=True)
    supported=PanelApertureForce(terminal_aperture=.75,terminal_support_margin_N=.5,stiction_assist=True,terminal_minimum_support_N=2.4)
    for i in range(501):
        t=i*.002
        original,_=baseline.update(t,.705,.7,2.25)
        actual,_=supported.update(t,.705,.7,2.25)
        assert actual==original
    previous=None
    for i in range(501,2001):
        # Overshoot causes the ordinary PI term to request its lowest load.
        t=i*.002
        original,_=baseline.update(t,.75,.795,2.25)
        actual,info=supported.update(t,.75,.795,2.25)
        assert 2.05<=actual<=6 and info['panel_force_integral_N']<=3
        if previous is not None: assert abs(actual-previous)<.03
        previous=actual
    assert original<2.1 and actual==2.4
    assert info['terminal_minimum_support_N']==2.4
    for value in [1.9,2.8,float('nan')]:
        with pytest.raises(ValueError,match='support floor'):
            PanelApertureForce(terminal_aperture=.75,terminal_minimum_support_N=value)
    with pytest.raises(ValueError,match='support floor'):
        PanelApertureForce(terminal_minimum_support_N=2.4)


def test_nine_newton_candidate_is_explicit_bounded_and_releases_integral_at_target():
    c=PanelApertureForce(terminal_aperture=1.2,stiction_assist=True,
        load_profile='bounded-9N-v1',terminal_minimum_support_N=2.4)
    for i in range(40001):
        target,info=c.update(i*.002,1.105,1.1,2.25)
        assert 2.05<=target<=9
    assert 7<target<=9 and info['panel_force_integral_N']==6.5
    assert info['panel_maximum_target_N']==9
    for i in range(40001,40201):target,info=c.update(i*.002,1.2,1.205,2.25)
    assert target==2.4
    assert PanelApertureForce().maximum_target_N==6
    with pytest.raises(ValueError):PanelApertureForce(load_profile='bounded-9N-v1')
