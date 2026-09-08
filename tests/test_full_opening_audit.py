import pytest

from doorbench.dexterous.full_opening_audit import full_opening_checks


def fixture():
    dt=.002
    steps=[dict(time_s=i*dt,angles=dict(leaf=1.2 if i==1500 else .08,operator=.85,latch=.012),
        surface=dict(total_normal_load_N=3.,palm_normal_load_N=3.),
        geometry=dict(time_s=i*dt,right_lever_clearance_m=.03,native_mirror_steps=0),
        teacher=dict(release=dict(release_fraction=1.))) for i in range(1,1501)]
    pads=[dict(sim_time_s=i*dt,valid_pad_grasp=i<=1000,contacts=[]) for i in range(1501)]
    return dict(steps=steps,pad_steps=pads,physics_dt=dt,maximum_seconds=55.,
                target_aperture=1.2,operation_started=1.,release_started=2.)


def score(args):
    return full_opening_checks({'sustained_pad_grasp':False,'native_motor_caps':True},**args)


def test_intentional_release_requires_replacement_measured_contact_chain():
    checks=score(fixture())
    assert all(checks.values()),checks
    assert 'sustained_pad_grasp' not in checks
    assert checks['qualified_grasp_before_intentional_release']


def test_one_two_millisecond_palm_unload_invalidates_final_hold():
    args=fixture()
    args['steps'][-100]['surface']['palm_normal_load_N']=0.
    assert not score(args)['sustained_left_palm_load']
    assert score(args)['usable_aperture_under_palm_load']


def test_grip_or_panel_support_must_precede_release():
    args=fixture()
    args['pad_steps'][999]['valid_pad_grasp']=False
    args['steps'][998]['surface']['total_normal_load_N']=0.
    checks=score(args)
    assert not checks['qualified_grasp_before_intentional_release']
    assert not checks['left_support_precedes_right_release']


def test_missing_samples_stale_geometry_or_skipped_crossing_fail():
    args=fixture();args['steps'].pop(700)
    assert not score(args)['complete_physics_steps']
    args=fixture();args['steps'][700]['geometry']['time_s']-=.002
    assert not score(args)['pose_joint_clock_aligned']
    args=fixture();args['steps'][-2]['angles']['leaf']=1.2
    assert not score(args)['complete_physics_steps']


def test_old_physical_failure_and_wrong_pad_are_never_erased():
    args=fixture();args['pad_steps'][900]['contacts']=[dict(pad_qualified=False)]
    checks=full_opening_checks({'upright':False,'sustained_pad_grasp':False},**args)
    assert not checks['upright']
    assert not checks['no_invalid_right_pad_patch']


def test_truncated_capture_cannot_become_aperture_success():
    args=fixture();args['steps'][-1]['angles']['leaf']=.5
    checks=score(args)
    assert not checks['complete_physics_steps']
    assert not checks['usable_aperture_under_palm_load']


@pytest.mark.parametrize('name,value',[('physics_dt',0.),('physics_dt',float('nan')),
    ('maximum_seconds',float('inf')),('target_aperture',-1.)])
def test_invalid_declared_timing_or_target_is_rejected(name,value):
    args=fixture();args[name]=value
    with pytest.raises(ValueError):score(args)


def test_nonfinite_measured_load_or_geometry_is_rejected():
    args=fixture();args['steps'][100]['surface']['palm_normal_load_N']=float('nan')
    with pytest.raises(ValueError):score(args)
    args=fixture();args['steps'][100]['geometry']['right_lever_clearance_m']=float('inf')
    with pytest.raises(ValueError):score(args)
