"""Mathematical signal/guard tests only: no motor or physical qualification."""
import numpy as np
import pytest

from doorbench.dexterous.guarded_ik_reference import GuardedIkReference


def make(initial=(0.,),**kwargs):
    n=len(initial)
    args=dict(start_time_s=0.,dt=.002,lower=[-2.]*n,upper=[2.]*n,
        maximum_velocity=[100.]*n,maximum_acceleration=[1e6]*n,
        minimum_segment_s=.02,maximum_segment_s=.5)
    args.update(kwargs)
    return GuardedIkReference(tuple('joint'+str(i) for i in range(n)),initial,**args)


def admit(sample):return True  # Analytic test fixture, never a live geometry gate.


def test_analytic_minimum_jerk_samples_and_exact_rest_endpoints():
    ref=make()
    rows=[ref.sample(i*.002,target=[1.] if i==0 else None,validate=admit) for i in range(11)]
    assert rows[0].position[0]==0 and rows[-1].position[0]==1
    assert rows[5].position[0]==pytest.approx(.5)
    assert rows[5].velocity[0]==pytest.approx(93.75)
    assert rows[5].acceleration[0]==pytest.approx(0,abs=1e-10)
    assert rows[1].position[0]==pytest.approx(.00856)
    for row in (rows[0],rows[-1]):
        np.testing.assert_array_equal(row.velocity,[0.]);np.testing.assert_array_equal(row.acceleration,[0.])


def test_one_progress_coordinate_keeps_complete_vector_on_segment_without_joint_clips():
    ref=make((.2,-.4),lower=[0,-1],upper=[1,0])
    for i in range(11):
        row=ref.sample(i*.002,target=[.8,-.1] if i==0 else None,validate=admit)
        assert (row.position[0]-.2)/.6==pytest.approx((row.position[1]+.4)/.3)
        assert 0<=row.position[0]<=1 and -1<=row.position[1]<=0


def test_new_refresh_at_endpoint_has_no_position_velocity_or_acceleration_jump():
    ref=make()
    for i in range(10):ref.sample(i*.002,target=[.1] if i==0 else None,validate=admit)
    at=ref.sample(.02,target=[-.2],validate=admit)
    np.testing.assert_array_equal(at.position,[.1])
    np.testing.assert_array_equal(at.velocity,[0.]);np.testing.assert_array_equal(at.acceleration,[0.])
    after=ref.sample(.022,validate=admit)
    assert -.2<after.position[0]<.1 and after.velocity[0]<0


def test_early_refreshes_coalesce_causally_without_resetting_active_segment():
    ref=make()
    rows=[]
    for i in range(11):
        target={0:[.1],3:[.9],5:[-.1]}.get(i)
        rows.append(ref.sample(i*.002,target=target,validate=admit))
    assert rows[3].queued_target and rows[5].queued_target
    assert rows[5].position[0]==pytest.approx(.05)
    assert rows[5].goal_time_s==0 and rows[5].latest_target_time_s==.01
    assert rows[10].position[0]==.1 and rows[10].goal_time_s==.01
    assert not rows[10].queued_target
    assert ref.sample(.022,validate=admit).velocity[0]<0  # Latest -.1, never queued .9.


def test_explicit_rate_and_acceleration_bounds_choose_sufficient_whole_segment_duration():
    ref=make(maximum_velocity=[.5],maximum_acceleration=[4.],maximum_segment_s=1.)
    first=ref.sample(0.,target=[.1],validate=admit)
    assert first.segment_duration_s>=.375  # Any minimum-jerk .1 step at .5 rad/s.
    count=round(first.segment_duration_s/.002)
    rows=[first]+[ref.sample(i*.002,validate=admit) for i in range(1,count+1)]
    assert max(abs(r.velocity[0]) for r in rows)<=.5
    assert max(abs(r.acceleration[0]) for r in rows)<=4.
    assert all(a.position[0]<=b.position[0] for a,b in zip(rows,rows[1:]))
    assert rows[-1].position[0]==.1


def test_large_step_does_not_silently_relax_caps_or_extend_maximum_duration():
    ref=make(maximum_velocity=[.1],maximum_acceleration=[1.],maximum_segment_s=.1)
    with pytest.raises(ValueError,match='duration'):ref.sample(0.,target=[1.],validate=admit)
    assert ref.last_time is None and ref.failure


def test_original_nonlinear_constraint_can_fail_between_valid_endpoints_and_stops_before_commit():
    ref=make((1.,0.))
    def unit_circle(sample):return bool(abs(np.linalg.norm(sample.position)-1.)<1e-6)
    first=ref.sample(0.,target=[0.,1.],validate=unit_circle)
    assert np.linalg.norm(first.position)==1.
    with pytest.raises(ValueError,match='guard rejected'):
        ref.sample(.002,validate=unit_circle)
    assert ref.last_time==0.
    np.testing.assert_array_equal(ref._position,[1.,0.])
    with pytest.raises(ValueError,match='terminal'):ref.sample(.002,validate=admit)


def test_every_motor_tick_requires_guard_and_returns_defensive_readonly_arrays():
    ref=make();calls=[]
    for i in range(11):
        row=ref.sample(i*.002,target=[.1] if i==0 else None,
            validate=lambda x:calls.append(x.time_s) or True)
        with pytest.raises(ValueError):row.position[0]=1.
    assert calls==[i*.002 for i in range(11)]
    with pytest.raises(ValueError,match='guard required'):ref.sample(.022,validate=None)


@pytest.mark.parametrize('bad',[.001,.004,-.002,float('nan'),float('inf'),'bad'])
def test_skipped_backwards_or_nonfinite_clock_is_terminal(bad):
    ref=make();ref.sample(0.,validate=admit)
    with pytest.raises((ValueError,TypeError)):ref.sample(bad,validate=admit)
    assert ref.failure is not None and ref.last_time==0.


@pytest.mark.parametrize('target',[[float('nan')],[float('inf')],[3.],[],[.1,.2]])
def test_invalid_or_outside_joint_box_endpoint_is_rejected_before_guard(target):
    ref=make()
    with pytest.raises(ValueError):ref.sample(0.,target=target,validate=lambda _:pytest.fail('Guard invoked'))
    assert ref.last_time is None


@pytest.mark.parametrize('options',[dict(dt=0),dict(minimum_segment_s=.001),dict(maximum_segment_s=.01),
    dict(maximum_velocity=[0.]),dict(maximum_acceleration=[-1.]),dict(lower=[.1]),
    dict(upper=[-1.]),dict(maximum_velocity=[float('nan')])])
def test_invalid_prospective_contract_rejects_at_construction(options):
    with pytest.raises(ValueError):make(**options)


def test_initial_reference_preserves_existing_position_servo_algebra():
    ref=make((.22381985422816136,))
    at=ref.sample(0.,target=[.19180852944251797],validate=admit)
    q=.23612524569034576;v=-2.828822612762451
    old=3000*(.22381985422816136-q)-26*v
    prospective=3000*(at.position[0]-q)+26*(at.velocity[0]-v)
    assert prospective==old  # Signal seam only; not a replayed physical command.
