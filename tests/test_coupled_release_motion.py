import numpy as np
import pytest

from doorbench.dexterous.coupled_release_motion import CoupledReferenceMotion


def test_initial_capture_and_returned_values_are_copies():
    initial=np.arange(8,dtype=float);motion=CoupledReferenceMotion(initial)
    initial[:]=0
    result,info=motion.update(50.,np.arange(8,dtype=float));result[:]=99
    assert np.array_equal(motion.value,np.arange(8,dtype=float))
    assert not info['limited']


def test_joint_acceleration_ramp_is_analytic_and_preserves_original_caps():
    motion=CoupledReferenceMotion(np.zeros(8));motion.update(0.,np.zeros(8))
    goal=np.r_[np.ones(6),10.,-10.]
    for i in range(1,301):
        value,info=motion.update(i*.002,goal)
        speed=min(3*i*.002,1.2)
        assert motion.velocity[6]==pytest.approx(speed)
        assert motion.velocity[7]==pytest.approx(-speed)
        assert info['joint_acceleration_rad_s2']<=3.+1e-10
        assert info['root_speed_m_s']==pytest.approx(.02)
        assert info['root_rotation_speed_rad_s']==pytest.approx(.03)
    # Integral of the acceleration ramp plus the original velocity cap.
    expected=.002*(.006*sum(range(1,201))+1.2*100)
    assert value[6]==pytest.approx(expected)
    assert value[7]==pytest.approx(-expected)


def test_direction_change_brakes_without_an_acceleration_jump():
    motion=CoupledReferenceMotion(np.zeros(7));motion.update(0.,np.zeros(7))
    for i in range(1,11):motion.update(i*.002,np.r_[np.zeros(6),1.])
    before=motion.velocity[6]
    _,info=motion.update(.022,np.r_[np.zeros(6),-1.])
    assert motion.velocity[6]==pytest.approx(before-.006)
    assert info['joint_acceleration_rad_s2']==pytest.approx(3.)


@pytest.mark.parametrize('time',[0.,-.002,.004,float('nan')])
def test_rejects_duplicate_backwards_missing_or_nonfinite_sample(time):
    motion=CoupledReferenceMotion(np.zeros(7));motion.update(0.,np.zeros(7))
    with pytest.raises(ValueError):motion.update(time,np.ones(7))


def test_rejects_first_reference_that_is_not_attained():
    with pytest.raises(ValueError):CoupledReferenceMotion(np.zeros(7)).update(0.,np.ones(7))


def test_motion_limits_do_not_claim_kinematic_admission():
    motion=CoupledReferenceMotion(np.zeros(7));motion.update(0.,np.zeros(7))
    value,info=motion.update(.002,np.ones(7))
    assert info['limited']
    assert value[6]==pytest.approx(.000012)
    assert 'passed' not in info


def test_stationary_goal_is_braked_without_overshoot():
    motion=CoupledReferenceMotion(np.zeros(7));motion.update(0.,np.zeros(7))
    goal=np.r_[np.zeros(6),.02]
    for i in range(1,501):
        value,info=motion.update(i*.002,goal)
        assert value[6]<=.02+1e-12
        assert info['joint_acceleration_rad_s2']<=3.+1e-10
    assert value[6]==pytest.approx(.02,abs=1e-9)


@pytest.mark.parametrize('index,delta',[(0,.000041),(3,.000061),(6,.000013)])
def test_projection_cannot_bypass_limits_or_commit_a_rejected_step(index,delta):
    motion=CoupledReferenceMotion(np.zeros(7));motion.update(50.,np.zeros(7))
    def invalid(candidate,previous,velocity,dt):
        previous[:]=999.;velocity[:]=999.  # Inputs are isolated copies.
        candidate[:]=0.;candidate[index]=delta
        return candidate,{}
    with pytest.raises(ValueError,match='original reference motion limits'):
        motion.update(50.002,np.zeros(7),project=invalid)
    assert motion.time==50.
    assert np.array_equal(motion.value,np.zeros(7))
    assert np.array_equal(motion.velocity,np.zeros(7))


def test_projection_root_limit_is_a_vector_ball_not_three_axis_limits():
    motion=CoupledReferenceMotion(np.zeros(7));motion.update(0.,np.zeros(7))
    def invalid(candidate,*args):
        candidate[:3]=.00003
        return candidate,{}
    with pytest.raises(ValueError,match='original reference motion limits'):
        motion.update(.002,np.zeros(7),project=invalid)


def test_projected_velocity_is_the_next_acceleration_reference():
    motion=CoupledReferenceMotion(np.zeros(7));motion.update(0.,np.zeros(7))
    def correction(candidate,*args):
        candidate[6]=.000004
        return candidate,{'method':'analytic fixture'}
    _,info=motion.update(.002,np.ones(7),project=correction)
    assert motion.velocity[6]==pytest.approx(.002)
    assert info['projection']['method']=='analytic fixture'
    value,info=motion.update(.004,np.ones(7))
    assert motion.velocity[6]==pytest.approx(.008)
    assert value[6]==pytest.approx(.00002)
    assert info['joint_acceleration_rad_s2']==pytest.approx(3.)


def test_projection_exception_and_initial_capture_do_not_mutate_motion():
    motion=CoupledReferenceMotion(np.zeros(7))
    def reject(*args):raise ValueError('infeasible pose')
    motion.update(0.,np.zeros(7),project=reject)  # Exact initial capture needs no solve.
    with pytest.raises(ValueError,match='infeasible pose'):
        motion.update(.002,np.ones(7),project=reject)
    assert motion.time==0.
    assert np.array_equal(motion.value,np.zeros(7))
