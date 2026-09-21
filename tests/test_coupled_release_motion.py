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
