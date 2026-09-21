import numpy as np
import pytest
from doorbench.dexterous.return_palm_feedback import integrate_correction


def test_correction_is_rate_limited_and_cannot_cross_original_joint_stop():
    q=integrate_correction(np.zeros(6),np.eye(6),np.ones(6),.01,
        np.full(6,-1.),np.array([.0002,1,1,1,1,1]))
    np.testing.assert_allclose(q,[.0002,.0006,.0006,.0006,.0006,.0006])
    q=integrate_correction(np.full(6,.06),np.eye(6),np.ones(6),.01,
        np.full(6,-1.),np.ones(6))
    np.testing.assert_allclose(q,.06)


def test_singular_arm_has_no_unbounded_correction_and_bad_clock_is_rejected():
    q=integrate_correction(np.zeros(7),np.zeros((6,7)),np.ones(6),.01,
        np.full(7,-1.),np.ones(7))
    np.testing.assert_array_equal(q,0.)
    for dt in [-.01,.1,float('nan')]:
        with pytest.raises(ValueError):
            integrate_correction(np.zeros(6),np.eye(6),np.ones(6),dt,-np.ones(6),np.ones(6))


def test_explicit_gain_six_doubles_unsaturated_analytic_correction():
    error=np.array([.001,-.002,.003,-.004,.005,-.006])
    args=(np.zeros(6),np.eye(6),error,.002,-np.ones(6),np.ones(6))
    old=integrate_correction(*args)
    expected=3.*error/(1.+1e-5)*.002
    np.testing.assert_array_equal(old,integrate_correction(*args,gain_s_inv=3.))
    np.testing.assert_allclose(old,expected,rtol=1e-15)
    np.testing.assert_array_equal(integrate_correction(*args,gain_s_inv=6.),2.*old)


def test_higher_gain_keeps_identical_rate_offset_and_joint_caps():
    for gain in (3.,6.):
        q=integrate_correction(np.zeros(6),np.eye(6),np.ones(6),.002,-np.ones(6),np.array([.00003,1,1,1,1,1]),gain_s_inv=gain)
        np.testing.assert_allclose(q,[.00003,.00012,.00012,.00012,.00012,.00012])
        q=integrate_correction(np.full(6,.05999),np.eye(6),np.ones(6),.002,-np.ones(6),np.ones(6),gain_s_inv=gain)
        np.testing.assert_array_equal(q,np.full(6,.06))


@pytest.mark.parametrize('gain',[0.,-1.,6.0001,float('nan'),float('inf'),True,'6',None])
def test_nonfinite_unbounded_or_implicit_gain_is_rejected(gain):
    with pytest.raises(ValueError,match='correction gain'):
        integrate_correction(np.zeros(6),np.eye(6),np.zeros(6),.002,-np.ones(6),np.ones(6),gain_s_inv=gain)


def test_gain_never_changes_exact_initial_zero_interval():
    previous=np.array([.01,-.02,.03,-.04,.05,-.01])
    for gain in (3.,6.):
        result=integrate_correction(previous,np.eye(6),np.ones(6),0.,-np.ones(6),np.ones(6),gain_s_inv=gain)
        np.testing.assert_array_equal(result,previous)


@pytest.mark.parametrize('gain',[float('nan'),7.,True,'6'])
def test_withdrawal_rejects_invalid_explicit_gain_before_loading_route(tmp_path,gain):
    import json
    from types import SimpleNamespace
    from doorbench.dexterous.standing_withdrawal import StandingWithdrawalTeacher
    path=tmp_path/'withdrawal.json'
    path.write_text(json.dumps({'schema':'doorbench.standing-withdrawal.v1','palm_correction_gain_s_inv':gain}))
    returned=SimpleNamespace(acquisition=object(),operation=object(),transfer=SimpleNamespace(left=object()))
    with pytest.raises(ValueError,match='correction gain'):
        StandingWithdrawalTeacher(returned,{},path)
