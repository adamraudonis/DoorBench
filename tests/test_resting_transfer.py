"""A measured spring-rest transition cannot be replaced by finger support."""
from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous.resting_transfer import RestingSupportWindow, RestingTransferBridge


def observe(window, time, *, palm=3., grasp=True, **angles):
    return window.observe(time, transfer_started=1., grasp_qualified=grasp,
        palm_load=palm, angles=dict(operator=0., latch=0., leaf=.08, **angles))


def test_half_second_requires_251_consecutive_actual_observations():
    gate=RestingSupportWindow()
    for i in range(250):
        assert not observe(gate, i*.002)
    assert observe(gate,.5)
    assert gate.verified_at==.5
    assert not observe(gate,.502,palm=0.)
    assert gate.verified_at==.5  # Historic evidence remains; current readiness does not.
    assert not observe(gate,.504)


@pytest.mark.parametrize('field,value',[('operator',.05001),('latch',.00101),('leaf',.07499)])
def test_outside_original_rest_bounds_cannot_authorize(field,value):
    gate=RestingSupportWindow()
    for i in range(251):
        angles=dict(operator=0.,latch=0.,leaf=.08);angles[field]=value
        assert not gate.observe(i*.002,transfer_started=1.,grasp_qualified=True,palm_load=3.,angles=angles)


def test_missing_interval_restarts_window_duplicates_cannot_manufacture_duration():
    gate=RestingSupportWindow()
    for i in range(250):observe(gate,i*.002)
    assert not observe(gate,.502)
    for _ in range(300):assert not observe(gate,.502)
    for i in range(1,250):assert not observe(gate,.502+i*.002)
    assert observe(gate,1.002)
    with pytest.raises(ValueError,match='backwards'):observe(gate,1.)


@pytest.mark.parametrize('palm',[None,-1.,float('nan'),float('inf')])
def test_missing_or_invalid_palm_is_rejected(palm):
    with pytest.raises(ValueError,match='Finite measured'):observe(RestingSupportWindow(),0.,palm=palm)


def test_bridge_leaves_motor_force_unchanged_and_never_claims_return():
    force=np.array([1.,-3.,2.]);seen=[]
    def delegate(*args,**kwargs):
        seen.append(kwargs)
        return force,{'phase':'transfer'}
    transfer=SimpleNamespace(acquisition=object(),operation=object(),started=1.,force=delegate)
    bridge=RestingTransferBridge(transfer)
    # 6 N total contact on fingers is insufficient with zero actual palm load.
    for i in range(251):
        output,info=bridge.force(i*.002,None,None,None,None,None,
            dict(operator=0.,latch=0.,leaf=.08),None,
            grasp_qualified=True,left_panel_load=6.,left_palm_load=0.)
    assert output is force and bridge.return_started is None
    assert info['measured_rest_ready'] is False
    assert seen[-1]==dict(grasp_qualified=True,left_panel_load=6.,left_palm_load=0.)


def test_withdrawal_rejects_finger_only_support_before_route_entry():
    from doorbench.dexterous.standing_withdrawal import StandingWithdrawalTeacher
    teacher=StandingWithdrawalTeacher.__new__(StandingWithdrawalTeacher)
    teacher.acquisition=object();teacher.palm_only_support=True;teacher.measured_rest=False
    teacher.started_withdrawal=None;teacher.start_time=5.;teacher.qualified_since=0.
    with pytest.raises(ValueError,match='Qualified resting grip'):
        teacher.force(5.,None,None,None,None,None,dict(operator=0.,latch=0.),None,
                      grasp_qualified=True,left_panel_load=6.,left_palm_load=0.)
    assert teacher.qualified_since is None
