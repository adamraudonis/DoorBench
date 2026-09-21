"""Synthetic CPU feedback histories; no physical palm-contact success claim."""
from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous.isaac_withdrawal_support import InheritedIsaacPalmSupport
from doorbench.dexterous.standing_support_feedback import StandingSupportFeedback


def predecessor():
    reads=[]
    left=SimpleNamespace(support_load_target=6.,filtered_palm_load=5.,
        hybrid_normal_target=6.,hybrid_blend=1.,offset=.001,
        palm=0,d=SimpleNamespace(site_xpos=np.zeros((1,3)),site_xmat=np.eye(3).reshape(1,9)),
        _read=lambda root,joints:reads.append((root,joints)))
    feedback=StandingSupportFeedback.__new__(StandingSupportFeedback)
    feedback.left=left;feedback.maximum_target_N=6.;feedback.started=32.
    feedback.previous=(43.998,np.zeros(3),np.eye(3));feedback.surface=np.eye(3)*.01
    transfer=SimpleNamespace(left=left,hybrid_support=True,support_feedback=feedback)
    return transfer,reads


def test_reuses_real_feedback_history_and_updates_once_with_palm_only_load():
    transfer,reads=predecessor();left=transfer.left;feedback=transfer.support_feedback
    previous=feedback.previous
    support=InheritedIsaacPalmSupport(transfer,6.)
    assert support.feedback is None and feedback.previous is previous and not reads
    support.begin(44.)
    assert support.feedback is feedback and feedback.previous is previous
    assert left.filtered_palm_load==5. and left.hybrid_blend==1. and feedback.started==32.
    info=support.update(44.,'actual-root',{'actual-joint':.1},[.00002,0,0,1,0,0,0],3.)
    assert reads==[('actual-root',{'actual-joint':.1})]
    assert left.filtered_palm_load==pytest.approx(5.+.002/.022*(3.-5.))
    assert left.hybrid_normal_target==6. and left.support_load_target==6.
    np.testing.assert_allclose(left.surface_velocity_world,[.01,0,0],atol=1e-12)
    assert info['previous_feedback_epoch']==43.998 and info['feedback_epoch']==44.
    assert info['palm_only_load_N']==3. and info['support_surface']=='left_palm_only'
    assert info['reused_predecessor_state'] and feedback.started==32.
    support.update(44.002,None,{},[.00004,0,0,1,0,0,0],7.)
    assert feedback.previous[0]==44.002 and left.hybrid_normal_target==6.


@pytest.mark.parametrize('change',['absent','stale','future','target','limit','blend','filter'])
def test_bad_actual_entry_history_rejected_stickily(change):
    transfer,_=predecessor();support=InheritedIsaacPalmSupport(transfer,6.)
    if change=='absent':transfer.support_feedback=None
    if change=='stale':transfer.support_feedback.previous=(43.996,None,None)
    if change=='future':transfer.support_feedback.previous=(44.,None,None)
    if change=='target':transfer.left.hybrid_normal_target=4.
    if change=='limit':transfer.support_feedback.maximum_target_N=8.
    if change=='blend':transfer.left.hybrid_blend=float('nan')
    if change=='filter':transfer.left.filtered_palm_load=-1.
    with pytest.raises(ValueError):support.begin(44.)
    with pytest.raises(ValueError):support.begin(44.)
    assert support.failure and support.started is None


@pytest.mark.parametrize('change',['duplicate','skipped','negative','missing','nonfinite','object','target','feedback_epoch'])
def test_runtime_missing_or_stale_feedback_cannot_resume(change):
    transfer,_=predecessor();support=InheritedIsaacPalmSupport(transfer,6.);support.begin(44.)
    t=44.;load=3.
    if change=='duplicate':t=43.998
    if change=='skipped':t=44.002
    if change=='negative':load=-.1
    if change=='missing':load=None
    if change=='nonfinite':load=float('nan')
    if change=='object':transfer.support_feedback=SimpleNamespace()
    if change=='target':transfer.left.support_load_target=4.
    if change=='feedback_epoch':transfer.support_feedback.previous=(43.996,None,None)
    with pytest.raises(ValueError):support.update(t,None,{},[0,0,0,1,0,0,0],load)
    with pytest.raises(ValueError):support.update(44.,None,{},[0,0,0,1,0,0,0],3.)
    assert support.previous==43.998


@pytest.mark.parametrize('target',[2.,8.01,float('nan'),True])
def test_no_new_support_target_range(target):
    transfer,_=predecessor()
    with pytest.raises(ValueError):InheritedIsaacPalmSupport(transfer,target)
