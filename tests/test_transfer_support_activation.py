"""Measured palm contact must admit and then retain hybrid support control."""
from types import SimpleNamespace
import pytest
from doorbench.dexterous.standing_transfer import StandingTransferTeacher
import doorbench.dexterous.standing_support_feedback as feedback


def test_contact_admission_and_retention(monkeypatch):
    calls=[]
    class Feedback:
        def __init__(self,left):calls.append('init')
        def update(self,*args):calls.append(args)
    monkeypatch.setattr(feedback,'StandingSupportFeedback',Feedback)
    t=StandingTransferTeacher.__new__(StandingTransferTeacher)
    t.hybrid_support=True;t.support_feedback=None
    t.left=SimpleNamespace(progress=.97,support_load_target=4.,_read=lambda *args:calls.append('read'))
    t.update_support(1,None,None,'pose',8.)
    t.left.progress=1.
    t.update_support(2,None,None,'pose',0.)
    assert calls==[]
    t.update_support(3,None,None,'pose',2.)
    t.update_support(3.002,None,None,'pose',0.)
    assert calls==['read','init',(3,'pose',2.,4.),'read',(3.002,'pose',0.,4.)]
    for load in (None,float('nan'),float('inf'),-1.):
        with pytest.raises(ValueError,match='palm-only'):t.update_support(4,None,None,'pose',load)


def test_default_does_not_change_existing_controller():
    t=StandingTransferTeacher.__new__(StandingTransferTeacher);t.hybrid_support=False
    t.update_support(0,None,None,None,None)
