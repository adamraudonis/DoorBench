from types import SimpleNamespace
import numpy as np
import pytest

from doorbench.dexterous.moving_body_recontact import MovingBodyPalmRecontact
from doorbench.dexterous.palm_recontact_teacher import TimedPalmRecontact


def test_body_and_arm_can_consume_one_target_without_advancing_twice(monkeypatch):
    obj=MovingBodyPalmRecontact.__new__(MovingBodyPalmRecontact)
    obj.body_planner=SimpleNamespace(scalar={'joint':0})
    obj.consumed=None;obj.latest={};obj.left=SimpleNamespace(info={});calls=[]
    def update(self,*args):
        calls.append(args);self.latest['whole_body_adaptation']={'time_s':args[0]}
    monkeypatch.setattr(TimedPalmRecontact,'update',update)
    root=np.r_[0,0,1,1,0,0,0,np.zeros(6)];leaf=[0,0,0,1,0,0,0]
    obj.update(1.,root,{'joint':.1},leaf,3.,.32)
    obj.update(1.,root.copy(),{'joint':.1},leaf,3.,.32)
    assert len(calls)==1
    with pytest.raises(ValueError,match='changed'):
        obj.update(1.,root,{'joint':.1001},leaf,3.,.32)
    with pytest.raises(ValueError,match='Complete'):
        obj.update(1.,root,{'joint':.1,'unknown':0.},leaf,3.,.32)
    obj.update(1.002,root,{'joint':.1},leaf,3.,.32)
    assert len(calls)==2


def test_transition_requires_immediately_preceding_commands():
    obj=MovingBodyPalmRecontact.__new__(MovingBodyPalmRecontact)
    obj.preceding=[{'time_s':.1},{'time_s':.102}]
    with pytest.raises(ValueError,match='immediately preceding'):
        obj.begin(.106,None,None,None,None)
