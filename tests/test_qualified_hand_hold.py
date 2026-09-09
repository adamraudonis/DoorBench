from types import SimpleNamespace
import numpy as np
import pytest
from doorbench.dexterous.qualified_hand_hold import QualifiedHandHold


def fixture():
    teacher=SimpleNamespace(fingers=np.array([0]),names=['finger','other'],matrix=np.eye(2),
        finger_inverse=np.array([[1.,0.]]),d=SimpleNamespace(qfrc_bias=np.array([.1,0.])),
        va=np.array([0,1]),kp=np.ones(2),damping=np.full(2,.1),bias=np.array([[0.,-1.,-.1],[0.,-1.,-.1]]),
        caps=np.array([[-1.,1.],[-2.,2.]]))
    return QualifiedHandHold(teacher),dict(finger=.2,other=0.),dict(finger=.2,other=0.)


def test_qualified_capture_preserves_command_and_never_overrides_other_motors():
    hold,q,v=fixture();force=np.array([.3,1.2])
    for t in np.arange(0.,.5,.01):hold.force(t,force,q,v,eligible=True)
    assert hold.started is None
    hold.force(.5,force,q,v,eligible=False)
    for t in np.arange(.51,1.02,.01):result,info=hold.force(t,force,q,v,eligible=True)
    assert hold.started==pytest.approx(1.01)
    np.testing.assert_allclose(result,force,atol=1e-12)
    changed=np.array([.8,-1.1])
    result,_=hold.force(2.02,changed,q,dict(finger=0.,other=0.),eligible=False)
    np.testing.assert_allclose(result,[.3,-1.1])
    assert hold.teacher.last_force[1]==-1.1


def test_missing_observations_cannot_qualify_the_capture():
    hold,q,v=fixture()
    hold.force(0.,np.array([.3,0.]),q,v,eligible=True)
    hold.force(.6,np.array([.3,0.]),q,v,eligible=True)
    assert hold.started is None and hold.since is None
    with pytest.raises(ValueError,match='clock'):hold.force(.5,np.array([.3,0.]),q,v,eligible=True)
