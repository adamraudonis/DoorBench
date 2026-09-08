import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from doorbench.dexterous.right_hand_release import axial_release_goal


def test_axial_release_rotates_with_measured_handle_and_preserves_fingers_separately():
    local=np.array([-.065,-.104,.083]);rotation=Rotation.from_euler('xyz',[.2,.3,-.1]).as_matrix()
    h=np.array([.4,-.1,1.,1,0,0,0.])
    p,r,u,b=axial_release_goal(h,local,rotation,6.)
    np.testing.assert_allclose(p,h[:3]+local+[-.14,0,0])
    np.testing.assert_allclose(r,rotation);assert u==b==1
    world=Rotation.from_euler('xyz',[.1,.7,-1.2]);q=world.as_quat();shift=np.array([-.2,.6,.1])
    moved=np.r_[world.apply(h[:3])+shift,q[3],q[:3]]
    p2,r2,_,_=axial_release_goal(moved,local,rotation,6.)
    np.testing.assert_allclose(p2,world.apply(p)+shift,atol=1e-12)
    np.testing.assert_allclose(r2,world.as_matrix()@r,atol=1e-12)


def test_release_quintic_has_smooth_start_and_stop():
    h=[0,0,0,1,0,0,0];p=[0,0,0];r=np.eye(3)
    x0=axial_release_goal(h,p,r,0)[0]
    x1=axial_release_goal(h,p,r,1e-4)[0]
    x2=axial_release_goal(h,p,r,6.-1e-4)[0]
    x3=axial_release_goal(h,p,r,6.)[0]
    assert np.linalg.norm(x1-x0)<1e-10
    assert np.linalg.norm(x3-x2)<1e-10
    np.testing.assert_allclose(axial_release_goal(h,p,r,20)[0],x3)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1])
def test_invalid_clock_rejected(bad):
    with pytest.raises(ValueError):axial_release_goal([0,0,0,1,0,0,0],[0,0,0],np.eye(3),bad)


def test_bad_transform_rejected():
    with pytest.raises(ValueError):axial_release_goal([0,0,0,0,0,0,0],[0,0,0],np.eye(3),0)
    with pytest.raises(ValueError):axial_release_goal([0,0,0,1,0,0,0],[0,0,0],np.diag([1,1,-1]),0)
