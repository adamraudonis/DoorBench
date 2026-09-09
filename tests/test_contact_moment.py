import numpy as np
import pytest
from doorbench.dexterous.contact_moment import contact_moment


def test_contact_order_and_tangential_resistance_are_preserved():
    # Contact normal +Y, tangent -X, hinge +Z. Both lever-arm terms matter.
    frame=np.array([[0,1,0],[-1,0,0],[0,0,1.]])
    a=contact_moment([.2,-.02,0],frame,[5,2,0,0,0,.03],[0,0,0],[0,0,1],body_index=1)
    b=contact_moment([.2,-.02,0],frame,[5,2,0,0,0,.03],[0,0,0],[0,0,1],body_index=0)
    assert a['normal_force_moment_Nm']==pytest.approx(1)
    assert a['tangential_force_moment_Nm']==pytest.approx(-.04)
    assert a['moment_about_hinge_Nm']==pytest.approx(.99)
    assert b['moment_about_hinge_Nm']==pytest.approx(-.99)


def test_bad_reference_frames_are_rejected():
    with pytest.raises(ValueError):contact_moment([0,0,0],np.eye(3),np.zeros(6),[0,0,0],[0,0,2],body_index=1)
    with pytest.raises(ValueError):contact_moment([0,0,0],2*np.eye(3),np.zeros(6),[0,0,0],[0,0,1],body_index=1)
