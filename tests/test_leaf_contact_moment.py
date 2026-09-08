import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.dexterous.audit_leaf_contact_moment import contact_moment_about_axis


def test_tangential_friction_reduces_opening_moment_with_recorded_side_sign():
    # Normal is world +Y; the second contact axis is world -X.
    frame=np.array([[0,1,0],[-1,0,0],[0,0,1.]])
    result,normal,force=contact_moment_about_axis(np.zeros(3),[0,0,1],[.2,-.03,1],frame,[4,.5,0,0,0,0],1)
    assert np.isclose(normal,.8)
    assert np.isclose(result,.785)
    np.testing.assert_allclose(force,[-.5,4,0])
    other,_,_=contact_moment_about_axis(np.zeros(3),[0,0,1],[.2,-.03,1],frame,[4,.5,0,0,0,0],0)
    assert other==-result


def test_global_frame_rotation_and_translation_do_not_change_hinge_moment():
    rot=Rotation.from_euler('xyz',[.4,-.7,.2]).as_matrix();offset=np.array([1.,2.,3.])
    anchor=np.array([.1,.2,.3]);point=np.array([.6,-.1,.9]);axis=np.array([0.,0.,1.]);frame=np.eye(3);wrench=np.array([3.,2.,1.,.2,-.1,.4])
    first=contact_moment_about_axis(anchor,axis,point,frame,wrench,1)
    second=contact_moment_about_axis(rot@anchor+offset,rot@axis,rot@point+offset,frame@rot.T,wrench,1)
    np.testing.assert_allclose(first[:2],second[:2],atol=1e-12)
    np.testing.assert_allclose(rot@first[2],second[2],atol=1e-12)


def test_nonfinite_force_rejected():
    with pytest.raises(ValueError):contact_moment_about_axis([0,0,0],[0,0,1],[0,0,0],np.eye(3),[np.nan,0,0,0,0,0],1)
