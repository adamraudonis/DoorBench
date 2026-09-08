from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from doorbench.dexterous.left_palm_orientation import palm_orientation_error,bind_attained_palm_orientation


def test_full_pose_detects_the_twist_that_normal_only_cannot_see():
    rotation=Rotation.from_rotvec([0,0,np.deg2rad(10)]).as_matrix()
    np.testing.assert_array_equal(palm_orientation_error(rotation,[0,0,1]),[0,0,0])
    np.testing.assert_allclose(palm_orientation_error(rotation,[0,0,1],np.eye(3)),[0,0,-np.deg2rad(10)],atol=1e-15)


def test_legacy_normal_residual_is_exactly_preserved():
    rotation=Rotation.from_rotvec([.02,.1,.2]).as_matrix();normal=np.array([0,0,1])
    np.testing.assert_array_equal(palm_orientation_error(rotation,normal),rotation[:,2]-normal)


def test_attained_binding_uses_current_fk_and_rotates_with_leaf():
    rotation=Rotation.from_rotvec([.1,.2,.3]).as_matrix();leaf=Rotation.from_rotvec([0,0,.2])
    reads=[];left=SimpleNamespace(_read=lambda root,joints:reads.append((root,joints)),d=SimpleNamespace(site_xmat=rotation.reshape(1,9)),palm=0)
    q=leaf.as_quat();pose=np.r_[0,0,0,q[3],q[:3]]
    bind_attained_palm_orientation(left,60.,[1],{'joint':2},pose)
    assert reads==[([1],{'joint':2})]
    np.testing.assert_allclose(leaf.as_matrix()@left.attained_palm_rotation_leaf,rotation,atol=1e-15)
    with pytest.raises(ValueError,match='rebind'):bind_attained_palm_orientation(left,61.,[1],{'joint':2},pose)


def test_improper_or_nonfinite_rotation_is_rejected():
    with pytest.raises(ValueError):palm_orientation_error(np.diag([1,1,-1]),[0,0,1],np.eye(3))
    with pytest.raises(ValueError):palm_orientation_error(np.eye(3),[0,0,1],np.full((3,3),np.nan))
