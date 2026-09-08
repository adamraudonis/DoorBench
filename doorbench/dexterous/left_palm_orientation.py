"""Optional full left-palm orientation target for an attained physical hold.

Only the analytic arm solver changes. Original contact forces, actuator caps,
physical joints, and plant state remain with the caller's unchanged controller.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .right_hand_release import _pose_matrix


def palm_orientation_error(actual_rotation, desired_normal, desired_rotation=None):
    actual = np.asarray(actual_rotation, float)
    normal = np.asarray(desired_normal, float)
    if actual.shape != (3,3) or normal.shape != (3,) or not np.isfinite(np.r_[actual.ravel(),normal]).all():
        raise ValueError('Require finite measured palm rotation and desired normal')
    if not np.allclose(actual.T@actual,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(actual),1.,atol=1e-6):
        raise ValueError('Require a proper measured palm rotation')
    if desired_rotation is None:
        # Preserve the previous five-dimensional position+normal residual.
        return actual[:,2]-normal
    desired = np.asarray(desired_rotation,float)
    if desired.shape!=(3,3) or not np.isfinite(desired).all() or not np.allclose(desired.T@desired,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(desired),1.,atol=1e-6):
        raise ValueError('Require a proper full desired palm rotation')
    return Rotation.from_matrix(desired@actual.T).as_rotvec()


def bind_attained_palm_orientation(left, t, root, joints, leaf_pose):
    """Bind once using explicit current measurements in the unstepped FK model."""
    if hasattr(left,'attained_palm_rotation_leaf'):
        raise ValueError('Do not silently rebind the attained orientation')
    if not np.isfinite(t) or t<0:
        raise ValueError('Require the current finite teacher clock')
    _, rotation = _pose_matrix(leaf_pose)
    left._read(root,joints)
    actual=left.d.site_xmat[left.palm].reshape(3,3).copy()
    palm_orientation_error(actual,actual[:,2],actual)
    left.attained_palm_rotation_leaf=rotation.T@actual
    left.attained_palm_orientation_time_s=float(t)
