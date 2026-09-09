import numpy as np
from scipy.spatial.transform import Rotation
from doorbench.dexterous.sensor_locomotion import pelvis_rotation_increment


def test_noncommuting_torso_and_base_rotations_are_separated():
    root0=Rotation.from_euler('xyz',[.1,-.2,1.7]).as_matrix()
    delta=Rotation.from_rotvec([.003,-.002,.001]).as_matrix()
    mount=Rotation.from_euler('xyz',[.2,.1,-.3]).as_matrix()
    rel0=Rotation.from_euler('z',.4).as_matrix()@mount
    rel1=Rotation.from_euler('z',.413).as_matrix()@mount
    observed=Rotation.from_matrix((root0@rel0).T@(root0@delta@rel1)).as_rotvec()/.002
    actual=pelvis_rotation_increment(rel0,rel1,observed,.002)
    np.testing.assert_allclose(actual,delta,atol=1e-14)
    assert np.linalg.norm(Rotation.from_rotvec(observed*.002).as_matrix()-delta)>.01
