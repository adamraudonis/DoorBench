import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.dexterous.audit_pose_gyro_run import reconstruct_rates


def test_rotating_mount_uses_local_frame_and_reset_interval():
    mount=Rotation.from_euler('x',90,degrees=True)
    q=Rotation.from_rotvec([[0,0,.3],[0,0,.302],[0,0,.304]]).as_quat()
    rate,_=reconstruct_rates([0,.002,.004],q,mount.as_quat()[[3,0,1,2]])
    np.testing.assert_allclose(rate,np.tile([0,1,0],(2,1)),atol=1e-6)
    wrong,_=reconstruct_rates([0,.002,.004],q,[1,0,0,0])
    assert not np.allclose(wrong,rate)


@pytest.mark.parametrize('times',[[.002,.004,.006],[0,.002,.006],[0,.002,.002]])
def test_reject_missing_reset_or_nonconsecutive_epoch(times):
    with pytest.raises(ValueError):reconstruct_rates(times,np.tile([0,0,0,1],(3,1)),[1,0,0,0])


def test_quaternion_sign_is_irrelevant_but_invalid_unit_rejected():
    q=Rotation.from_rotvec([[0,0,0],[0,.001,0],[0,.002,0]]).as_quat()
    rate,_=reconstruct_rates([0,.002,.004],q,[1,0,0,0])
    q[1]*=-1
    other,_=reconstruct_rates([0,.002,.004],q,[1,0,0,0])
    np.testing.assert_array_equal(rate,other)
    q[1]*=2
    with pytest.raises(ValueError):reconstruct_rates([0,.002,.004],q,[1,0,0,0])
