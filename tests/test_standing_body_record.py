import numpy as np
import pytest
from doorbench.dexterous.standing_body_record import pack_standing_body_poses, PLANNER_BODIES


def test_copies_ordered_measured_poses_without_normalizing():
    poses=np.zeros((5,7),dtype=np.float32);poses[:,0]=np.arange(5);poses[:,3]=1
    handle=np.array([9,0,0,1,0,0,0],dtype=np.float32)
    out=pack_standing_body_poses(poses,handle)
    np.testing.assert_array_equal(out[:,0],[0,1,2,3,4,9])
    assert PLANNER_BODIES==('robot/left_ankle_link','robot/right_ankle_link','robot/torso_link','robot/rh_palm','robot/lh_palm','leaf_handle')
    poses[:]=99;handle[:]=99
    assert out[0,3]==1 and out[-1,0]==9
    assert out.nbytes==168


@pytest.mark.parametrize('fault',['shape','nan','quaternion'])
def test_rejects_missing_or_invalid_measurements(fault):
    poses=np.zeros((5,7));poses[:,3]=1;handle=poses[0].copy()
    if fault=='shape':poses=poses[:4]
    if fault=='nan':poses[0,0]=float('nan')
    if fault=='quaternion':handle[3]=.5
    with pytest.raises(ValueError):pack_standing_body_poses(poses,handle)
