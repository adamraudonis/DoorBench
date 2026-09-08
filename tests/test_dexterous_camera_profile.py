import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from doorbench.dexterous.camera_profile import apply_camera_profile


def fixture():
    return dict(cameras=[dict(name='eye',body_name='head',position_body_m=[.1,0,.65],
        quaternion_wxyz_body=[-.5,-.5,.5,.5],convention='opengl',fovy_degrees=45)],
        sensors=[dict(name='pad',dimension=24)]),dict(schema_version='doorbench.robot-camera-profile.v1',
        name='fixed-down',camera_overrides={'eye':dict(pitch_down_degrees=45,fovy_degrees=100)})


def test_camera_looks_forward_and_down_without_moving_mount_or_touch():
    layout,profile=fixture();before=copy.deepcopy(layout)
    result=apply_camera_profile(layout,profile);camera=result['cameras'][0]
    direction=Rotation.from_quat(np.roll(camera['quaternion_wxyz_body'],-1)).apply([0,0,-1])
    np.testing.assert_allclose(direction,[2**-.5,0,-2**-.5],atol=1e-12)
    assert camera['position_body_m']==before['cameras'][0]['position_body_m']
    assert camera['body_name']=='head' and result['sensors']==before['sensors']
    assert result['native_cameras']==before['cameras'] and layout==before


def test_camera_profile_cannot_supply_world_target_or_unknown_camera():
    layout,profile=fixture();profile['camera_overrides']['eye']['target_world']=[0,0,0]
    with pytest.raises(ValueError):apply_camera_profile(layout,profile)
    _,profile=fixture();profile['camera_overrides']['oracle']=profile['camera_overrides'].pop('eye')
    with pytest.raises(ValueError):apply_camera_profile(layout,profile)


def test_camera_profile_rejects_invalid_projection_or_rotation():
    layout,profile=fixture();profile['camera_overrides']['eye']['fovy_degrees']=float('nan')
    with pytest.raises(ValueError):apply_camera_profile(layout,profile)
    layout,profile=fixture();layout['cameras'][0]['quaternion_wxyz_body']=[0,0,0,0]
    with pytest.raises(ValueError):apply_camera_profile(layout,profile)
