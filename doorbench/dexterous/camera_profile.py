"""Fixed, declared robot camera calibration changes; no scene or task inputs."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation


def apply_camera_profile(layout, profile):
    if profile.get('schema_version') != 'doorbench.robot-camera-profile.v1':
        raise ValueError('Unknown robot camera profile version')
    result=copy.deepcopy(layout)
    cameras={camera['name']:camera for camera in result['cameras']}
    overrides=profile['camera_overrides']
    if not overrides or set(overrides)-set(cameras):
        raise ValueError('Camera profile must refer to existing fixed robot cameras')
    for name,settings in overrides.items():
        if set(settings) != {'pitch_down_degrees','fovy_degrees'}:
            raise ValueError('Only fixed pitch and field of view are supported')
        pitch,fov=float(settings['pitch_down_degrees']),float(settings['fovy_degrees'])
        if not np.isfinite([pitch,fov]).all() or not -90<=pitch<=90 or not 1<fov<180:
            raise ValueError('Invalid fixed camera calibration')
        camera=cameras[name]
        if camera['convention']!='opengl':raise ValueError('Expected OpenGL camera convention')
        quaternion=np.asarray(camera['quaternion_wxyz_body'])
        if quaternion.shape!=(4,) or not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion),1):
            raise ValueError('Camera quaternion must be finite and normalized')
        rotation=Rotation.from_quat(np.roll(quaternion,-1))*Rotation.from_euler('x',-pitch,degrees=True)
        camera['quaternion_wxyz_body']=np.roll(rotation.as_quat(),1).tolist()
        camera['fovy_degrees']=fov
    result['camera_profile']=copy.deepcopy(profile)
    result['native_cameras']=copy.deepcopy(layout['cameras'])
    return result
