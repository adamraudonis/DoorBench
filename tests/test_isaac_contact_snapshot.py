import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
s=importlib.util.spec_from_file_location('snapshot',Path(__file__).resolve().parents[1]/'scripts/dexterous/render_isaac_contact_snapshot.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_measured_xyzw_transform_preserves_known_world_contact():
    pos,rot=m.measured_body_pose([1,2,3,*Rotation.from_euler('z',90,degrees=True).as_quat()])
    np.testing.assert_allclose(pos+rot@np.array([.01,0,0]),[1,2.01,3],atol=1e-12)


@pytest.mark.parametrize('pose',[[0]*7,[0,0,0,0,0,0,2],[float('nan'),0,0,0,0,0,1],[0]*6])
def test_measured_pose_rejects_missing_or_invalid_orientation(pose):
    with pytest.raises(ValueError):m.measured_body_pose(pose)
