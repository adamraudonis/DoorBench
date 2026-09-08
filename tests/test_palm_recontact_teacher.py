import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from doorbench.dexterous.palm_recontact_teacher import TimedPalmRecontact, reproject_palm
from doorbench.dexterous.screened_panel_path import ScreenedPanelPath


def test_recontact_progresses_with_an_entirely_stationary_leaf():
    teacher = TimedPalmRecontact.__new__(TimedPalmRecontact)
    teacher.started = 10.
    teacher._advance_time = None
    x = np.linspace(0.,1.,9)
    q = np.zeros((len(x),31))
    q[:,0] = .002*x
    q[:,6] = .1*x
    teacher.path = ScreenedPanelPath(x,q,6.)
    for t in 10.+np.arange(3001)*.002:
        teacher.advance(t,.32)
    assert teacher.latest['aperture'] == .32
    assert teacher.latest['progress'] == 1.
    np.testing.assert_allclose(teacher.latest['coordinate'][[0,6]],[.002,.1])
    np.testing.assert_allclose(teacher.latest['velocity'],0.,atol=1e-15)


def test_leaf_reprojection_preserves_palm_local_pose():
    initial = np.array([.2,-.1,0.,1.,0.,0.,0.])
    rotation = Rotation.from_euler('z',.2).as_matrix()
    quat = Rotation.from_matrix(rotation).as_quat()
    current = np.r_[.25,-.12,.01,quat[3],quat[:3]]
    palm = np.array([.3,-.15,.9])
    p,r = reproject_palm(palm,np.eye(3),initial,current)
    np.testing.assert_allclose(rotation.T@(p-current[:3]),palm-initial[:3],atol=1e-15)
    np.testing.assert_allclose(rotation.T@r,np.eye(3),atol=1e-15)


@pytest.mark.parametrize('leaf',[np.zeros(7),np.ones(6),np.array([np.nan,0,0,1,0,0,0])])
def test_corrupt_leaf_measurements_are_rejected(leaf):
    with pytest.raises(ValueError):
        reproject_palm(np.zeros(3),np.eye(3),[0,0,0,1,0,0,0],leaf)


def test_angular_progress_cannot_be_substituted_for_recontact():
    teacher = TimedPalmRecontact.__new__(TimedPalmRecontact)
    plan = dict(initial_leaf_angle_rad=.32,final_leaf_angle_rad=.32,normal_recontact_m=.0045)
    assert teacher.make_phase(plan,0.,None,.1).lead_at(.32) == 0.
    with pytest.raises(ValueError):
        teacher.make_phase(plan,.005,None,.1)
    with pytest.raises(ValueError):
        teacher.make_phase(dict(plan,final_leaf_angle_rad=1.2),0.,None,.1)
