"""The detached orientation calculation must expose common-foot-roll ambiguity."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.dexterous.diagnose_support_orientation import weighted_support_orientation


def test_fixed_feet_recover_root_in_the_correct_frame():
    root=Rotation.from_rotvec([.02,-.01,.3]).as_matrix()
    calibrated=Rotation.from_rotvec([[.001,.002,.1],[-.001,-.002,-.1]]).as_matrix()
    relative=root.T@calibrated
    estimate,weights=weighted_support_orientation(calibrated,relative,[100,300])
    np.testing.assert_allclose(estimate,root,atol=1e-15)
    np.testing.assert_array_equal(weights,[.25,.75])


def test_common_foot_roll_is_indistinguishable_from_root_error_despite_agreement():
    calibrated=np.repeat(np.eye(3)[None],2,axis=0)
    root=Rotation.from_rotvec([.003,0,0]).as_matrix()
    actual_feet=Rotation.from_rotvec([.009,0,0]).as_matrix()
    relative=np.repeat((root.T@actual_feet)[None],2,axis=0)
    estimate,_=weighted_support_orientation(calibrated,relative,[250,250])
    assert Rotation.from_matrix(estimate@root.T).magnitude()==pytest.approx(.009)
    np.testing.assert_array_equal(relative[0],relative[1])


def test_touch_weights_clip_at_original_limit_and_cold_fallback_is_explicit():
    rotations=np.repeat(np.eye(3)[None],2,axis=0)
    _,w=weighted_support_orientation(rotations,rotations,[0,0]);np.testing.assert_array_equal(w,[.5,.5])
    _,w=weighted_support_orientation(rotations,rotations,[1000,250]);np.testing.assert_allclose(w,[2/3,1/3])
    with pytest.raises(ValueError):weighted_support_orientation(rotations,rotations,[np.nan,1])
