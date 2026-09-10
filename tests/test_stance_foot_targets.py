import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from doorbench.dexterous.stance import StanceController
from doorbench.dexterous.standing_panel_reference import StandingPanelReference


def stance():
    controller=StanceController.__new__(StanceController)
    controller.target_rotation=Rotation.from_euler('z',.1).as_matrix()
    controller.fixed_foot_rotations=None
    return controller


def test_legacy_targets_follow_root_without_opt_in():
    c=stance();c.target_rotation=Rotation.from_euler('z',.3).as_matrix()
    for i in range(2):np.testing.assert_array_equal(c.foot_target_rotation(i),c.target_rotation)


def test_freeze_is_continuous_and_independent_of_future_pelvis_target():
    c=stance();before=c.target_rotation.copy();c.freeze_foot_targets()
    for i in range(2):np.testing.assert_array_equal(c.foot_target_rotation(i),before)
    c.target_rotation[:]=Rotation.from_euler('xyz',[.02,-.03,.31]).as_matrix()
    for i in range(2):np.testing.assert_array_equal(c.foot_target_rotation(i),before)
    assert not np.allclose(c.target_rotation,before)


def test_later_handoff_preserves_distinct_existing_foot_targets():
    c=stance();c.fixed_foot_rotations=Rotation.from_euler('z',[[.05],[-.05]]).as_matrix()
    before=c.fixed_foot_rotations.copy();c.freeze_foot_targets()
    c.target_rotation=Rotation.from_euler('z',.7).as_matrix()
    np.testing.assert_array_equal(c.fixed_foot_rotations,before)


@pytest.mark.parametrize('bad',[1,'true',None])
def test_panel_rejects_ambiguous_opt_in_before_loading(bad):
    with pytest.raises(ValueError,match='Explicit boolean'):
        StandingPanelReference(None,None,None,fixed_foot_targets=bad)
