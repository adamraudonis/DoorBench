import numpy as np
import pytest
from doorbench.dexterous.reset import check_joint_reset


def test_ankle_cannot_be_initialized_beyond_its_hard_stop():
    names=['shoulder','knee','ankle'];limits=[[-2.87,2.87],[-.26,2.05],[-.87,.52]]
    check_joint_reset(names,[.2,1.5,-.75],limits)
    with pytest.raises(ValueError,match='ankle'):
        check_joint_reset(names,[.2,1.8,-.894],limits)


def test_nan_and_mismatched_import_cannot_pass_reset_validation():
    with pytest.raises(ValueError,match='nonfinite'):check_joint_reset(['wrist'],[np.nan],[[-1,1]])
    with pytest.raises(ValueError,match='dimensions'):check_joint_reset(['wrist'],[0,0],[[-1,1]])
