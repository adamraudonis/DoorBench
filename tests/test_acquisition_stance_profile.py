import pytest
from doorbench.dexterous.acquisition_teacher import AcquisitionTeacher


def test_unknown_profile_is_rejected_before_model_loading():
    with pytest.raises(ValueError,match='stance profile'):
        AcquisitionTeacher(None,None,None,stance_profile='unbounded')


def test_unknown_pressure_segment_is_rejected_before_model_loading():
    with pytest.raises(ValueError,match='pressure segment'):
        AcquisitionTeacher(None,None,None,pressure_segment='dorsal')


def test_fixed_landed_profile_rejects_ignored_solver_overrides():
    with pytest.raises(ValueError,match='already fixes'):
        AcquisitionTeacher(None,None,None,stance_profile='landed-foot-v1',stance_solver_settings={'max_iter':100})
