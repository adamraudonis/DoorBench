import mujoco
import numpy as np
import pytest
from doorbench.dexterous.native_warning_audit import warning_interval,warning_counts


def test_real_empty_data_counters():
    m=mujoco.MjModel.from_xml_string('<mujoco/>');d=mujoco.MjData(m)
    value=warning_counts(d)
    assert warning_interval(value,value)['passed']


@pytest.mark.parametrize('index',range(int(mujoco.mjtWarning.mjNWARNING)))
def test_every_warning_category_fails_including_capacity(index):
    before=np.zeros(int(mujoco.mjtWarning.mjNWARNING),dtype=int);after=before.copy();after[index]+=1
    result=warning_interval(before,after)
    assert not result['passed'] and result['deltas'][index]==1


def test_prior_initialization_warning_is_distinct_from_interval_warning():
    before=np.ones(int(mujoco.mjtWarning.mjNWARNING),dtype=int)
    assert warning_interval(before,before)['passed']
    with pytest.raises(ValueError):warning_interval(before,before-1)
    with pytest.raises(ValueError):warning_interval(before,before.astype(float))
