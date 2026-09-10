import pytest
from doorbench.dexterous.interval_clock import validate_step_epochs


def records(t):
    return ({'interval_start_s':t,'geometry_time_s':t,'interval_end_s':t+.002},
            {'contact_geometry_time_s':t,'measurement_pose_time_s':t+.002})


def test_four_hundred_seconds_of_recorded_simulator_clock():
    expected=actual=0.
    for _ in range(200000):
        expected=validate_step_epochs(*records(actual),expected)
        actual+=.002
    assert expected==actual
    assert actual-400.>1e-9  # Former index*dt audit rejected this valid clock.


@pytest.mark.parametrize('field',['interval_start_s','geometry_time_s','interval_end_s','contact_geometry_time_s','measurement_pose_time_s'])
@pytest.mark.parametrize('offset',[.002,-.002,2e-9,float('nan'),float('inf')])
def test_rejects_wrong_or_nonfinite_epoch(field,offset):
    raw,end=records(399.)
    obj=raw if field in raw else end
    obj[field]+=offset
    with pytest.raises(ValueError):validate_step_epochs(raw,end,399.)
