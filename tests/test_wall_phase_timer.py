import pytest
from doorbench.dexterous.wall_phase_timer import WallPhaseTimer


def test_phase_partition_accumulates_and_excludes_between_step_time():
    ticks=iter([0., 2., 5., 20., 24., 25.])
    timer=WallPhaseTimer(lambda: next(ticks))
    timer.start();timer.mark('control');timer.finish('physics')
    timer.start();timer.mark('control');timer.finish('physics')
    receipt=timer.receipt()
    assert receipt['completed_steps']==2
    assert receipt['phase_seconds']=={'control':6., 'physics':4.}
    assert receipt['total_seconds']==10.
    with pytest.raises(ValueError):timer.mark('missing start')
