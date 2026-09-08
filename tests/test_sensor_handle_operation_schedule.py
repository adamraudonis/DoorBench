"""A static press cannot overwrite or reset the qualified acquisition prefix."""
import inspect
import numpy as np
import pytest
from doorbench.dexterous.sensor_acquisition_schedule import ScriptedAcquisitionSchedule
from doorbench.dexterous.sensor_handle_operation_schedule import ScriptedHandleOperationSchedule


def make():
    acquisition=ScriptedAcquisitionSchedule(['torso','rh_FFJ1','rh_FFJ2'],['torso','rh_FFJ1','rh_FFJ2'],[[-.9,.1,.2],[-.2,.2,.4]])
    end=acquisition.goals(19.)['torso']
    return acquisition,ScriptedHandleOperationSchedule(acquisition,['torso'],[[end],[end+.1]])


def test_all_9501_acquisition_commands_and_uncommanded_fingers_preserved():
    acq,full=make()
    for t in np.arange(0,19.002,.002):assert full.goals(float(t))==acq.goals(float(t))
    for t in np.arange(19.,30.002,.002):
        g=full.goals(float(t));assert g['rh_FFJ1']==acq.goals(19.)['rh_FFJ1'];assert g['rh_FFJ2']==acq.goals(19.)['rh_FFJ2']
    assert full.goals(30.)['torso']==pytest.approx(-.1)
    assert full.duration_s==30.
    assert list(inspect.signature(full.goals).parameters)==['t']


@pytest.mark.parametrize('kind',['jump','missing','duplicate','nonfinite','duration'])
def test_invalid_motor_plan_is_rejected_before_runtime(kind):
    acq,_=make();names=['torso'];path=[[-.2],[-.1]];kw={}
    if kind=='jump':path[0][0]=0.
    if kind=='missing':names=['root_x']
    if kind=='duplicate':names=['torso','torso'];path=[[-.2,-.2],[-.1,-.1]]
    if kind=='nonfinite':path[1][0]=np.nan
    if kind=='duration':kw['press_seconds']=40.
    with pytest.raises(ValueError):ScriptedHandleOperationSchedule(acq,names,path,**kw)
