import numpy as np
import pytest
from doorbench.dexterous.screened_panel_teacher import validate_attained_panel_state


def state():
    joints={f'finger_{i}':float(i)/100. for i in range(69)}
    root=np.r_[.1,-.5,.87,1.,0.,0.,0.,np.zeros(6)]
    plan=dict(initial_robot_joints=joints.copy(),initial_leaf_angle_rad=.28)
    return plan,root,joints


def test_identical_complete_attained_state_is_admitted():
    plan,root,joints=state()
    validate_attained_panel_state(plan,root[:7],root,joints,.28)


@pytest.mark.parametrize('fault',['root','finger','missing','angle','nan_angle','nan_velocity','short_root'])
def test_other_state_and_nonfinite_measurements_are_rejected(fault):
    plan,root,joints=state();initial=root[:7].copy();angle=.28
    if fault=='root':root[1]+=.001
    if fault=='finger':joints['finger_68']+=.001
    if fault=='missing':del joints['finger_68']
    if fault=='angle':angle+=.001
    if fault=='nan_angle':angle=np.nan
    if fault=='nan_velocity':root[-1]=np.nan
    if fault=='short_root':root=root[:7]
    with pytest.raises(ValueError):validate_attained_panel_state(plan,initial,root,joints,angle)
