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


@pytest.mark.parametrize('value',[np.nan,np.inf,-1.,0.,8.001,True])
def test_invalid_feedforward_rejected_before_a_robot_is_loaded(value):
    from doorbench.dexterous.screened_panel_teacher import ScreenedWholeBodyPanel
    with pytest.raises(ValueError):
        ScreenedWholeBodyPanel(None,'nonexistent-plan.json',normal_feedforward_N=value)


def test_declared_feedforward_is_installed_on_actual_arm_force_adapter_at_handoff():
    from types import SimpleNamespace
    from doorbench.dexterous.screened_panel_teacher import ScreenedWholeBodyPanel
    panel=ScreenedWholeBodyPanel.__new__(ScreenedWholeBodyPanel)
    joints={'lh_LFJ5':.2,'rh_LFJ5':.1}
    root=np.r_[.1,-.5,.87,1.,0.,0.,0.,np.zeros(6)]
    panel.started=None
    panel.plan=dict(initial_robot_joints=joints.copy(),initial_leaf_angle_rad=.28)
    panel.initial_root=root[:7].copy()
    panel.left=SimpleNamespace(started=0.,progress=1.,contact_force=8.)
    panel.teacher=SimpleNamespace(names=list(joints),path=np.zeros((1,2)))
    panel.advance=lambda t,angle:None
    panel.normal_feedforward_N=4.
    panel.begin(1.,root,joints,None,.28)
    assert panel.left.contact_force==4.
    np.testing.assert_array_equal(panel.teacher.path[-1],[.2,.1])
