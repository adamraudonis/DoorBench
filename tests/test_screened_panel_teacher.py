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


def test_increased_lead_requires_the_matching_shifted_geometry_evidence():
    from doorbench.dexterous.screened_panel_teacher import validate_tracking_lead_receipt
    plan={'screen_receipt':{'screen_sha256':'exact-original-path'}}
    receipt=dict(passed=True,samples=2001,screen_sha256='exact-original-path',actual_leaf_lag_rad=.01,lag_start_angle_rad=.4)
    validate_tracking_lead_receipt(plan,receipt,.01,.4)
    for changes in ({'passed':False},{'samples':2000},{'screen_sha256':'other-path'},
                    {'actual_leaf_lag_rad':.009},{'actual_leaf_lag_rad':np.nan},{'lag_start_angle_rad':.41}):
        with pytest.raises(ValueError):validate_tracking_lead_receipt(plan,dict(receipt,**changes),.01,.4)
    with pytest.raises(ValueError):validate_tracking_lead_receipt(plan,receipt,.01,None)
    with pytest.raises(ValueError):validate_tracking_lead_receipt(plan,None,.01,.4)


def test_phase_consumes_the_audited_reduced_motion_rates():
    from doorbench.dexterous.screened_panel_teacher import ScreenedWholeBodyPanel
    plan=dict(initial_leaf_angle_rad=.4,final_leaf_angle_rad=.75,initial_leaf_velocity_rad_s=.04,
              screen_receipt=dict(reference_phase_envelope=dict(aperture_speed_limit_rad_s=.065,aperture_acceleration_limit_rad_s2=.01)))
    phase=ScreenedWholeBodyPanel.make_phase(None,plan,.005,None,.1)
    assert phase.speed==.065 and phase.acceleration==.01
    plan['screen_receipt']['reference_phase_envelope']['aperture_speed_limit_rad_s']=.2
    import pytest
    with pytest.raises(ValueError,match='audited'):ScreenedWholeBodyPanel.make_phase(None,plan,.005,None,.1)
