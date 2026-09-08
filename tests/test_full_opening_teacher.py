from collections import deque
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

# Load this owned module against the integration tree during isolated development.
# The actual public imports are used in the uninterrupted native regression.
spec=importlib.util.spec_from_file_location('_full_opening_teacher_test',Path(__file__).resolve().parents[1]/'doorbench/dexterous/full_opening_teacher.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
FullOpeningTeacher=module.FullOpeningTeacher


def clock_only():
    obj=FullOpeningTeacher.__new__(FullOpeningTeacher)
    obj.history=deque();obj.last_time=None;obj.qualification_seconds=.5;obj.physics_dt=.002
    obj.initial_contract=None;obj.all_physics_qualified=True;obj.all_pad_patches_qualified=True
    return obj


def evidence(**changes):
    return dict(grasp_qualified=True,physics_qualified=True,right_pad_patches_valid=True,
                hand_contact_count=0,left_panel_load_N=3.,left_palm_load_N=2.5,
                right_lever_clearance_m=.02,**changes)


def test_hold_requires_every_actual_physics_sample_and_breaks_on_gap():
    obj=clock_only();angles=dict(operator=0.,leaf=0.,latch=0.)
    for i in range(251):obj._record_evidence(i*.002,angles,evidence())
    assert obj._qualified(lambda row:row['grasp_qualified'])
    obj._record_evidence(.504,angles,evidence())
    assert not obj._qualified(lambda row:row['grasp_qualified'])


def test_missing_contact_or_physics_is_not_filled_from_reference():
    obj=clock_only();angles=dict(operator=0.,leaf=0.,latch=0.)
    for i in range(251):
        row=evidence()
        if i==100:row['grasp_qualified']=False;row['physics_qualified']=False
        obj._record_evidence(i*.002,angles,row)
    assert not obj._qualified(lambda row:row['grasp_qualified'])
    assert not obj.all_physics_qualified


def test_evidence_rejects_opaque_handles_and_unknown_fields():
    obj=clock_only();row=evidence();row['active_simulator']=object()
    with pytest.raises(ValueError):obj._record_evidence(0.,dict(operator=0.,leaf=0.),row)
    assert not obj.history


@pytest.mark.parametrize('field',['left_panel_load_N','left_palm_load_N','right_lever_clearance_m'])
def test_nonfinite_evidence_rejected(field):
    row=evidence();row[field]=np.nan
    with pytest.raises(ValueError):clock_only()._record_evidence(0.,dict(operator=0.,leaf=0.),row)


def test_out_of_order_evidence_rejected():
    obj=clock_only();angles=dict(operator=0.,leaf=0.)
    obj._record_evidence(.2,angles,evidence())
    with pytest.raises(ValueError):obj._record_evidence(.1,angles,evidence())


def test_stale_measured_pose_cannot_be_presented_as_current_joint_state():
    obj=clock_only();pose=[0,0,0,1,0,0,0]
    with pytest.raises(ValueError,match='share the current step'):
        obj.force(1.,None,{}, {},pose,pose,dict(operator=0.,leaf=0.,latch=0.),{},
                  evidence=evidence(),right_palm_pose=pose,pose_time_s=.998)


def test_repeating_measurement_cannot_manufacture_a_contact_hold():
    obj=clock_only();angles=dict(operator=0.,leaf=0.)
    obj._record_evidence(0.,angles,evidence())
    with pytest.raises(ValueError,match='exactly once'):
        obj._record_evidence(0.,angles,evidence())


def test_dense_sample_count_cannot_replace_the_actual_hold_duration():
    obj=clock_only();angles=dict(operator=0.,leaf=0.)
    for i in range(251):obj._record_evidence(i*.001,angles,evidence())
    assert not obj._qualified(lambda row:row['grasp_qualified'])


def operation_only(*, early=False,gain=0.):
    obj=clock_only()
    obj.acquisition=SimpleNamespace(positions=np.zeros((1,3)),rotations=np.array([np.eye(3)]))
    obj.initial_handle=0.;obj.operation_started=0.;obj.operation_last_time=0.
    obj.press_seconds=5.;obj.opening_seconds=3.;obj.open_started=None
    obj.open_on_latch_clear=early;obj.operator_compliance_gain=gain
    obj.operator_compliance_limit=.15;obj.operator_compliance=0.
    obj.freeze_compliance_on_release=True
    obj.geometry=dict(operator_origin=np.zeros(3),operator_axis=np.array([1.,0.,0.]),
                      leaf_origin=np.zeros(3),leaf_axis=np.array([0.,0.,1.]))
    obj.p_relative=np.array([0.,.1,0.]);obj.r_relative=np.eye(3)
    obj.operation_info=dict(goal_leaf_rad=0.);obj.handoffs={}
    return obj


def test_early_opening_is_opt_in_and_retains_both_actual_release_thresholds():
    pose=[0,0,0,1,0,0,0]
    ordinary=operation_only();early=operation_only(early=True)
    for obj in (ordinary,early):
        obj._operation_targets(1.,pose,pose,dict(operator=.79,leaf=0.,latch=.012))
        assert obj.open_started is None
        obj._operation_targets(1.002,pose,pose,dict(operator=.81,leaf=0.,latch=.0109))
        assert obj.open_started is None
        obj._operation_targets(1.004,pose,pose,dict(operator=.81,leaf=0.,latch=.012))
    assert ordinary.open_started is None
    assert early.open_started==1.004
    assert early.operator_compliance==0.


def test_compensation_is_bounded_and_freezes_on_actual_latch_release():
    obj=operation_only(early=True,gain=.5);pose=[0,0,0,1,0,0,0]
    for t in np.arange(.002,4.,.002):
        obj._operation_targets(float(t),pose,pose,dict(operator=0.,leaf=0.,latch=0.))
    assert obj.operator_compliance==.15
    obj._operation_targets(4.,pose,pose,dict(operator=.81,leaf=0.,latch=.012))
    frozen=obj.operator_compliance
    assert obj.open_started==4.
    for t in np.arange(4.002,6.,.002):
        obj._operation_targets(float(t),pose,pose,dict(operator=0.,leaf=.08,latch=.012))
        assert obj.operator_compliance==frozen
    assert obj.operation_info['goal_handle_rad']==pytest.approx(.87)
