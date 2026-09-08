from collections import deque
import importlib.util
from pathlib import Path
import sys

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
