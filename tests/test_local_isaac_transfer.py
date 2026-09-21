"""Local measured-state planning preserves original qualification boundaries."""
import copy

import numpy as np
import pytest

from scripts.dexterous.plan_local_isaac_transfer import local_summary, numerical_left_preferences
from doorbench.dexterous.qualified_isaac_grasp import validate_reports, EXTRACTED_FILES, AUDITED_FILES
from doorbench.dexterous.landed_left_planner import JOINT_NAMES


def records():
    report=dict(passed=True,checks={'real_physics':True},duration_s=1.,physics_dt_s=.002)
    audit=dict(accounting_passed=True,task_passed=True,independent_raw_contact_audit_complete=True,
        checks={'raw_contacts':True},invalid_loaded_patches=0,physical_intervals=500,raw_intervals=500,
        final_half_second_failed_samples=[],input_sha256={name:'a'*64 for name in AUDITED_FILES})
    extracted=dict(binding={'time_s':1.},input_sha256={name:'b'*64 for name in EXTRACTED_FILES})
    return report,audit,extracted


def test_local_summary_is_explicit_derivation_and_passes_same_original_gates():
    report,audit,extracted=records()
    before=copy.deepcopy((report,audit,extracted))
    summary=local_summary(report,audit)
    validate_reports(report,audit,summary,extracted)
    assert 'Derived local direct-run' in summary['provenance']
    assert (report,audit,extracted)==before


@pytest.mark.parametrize('change',['physical','physical_check','accounting','raw','patch','hold','intervals','epoch','binding'])
def test_failed_incomplete_or_wrong_epoch_source_cannot_be_promoted(change):
    report,audit,extracted=records()
    if change=='physical':report['passed']=False
    if change=='physical_check':report['checks']['real_physics']=False
    if change=='accounting':audit['accounting_passed']=False
    if change=='raw':audit['independent_raw_contact_audit_complete']=False
    if change=='patch':audit['invalid_loaded_patches']=1
    if change=='hold':audit['final_half_second_failed_samples']=[499]
    if change=='intervals':audit['raw_intervals']=499
    if change=='epoch':extracted['binding']['time_s']=.998
    if change=='binding':extracted['input_sha256'].pop('acquisition-physics.npz')
    with pytest.raises(ValueError):validate_reports(report,audit,local_summary(report,audit),extracted)


def test_native_route_contributes_only_lh_coordinates_not_root_or_qualification():
    targets=[dict(phase='left_reach',position=[u,-.05,1.],normal=[0,-1,0],nominal=[u]*8) for u in (0.,1.)]
    route=dict(schema='doorbench.standing-transfer.v1',left_joint_names=JOINT_NAMES,left_targets=targets,
        geometric_screen_passed=False,root_path=[[999]*7],joint_path=[[999]*69],
        attained_source_qualification={'passed':False},attained_time_s=-5)
    before=copy.deepcopy(route)
    numeric=numerical_left_preferences(route)
    assert set(numeric)=={'position','normal','nominal'}
    np.testing.assert_array_equal(numeric['position'][-1],targets[-1]['position'])
    assert route==before


def test_unmatched_lh_joint_order_rejected_even_if_native_route_claims_pass():
    route=dict(schema='doorbench.standing-transfer.v1',left_joint_names=['wrong']*8,left_targets=[],geometric_screen_passed=True)
    with pytest.raises(ValueError):numerical_left_preferences(route)
