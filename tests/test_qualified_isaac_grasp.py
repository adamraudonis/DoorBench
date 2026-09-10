from copy import deepcopy
import pytest
from doorbench.dexterous.qualified_isaac_grasp import validate_reports,EXTRACTED_FILES,AUDITED_FILES


def fixture():
    return [dict(passed=True,checks={'original_check':True},duration_s=36.,physics_dt_s=.002),
        dict(accounting_passed=True,task_passed=True,independent_raw_contact_audit_complete=True,checks={'raw':True},invalid_loaded_patches=0,physical_intervals=18000,raw_intervals=18000,final_half_second_failed_samples=[],input_sha256=dict.fromkeys(AUDITED_FILES,'hash')),
        dict.fromkeys(('passed','isaac_runtime_passed','independent_audit_passed','all_loaded_handle_patches_qualified'),True),
        dict(binding={'time_s':36.},input_sha256=dict.fromkeys(EXTRACTED_FILES,'hash'))]


def test_all_evidence_required_before_admission():
    validate_reports(*fixture())
    for index,key,value in [(0,'passed',False),(1,'task_passed',False),(1,'invalid_loaded_patches',1),(1,'raw_intervals',17999),(1,'final_half_second_failed_samples',[35.9]),(2,'all_loaded_handle_patches_qualified',False),(3,'binding',{'time_s':35.})]:
        args=deepcopy(fixture());args[index][key]=value
        with pytest.raises(ValueError):validate_reports(*args)


def test_incomplete_or_unexpected_hash_inventory_rejected():
    for index in (1,3):
        args=fixture();args[index]['input_sha256']['../unrelated']='hash'
        with pytest.raises(ValueError,match='file bindings'):validate_reports(*args)


def test_transfer_endpoint_requires_complete_bound_transfer_audit():
    from doorbench.dexterous.qualified_isaac_grasp import validate_transfer_evidence,TRANSFER_CHECKS
    report={'checks':dict.fromkeys(TRANSFER_CHECKS,True)}
    coordinator={'independent_transfer_passed':True}
    hashes={'/trial/operation-report.json':'report','/trial/standing-transfer-steps.json.gz':'stream'}
    audit=dict(passed=True,producer_matches=True,checks=dict.fromkeys(TRANSFER_CHECKS,True),input_sha256=hashes)
    validate_transfer_evidence(report,coordinator,audit,hashes)
    for key,value in [('passed',False),('producer_matches',False),('checks',{}),('input_sha256',{})]:
        changed=deepcopy(audit);changed[key]=value
        with pytest.raises(ValueError):validate_transfer_evidence(report,coordinator,changed,hashes)
    for check in TRANSFER_CHECKS:
        changed=deepcopy(audit);changed['checks'][check]=False
        with pytest.raises(ValueError):validate_transfer_evidence(report,coordinator,changed,hashes)
    with pytest.raises(ValueError):validate_transfer_evidence(report,{},audit,hashes)
    wrong=dict(hashes);wrong['/trial/standing-transfer-steps.json.gz']='changed bytes'
    with pytest.raises(ValueError):validate_transfer_evidence(report,coordinator,audit,wrong)

    relocated={k.replace('/trial/','/restored/trial/'):v for k,v in hashes.items()}
    validate_transfer_evidence(report,coordinator,audit,relocated)
    duplicated=deepcopy(audit);duplicated['input_sha256']['/other/operation-report.json']='report'
    with pytest.raises(ValueError):validate_transfer_evidence(report,coordinator,duplicated,hashes)
