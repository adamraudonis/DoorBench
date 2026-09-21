import copy
import numpy as np
import pytest

from scripts.dexterous.audit_isaac_acquisition_contacts import final_hold_observation
from doorbench.dexterous.isaac_withdrawal_audit import validate_record,validate_phase_record,validate_contact_receipt,sha
from doorbench.dexterous.qualified_isaac_grasp import AUDITED_FILES


def records():
    configuration=dict(door_joint_names=['leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide'])
    physics=dict(time_s=np.array([.002]),standing_leaf_pose=np.array([[0,0,0,1,0,0,0]],float),door=np.array([[.08,0.,0.]]))
    surface=dict(time_s=.002,leaf_pose=[0,0,0,1,0,0,0],surface={'palm_normal_load_N':3.})
    pad=dict(sim_time_s=.002,valid_pad_grasp=True,contacts=[],raw_evidence=dict(body_transforms_xyzw={'/World/H1/rh_palm':[0,0,0,0,0,0,1]}))
    row=dict(sim_time_s=.002,door_q=.08,handle_angle_rad=0.,bolt_slide_m=0.,pad_grasp={'valid_pad_grasp':True,'contacts':[]},
        left_surface=copy.deepcopy(surface['surface']),leaf_pose=surface['leaf_pose'].copy(),
        geometry={'body_poses_xyz_wxyz':{'rh_palm':[0,0,0,1,0,0,0]}})
    return row,pad,surface,physics,0,configuration


def test_original_contact_audit_observation_is_unchanged():
    for value in (True,False):
        assert final_hold_observation({'checks':{'sustained_pad_grasp':value}},{'args':{}}) is value


def test_release_contact_audit_requires_explicit_separate_observation():
    configuration={'args':{'standing_withdrawal_route':'new.json'}}
    report={'checks':{},'standing_withdrawal':{},'observed_final_pad_grasp_hold':False}
    assert final_hold_observation(report,configuration) is False
    for changed in ({'observed_final_pad_grasp_hold':None},{'standing_withdrawal':None},{'checks':{'sustained_pad_grasp':True}}):
        with pytest.raises(ValueError):final_hold_observation({**report,**changed},configuration)


def test_independent_release_record_binds_original_actual_streams():
    validate_record(*records())


@pytest.mark.parametrize('change',['clock','pad','surface','leaf','mechanism','geometry'])
def test_actual_stream_mismatch_is_rejected(change):
    args=list(records());row,pad,surface,physics,index,configuration=args
    if change=='clock':row['sim_time_s']=.004
    elif change=='pad':row['pad_grasp']['valid_pad_grasp']=False
    elif change=='surface':row['left_surface']['palm_normal_load_N']=4.
    elif change=='leaf':physics['standing_leaf_pose'][0,0]=.01
    elif change=='mechanism':physics['door'][0,0]=.09
    else:row['geometry']['body_poses_xyz_wxyz']['rh_palm'][0]=.000003
    with pytest.raises(ValueError):validate_record(*args)


def test_contact_receipt_requires_original_bound_files_profile_and_all_intervals(tmp_path):
    for name in AUDITED_FILES:(tmp_path/name).write_text(name)
    receipt=dict(schema='doorbench.isaac-acquisition-contact-audit.v1',accounting_passed=True,
        independent_raw_contact_audit_complete=True,checks={'clock':True},physical_intervals=1000,
        raw_intervals=1000,grasp_profile='volar-phalange-v1',
        input_sha256={name:sha(tmp_path/name) for name in AUDITED_FILES})
    config={'args':{'grasp_profile':'volar-phalange-v1'}}
    assert len(validate_contact_receipt(receipt,tmp_path,config,1000))==4
    for changed in ({'input_sha256':{}},{'raw_intervals':999},{'physical_intervals':999},
            {'grasp_profile':'distal-pad-v1'},{'checks':{}}):
        with pytest.raises(ValueError):validate_contact_receipt({**receipt,**changed},tmp_path,config,1000)
    (tmp_path/'operation-report.json').write_text('another report')
    with pytest.raises(ValueError):validate_contact_receipt(receipt,tmp_path,config,1000)


def test_phase_marker_must_start_at_actual_command_epoch_and_stay_fixed():
    row=dict(sim_time_s=1.002,teacher=dict(phase='standing_withdrawal',withdrawal_started_s=1.,
        withdrawal_progress=0.,release_started_s=1.))
    assert validate_phase_record(row,start=1.,duration=1.,release=None)==1.
    with pytest.raises(ValueError):validate_phase_record(row,start=1.,duration=1.,release=.9)
    row['teacher']['release_started_s']=None
    with pytest.raises(ValueError):validate_phase_record(row,start=1.,duration=1.,release=1.)


def test_relabeling_stage_or_completion_cannot_replace_actual_phase_progress():
    before=dict(sim_time_s=1.,teacher=dict(withdrawal_started_s=None,release_started_s=None,withdrawal_progress=None))
    assert validate_phase_record(before,start=1.,duration=1.,release=None) is None
    before['teacher']['withdrawal_progress']=.001
    with pytest.raises(ValueError):validate_phase_record(before,start=1.,duration=1.,release=None)
    end=dict(sim_time_s=2.,teacher=dict(phase='standing_withdrawal',withdrawal_started_s=1.,
        withdrawal_progress=.9,release_started_s=None))
    with pytest.raises(ValueError):validate_phase_record(end,start=1.,duration=1.,release=None)
