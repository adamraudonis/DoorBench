import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('panel_phase_audit',Path(__file__).resolve().parents[1]/'scripts/dexterous/audit_panel_phase_moment.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def row():
    sizes={'previous_contact_interval_s':2,'body_coordinates':6,'planned_coordinates':31,
           'planned_velocity':31,'planned_acceleration':31,'latched_motor_targets':61,
           'left_arm_targets':7,'left_arm_target_velocity':7}
    item={key:np.zeros(size).tolist() for key,size in sizes.items()}
    for key in ('episode_time_s','local_time_s','opening_clock_offset_s','pose_time_s',
                'reference_aperture_rad','actual_aperture_rad','normal_feedforward_N','tracking_lead_rad'):item[key]=0.
    return item


@pytest.mark.parametrize('key',['body_coordinates','latched_motor_targets','left_arm_targets','left_arm_target_velocity','episode_time_s'])
def test_nonfinite_evidence_cannot_hide_in_comparisons(key):
    item=row()
    if isinstance(item[key],list):item[key][0]=np.nan
    else:item[key]=np.nan
    with pytest.raises(ValueError):module.validate_panel_reference_row(item)


def test_actual_command_path_must_match_its_rate_claim():
    previous=row();item=copy.deepcopy(previous);item['episode_time_s']=item['local_time_s']=.002
    item['actual_base_correction']={'time_s':.002};item['left_arm_target_velocity'][0]=.006;item['left_arm_targets'][0]=.000006
    speed,acceleration=module.validate_panel_reference_row(item,previous)
    assert speed==.006 and acceleration==3.
    item['left_arm_targets'][0]+=.0001
    with pytest.raises(ValueError):module.validate_panel_reference_row(item,previous)


def test_normal_admittance_report_cannot_claim_a_bounded_teleported_offset():
    previous=row();previous['normal_admittance']=dict(time_s=0.,normal_offset_m=0.,normal_offset_velocity_m_s=0.,normal_offset_acceleration_m_s2=0.)
    item=copy.deepcopy(previous);item['episode_time_s']=item['local_time_s']=.002
    item['normal_admittance'].update(time_s=.002,normal_offset_m=.0001)
    with pytest.raises(ValueError):module.validate_panel_reference_row(item,previous)
    item['normal_admittance']['normal_offset_m']=0.
    module.validate_panel_reference_row(item,previous)


def test_corrected_waist_target_cannot_jump_behind_valid_seven_arm_telemetry():
    previous=row();names=['torso']+['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1']
    previous.update(corrected_chain_joint_names=names,corrected_chain_targets=[0.]*8,corrected_chain_velocity=[0.]*8)
    item=copy.deepcopy(previous);item['episode_time_s']=item['local_time_s']=.002
    item['corrected_chain_targets'][0]=.01
    with pytest.raises(ValueError):module.validate_panel_reference_row(item,previous)
    item['corrected_chain_targets'][0]=0.
    module.validate_panel_reference_row(item,previous)
