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
