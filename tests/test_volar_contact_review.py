"""The candidate contact expansion must still reject the original bad anatomy."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('volar_review',Path(__file__).resolve().parents[1]/'scripts/dexterous/review_volar_contacts.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def patch():
    return dict(body='/World/H1/pelvis/rh_ffmiddle',body_position_m=[.004,-.007,.019],
        hand_outward_normal_body=[0.,-1.,0.],axial_clearance_m=.008,
        on_lever_cylindrical_side=True,inward_radial_normal_alignment=1.)


def test_candidate_allows_inner_middle_surface_without_requiring_fingertip():
    assert module.qualified_patch(patch())


@pytest.mark.parametrize('change', ['dorsal_point','dorsal_normal','lateral_normal','past_segment',
    'joint_origin','lever_cap','hub_surface','nonradial_normal','knuckle','thumb_middle'])
def test_candidate_preserves_anatomy_and_lever_surface_rejections(change):
    p=copy.deepcopy(patch())
    if change=='dorsal_point':p['body_position_m'][1]=.007
    elif change=='dorsal_normal':p['hand_outward_normal_body']=[0.,1.,0.]
    elif change=='lateral_normal':p['hand_outward_normal_body']=[1.,0.,0.]
    elif change=='past_segment':p['body_position_m'][2]=.026
    elif change=='joint_origin':p['body_position_m'][2]=0.
    elif change=='lever_cap':p['axial_clearance_m']=.0005
    elif change=='hub_surface':p['on_lever_cylindrical_side']=False
    elif change=='nonradial_normal':p['inward_radial_normal_alignment']=.79
    elif change=='knuckle':p['body']='/World/H1/pelvis/rh_ffknuckle'
    elif change=='thumb_middle':p['body']='/World/H1/pelvis/rh_thmiddle'
    assert not module.qualified_patch(p)
