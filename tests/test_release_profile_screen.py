import pytest

from scripts.dexterous.audit_standing_ungrip import release_surface_scores


@pytest.mark.parametrize('digit,segment,point,normal,expected',[
    ('ff','middle',[0.,-.004,.015],[0.,-1.,0.],(True,False)),
    ('ff','proximal',[0.,-.004,.015],[0.,-1.,0.],(True,False)),
    ('th','middle',[0.,-.004,.015],[0.,-1.,0.],(False,False)),
    ('ff','distal',[0.,-.004,.015],[0.,-1.,0.],(True,True)),
    ('ff','proximal',[0.,-.004,.001],[0.,-1.,0.],(False,False)),
    ('ff','middle',[0.,.004,.015],[0.,1.,0.],(False,False)),
])
def test_selected_volar_and_unchanged_distal_counter_score(digit,segment,point,normal,expected):
    assert release_surface_scores(digit,segment,point,normal,.002,1.,profile='volar-phalange-v1')==expected


@pytest.mark.parametrize('clearance,alignment',[(.0009,1.),(.002,.8)])
def test_profile_does_not_relax_axial_side_or_radial_normal_gate(clearance,alignment):
    assert release_surface_scores('ff','middle',[0.,-.004,.015],[0.,-1.,0.],
        clearance,alignment,profile='volar-phalange-v1')==(False,False)
