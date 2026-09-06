"""Independent evidence tests: panel scope must follow geometry, not body count."""
import copy

import pytest

from doorbench.build import build_model
from doorbench.mechanical_audit import audit_model, material_inventory, rectangle_union_area
from doorbench.spec import generate_all


@pytest.fixture(scope='module')
def cases():
    ids={'db0008_sliding_bypass','db0004_bifold','db0010_swing_double','db0037_strip_curtain','db0095_dutch','db0148_garage_sectional','db0202_turnstile_tripod','db0066_revolving'}
    return {s['id']:(s,build_model(s).to_dict('full')) for s in generate_all() if s['id'] in ids}


def test_projection_unions_duplicates_and_overlapping_proxies():
    assert rectangle_union_area([(0,0,2,1),(0,0,2,1),(1,0,3,1)])==3


@pytest.mark.parametrize('door,ratio',[
    ('db0008_sliding_bypass',3),('db0004_bifold',2),('db0010_swing_double',2),
    ('db0037_strip_curtain',5),('db0095_dutch',1),('db0148_garage_sectional',1),
])
def test_actual_material_scope_not_just_leaf_body_count(cases,door,ratio):
    spec,model=cases[door];inventory=material_inventory(spec,model)
    assert inventory['area_multiple']==pytest.approx(ratio,rel=.025)


def test_revolving_three_wings_share_one_rotor_body(cases):
    spec,model=cases['db0066_revolving'];inventory=material_inventory(spec,model)
    assert inventory['leaf_body_count']==1 and inventory['material_panel_count']==3
    assert inventory['stock_material_mass_kg']==pytest.approx(inventory['glass_volume_m3']*2500)


def test_tripod_formula_already_contains_three_arms(cases):
    spec,model=cases['db0202_turnstile_tripod'];inventory=material_inventory(spec,model)
    assert inventory['material_panel_count']==3
    # Original 38 mm OD, 1.5 mm wall tubes: ~5.04 kg, including 3 kg hub.
    assert 5.0<inventory['stock_material_mass_kg']<5.1


def test_three_panels_cannot_share_one_panels_mass(cases):
    spec,source=cases['db0008_sliding_bypass'];model=copy.deepcopy(source)
    expected=material_inventory(spec,model)['stock_material_mass_kg']
    moving=[b for b in model['bodies'] if not b['static']]
    for b in moving:b['mass']=expected/(3*len(moving))
    issues=audit_model(spec,model)['issues']
    assert any(i['code']=='moving_mass_below_material_estimate' and i['ratio']==pytest.approx(1/3) for i in issues)
    for b in moving:b['mass']=expected/len(moving)
    assert not any(i['code']=='moving_mass_below_material_estimate' for i in audit_model(spec,model)['issues'])


def test_missing_operator_geometry_and_contacts_are_not_silently_accepted(cases):
    spec,source=cases['db0008_sliding_bypass'];model=copy.deepcopy(source)
    for b in model['bodies']:
        b['geoms']=[g for g in b['geoms'] if g['semantic']!='operator']
        b['sites']=[s for s in b['sites'] if s['role'] not in ('grip','push')]
    codes={i['code'] for i in audit_model(spec,model)['issues']}
    assert {'specified_operator_has_no_operator_geometry','specified_operator_has_no_grip_or_push_site','specified_operator_has_no_operator_collider'}<=codes
