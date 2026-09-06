"""Mass moves with newly articulated material without being counted twice."""
from copy import deepcopy
import pytest
from doorbench.ir import Body,Joint,Model
from doorbench.geometry import common as C
from doorbench.mass_reconciliation import reconcile_moving_mass


def fixture():
    model=Model('material-transfer-fixture')
    leaf=Body('leaf',None,semantic='leaf');leaf.joint=Joint('leaf_hinge','hinge')
    leaf.geoms=[C.box('material',(0,0,0),(.1,.1,.1),'wood',1250)]
    model.add_body(leaf)
    for name in ('arm_a','arm_b'):
        arm=Body(name,'leaf',semantic='mechanism');arm.joint=Joint(name+'_hinge','hinge')
        arm.geoms=[C.box(name+'_material',(0,0,0),(.05,.05,.05),'wood',1000)]
        model.add_body(arm)
    model.meta['material_transfer_bodies']={'arm_a':'leaf','arm_b':'leaf'}
    row={'body':'leaf','slab_kg':10.,'glass_kg':0.,'hardware_kg':1.,'total_kg':11.,'hardware_parts':{'operator':1.}}
    physics={'mass':{**deepcopy(row),'per_body':[row]}}
    physics['mass'].pop('body')
    return model,physics


def test_material_transfer_preserves_total_and_native_authored_arm_masses():
    model,physics=fixture();reconcile_moving_mass(model,physics)
    assert model.body('leaf').inertial('full')[0]==pytest.approx(9.)
    assert [model.body(n).inertial('full')[0] for n in ('arm_a','arm_b')]==pytest.approx([1.,1.])
    assert physics['mass']['total_kg']==11.
    assert physics['mass']['slab_kg']==10.
    assert physics['mass']['transferred_material_kg']==pytest.approx(2.)
    before=deepcopy(physics);reconcile_moving_mass(model,physics)
    assert physics==before
    assert model.meta['mass_reconciliation']['panels'][0]['transferred_material_bodies_kg']==pytest.approx({'arm_a':1.,'arm_b':1.})


def test_new_hardware_is_added_separately_from_existing_material_transfer():
    model,physics=fixture();pin=Body('new_pin','arm_a',semantic='mechanism');pin.joint=Joint('pin_hinge','hinge')
    pin.geoms=[C.box('steel_pin',(0,0,0),(.005,.005,.005),'steel',7850)]
    model.add_body(pin);model.meta['mechanism_mass_bodies']=['new_pin']
    reconcile_moving_mass(model,physics)
    assert physics['mass']['total_kg']==pytest.approx(11.+.00785)
    assert model.body('leaf').inertial('full')[0]==pytest.approx(9.)
    assert model.body('new_pin').inertial('full')[0]==pytest.approx(.00785)
    reconcile_moving_mass(model,physics)
    assert physics['mass']['total_kg']==pytest.approx(11.+.00785)


@pytest.mark.parametrize('mapping',(['arm_a'],{'missing':'leaf'},{'leaf':'leaf'},{'arm_a':'missing'},{'arm_a':3}))
def test_rejects_malformed_or_unreal_material_transfer(mapping):
    model,physics=fixture();model.meta['material_transfer_bodies']=mapping
    with pytest.raises(ValueError,match='[Mm]aterial'):reconcile_moving_mass(model,physics)


def test_rejects_cross_leaf_double_budget_and_insufficient_material():
    model,physics=fixture();model.meta['mechanism_mass_bodies']=['arm_a']
    with pytest.raises(ValueError,match='cannot also add'):reconcile_moving_mass(model,physics)
    model,physics=fixture();model.body('arm_a').parent=None
    with pytest.raises(ValueError,match='ancestor leaf'):reconcile_moving_mass(model,physics)
    model,physics=fixture();model.body('arm_a').geoms[0].density=20000
    with pytest.raises(ValueError,match='exceeds'):reconcile_moving_mass(model,physics)
