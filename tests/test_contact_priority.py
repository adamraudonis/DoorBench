"""Opt-in contact material mixing preserves collision and default exports."""
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest

from doorbench.ir import Model, Body, Joint
from doorbench.geometry import common as C
from doorbench.export.mjcf import build_mjcf


def fixture(priority):
    model=Model('contact_priority')
    material=C.mat_rgba(model,'steel',(.5,.5,.5,1))
    floor=Body('floor',None,static=True)
    floor.geoms=[C.box('floor_stock',(0,0,-.02),(.1,.1,.02),material,7900)]
    floor.geoms[0].solref=(.005,1.)
    head=Body('head',None,joint=Joint('head_z','slide',(0,0,1),range=None))
    head.geoms=[C.sphere('steel_head',(0,0,.004),.005,material,7900,True)]
    head.geoms[0].solref=(.0002,1.)
    head.geoms[0].contact_priority=priority
    model.add_body(floor);model.add_body(head)
    return model,head.geoms[0]


@pytest.mark.parametrize('priority',(0,1,2147483647))
def test_priority_roundtrip_and_native_contact_material_mixing(priority):
    model,head=fixture(priority)
    assert model.validate()
    tree=build_mjcf(model)
    node=tree.find(".//geom[@name='steel_head']")
    assert node.get('priority')==(str(priority) if priority else None)
    assert head.to_dict().get('contact_priority')==(priority if priority else None)
    native=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'))
    data=mujoco.MjData(native);mujoco.mj_forward(native,data)
    assert native.geom_priority[native.geom('steel_head').id]==priority
    assert data.ncon==1 and data.contact[0].dist<0
    # A priority selects the head's stiff material; zero retains the normal
    # equal-priority average. Neither setting removes the measured contact.
    np.testing.assert_allclose(data.contact[0].solref,
        (.0002,1.) if priority else ((.005+.0002)/2,1.),rtol=0,atol=1e-12)


@pytest.mark.parametrize('priority',(-1,True,False,1.5,float('nan'),float('inf'),2147483648,'1',None))
def test_invalid_priority_rejected_by_model_json_and_native_export(priority):
    model,head=fixture(priority)
    for action in (model.validate,head.to_dict,lambda:build_mjcf(model)):
        with pytest.raises(ValueError,match='contact_priority'):
            action()
