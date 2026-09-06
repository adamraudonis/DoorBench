"""Optional full implicit integration is serialized without changing defaults."""
from xml.etree import ElementTree as ET
import mujoco
import pytest
from doorbench.ir import Model,Body,Joint
from doorbench.geometry import common as C
from doorbench.export.mjcf import build_mjcf


def fixture():
    model=Model('integrator')
    material=C.mat_rgba(model,'steel',(.5,.5,.5,1))
    body=Body('pendulum',None,joint=Joint('pivot','hinge',(1,0,0),range=None))
    body.geoms=[C.box('link',(0,0,-.1),(.01,.01,.1),material,7900)]
    model.add_body(body)
    return model


@pytest.mark.parametrize('value',(None,'implicitfast','implicit'))
def test_default_and_opt_in_integrator_reach_native_model_in_every_tier(value):
    model=fixture()
    if value is not None:model.meta['native_integrator']=value
    for tier in ('full','simple','minimal'):
        xml=build_mjcf(model,tier=tier)
        assert xml.find('option').get('integrator')==(value or 'implicitfast')
        native=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
        assert native.opt.integrator==getattr(mujoco.mjtIntegrator,'mjINT_'+(value or 'implicitfast').upper())


@pytest.mark.parametrize('value',(None,True,3,'Euler','RK4','Implicit','implicit\n',[],{}))
def test_invalid_integrator_metadata_is_rejected(value):
    model=fixture();model.meta['native_integrator']=value
    with pytest.raises(ValueError,match='native_integrator'):build_mjcf(model)
