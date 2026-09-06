"""Export real small moving hardware without rounding away its inertia."""
from xml.etree import ElementTree as ET
import numpy as np
import mujoco
from doorbench.ir import Body, Geom, Joint, Model
from doorbench.export.urdf import build_urdf


def test_two_millimetre_steel_pin_retains_physical_tensor_and_loads():
    model=Model('small_steel_hardware')
    model.add_body(Body('world_env',None,static=True))
    pin=model.add_body(Body('steel_eye','world_env',pos=(0,0,1),joint=Joint('eye_swivel','hinge')))
    pin.geoms.append(Geom('eye','sphere',(.002,),density=7850.,visual=False))
    mass,_,tensor=pin.inertial()
    assert mass>0 and 0<tensor[0,0]<5e-10
    urdf=build_urdf(model)
    inertia=urdf.find("./link[@name='steel_eye']/inertial/inertia")
    assert inertia is not None
    for name,i,j in (('ixx',0,0),('ixy',0,1),('ixz',0,2),('iyy',1,1),('iyz',1,2),('izz',2,2)):
        assert float(inertia.attrib[name])==float(tensor[i,j])
    native=mujoco.MjModel.from_xml_string(ET.tostring(urdf,encoding='unicode'))
    assert np.all(native.body_inertia[native.body('steel_eye').id]>0)
    np.testing.assert_allclose(native.body_inertia[native.body('steel_eye').id],np.diag(tensor),rtol=1e-12,atol=0.)
