"""Flattened stationary assemblies retain every authored rigid transform."""
from xml.etree import ElementTree as ET
import numpy as np
import mujoco
import pytest
from doorbench.ir import Model,Body,Joint,Site,quat_from_axis_angle,quat_to_mat
from doorbench.geometry import common as C
from doorbench.export.mjcf import build_mjcf


@pytest.mark.parametrize('retained',[False,True])
def test_nested_rotated_static_frames_preserve_geometry_sites_and_moving_children(retained):
    model=Model('static_frames');mat=C.mat_rgba(model,'steel',(.4,.4,.4,1))
    frame=Body('frame',None,(1.,-.4,1.7),quat_from_axis_angle((0,0,1),.7),static=True)
    frame.geoms=[C.box('housing',(0,.2,0),(.05,.05,.05),mat,7850)]
    child=Body('mount',frame.name,(.2,0,.1),quat_from_axis_angle((1,0,0),-.4),static=True)
    child.sites=[Site('anchor',(.01,.03,0))]
    moving=Body('pawl',child.name,(.03,-.02,.06),quat_from_axis_angle((0,1,0),.3),
                joint=Joint('pivot','hinge',(0,0,1),range=(-1,1),role='mechanism'))
    moving.geoms=[C.box('pawl_tip',(.03,0,0),(.02,.003,.006),mat,7850)]
    moving.sites=[Site('tip',(.05,0,0))]
    for b in (frame,child,moving):model.add_body(b)
    if retained:model.meta['native_fixed_body_names']=['frame','mount']
    native=mujoco.MjModel.from_xml_string(ET.tostring(build_mjcf(model),encoding='unicode'));d=mujoco.MjData(native)
    for angle in (0.,.35,-.2):
        d.qpos[0]=angle;mujoco.mj_forward(native,d)
        for body,name,local in [(frame,'housing',frame.geoms[0].pos),(child,'anchor',child.sites[0].pos)]:
            p,q=model.world_transform(body.name);expected=np.asarray(p)+quat_to_mat(q)@local
            actual=d.geom_xpos[native.geom(name).id] if name=='housing' else d.site_xpos[native.site(name).id]
            np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=0)
        p,q=model.world_transform(moving.name);R=quat_to_mat(q)@quat_to_mat(quat_from_axis_angle((0,0,1),angle))
        np.testing.assert_allclose(d.site_xpos[native.site('tip').id],np.asarray(p)+R@moving.sites[0].pos,atol=3e-6,rtol=0)
    if not retained:
        assert native.nbody==2 and native.body_parentid[native.body('pawl').id]==0
        assert native.geom_bodyid[native.geom('housing').id]==0
    assert native.body_mass[native.body('pawl').id]==pytest.approx(moving.inertial()[0],abs=5e-7)
