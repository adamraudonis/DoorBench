"""Native routed-cable limits, pulley purchase, and exact tangent recording."""
import math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import pytest
from doorbench.ir import Model,Body,Joint,Site,SpatialCable,SpatialSpring,ALL_TIERS,QUAT_ID
from doorbench.geometry import common as C
from doorbench.export.mjcf import build_mjcf
from doorbench.export.urdf import build_urdf
from doorbench.spatial_cables import native_cable_paths


def fixture():
    model=Model('purchase');material=C.mat_rgba(model,'steel',(.5,.5,.5,1))
    world=Body('world_env',None,static=True)
    world.sites=[Site('a',(0,-.1,0)),Site('b',(0,.1,0)),Site('rear',(4,0,0))]
    model.add_body(world)
    body=Body('pulley',None,(1,0,0),QUAT_ID,Joint('slide','slide',(1,0,0),range=(-.1,1.)))
    body.geoms=[C.cyl('sheave',(0,0,0),.1,.01,material,(0,0,1),mass=1)]
    body.sites=[Site('side',(.2,0,0)),Site('spring_end',(0,0,0))]
    model.add_body(body)
    model.spatial_cables=[SpatialCable('cable',({'site':'a'},{'geom':'sheave','sidesite':'side'},{'site':'b'}),2+math.pi*.1)]
    model.spatial_springs=[SpatialSpring('spring',('spring_end','rear'),100.,1.)]
    return model


def compile_model(model):
    model.validate();native=mujoco.MjModel.from_xml_string(ET.tostring(build_mjcf(model),encoding='unicode'))
    data=mujoco.MjData(native);mujoco.mj_forward(native,data)
    return native,data


def test_routed_cable_has_two_to_one_purchase_and_tension_only_reaction():
    model=fixture();m,d=compile_model(model)
    tid=m.tendon('cable').id
    assert d.ten_length[tid]==pytest.approx(2+math.pi*.1,abs=1e-6)
    assert np.asarray(d.ten_J).reshape(m.ntendon,m.nv)[tid,0]==pytest.approx(2.)
    assert d.qfrc_spring[0]==pytest.approx(200.)
    for _ in range(1000):mujoco.mj_step(m,d)
    assert 0<=d.qpos[0]<.002
    assert d.qfrc_constraint[0]==pytest.approx(-200.,abs=1.)
    # Shortened cable is slack; the cable cannot push the carriage backwards.
    d.qpos[0]=-.05;d.qvel[:]=0;mujoco.mj_forward(m,d)
    assert d.ten_length[tid]<m.tendon_range[tid,1]-.05
    assert d.qfrc_constraint[0]==pytest.approx(0.,abs=1e-8)


def test_native_routing_records_tangent_points_not_pulley_centres():
    m,d=compile_model(fixture());before=d.qpos.copy()
    report=native_cable_paths(m,d,['cable']);cable=report['cables'][0]
    wrapped=[n['point'] for n in cable['nodes'] if n['geom_name']=='sheave']
    assert len(wrapped)==2
    for point in wrapped:assert np.linalg.norm(np.array(point)-d.geom_xpos[m.geom('sheave').id])==pytest.approx(.1)
    assert cable['wrap_geometries']['sheave']['side_point']==pytest.approx([1.2,0,0])
    assert cable['wrap_geometries']['sheave']['axis']==pytest.approx([0,0,1])
    assert np.array_equal(before,d.qpos)


@pytest.mark.parametrize('mutation', ['missing_site','missing_geom','missing_side','adjacent_geoms','negative_length','bad_endpoint','wrong_geom_type','tier_site'])
def test_cable_validation_rejects_invalid_routes(mutation):
    model=fixture();c=model.spatial_cables[0]
    if mutation=='missing_site':c.path=({'site':'missing'},*c.path[1:])
    elif mutation=='missing_geom':c.path=(c.path[0],{'geom':'missing'},c.path[-1])
    elif mutation=='missing_side':c.path=(c.path[0],{'geom':'sheave','sidesite':'missing'},c.path[-1])
    elif mutation=='adjacent_geoms':c.path=(c.path[0],c.path[1],c.path[1],c.path[-1])
    elif mutation=='negative_length':c.max_length=-1
    elif mutation=='bad_endpoint':c.path=c.path[1:]
    elif mutation=='wrong_geom_type':
        geom=model.body('pulley').geoms[0];geom.type='box';geom.size=(.1,.1,.01)
    elif mutation=='tier_site':model.body('world_env').sites[0].tiers=frozenset({'full'})
    with pytest.raises(AssertionError):model.validate()
    with pytest.raises(ValueError,match='Invalid spatial cable'):build_mjcf(model,'minimal' if mutation=='tier_site' else 'full')


def test_serialization_and_urdf_do_not_claim_native_cable_support():
    model=fixture();payload=model.to_dict()['spatial_cables'][0]
    assert payload['path'][1]=={'geom':'sheave','sidesite':'side'}
    assert payload['max_length']==pytest.approx(2+math.pi*.1)
    node=build_urdf(model).find('doorbench:spatial_cable')
    assert node is not None and node.get('native_support')=='false'
    empty=Model('empty');assert 'spatial_cables' not in empty.to_dict()
