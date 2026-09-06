"""A material-chain root needs seven pose values, not a fake scalar slider."""
import json,math
from xml.etree import ElementTree as ET
import numpy as np
import mujoco
import pytest
from doorbench.ir import Model,Body,Joint,Equality
from doorbench.geometry import common as C
from doorbench.export.mjcf import build_mjcf
from doorbench.export.urdf import build_urdf


def fixture():
    model=Model('free_chain_root');mat=C.mat_rgba(model,'steel',(.5,.5,.5,1))
    model.add_body(Body('world_env',None,static=True))
    leaf=model.add_body(Body('leaf',None,(-1,0,1),joint=Joint('door_hinge','hinge')))
    leaf.geoms=[C.box('slab',(0,0,0),(.1,.02,.3),mat,7850)]
    model.meta['primary_joint']='door_hinge'
    root=model.add_body(Body('chain_link_0',None,(.3,.2,1.2),(math.cos(.3),math.sin(.3),0,0),
        joint=Joint('chain_free','free',range=None,role='mechanism',robot_interactive=False)))
    root.geoms=[C.box('first_link',(0,0,-.02),(.003,.006,.02),mat,7850)]
    child=model.add_body(Body('chain_link_1',root.name,(0,0,-.04),joint=Joint('chain_pin','hinge',(-1,0,0),range=(-1,1),role='mechanism',robot_interactive=False)))
    child.geoms=[C.box('second_link',(0,0,-.02),(.003,.006,.02),mat,7850)]
    return model


def test_free_root_native_pose_width_rest_inertia_and_reset():
    model=fixture();model.validate();before=model.body('chain_link_0').to_dict();model.bake_initial()
    assert model.body('chain_link_0').to_dict()==before
    xml=build_mjcf(model);free=xml.find('.//freejoint')
    assert free is not None and free.attrib=={'name':'chain_free'}
    m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));d=mujoco.MjData(m)
    assert (m.nq,m.nv,m.njnt)==(9,8,3)
    assert m.jnt_qposadr.tolist()==[0,1,8] and m.jnt_dofadr.tolist()==[0,1,7]
    np.testing.assert_allclose(m.qpos0[1:4],model.body('chain_link_0').pos,atol=1e-7)
    np.testing.assert_allclose(m.qpos0[4:8],model.body('chain_link_0').quat,atol=1e-6)
    assert np.all(m.dof_damping[1:7]==0) and np.all(m.dof_armature[1:7]==0)
    assert m.body_mass[m.body('chain_link_0').id]==pytest.approx(model.body('chain_link_0').inertial()[0],abs=5e-7)
    for _ in range(50):mujoco.mj_step(m,d)
    assert d.qpos[3]<m.qpos0[3]  # Genuine free fall, no hidden world slider or weld.
    assert np.linalg.norm(d.qpos[4:8])==pytest.approx(1.)
    mujoco.mj_resetData(m,d);np.testing.assert_array_equal(d.qpos,m.qpos0)
    payload=model.to_dict()['bodies'][2]['joint']
    assert payload['qpos_width']==7 and payload['qvel_width']==6 and not payload['robot_interactive']


def test_floating_urdf_loads_and_has_no_scalar_axis_limits():
    model=fixture();xml=build_urdf(model);joint=xml.find("joint[@name='chain_free']")
    assert joint is not None and joint.get('type')=='floating'
    assert joint.find('axis') is None and joint.find('limit') is None and joint.find('dynamics') is None
    m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    j=m.joint('chain_free').id;assert m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE
    adr=m.jnt_qposadr[j];np.testing.assert_allclose(m.qpos0[adr:adr+3],model.body('chain_link_0').pos,atol=1e-6)


@pytest.mark.parametrize('bad',['parent','static','range','interactive','initial','modeled','stiffness','zero_mass','bad_quat','actuator','coupling'])
def test_free_root_rejects_scalar_or_fabricated_body_semantics(bad):
    model=fixture();body=model.body('chain_link_0');joint=body.joint
    if bad=='parent':body.parent='world_env'
    elif bad=='static':body.static=True
    elif bad=='range':joint.range=(0,1)
    elif bad=='interactive':joint.robot_interactive=True
    elif bad=='initial':joint.initial=.2
    elif bad=='modeled':joint.modeled_at=.2
    elif bad=='stiffness':joint.stiffness=1.
    elif bad=='zero_mass':body.geoms=[]
    elif bad=='bad_quat':body.quat=(2,0,0,0)
    elif bad=='actuator':model.meta['actuators']=[{'joint':'chain_free','kind':'motor','name':'bad'}]
    elif bad=='coupling':model.equalities=[Equality(name='bad',kind='joint',a='chain_free',b='door_hinge')]
    with pytest.raises(AssertionError):model.validate()
    with pytest.raises(AssertionError):build_mjcf(model)
    with pytest.raises(AssertionError):build_urdf(model)


def test_usd_marks_free_chain_root_as_unsupported_in_both_layouts(tmp_path):
    from pxr import Usd
    from doorbench.export.usd import write_usd,write_usd_rl
    model=fixture();hardware=tmp_path/'hardware';hardware.mkdir()
    write_usd(model,str(tmp_path),str(hardware));write_usd_rl(model,str(tmp_path),str(hardware))
    full=Usd.Stage.Open(str(tmp_path/'door.usda'));meta=json.loads(full.GetDefaultPrim().GetAttribute('doorbench:meta').Get())
    assert meta['mechanical_parity_supported'] is False
    assert meta['unsupported_free_roots'][0]['joint']=='chain_free'
    canonical=Usd.Stage.Open(str(tmp_path/'door_rl.usda'));rl=json.loads(canonical.GetDefaultPrim().GetAttribute('doorbench:rl').Get())
    assert rl['mechanical_parity_supported'] is False and rl['unsupported_free_roots'][0]['body']=='chain_link_0'
