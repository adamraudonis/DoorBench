"""Mechanical regressions for retractable tilt-up doors and independent bifolds."""
import json
import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from doorbench.build import export_door
from doorbench.clearance import Clearance
from doorbench.geometry.garage_tiltup import linkage_dimensions, linkage_pose, resolve_garage_configuration, projected_static_resistance
from doorbench.ir import Body, Joint, Model, Site, SpatialSpring, QUAT_ID, ALL_TIERS
from doorbench.geometry import common as C
from doorbench.export.mjcf import build_mjcf
from doorbench.spec import generate_all

SPECS = [s for s in generate_all() if s['family'] in ('garage_tiltup','bifold')]
GARAGES = [s for s in SPECS if s['family']=='garage_tiltup']
FOLDS = [s for s in SPECS if s['family']=='bifold']


@pytest.fixture(scope='module')
def exports(tmp_path_factory):
    root=tmp_path_factory.mktemp('garage-folding')
    for spec in SPECS:
        export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
        if spec['family']=='garage_tiltup' and spec['lock'].get('engaged'):
            unlocked=copy.deepcopy(spec);unlocked['lock']['engaged']=False
            export_door(unlocked,str(root/'unlocked/doors'),str(root/'unlocked/hardware'),formats=('mjcf','json'))
    return root


def _load(exports,spec,tier='full'):
    root=exports/'doors'/spec['id']
    filename='door.xml' if tier=='full' else f'door_{tier}.xml'
    m=mujoco.MjModel.from_xml_path(str(root/filename))
    return m,mujoco.MjData(m),json.loads((root/'model.json').read_text())


def _connections(model,data):
    errors=[]
    for e in range(model.neq):
        if model.eq_active0[e] and model.eq_type[e]==mujoco.mjtEq.mjEQ_CONNECT:
            a,b=model.eq_obj1id[e],model.eq_obj2id[e]
            pa=data.xpos[a]+data.xmat[a].reshape(3,3)@model.eq_data[e,:3]
            pb=data.xpos[b]+data.xmat[b].reshape(3,3)@model.eq_data[e,3:6]
            errors.append(np.linalg.norm(pa-pb))
    return errors


def _minimum_leaf_height(model,data,description):
    names={g['name'] for b in description['bodies'] if b['semantic']=='leaf' for g in b['geoms']
           if g.get('semantic','leaf') in ('leaf','glass')}
    minima=[]
    for name in names:
        try:gid=model.geom(name).id
        except KeyError:continue
        rot=data.geom_xmat[gid].reshape(3,3)
        center=data.geom_xpos[gid]+rot@model.geom_aabb[gid,:3]
        ext=np.abs(rot)@model.geom_aabb[gid,3:]
        minima.append(center[2]-ext[2])
    return min(minima)


@pytest.mark.parametrize('spec',GARAGES,ids=lambda s:s['id'])
def test_garage_has_supported_overhead_panel_and_real_springs_in_all_tiers(exports,spec):
    for tier in ('full','simple','minimal'):
        m,d,description=_load(exports,spec,tier)
        meta=description['meta'];mechanism=meta['garage_tiltup_linkage']
        assert len(mechanism['arm_joints'])==2 and m.ntendon==2
        j=m.joint(mechanism['primary_joint']).id
        # Full nominal geometry is checked separately from any locked range.
        for angle in np.linspace(0,mechanism['nominal_range_rad'][1],89):
            d.qpos[:]=m.qpos0;d.qpos[m.jnt_qposadr[j]]=angle
            resolve_garage_configuration(m,d.qpos,meta);mujoco.mj_forward(m,d)
            assert max(_connections(m,d))<1e-6
            assert d.qpos[m.jnt_qposadr[m.joint(mechanism['carriage_joint']).id]]>=-1e-8
            # Both extension springs stay in tension throughout this branch.
            assert np.all(d.ten_length>m.tendon_lengthspring[:,0])
        assert _minimum_leaf_height(m,d,description)>1.8
        assert d.xpos[m.body('door').id,2]>1.9
        assert description['meta']['mechanical_export_support']['usd'].startswith('unsupported')


@pytest.mark.parametrize('spec',GARAGES,ids=lambda s:s['id'])
def test_garage_clearance_and_running_sweep_include_actual_linkage(exports,spec):
    gate=Clearance(str(exports/'doors'/spec['id']))
    assert gate.run(88)['ok']
    assert gate.run_running(88)['ok']


def test_old_stationary_half_height_hinge_blocks_passage():
    # Reproduce the user's fault: an 88 degree open slab still lies at waist height.
    m=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
      <body pos="0 0 1.065"><joint name="door_hinge" axis="-1 0 0" range="0 1.536"/>
      <geom name="slab" type="box" size="1.22 .02 1.065" mass="50"/></body>
      </worldbody></mujoco>''')
    d=mujoco.MjData(m);d.qpos[0]=math.radians(88);mujoco.mj_forward(m,d)
    desc={'bodies':[{'semantic':'leaf','geoms':[{'name':'slab'}]}]}
    assert _minimum_leaf_height(m,d,desc)<1.1


@pytest.mark.parametrize('spec',GARAGES,ids=lambda s:s['id'])
def test_native_tiltup_reaches_overhead_without_pose_resets_or_track_pinch(exports,spec):
    """A bounded virtual motor proves load-bearing loop travel, not human strength.

    This catches the actual 40 mm wheel/40 mm channel pinch that static sweeps
    and prescribed configurations missed. Locks are released only for this
    full nominal mechanical-travel test; the ordinary QA still tests locks.
    """
    # Release the actual hasp/padlock assembly via a separately authored
    # unlocked fixture; changing only the door's range cannot remove a lock.
    m,d,description=_load(exports/'unlocked' if spec['lock'].get('engaged') else exports,spec)
    j=m.joint('door_hinge').id;qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j]
    target=math.radians(88);m.jnt_range[j,1]=target
    scale=float(m.body_subtreemass[m.body('door').id])/75
    integral=0.;max_error=0.;depth=0.
    for i in range(round(18/m.opt.timestep)):
        u=min(1.,i*m.opt.timestep/12)
        desired=target*(10*u**3-15*u**4+6*u**5)
        error=desired-d.qpos[qa];integral+=error*m.opt.timestep
        d.qfrc_applied[va]=np.clip(scale*(2000*error+2000*integral-250*d.qvel[va]),-1000*scale,1000*scale)
        mujoco.mj_step(m,d)
        max_error=max(max_error,max(_connections(m,d)))
        depth=max(depth,max((-c.dist for c in d.contact),default=0.))
    assert math.degrees(d.qpos[qa])>87.5
    assert _minimum_leaf_height(m,d,description)>1.8
    assert max_error<.0002
    assert depth<1e-6
    assert not any(w.number for w in d.warning)


def test_projected_native_passive_load_uses_loop_tangent_without_mutating_data(exports):
    m,d,description=_load(exports,GARAGES[0]);meta=description['meta']
    qa=m.jnt_qposadr[m.joint('door_hinge').id]
    for angle in (0.,.1,.7,1.5):
        d.qpos[qa]=angle;resolve_garage_configuration(m,d.qpos,meta);mujoco.mj_forward(m,d)
        before=d.qpos.copy();vbefore=d.qvel.copy()
        report=projected_static_resistance(m,d,meta)
        numeric=np.zeros(m.nv);epsilon=1e-6
        plus=before.copy();minus=before.copy();plus[qa]+=epsilon;minus[qa]-=epsilon
        delta=(resolve_garage_configuration(m,plus,meta)-resolve_garage_configuration(m,minus,meta))/(2*epsilon)
        for joint in range(m.njnt):numeric[m.jnt_dofadr[joint]]=delta[m.jnt_qposadr[joint]]
        assert np.allclose(report['tangent_dof'],numeric,atol=1e-7)
        assert report['static_resistance']==pytest.approx(numeric@(d.qfrc_bias-d.qfrc_passive),abs=1e-5)
        assert report['frictionloss']==pytest.approx(np.abs(numeric)@m.dof_frictionloss)
        assert np.array_equal(before,d.qpos) and np.array_equal(vbefore,d.qvel)
        # An arbitrarily strong external drive cannot pollute this load estimate.
        d.qfrc_applied[:]=1e8
        assert projected_static_resistance(m,d,meta)==report


def test_padlocked_hasp_physically_retains_panel_even_without_primary_range_stop(exports):
    spec=next(s for s in GARAGES if s['id']=='db0651_garage_tiltup')
    m,d,description=_load(exports,spec)
    assert description['meta']['garage_lock_hardware']['kind']=='rear_slotted_hasp_padlock'
    pj=m.joint('door_hinge').id;hj=m.joint('garage_lock_hasp_hinge').id
    m.jnt_range[pj,1]=math.radians(88)
    m.jnt_limited[hj]=False
    peak=0.;hasp_contacts=False
    for _ in range(2000):
        d.qfrc_applied[m.jnt_dofadr[pj]]=150
        mujoco.mj_step(m,d);peak=max(peak,d.qpos[m.jnt_qposadr[pj]])
        hasp_contacts |= any('garage_lock_hasp' in m.geom(c.geom1).name or
                             'garage_lock_hasp' in m.geom(c.geom2).name for c in d.contact)
    assert peak<math.radians(4) and hasp_contacts
    assert not any(w.number for w in d.warning)


def test_garage_slide_bolt_contact_blocks_then_actual_retraction_releases(tmp_path):
    spec=copy.deepcopy(next(s for s in GARAGES if s['id']=='db0173_garage_tiltup'))
    spec['lock']['engaged']=True
    export_door(spec,str(tmp_path/'doors'),str(tmp_path/'hardware'),formats=('mjcf','json'))
    m,d,description=_load(tmp_path,spec)
    pj=m.joint('door_hinge').id;bj=m.joint('garage_slide_lock_slide').id
    assert m.jnt_range[pj,1]>1.5  # No artificial primary stop holds this test.
    for _ in range(1500):
        d.qfrc_applied[m.jnt_dofadr[pj]]=400;mujoco.mj_step(m,d)
    assert d.qpos[m.jnt_qposadr[pj]]<math.radians(2)
    assert d.qpos[m.jnt_qposadr[bj]]<.001
    assert any('garage_slide_lock_rod' in (m.geom(c.geom1).name,m.geom(c.geom2).name) for c in d.contact)
    for i in range(6000):
        d.qfrc_applied[:]=0;d.qfrc_applied[m.jnt_dofadr[bj]]=60
        if i>500:d.qfrc_applied[m.jnt_dofadr[pj]]=500
        mujoco.mj_step(m,d)
    assert d.qpos[m.jnt_qposadr[bj]]>.06
    assert d.qpos[m.jnt_qposadr[pj]]>math.radians(20)
    assert not any(w.number for w in d.warning)


@pytest.mark.parametrize('spec',FOLDS,ids=lambda s:s['id'])
def test_bifold_banks_are_independent_guided_and_have_useful_outward_grip(exports,spec):
    gate=Clearance(str(exports/'doors'/spec['id']));m,d,description=_load(exports,spec)
    banks=description['meta']['folding_banks']
    assert len(banks)==spec['leaf']['count']//2
    assert all(b['grip_site'] for b in banks)
    for bank in banks:
        pivot=m.joint(bank['pivot_joint']).id;fold=m.joint(bank['fold_joint']).id
        # A signed constrained Jacobian verifies an outward force opens the bank.
        mujoco.mj_forward(m,d);jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
        mujoco.mj_jacSite(m,d,jp,jr,m.site(bank['grip_site']).id)
        outward=-(jp[1,m.jnt_dofadr[pivot]]-2*jp[1,m.jnt_dofadr[fold]])
        assert outward>.025, (spec['id'],outward)
        for angle in np.linspace(0,bank['open_q'],45):
            q=m.qpos0.copy();q[m.jnt_qposadr[pivot]]=angle
            d.qpos[:]=gate.resolve(q);mujoco.mj_forward(m,d)
            assert abs(d.site_xpos[m.site(bank['guide_site']).id,1])<1e-6
            for other in banks:
                if other is not bank:
                    assert d.qpos[m.jnt_qposadr[m.joint(other['pivot_joint']).id]]==0
        d.qpos[:]=m.qpos0
    assert gate.run(45)['ok']
    assert gate.run_running(45)['ok']


def test_spatial_spring_applies_actual_endpoint_force_and_validates_sites():
    model=Model('spring_test');mat=C.mat_rgba(model,'steel',(.5,.5,.5,1))
    world=Body('world_env',None,(0,0,0),QUAT_ID,None,[],[],ALL_TIERS,'wall','World',static=True)
    world.sites.append(Site('anchor',(0,0,0)));model.add_body(world)
    load=Body('load',None,(1,0,0),QUAT_ID,Joint('slide','slide',(1,0,0),(0,0,0),(-.5,.5)),[],[],ALL_TIERS,'leaf','Load')
    load.sites.append(Site('end',(0,0,0)));load.geoms.append(C.box('mass',(0,0,0),(.02,.02,.02),mat,mass=1.));model.add_body(load)
    model.spatial_springs.append(SpatialSpring('extension',('anchor','end'),100.,.75))
    model.validate();m=mujoco.MjModel.from_xml_string(ET.tostring(build_mjcf(model),encoding='unicode'));d=mujoco.MjData(m)
    for q,length,force in [(0.,1.,-25.),(.2,1.2,-45.)]:
        d.qpos[0]=q;mujoco.mj_forward(m,d)
        assert d.ten_length[0]==pytest.approx(length)
        assert d.qfrc_spring[0]==pytest.approx(force)
    model.spatial_springs[0].sites=('anchor','missing')
    with pytest.raises(AssertionError,match='missing spatial spring site'):model.validate()
    with pytest.raises(ValueError,match='Missing spatial spring endpoint'):build_mjcf(model)
