"""Native hatch load paths, real pull cavities, and hold/release regressions."""
import copy
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from doorbench.build import export_door
from doorbench.geometry.hatch_supports import resolve_hatch_configuration
from doorbench.spec import generate_all

SPECS=[s for s in generate_all() if s['family'] in ('hatch_floor','hatch_ceiling')]


@pytest.fixture(scope='module')
def exports(tmp_path_factory):
    root=tmp_path_factory.mktemp('hatches')
    for original in SPECS:
        spec=copy.deepcopy(original)
        # Exercise support travel independently from a separately authored
        # padlock/bolt. The source corpus's locked state is never changed.
        spec['lock']={'model':'none','engaged':False,'robot_side_release':True}
        spec['latch']={'model':'none'}
        export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
    return root


def load(root,spec,tier='full'):
    folder=root/'doors'/spec['id'];desc=json.loads((folder/'model.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(folder/('door.xml' if tier=='full' else f'door_{tier}.xml')))
    return model,mujoco.MjData(model),desc


def loops(m,d):
    errors=[]
    for e in range(m.neq):
        if m.eq_type[e]!=mujoco.mjtEq.mjEQ_CONNECT:continue
        a,b=m.eq_obj1id[e],m.eq_obj2id[e]
        p=d.xpos[a]+d.xmat[a].reshape(3,3)@m.eq_data[e,:3]
        q=d.xpos[b]+d.xmat[b].reshape(3,3)@m.eq_data[e,3:6]
        errors.append(float(np.linalg.norm(p-q)))
    return errors


@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_all_hatches_native_full_travel_and_true_pulls(exports,spec):
    for tier in ('full','simple','minimal'):
        m,d,desc=load(exports,spec,tier);meta=desc['meta']
        j=m.joint('hatch_hinge').id;adr=m.jnt_qposadr[j]
        support=meta.get('hatch_support');expected=int(spec['closer']['model']=='gas_strut')+int(spec['kinematics']['stop']=='prop_arm')
        assert m.neq==expected
        assert m.jnt_stiffness[j]==0  # no double-counted phantom torsion spring
        for q in np.linspace(0,math.radians(spec['kinematics']['max_open_deg']),81):
            d.qpos[:]=m.qpos0;d.qpos[adr]=q
            resolve_hatch_configuration(m,d.qpos,meta);mujoco.mj_forward(m,d)
            assert max(loops(m,d),default=0)<1e-6
            for c in d.contact:
                assert c.dist>-.0005,(spec['id'],q,m.geom(c.geom1).name,m.geom(c.geom2).name,c.dist)
        # Actual grip moves with its mounting body and is on the approached
        # face. No leftover vertical-door site or invented through-slab reach.
        d.qpos[:]=m.qpos0;mujoco.mj_forward(m,d)
        grip=m.site(meta['hatch_hand_access']['opening_contact']).id
        face=-1 if spec['family']=='hatch_ceiling' else 1
        lidz=d.xpos[m.body('hatch').id,2]
        if spec['operator']['model'] in ('hatch_ring','pull_ring'):
            assert face*(d.site_xpos[grip,2]-lidz)==pytest.approx(spec['leaf']['thickness']/2-.006,abs=1e-6)
        else:assert face*(d.site_xpos[grip,2]-lidz)>spec['leaf']['thickness']/2
        if spec['family']=='hatch_ceiling':assert meta['hatch_hand_access']['requires_elevation_aid']
        if spec['operator']['model'] in ('hatch_ring','pull_ring'):
            ring=m.body('ring').id;slabs=[g for g in range(m.ngeom) if m.geom(g).name.startswith('hatch_slab')]
            for angle in np.linspace(0,math.pi/2,25):
                d.qpos[m.jnt_qposadr[m.joint('ring_hinge').id]]=angle;mujoco.mj_forward(m,d)
                # Parent-child filtering cannot hide a ring embedded in a
                # shallow fake recess: direct convex distance sees the slab.
                for g in range(m.ngeom):
                    if m.geom_bodyid[g]!=ring:continue
                    for slab in slabs:
                        assert mujoco.mj_geomDistance(m,d,g,slab,.02,None)>-.0001
        if support:
            records=[support]+([support['gas_assist']] if support.get('gas_assist') else [])
            for record in records:
                if record['gas_force_at_extension_N']:
                    assert record['support_release_joint'] is None
                    slider=m.joint(record['slide_joint']).id
                    for fraction in (0.,.5,1.):
                        d.qpos[:]=m.qpos0;d.qpos[m.jnt_qposadr[slider]]=fraction*record['stroke_m']
                        mujoco.mj_forward(m,d)
                        force=d.qfrc_passive[m.jnt_dofadr[slider]]
                        assert force==pytest.approx(spec['closer']['gas_force_N']*(1.1-.1*fraction),abs=.002)


def cycle(m,d,meta,*,disable_pin=False):
    """Finite applied torques/forces only after reset; no per-step pose setting."""
    r=meta['hatch_support'];maximum=r['nominal_angle_rad']
    j=m.joint('hatch_hinge').id;qa=m.jnt_qposadr[j];qv=m.jnt_dofadr[j]
    pin=m.joint(r['support_release_joint']).id;pa=m.jnt_qposadr[pin];pv=m.jnt_dofadr[pin]
    if disable_pin:
        gid=m.geom('stay_lock_pin').id;m.geom_contype[gid]=m.geom_conaffinity[gid]=0
    # A virtual test motor lifts the hatch. This is an assembly test, not a
    # humanoid-strength or safe-operating-force claim.
    for k in range(round(5/m.opt.timestep)):
        u=min(1.,k*m.opt.timestep/4);target=maximum*u*u*(3-2*u)
        d.qfrc_applied[:]=0
        d.qfrc_applied[qv]=np.clip(4000*(target-d.qpos[qa])-150*d.qvel[qv]+d.qfrc_bias[qv],-750,750)
        mujoco.mj_step(m,d)
    engagement=float(d.qpos[pa]);opened=float(d.qpos[qa]);contact_force=0.
    for k in range(round(1/m.opt.timestep)):
        d.qfrc_applied[:]=0;d.qfrc_applied[qv]=-abs(d.qfrc_bias[qv])-100
        mujoco.mj_step(m,d)
        for i,c in enumerate(d.contact):
            if m.geom(c.geom1).name=='stay_lock_pin' or m.geom(c.geom2).name=='stay_lock_pin':
                force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
                contact_force=max(contact_force,float(np.linalg.norm(force[:3])))
    held=float(d.qpos[qa])
    # Lift slightly back to the nominal stop to unload the locking pin, then
    # pull it against its known spring. Keep it withdrawn during lowering.
    for k in range(round(2/m.opt.timestep)):
        d.qfrc_applied[:]=0
        if k*m.opt.timestep<1:
            d.qfrc_applied[qv]=np.clip(4000*(maximum-d.qpos[qa])-150*d.qvel[qv]+d.qfrc_bias[qv],-750,750)
        else:d.qfrc_applied[qv]=-abs(d.qfrc_bias[qv])-100
        d.qfrc_applied[pv]=np.clip(5000*(.016-d.qpos[pa])-20*d.qvel[pv]+600*d.qpos[pa],-50,50)
        mujoco.mj_step(m,d)
    return {'engagement':engagement,'opened':opened,'held':held,'final':float(d.qpos[qa]),'contact_force':contact_force}


@pytest.mark.parametrize('spec',[s for s in SPECS if s['kinematics']['stop']=='prop_arm'],ids=lambda s:s['id'])
def test_native_lift_pin_carries_load_and_force_releases(exports,spec):
    m,d,desc=load(exports,spec);result=cycle(m,d,desc['meta'])
    assert result['engagement']<.003,result
    assert result['opened']>math.radians(spec['kinematics']['max_open_deg'])-.02,result
    assert result['opened']-result['held']<math.radians(1.),result
    assert result['contact_force']>100,result
    assert result['held']-result['final']>math.radians(30),result


def test_missing_physical_pin_cannot_pass_hold_open(exports):
    spec=next(s for s in SPECS if s['id']=='db0360_hatch_floor')
    m,d,desc=load(exports,spec);result=cycle(m,d,desc['meta'],disable_pin=True)
    assert result['contact_force']==0
    assert result['opened']-result['held']>math.radians(20)


def test_closed_solid_rod_blocks_pin_until_real_slot_arrives(exports):
    spec=next(s for s in SPECS if s['id']=='db0360_hatch_floor')
    m,d,desc=load(exports,spec)
    for _ in range(round(.6/m.opt.timestep)):mujoco.mj_step(m,d)
    p=m.joint('hatch_stay_release').id
    assert d.qpos[m.jnt_qposadr[p]]>.012
    assert abs(d.qpos[m.jnt_qposadr[m.joint('hatch_hinge').id]])<.01


@pytest.mark.parametrize('spec', SPECS, ids=lambda s:s['id'])
def test_ring_pivot_and_release_knob_have_real_connected_stock(exports, spec):
    m,d,desc=load(exports,spec);mujoco.mj_forward(m,d)
    if desc['meta']['hatch_hand_access']['opening_contact']=='grip_ring':
        pin=m.geom('ring_pivot_pin').id
        ears=[g for g in range(m.ngeom) if m.geom(g).name.startswith('ring_bearing_')]
        assert len(ears)==12
        for angle in np.linspace(0,math.pi/2,13):
            d.qpos[m.jnt_qposadr[m.joint('ring_hinge').id]]=angle
            mujoco.mj_forward(m,d)
            gaps=[mujoco.mj_geomDistance(m,d,pin,g,.03,None) for g in ears]
            assert min(gaps)==pytest.approx(.0005,abs=2e-6)
        for side in (-1,1):
            assert mujoco.mj_geomDistance(m,d,pin,m.geom(f'ring_side_{side}').id,.03,None)<0
    support=desc['meta'].get('hatch_support')
    if support and support.get('support_release_joint'):
        assert mujoco.mj_geomDistance(m,d,m.geom('stay_lock_pin').id,
                                     m.geom('stay_release_knob').id,.03,None)<0
