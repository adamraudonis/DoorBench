"""Physical hand-chain topology and load-bearing installation regression."""
import json
import copy
from pathlib import Path
import mujoco
import numpy as np
import pytest
from doorbench.spec import generate_all
from doorbench import physics
from doorbench.build import build_model,write_hardware_meshes
from doorbench.export.mjcf import write_mjcf
from doorbench.geometry.rollup_hoist import add_chain_hoist,hand_chain_dimensions
from doorbench.mass_reconciliation import reconcile_moving_mass
from doorbench.rollup_hoist import compile_hoist,hoist_control,prepare_hoist_open

SPECS=[s for s in generate_all()if s['family']=='rollup' and s['kinematics']['opener']=='chain_hoist']


def make_fixture(spec):
    phys=physics.derive(spec);model=build_model(spec,phys)
    return model,phys


@pytest.fixture(scope='module')
def fixtures(tmp_path_factory):
    root=tmp_path_factory.mktemp('hoist');rows={}
    for spec in SPECS:
        model,phys=make_fixture(spec);p=root/spec['id'];p.mkdir()
        write_hardware_meshes(model,str(root/'hardware'));write_mjcf(model,str(p),mesh_dir_rel='../hardware')
        rows[spec['id']]=(model,phys,p)
    return rows


@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_every_hoist_has_a_real_free_material_loop_and_clear_closed_bearings(fixtures,spec):
    model,phys,path=fixtures[spec['id']];hoist=model.meta['rollup_hoist'];params=hoist['parameters']
    assert model.meta['rollup_curtain']['drive']['mode']=='manual_chain'
    assert model.meta['rollup_curtain']['drive']['chain_hoist_supported'] is True
    assert not any(row['component']=='chain_hoist'for row in model.meta.get('mechanical_incomplete',[]))
    points=np.array(params['points_yz']);lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    np.testing.assert_allclose(lengths,params['pitch_m'],atol=1e-12)
    np.testing.assert_allclose(points[0],points[-1],atol=1e-12)
    assert params['idler_z_m']-params['pitch_radius_m']>.3
    assert model.body('hoist_chain_link_0').parent is None
    assert model.body('hoist_chain_link_0').inertial()[0]>.02
    assert not any('chain_y' in b.name or 'chain_z' in b.name for b in model.bodies)
    assert set(hoist['material_bodies'])<=set(model.meta['mechanism_mass_bodies'])
    gear=next(e for e in model.equalities if e.name=='hoist_gear_ratio')
    assert gear.a=='curtain_drum_hinge' and gear.b=='hoist_input_hinge' and gear.polycoeff==(0,-.25,0,0,0)
    assert hoist['opening_pull_strand_y_sign']==-hoist['closing_pull_strand_y_sign']==-1
    for tier in ('full','simple','minimal'):
        m=mujoco.MjModel.from_xml_path(str(path/('door.xml'if tier=='full'else f'door_{tier}.xml')));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        root=m.joint('hoist_chain_free').id
        assert m.jnt_type[root]==mujoco.mjtJoint.mjJNT_FREE
        assert m.nq==m.nv+1 and m.nq==m.njnt+6
        assert m.body_mass[m.body('hoist_chain_link_0').id]>.02
        assert np.all(m.dof_armature[m.jnt_dofadr[root]:m.jnt_dofadr[root]+6]==0)
        assert m.body_mass.sum()==pytest.approx(phys['mass']['total_kg'],abs=.001)
        bad=[(m.geom(c.geom1).name,m.geom(c.geom2).name,float(c.dist))for c in d.contact if c.dist<-.00005]
        assert not bad
        assert np.linalg.norm(d.site_xpos[m.site('hoist_chain_loop_end').id]-d.site_xpos[m.site('hoist_chain_node_0').id])<.00005
        for prefix in ('hoist_upper','hoist_lower'):
            # Each bearing reaches a real static bracket; world anchoring
            # alone must not conceal a floating inboard support.
            for side,post in ((-1,'inboard_bearing_post'),(1,'bearing_bridge')):
                bearings=[i for i in range(m.ngeom)if m.geom(i).name.startswith(f'{prefix}_bearing_{side}_')]
                support=m.geom(f'{prefix}_{post}').id
                assert bearings
                assert min(mujoco.mj_geomDistance(m,d,g,support,1.,None)for g in bearings)<1e-6


def test_new_hoist_geometric_mass_is_preserved_by_reconciliation(fixtures):
    model,phys,_=fixtures['db0419_rollup'];before=phys['mass']['total_kg']
    actual={n:model.body(n).inertial()[0]for n in model.meta['mechanism_mass_bodies']if n.startswith('hoist_')}
    assert sum(actual.values())>5.
    reconcile_moving_mass(model,phys)
    assert phys['mass']['total_kg']==pytest.approx(before)
    assert actual=={n:model.body(n).inertial()[0]for n in actual}


def test_control_selects_real_opposite_strands_and_never_writes_native_state(fixtures):
    model,_,path=fixtures['db0419_rollup'];m=mujoco.MjModel.from_xml_path(str(path/'door.xml'))
    d=mujoco.MjData(m);mujoco.mj_forward(m,d);rules=compile_hoist(m,model.meta)
    before={n:getattr(d,n).copy()for n in ('qpos','qvel','qfrc_applied','qfrc_passive','ctrl')}
    for opening in (False,True):
        action=hoist_control(m,d,rules,opening=opening,elapsed_s=2.)
        sid=m.site(action['site']).id
        assert sid==action['site_id'] and m.site_bodyid[sid]==action['body_id']
        assert .55<d.site_xpos[sid,2]<1.7
        assert (-1 if opening else 1)*(d.site_xpos[sid,1]-rules.wheel_y)>.06
        assert np.linalg.norm(action['force_N'])<=120
        if opening:assert action['force_N'][2]<-100
        zero=hoist_control(m,d,rules,opening=opening,elapsed_s=0.)
        assert zero['force_N']==[0.,0.,0.]
    for name,value in before.items():assert np.array_equal(getattr(d,name),value)
    # Extreme link speed is still bounded, and braking acts at that same
    # material grip rather than being a fictitious opposing shaft torque.
    d.qvel[:]=20.
    assert np.linalg.norm(hoist_control(m,d,rules,elapsed_s=2.)['force_N'])<=120
    with pytest.raises(ValueError):hoist_control(m,d,rules,elapsed_s=float('nan'))


def test_wrong_native_ratio_and_detached_grip_fail_binding(fixtures):
    model,_,path=fixtures['db0419_rollup'];m=mujoco.MjModel.from_xml_path(str(path/'door.xml'))
    meta=copy.deepcopy(model.meta);meta['rollup_hoist']['material_bodies'][0]='hoist_input'
    with pytest.raises(ValueError,match='attached'):compile_hoist(m,meta)
    m.eq_data[m.equality('hoist_gear_ratio').id,1]=-.5
    with pytest.raises(ValueError,match='constraint'):compile_hoist(m,model.meta)


def test_partial_native_initializer_is_not_promoted_and_cache_is_model_bound(fixtures):
    model,_,path=fixtures['db0419_rollup'];m=mujoco.MjModel.from_xml_path(str(path/'door.xml'))
    initial=m.qpos0.copy();masses=m.body_mass.copy();friction=m.dof_frictionloss.copy()
    result=prepare_hoist_open(m,model.meta,time_limit_s=.003)
    assert not result['ok'] and result['reason']=='native_goal_not_reached'
    assert 'qpos' not in result and 'qvel' not in result
    assert result['elapsed_native_s']==pytest.approx(.003)
    assert prepare_hoist_open(m,model.meta,time_limit_s=.003)['cache_hit']
    changed=initial.copy();changed[m.jnt_qposadr[m.joint('hoist_chain_free').id]]+=1e-6
    assert not prepare_hoist_open(m,model.meta,changed,time_limit_s=.003)['cache_hit']
    assert np.array_equal(m.qpos0,initial) and np.array_equal(m.body_mass,masses) and np.array_equal(m.dof_frictionloss,friction)
    invalid=initial.copy();invalid[m.jnt_qposadr[m.joint('hoist_chain_free').id]+3:m.jnt_qposadr[m.joint('hoist_chain_free').id]+7]=0
    with pytest.raises(ValueError,match='quaternion'):prepare_hoist_open(m,model.meta,invalid,time_limit_s=.003)


def test_initializer_refuses_unrelated_robot_without_advancing_it(fixtures):
    model,_,path=fixtures['db0419_rollup']
    xml=(path/'door.xml').read_text().replace('</worldbody>',
        '<body name="unrelated_robot" pos="0 0 10"><freejoint name="robot_free"/><inertial pos="0 0 0" mass="10" diaginertia=".04 .04 .04"/><geom type="sphere" size=".1" mass="10"/></body></worldbody>')
    extra=path/'robot-fixture.xml';extra.write_text(xml)
    m=mujoco.MjModel.from_xml_path(str(extra));before=m.qpos0.copy()
    result=prepare_hoist_open(m,model.meta,time_limit_s=.003)
    assert not result['ok'] and result['reason']=='additional_dynamic_bodies_require_door_only_initialization'
    assert result['elapsed_native_s']==0 and result['unsupported_joint_names']==['robot_free']
    assert np.array_equal(m.qpos0,before)


def test_actual_chain_pull_transmits_load_and_missing_gears_cannot_lift(fixtures):
    from doorbench.hoist_keeper import compile_keeper,begin_keeper_transition,keeper_transition_action,keeper_open_force
    model,_,path=fixtures['db0419_rollup'];results=[]
    for connected in (True,False):
        m=mujoco.MjModel.from_xml_path(str(path/'door.xml'));rules=compile_hoist(m,model.meta)
        d=mujoco.MjData(m)
        mujoco.mj_forward(m,d);keeper=compile_keeper(m,model.meta)
        transition=begin_keeper_transition(m,d,rules,keeper,mode='release')
        for _ in range(round(3./m.opt.timestep)):
            mujoco.mj_forward(m,d)
            release=keeper_transition_action(m,d,rules,keeper,transition);transition=release['next_state']
            assert not release['failed'],release['reason']
            if release['done']:break
            d.qfrc_applied[:]=0
            for name,force in release['site_forces'].items():
                sid=m.site(name).id
                mujoco.mj_applyFT(m,d,np.array(force),np.zeros(3),d.site_xpos[sid],m.site_bodyid[sid],d.qfrc_applied)
            mujoco.mj_step(m,d)
        assert transition['done'] and d.qpos[keeper.qpos]>.078
        start=float(d.time)
        if not connected:d.eq_active[m.equality('hoist_gear_ratio').id]=False
        depth=0.;loop=0.;gear=0.;sites=set()
        for _ in range(round(4./m.opt.timestep)):
            mujoco.mj_forward(m,d);action=hoist_control(m,d,rules,elapsed_s=float(d.time)-start);sid=action['site_id']
            assert np.linalg.norm(action['force_N'])<=120.
            d.qfrc_applied[:]=0
            mujoco.mj_applyFT(m,d,np.array(action['force_N']),np.zeros(3),d.site_xpos[sid],action['body_id'],d.qfrc_applied)
            for name,force in keeper_open_force(m,d,keeper).items():
                pin_site=m.site(name).id
                mujoco.mj_applyFT(m,d,np.array(force),np.zeros(3),d.site_xpos[pin_site],m.site_bodyid[pin_site],d.qfrc_applied)
            mujoco.mj_step(m,d);sites.add(sid)
            depth=max(depth,max((-float(c.dist)for c in d.contact),default=0.))
            loop=max(loop,float(np.linalg.norm(d.site_xpos[rules.loop_start]-d.site_xpos[rules.loop_end])))
            gear=max(gear,abs(float(d.qpos[rules.output_qpos]-rules.ratio*d.qpos[rules.input_qpos])))
        mujoco.mj_forward(m,d);height=float(d.site_xpos[rules.bottom_site,2]);results.append(height)
        assert depth<.001 and loop<.001 and not any(w.number for w in d.warning)
        assert len(sites)>10
        if connected:assert gear<.005 and height>.3
        else:assert gear>1. and height<.04
    assert results[0]-results[1]>.3
