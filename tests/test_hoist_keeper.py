"""Positive load path, original masses, spring force and actual-site handoff."""
from dataclasses import replace
import copy
import mujoco
import numpy as np
import pytest
from doorbench.spec import generate_all
from doorbench.build import build_model,write_hardware_meshes
from doorbench.physics import derive
from doorbench.mass_reconciliation import reconcile_moving_mass
from doorbench.geometry.hoist_keeper import add_chain_keeper
from doorbench.export.mjcf import write_mjcf
from doorbench.rollup_hoist import compile_hoist,hoist_control
from doorbench.hoist_keeper import compile_keeper,keeper_site_force,keeper_open_force,begin_keeper_transition,keeper_transition_action,_hand_rules
from doorbench.native_warnings import capture_native_warnings

SPECS=[s for s in generate_all() if s['family']=='rollup' and s['kinematics']['opener']=='chain_hoist']


@pytest.fixture(autouse=True)
def no_uncounted_native_messages():
    with capture_native_warnings() as messages:
        yield
    assert not messages,messages


@pytest.fixture(scope='module')
def fixtures(tmp_path_factory):
    root=tmp_path_factory.mktemp('positive-keeper');rows={}
    assert len(SPECS)==6
    for s in SPECS:
        from doorbench.geometry import hoist_keeper
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(hoist_keeper,'add_chain_keeper',lambda model,spec:None)
            original=build_model(s,derive(s))
        before={b.name:b.inertial()[0] for b in original.bodies}
        phys=derive(s);ir=build_model(s,phys)
        assert all(ir.body(n).inertial()[0]==pytest.approx(mass,abs=1e-12) for n,mass in before.items())
        path=root/s['id'];path.mkdir();write_hardware_meshes(ir,str(root/'hardware'));write_mjcf(ir,str(path),mesh_dir_rel='../hardware')
        rows[s['id']]=(ir,phys,path)
    return rows


@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_six_real_captured_pins_preserve_original_mass_and_clear_both_states(fixtures,spec):
    ir,phys,path=fixtures[spec['id']];row=ir.meta['rollup_hoist']['keeper']
    assert row['body'] in ir.meta['mechanism_mass_bodies']
    assert ir.body(row['body']).inertial()[0]>.2
    assert not any(e.a==row['joint'] or e.b==row['joint'] for e in ir.equalities)
    assert ir.body(row['body']).joint.frictionloss==.2
    for tier in ('full','simple','minimal'):
        m=mujoco.MjModel.from_xml_path(str(path/('door.xml' if tier=='full' else f'door_{tier}.xml')));d=mujoco.MjData(m);k=compile_keeper(m,ir.meta)
        assert m.body_mass.sum()==pytest.approx(phys['mass']['total_kg'],abs=.001)
        for q,expected in ((0.,-2.),(.08,-18.)):
            d.qpos[k.qpos]=q;mujoco.mj_forward(m,d)
            assert d.qfrc_passive[k.dof]==pytest.approx(expected,abs=1e-8)
            assert max((-float(c.dist) for c in d.contact),default=0.)<.00005
        # A pin crosses the central chain gap and enters a bored receiver;
        # it cannot rely on an unsupported cantilever tip or guide friction.
        pin=ir.body(row['body']).geoms[0]
        assert pin.pos[1]-pin.size[1]<-.033 and pin.size[0]==.005
        assert any(g.name.startswith('hoist_keeper_receiver_') for g in ir.body(row['fixed_body']).geoms)
        assert all(g.friction==(.12,.001,.0001) for g in ir.body(row['fixed_body']).geoms)


def test_keeper_rejects_absent_pin_spring_wrong_input_and_duplicate_install(fixtures):
    ir,_,path=fixtures['db0419_rollup'];m=mujoco.MjModel.from_xml_path(str(path/'door.xml'))
    with pytest.raises(ValueError,match='already'):add_chain_keeper(ir,SPECS[2])
    meta=copy.deepcopy(ir.meta);meta['rollup_hoist']['keeper']['grip_site']='hoist_chain_grip_0'
    with pytest.raises(ValueError,match='attached grip'):compile_keeper(m,meta)
    k=compile_keeper(m,ir.meta);m.geom_contype[k.pin_geom]=0
    with pytest.raises(ValueError,match='active native contact'):compile_keeper(m,ir.meta)
    m.geom_contype[k.pin_geom]=1;m.tendon_stiffness[m.tendon('hoist_keeper_return').id]=0
    with pytest.raises(ValueError,match='return spring'):compile_keeper(m,ir.meta)


def test_actual_site_controller_is_read_only_bounded_and_keeps_grip_out_of_housing(fixtures):
    ir,_,path=fixtures['db0419_rollup'];m=mujoco.MjModel.from_xml_path(str(path/'door.xml'));d=mujoco.MjData(m);mujoco.mj_forward(m,d);h=compile_hoist(m,ir.meta);k=compile_keeper(m,ir.meta)
    before={n:getattr(d,n).copy() for n in ('qpos','qvel','qfrc_applied','qfrc_passive','ctrl')}
    s=begin_keeper_transition(m,d,h,k,mode='release');original=copy.deepcopy(s)
    action=keeper_transition_action(m,d,h,k,s)
    assert s==original and not action['done'] and not action['failed']
    assert s['phase']=='settle_on_floor' and s['initial_floor_reaction_N']>100.
    assert action['site_forces']=={}  # First settle the actual floor-supported curtain.
    for name,force in action['site_forces'].items():
        sid=m.site(name).id;assert np.linalg.norm(force)<=120
        if name!=k.grip_name:assert not k.excluded_grip_z[0]<=d.site_xpos[sid,2]<=k.excluded_grip_z[1]
    assert keeper_open_force(m,d,k)[k.grip_name][1]<=120
    for name,value in before.items():np.testing.assert_array_equal(getattr(d,name),value)
    with pytest.raises(ValueError,match='floor-supported'):begin_keeper_transition(m,d,h,k,mode='engage')
    for value in (-.001,.081,float('nan'),True):
        with pytest.raises(ValueError):keeper_site_force(m,d,k,value)
    s['last_time_s']=1.
    with pytest.raises(ValueError,match='backwards'):keeper_transition_action(m,d,h,k,s)


def test_real_positive_pin_arrests_opening_and_removed_pin_cannot(fixtures):
    ir,_,path=fixtures['db0419_rollup'];results=[]
    for connected in (True,False):
        m=mujoco.MjModel.from_xml_path(str(path/'door.xml'));d=mujoco.MjData(m);h=compile_hoist(m,ir.meta);k=compile_keeper(m,ir.meta)
        if not connected:m.geom_contype[k.pin_geom]=m.geom_conaffinity[k.pin_geom]=0
        peak_load=depth=0.
        for _ in range(round(1.5/m.opt.timestep)):
            mujoco.mj_fwdPosition(m,d);mujoco.mj_fwdVelocity(m,d)
            action=hoist_control(m,d,_hand_rules(d,h,k),elapsed_s=float(d.time));sid=action['site_id']
            force=np.array([0.,0.,-100.*min(1.,float(d.time)/.5)])
            d.qfrc_applied[:]=0;mujoco.mj_applyFT(m,d,force,np.zeros(3),d.site_xpos[sid],action['body_id'],d.qfrc_applied);mujoco.mj_step(m,d)
            depth=max(depth,max((-float(c.dist) for c in d.contact),default=0.))
            for ci,c in enumerate(d.contact):
                other=c.geom2 if c.geom1==k.pin_geom else c.geom1 if c.geom2==k.pin_geom else -1
                if other in k.chain_geoms:
                    load=np.zeros(6);mujoco.mj_contactForce(m,d,ci,load);peak_load=max(peak_load,float(load[0]))
        mujoco.mj_forward(m,d);results.append(float(d.site_xpos[h.bottom_site,2]))
        assert depth<.001 and not any(w.number for w in d.warning)
        if connected:assert peak_load>50.
    assert results[0]<.05
    assert results[1]-results[0]>.10
