"""Source-bound physical vault transmissions; no released asset mutation."""
import copy
import json
import math
from unittest.mock import patch
import mujoco
import numpy as np
import pytest
from doorbench.build import build_model,write_hardware_meshes
from doorbench.export.mjcf import write_mjcf
from doorbench.geometry import hinged
from doorbench.geometry.vault_hardware import rebuild_vault_hardware,resolve_vault_configuration,_account_prepared_stock
from doorbench.mass_reconciliation import reconcile_moving_mass
from doorbench.physics import derive
from doorbench.spec import generate_all
from doorbench.clearance import gate_model
from doorbench.vault_hardware_qa import run_vault_native_qa

SPECS=[s for s in generate_all() if s['family'] in ('vault','blast')]
TIERS=('full','simple','minimal')

@pytest.fixture(scope='module')
def fixtures(tmp_path_factory):
    root=tmp_path_factory.mktemp('vault-native');out={};old=hinged.build_vault
    def prototype(spec,phys,model):
        leaf=old(spec,phys,model)
        if not model.meta.get('vault_boltwork'):rebuild_vault_hardware(model,spec,phys)
        return leaf
    with patch.object(hinged,'build_vault',prototype):
        for s in SPECS:
            phys=derive(s);original=copy.deepcopy(phys);ir=build_model(s,phys);p=root/s['id'];p.mkdir()
            write_hardware_meshes(ir,str(root/'hardware'));write_mjcf(ir,str(p),mesh_dir_rel='../hardware')
            (p/'model.json').write_text(json.dumps(ir.to_dict(),indent=2))
            (p/'spec.json').write_text(json.dumps(s,indent=2))
            out[s['id']]=(ir,phys,original,p)
    assert len(out)==14
    return out

def native(path,tier):
    return mujoco.MjModel.from_xml_path(str(path/('door.xml' if tier=='full' else f'door_{tier}.xml')))

@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_prepared_material_and_actual_mechanism_BOM_are_not_double_counted(fixtures,spec):
    ir,phys,original,p=fixtures[spec['id']];account=ir.meta['vault_material_accounting'];row=phys['mass']['per_body'][0]
    removed=account['removed_stock_kg'];replaced=sum(account['replaced_catalogue_kg'].values())
    assert removed>0 and removed==pytest.approx(account['removed_geometry_volume_m3']*original['mass']['slab_kg']/(spec['leaf']['width']*spec['leaf']['height']*spec['leaf']['thickness']),abs=1e-8)
    assert row['slab_kg']==pytest.approx(original['mass']['slab_kg']-removed,abs=1e-8)
    assert row['hardware_parts']['operator']==row['hardware_parts']['hinges_half']==0
    added=phys['mass']['geometry_backed_mechanisms_kg']
    assert phys['mass']['total_kg']==pytest.approx(original['mass']['total_kg']-removed-replaced+added,abs=1e-7)
    assert phys['per_body_dynamics']['leaf']['mass']['total_kg']==pytest.approx(original['mass']['total_kg']-removed-replaced,abs=1e-8)
    before=phys['mass']['total_kg'];reconcile_moving_mass(ir,phys);assert phys['mass']['total_kg']==pytest.approx(before,abs=1e-9)
    for body in ir.bodies:
        if body.name in ir.meta['mechanism_mass_bodies']:
            assert body.mass_override is None
            assert body.inertial('full')[0]==pytest.approx(sum(g.mass() for g in body.geoms),abs=1e-9)
    for tier in TIERS:
        m=native(p,tier)
        assert m.body_mass.sum()==pytest.approx(before,abs=.001)
        j=m.joint(ir.meta['primary_joint']).id
        assert m.dof_frictionloss[m.jnt_dofadr[j]]==pytest.approx(original['hinge']['coulomb_torque_Nm']+.5*original['hinge']['stick_torque_Nm'],abs=.001)
    with pytest.raises(ValueError,match='fresh'):_account_prepared_stock(ir,phys,spec)

@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_surface_grips_are_reachable_and_have_actual_leaf_or_operator_moments(fixtures,spec):
    ir,_,_,p=fixtures[spec['id']];m=native(p,'full');d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    groups=ir.meta['vault_boltwork']['groups'];bindings=[]
    for row in groups:
        b=m.body(row['operator_body']).id
        bindings.extend((i,row['operator_joint']) for i in range(m.nsite) if m.site_bodyid[i]==b and 'grip' in m.site(i).name)
    bindings.extend((m.site(n).id,ir.meta['primary_joint']) for n in ('vault_leaf_grip_n','vault_leaf_grip_p'))
    for sid,joint in bindings:
        normal=d.site_xmat[sid].reshape(3,3)[:,2];point=d.site_xpos[sid];gid=np.array([-1],np.int32)
        dist=mujoco.mj_ray(m,d,point+.15*normal,-normal,None,1,-1,gid)
        assert dist==pytest.approx(.15,abs=2e-6),(spec['id'],m.site(sid).name,dist,m.geom(gid[0]).name)
        assert any(x in m.geom(gid[0]).name for x in ('wheel_rim','lever_','vault_pull_bar'))
        j=m.joint(joint).id;moment=np.linalg.norm(np.cross(d.xaxis[j],point-d.xanchor[j]))
        assert moment>(.6 if joint==ir.meta['primary_joint'] else .19)

@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_all_tier_real_pin_loops_and_full_operator_sweep_clearance(fixtures,spec):
    ir,_,_,p=fixtures[spec['id']]
    assert not any(e.name.startswith(('wheel_bolt_','lever_bolt_')) for e in ir.equalities)
    for tier in TIERS:
        file=p/('door.xml' if tier=='full' else f'door_{tier}.xml');m=gate_model(str(file));d=mujoco.MjData(m)
        for f in np.linspace(0,1,49):
            d.qpos[:]=m.qpos0
            for row in ir.meta['vault_boltwork']['groups']:
                d.qpos[m.jnt_qposadr[m.joint(row['operator_joint']).id]]=row['operator_nominal_range'][1]*f
            resolve_vault_configuration(m,d.qpos,ir.meta);mujoco.mj_forward(m,d)
            equal=d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            assert max(abs(d.efc_pos[equal]),default=0.)<1e-6
            bad=[(-c.dist,m.geom(c.geom1).name,m.geom(c.geom2).name) for c in d.contact if c.dist<-.0005]
            assert not bad,(spec['id'],tier,f,bad[:5])

@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
@pytest.mark.parametrize('tier',TIERS)
def test_native_release_load_rethrow_and_removed_transmission_controls(fixtures,spec,tier):
    ir,_,_,p=fixtures[spec['id']];r=run_vault_native_qa(native(p,tier),ir.meta)
    (p/f'native-proof-{tier}.json').write_text(json.dumps(r,indent=2))
    assert r['ok'],[(x['check'],x['phase']) for x in r['failures']]

@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_actual_journals_and_mounts_are_connected_with_open_running_bores(fixtures,spec):
    from doorbench.vault_hardware_qa import run_vault_mount_qa
    ir,_,_,p=fixtures[spec['id']]
    for tier in TIERS:
        report=run_vault_mount_qa(native(p,tier),ir.meta)
        (p/f'mount-proof-{tier}.json').write_text(json.dumps(report,indent=2))
        assert report['ok'],report['failures']


def test_detached_frame_block_and_misplaced_bolt_guide_cannot_pass(fixtures):
    from doorbench.vault_hardware_qa import run_vault_mount_qa
    ir,_,_,p=fixtures['db0179_vault'];m=native(p,'full')
    m.geom_pos[m.geom(ir.meta['vault_crane_journals'][0]['frame_blocks'][0]).id,2]+=1.
    report=run_vault_mount_qa(m,ir.meta)
    assert not report['ok'] and any(x['check']=='journal_in_frame_block' for x in report['failures'])
    m=native(p,'full');r=ir.meta['vault_boltwork']['groups'][0]
    for name in r['guide_geoms']:
        if name.startswith(r['bolt_geoms'][0]+'_guide_0_'):m.geom_pos[m.geom(name).id,1]+=.1
    report=run_vault_mount_qa(m,ir.meta)
    assert not report['ok'] and any(x['check']=='bolt_guide_radial_bore' for x in report['failures'])

@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_fully_retracted_leaf_clears_its_entire_authored_range(fixtures,spec):
    ir,_,_,p=fixtures[spec['id']];m=gate_model(str(p/'door.xml'));d=mujoco.MjData(m);j=m.joint(ir.meta['primary_joint']).id
    for angle in np.linspace(*ir.meta['vault_primary_nominal_range'],361):
        d.qpos[:]=m.qpos0;d.qpos[m.jnt_qposadr[j]]=angle
        for row in ir.meta['vault_boltwork']['groups']:
            d.qpos[m.jnt_qposadr[m.joint(row['operator_joint']).id]]=row['operator_nominal_range'][1]
        resolve_vault_configuration(m,d.qpos,ir.meta);mujoco.mj_forward(m,d)
        bad=[(-c.dist,m.geom(c.geom1).name,m.geom(c.geom2).name) for c in d.contact if c.dist<-.0005]
        assert not bad,(spec['id'],angle,bad[:5])

@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_closed_bolts_arrest_before_one_centiradian_and_sweep_stops_there(fixtures,spec):
    from doorbench.vault_hardware_qa import first_vault_contact_angle
    from doorbench.clearance import Clearance
    ir,_,_,p=fixtures[spec['id']];m=native(p,'full');r=first_vault_contact_angle(m,ir.meta)
    assert r['ok'] and 0<r['angle_rad']<.01,r
    gate=Clearance(str(p));result=gate.run();assert result['ok'],result['failures']
    lo,hi=gate._latched_range(ir.meta['primary_joint'],m.qpos0,0,3.)
    assert lo==0 and hi==pytest.approx(r['angle_rad'],abs=1e-9)
    expected={g['operator_joint'] for g in ir.meta['vault_boltwork']['groups']}
    assert set(result['mech_joints'])==expected
    for row in ir.meta['vault_boltwork']['groups']:
        assert gate._operating_range(row['operator_joint'],-100,100)==tuple(row['operator_nominal_range'])
    # Explicitly disabling native collision cannot be reclassified as a
    # functioning lock merely because its visual shape still has a distance.
    m.geom_contype[m.geom(ir.meta['vault_boltwork']['groups'][0]['bolt_geoms'][0]).id]=0
    with pytest.raises(ValueError,match='native collider'):first_vault_contact_angle(m,ir.meta)

@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_runtime_contact_selection_keeps_leaf_pull_separate_from_release_inputs(fixtures,spec):
    from types import SimpleNamespace
    from doorbench.benchmark.interactions import ContactSites
    ir,_,_,p=fixtures[spec['id']];m=native(p,'full');d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    env=SimpleNamespace(m=m,d=d,mj=mujoco,spec=spec,model_json=ir.to_dict(),meta=ir.meta)
    contacts=ContactSites(env)
    assert m.site(contacts.select(ir.meta['primary_joint'])).name=='vault_leaf_grip_n'
    for row in ir.meta['vault_boltwork']['groups']:
        sid=contacts.select(row['operator_joint']);assert sid is not None
        assert m.site_bodyid[sid]==m.body(row['operator_body']).id
        assert d.site_xmat[sid].reshape(3,3)[1,2]<-.999


def test_a_floating_closing_rebate_is_not_a_valid_frame_load_path(fixtures):
    from doorbench.vault_hardware_qa import run_vault_mount_qa
    ir,_,_,p=fixtures['db0124_vault'];m=native(p,'full')
    g=m.geom(ir.meta['vault_closing_stops'][0]).id;m.geom_pos[g,0]-=ir.meta['u']*.1
    report=run_vault_mount_qa(m,ir.meta)
    assert not report['ok'] and any(r['check']=='closing_rebate_to_jamb' for r in report['failures'])
