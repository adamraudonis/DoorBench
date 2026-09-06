"""Native input-only transmission checks; no prescribed poses during dynamics."""
import math
import mujoco
import numpy as np
import pytest

from doorbench.build import build_model,write_hardware_meshes
from doorbench.export.mjcf import write_mjcf
from doorbench.geometry.marine_linkage import resolve_marine_configuration
from doorbench.mass_reconciliation import reconcile_moving_mass
from doorbench.marine_dog_qa import run_marine_dog_qa
from doorbench.physics import derive
from doorbench.spec import generate_all

SPECS=[s for s in generate_all() if s['family']=='ship_watertight']
WHEELS=[s for s in SPECS if s['kinematics'].get('wheel_dogging')]


@pytest.fixture(scope='module')
def fixtures(tmp_path_factory):
    root=tmp_path_factory.mktemp('marine-native');out={}
    for s in SPECS:
        phys=derive(s);ir=build_model(s,phys);path=root/s['id'];path.mkdir()
        write_hardware_meshes(ir,str(root/'hardware'));write_mjcf(ir,str(path),mesh_dir_rel='../hardware')
        out[s['id']]=(ir,phys,path)
    return out


def _loops(m,d,names):
    errors=[]
    for name in names:
        e=m.equality(name).id;a,b=m.eq_obj1id[e],m.eq_obj2id[e]
        errors.append(np.linalg.norm(d.xpos[a]+d.xmat[a].reshape(3,3)@m.eq_data[e,:3]
            -d.xpos[b]-d.xmat[b].reshape(3,3)@m.eq_data[e,3:6]))
    return max(errors,default=0.)


@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_native_mass_replaces_only_operator_allowance_and_locked_dogs_carry_load(fixtures,spec):
    ir,phys,path=fixtures[spec['id']]
    row=phys['mass']['per_body'][0]
    assert row['hardware_parts']['operator']==0
    assert row['catalogue_operator_replaced_kg']==(5 if spec in WHEELS else 1.2)
    assert row['hardware_parts']['hinges_half']==1.5
    assert row['hardware_parts']['warning_placard']==.1
    before=phys['mass']['total_kg'];reconcile_moving_mass(ir,phys)
    assert phys['mass']['total_kg']==pytest.approx(before,abs=1e-10)
    m=mujoco.MjModel.from_xml_path(str(path/'door.xml'));d=mujoco.MjData(m)
    assert m.body_mass.sum()==pytest.approx(before,abs=.001)
    j=m.joint('leaf_hinge').id;a=m.jnt_qposadr[j];v=m.jnt_dofadr[j]
    peak=depth=0.;cleat_load=False
    for _ in range(round(1/m.opt.timestep)):
        d.qfrc_applied[:]=0;d.qfrc_applied[v]=80;mujoco.mj_step(m,d)
        peak=max(peak,abs(d.qpos[a]));depth=max(depth,max((-c.dist for c in d.contact),default=0.))
        for k,c in enumerate(d.contact):
            if not any(m.geom(g).name.startswith('cleat_') for g in (c.geom1,c.geom2)):continue
            force=np.zeros(6);mujoco.mj_contactForce(m,d,k,force)
            cleat_load |= force[0]>1
    assert cleat_load and peak<.01 and depth<.001
    assert not any(w.number for w in d.warning)


@pytest.mark.parametrize('spec',WHEELS,ids=lambda s:s['id'])
def test_exact_inspection_branch_stays_nonsingular_and_closes_real_pins(fixtures,spec):
    ir,_,path=fixtures[spec['id']];row=ir.meta['marine_dog_linkage']
    assert not any(e.name.startswith('wheel_dog_') for e in ir.equalities)
    assert len(row['connect_equalities']) == spec['kinematics']['dogs'] == 4
    assert set(ir.meta['dog_joints']) == set(row['dog_joints'])
    # A wheel-operated linkage must not also receive independent lever grips
    # and their over-centre springs in the connecting-rod sweep.
    assert not ir.meta.get('marine_dog_retention')
    assert not any('lever' in g.name for b in ir.bodies
                   if b.name in [f'dog_{k}' for k in range(4)] for g in b.geoms)
    for tier in ('full','simple','minimal'):
        m=mujoco.MjModel.from_xml_path(str(path/('door.xml' if tier=='full' else f'door_{tier}.xml')))
        d=mujoco.MjData(m);a=m.jnt_qposadr[m.joint(row['input_joint']).id]
        for q in np.linspace(*row['input_range_rad'],91):
            d.qpos[:]=m.qpos0;d.qpos[a]=q;resolve_marine_configuration(m,d.qpos,ir.meta)
            mujoco.mj_forward(m,d)
            assert _loops(m,d,row['connect_equalities'])<1e-9
            assert max((-c.dist for c in d.contact),default=0.)<.001
        for crank in row['cranks']:
            delta=np.asarray(crank['rod_vector_m'])[[0,2]];delta/=np.linalg.norm(delta)
            p=np.asarray(crank['crank_offset_xz_m'])/.035
            for angle in np.linspace(0,math.pi/2,91):
                # Cross-product magnitude is the crank moment arm relative
                # to the rod: never use a zero-leverage toggle as a shortcut.
                theta=-ir.meta['u']*angle
                rotated=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])@p
                assert abs(float(delta[0]*rotated[1]-delta[1]*rotated[0]))>.70


def _cycle(m,row,*,return_cycle=True,pull_leaf_after=False):
    d=mujoco.MjData(m);j=m.joint(row['input_joint']).id;a=m.jnt_qposadr[j];v=m.jnt_dofadr[j]
    site=m.site('wheel_grip_n').id;body=m.site_bodyid[site];maximum=row['input_range_rad'][1]
    report={'max_force_N':0.,'max_depth_m':0.,'max_loop_m':0.,'max_gear_rad':0.,'released_dogs':None}
    for step in range(round((12 if return_cycle else 6)/m.opt.timestep)):
        mujoco.mj_forward(m,d);time=d.time;opening=time<6
        f=min(1,max(0,(time if opening else time-6)/4));smooth=f**3*(10-15*f+6*f*f)
        target=maximum*(smooth if opening else 1-smooth)
        tau=np.clip(12*(target-d.qpos[a])-1.5*d.qvel[v],-21.6,21.6)
        tangent=np.cross(d.xaxis[j],d.site_xpos[site]-d.xanchor[j])
        force=tangent*tau/np.dot(tangent,tangent)
        assert np.linalg.norm(force)<=120+1e-9
        d.qfrc_applied[:]=0
        mujoco.mj_applyFT(m,d,force,np.zeros(3),d.site_xpos[site],body,d.qfrc_applied)
        # The physical wheel/shaft is the only directly driven mechanism.
        for name in [row['output_joint'],*row['dog_joints'],*row['rod_joints']]:
            assert abs(d.qfrc_applied[m.jnt_dofadr[m.joint(name).id]])<1e-12
        mujoco.mj_step(m,d)
        report['max_force_N']=max(report['max_force_N'],float(np.linalg.norm(force)))
        report['max_depth_m']=max(report['max_depth_m'],max((-c.dist for c in d.contact),default=0.))
        active=[n for n in row['connect_equalities'] if m.eq_active0[m.equality(n).id]]
        report['max_loop_m']=max(report['max_loop_m'],_loops(m,d,active))
        report['max_gear_rad']=max(report['max_gear_rad'],abs(d.qpos[m.jnt_qposadr[m.joint(row['output_joint']).id]]-d.qpos[a]/6))
        if time<6:report['released_dogs']=[float(d.qpos[m.jnt_qposadr[m.joint(n).id]]) for n in row['dog_joints']]
    report['final_dogs']=[float(d.qpos[m.jnt_qposadr[m.joint(n).id]]) for n in row['dog_joints']]
    if pull_leaf_after:
        j=m.joint('leaf_hinge').id;peak=0.
        for _ in range(round(1/m.opt.timestep)):
            d.qfrc_applied[:]=0;d.qfrc_applied[m.jnt_dofadr[j]]=80
            mujoco.mj_step(m,d);peak=max(peak,abs(float(d.qpos[m.jnt_qposadr[j]])))
        report['forced_leaf_peak_rad']=peak
    report['warnings']=[int(w.number) for w in d.warning]
    return report


@pytest.mark.parametrize('spec',WHEELS,ids=lambda s:s['id'])
def test_native_wheel_grip_alone_releases_and_returns_all_dogs(fixtures,spec):
    ir,_,path=fixtures[spec['id']];m=mujoco.MjModel.from_xml_path(str(path/'door.xml'))
    report=_cycle(m,ir.meta['marine_dog_linkage'])
    assert min(report['released_dogs'])>1.5
    assert max(abs(q) for q in report['final_dogs'])<.02
    assert report['max_loop_m']<.001 and report['max_gear_rad']<.001
    assert report['max_depth_m']<.001 and not any(report['warnings'])


def test_disconnected_rod_cannot_be_hidden_by_remote_dog_equalities(fixtures):
    ir,_,path=fixtures[WHEELS[0]['id']];row=ir.meta['marine_dog_linkage']
    m=mujoco.MjModel.from_xml_path(str(path/'door.xml'))
    m.eq_active0[m.equality('marine_rod_0_pin').id]=False
    # Removing only the equality leaves a real retained pin inside a real
    # bored rod eye: native contact can still transmit the force. Remove the
    # physical rod as well to represent an actually broken load path.
    removed=m.geom_bodyid==m.body('marine_rod_0').id
    m.geom_contype[removed]=0;m.geom_conaffinity[removed]=0
    report=_cycle(m,row,return_cycle=False,pull_leaf_after=True)
    # Gravity may rotate a disconnected crank part way. It must remain
    # engaged, and the actual resulting lock must still block the leaf.
    assert report['released_dogs'][0]<1.
    assert min(report['released_dogs'][1:])>1.45
    assert report['forced_leaf_peak_rad']<.01


@pytest.mark.parametrize('spec',[s for s in SPECS if s not in WHEELS],ids=lambda s:s['id'])
def test_individual_dogs_release_one_at_a_time_using_the_actual_hand_grips(fixtures,spec):
    ir,_,path=fixtures[spec['id']];m=mujoco.MjModel.from_xml_path(str(path/'door.xml'));d=mujoco.MjData(m)
    peak=0.
    for row in ir.meta['marine_dog_mounts']:
        j=m.joint(row['joint']).id;a=m.jnt_qposadr[j];v=m.jnt_dofadr[j]
        site=m.site(row['body']+'_grip').id;body=m.site_bodyid[site]
        for _ in range(round(2/m.opt.timestep)):
            mujoco.mj_forward(m,d)
            tau=np.clip(50*(1.56-d.qpos[a])-3*d.qvel[v]+m.dof_frictionloss[v],-21.6,21.6)
            tangent=np.cross(d.xaxis[j],d.site_xpos[site]-d.xanchor[j])
            force=tangent*tau/np.dot(tangent,tangent)
            peak=max(peak,float(np.linalg.norm(force)));d.qfrc_applied[:]=0
            mujoco.mj_applyFT(m,d,force,np.zeros(3),d.site_xpos[site],body,d.qfrc_applied)
            mujoco.mj_step(m,d)
        assert d.qpos[a]>1.5
    assert peak<=120+1e-6
    assert all(d.qpos[m.jnt_qposadr[m.joint(r['joint']).id]]>1.45 for r in ir.meta['marine_dog_mounts'])
    assert not any(w.number for w in d.warning)


@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_native_gate_includes_hand_free_retention_at_both_endpoints(fixtures,spec):
    ir,_,path=fixtures[spec['id']]
    report=run_marine_dog_qa(mujoco.MjModel.from_xml_path(str(path/'door.xml')),ir.meta)
    assert report['ok'],report['failures']
    assert len(report['hand_release_holds'])==2
    assert all(row['duration_s']==2 for row in report['hand_release_holds'])
    assert report['peak_force_N']<=120


def test_removed_spring_reproduces_gravity_undogging_without_mass_or_friction_waiver(fixtures):
    ir,_,path=fixtures['db0285_ship_watertight']
    m=mujoco.MjModel.from_xml_path(str(path/'door.xml'));d=mujoco.MjData(m)
    j=m.joint('dog_0_hinge').id;v=m.jnt_dofadr[j];a=m.jnt_qposadr[j]
    assert m.dof_frictionloss[v]==1.5
    tendon=m.tendon('dog_0_retention_spring').id
    mujoco.mj_forward(m,d)
    # Actual gravity exceeds the original bearing friction. An explicitly
    # attached spring supplies the restoring load; no target servo is active.
    assert abs(d.qfrc_bias[v])>2.5
    assert d.qfrc_passive[v]<-3.
    for _ in range(round(2/m.opt.timestep)):mujoco.mj_step(m,d)
    assert abs(d.qpos[a])<.01
    m.tendon_stiffness[tendon]=0;m.tendon_damping[tendon]=0
    for _ in range(round(3/m.opt.timestep)):mujoco.mj_step(m,d)
    assert d.qpos[a]>.5
    # The full gate must retain this physical failure; it may not quietly
    # declare success from the final hand-held pose.
    report=run_marine_dog_qa(m,ir.meta)
    assert not report['ok'] and 'not_all_dogs_returned' in report['failures']
