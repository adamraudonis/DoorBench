"""Sectional panels must pass below an intact lintel using actual native forces."""
import json
import copy
import math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import pytest

from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.geometry.sectional import track_dimensions,track_path,track_progress,inspection_pose,resolve_sectional_configuration
from doorbench.native_warnings import capture_native_warnings

SPECS=[s for s in generate_all() if s['family']=='garage_sectional']


@pytest.fixture(autouse=True)
def no_uncounted_native_solver_messages():
    with capture_native_warnings() as messages:
        yield
    assert not messages,messages


@pytest.fixture(scope='module')
def exports(tmp_path_factory):
    root=tmp_path_factory.mktemp('sectional')
    for spec in SPECS:
        export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('json','mjcf'))
        if spec['lock'].get('engaged'):
            unlocked=copy.deepcopy(spec);unlocked['lock']['engaged']=False
            export_door(unlocked,str(root/'unlocked/doors'),str(root/'unlocked/hardware'),formats=('json','mjcf'))
    return root


def load(root,spec,tier='full'):
    p=root/'doors'/spec['id'];m=mujoco.MjModel.from_xml_path(str(p/('door.xml' if tier=='full' else f'door_{tier}.xml')))
    return m,mujoco.MjData(m),json.loads((p/'model.json').read_text())


def lowest_panel(model,data,description):
    names={g['name'] for b in description['bodies'] if b['semantic']=='leaf' for g in b['geoms'] if g['semantic'] in ('leaf','glass')}
    low=[]
    for name in names:
        try:gid=model.geom(name).id
        except KeyError:continue
        rot=data.geom_xmat[gid].reshape(3,3)
        low.append((data.geom_xpos[gid]+rot@model.geom_aabb[gid,:3])[2]-(np.abs(rot)@model.geom_aabb[gid,3:])[2])
    return min(low)


def astragal_floor_reaction(model,data,description):
    meta=description['meta']['sectional_bottom_seal'];floor=model.geom(meta['floor_geom']).id
    seals={model.geom(name).id for name in meta['contact_geoms']};total=0.
    for k,c in enumerate(data.contact):
        if floor not in (c.geom1,c.geom2) or not seals.intersection((c.geom1,c.geom2)):continue
        force=np.zeros(6);mujoco.mj_contactForce(model,data,k,force)
        world=c.frame.reshape(3,3).T@force[:3]
        total+=float(world[2] if c.geom2 in seals else -world[2])
    return total


@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_closed_door_settles_on_its_actual_bottom_seal_not_coordinate_limit(exports,spec):
    rows=[]
    states=[('source',exports)]
    if spec['lock'].get('engaged'):states.append(('explicitly_unlocked',exports/'unlocked'))
    for state,root,tier in [(state,root,tier) for state,root in states for tier in ('full','simple','minimal')]:
        m,d,desc=load(root,spec,tier);seal=desc['meta']['sectional_bottom_seal']
        assert m.body(seal['body']).mass[0]==pytest.approx(seal['mass_kg'],abs=1e-6)
        assert seal['body'] in desc['meta']['mechanism_mass_bodies']
        q=m.jnt_qposadr[m.joint('door_slide').id];floor=m.geom('floor').id
        mujoco.mj_forward(m,d)
        gap=min(mujoco.mj_geomDistance(m,d,floor,m.geom(name).id,.1,None) for name in seal['contact_geoms'])
        assert gap==pytest.approx(.0005,abs=1e-6)
        for _ in range(round(1./m.opt.timestep)):mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d);reaction=astragal_floor_reaction(m,d,desc)
        assert abs(d.qpos[q])<.003,(spec['id'],tier,d.qpos[q])
        # An engaged side bolt may carry the door before the rubber is loaded.
        # Every source also has an unlocked proof of actual seal support.
        if state=='explicitly_unlocked' or not spec['lock'].get('engaged'):
            assert reaction>5.,(spec['id'],tier,reaction)
        assert not any(d.efc_type[k]==mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT and
            d.efc_id[k]==m.joint('door_slide').id for k in range(d.nefc))
        depth=max((-c.dist for c in d.contact),default=0.)
        assert depth<.001 and not any(w.number for w in d.warning)
        rows.append({'tier':tier,'state':state,'settle_drift_m':float(d.qpos[q]),'upward_seal_floor_reaction_N':reaction,
            'initial_gap_m':gap,'max_final_penetration_m':depth,'seal_mass_kg':seal['mass_kg']})
    (exports/'doors'/spec['id']/'astragal-settle.json').write_text(json.dumps(rows,indent=2)+'\n')


def test_removed_astragal_reproduces_twenty_mm_drop_onto_numerical_limit(exports):
    spec=next(s for s in SPECS if s['id']=='db0806_garage_sectional')
    m,d,desc=load(exports,spec);meta=desc['meta']['sectional_bottom_seal']
    # Remove only the rubber contact while retaining its exact native inertia,
    # all tracks, springs (none in this failed source), locks and force limits.
    for name in meta['contact_geoms']:
        g=m.geom(name).id;m.geom_contype[g]=0;m.geom_conaffinity[g]=0
    for _ in range(round(1./m.opt.timestep)):mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d);j=m.joint('door_slide').id
    assert d.qpos[m.jnt_qposadr[j]]<-.0199
    assert astragal_floor_reaction(m,d,desc)==0.
    loads=[float(d.efc_force[k]) for k in range(d.nefc) if
        d.efc_type[k]==mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT and d.efc_id[k]==j]
    assert loads and max(loads)>500.


@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_all_sections_are_real_panels_behind_intact_header_in_every_tier(exports,spec):
    for tier in ('full','simple','minimal'):
        m,d,desc=load(exports,spec,tier);mechanism=desc['meta']['sectional_track'];path=mechanism['path']
        assert [b['name'] for b in desc['bodies'] if b['semantic']=='leaf']==[f'section_{i}' for i in range(path['panel_count'])]
        header=m.geom('wall_header').id
        assert m.geom_pos[header,2]-m.geom_size[header,2]==pytest.approx(spec['opening']['height'],abs=1e-6)
        assert len(desc['spatial_cables'])==2
        assert len(desc.get('spatial_springs',[]))==(2 if spec['kinematics']['counterbalance_fraction'] else 0)
        assert mechanism['counterbalance']['fraction']==spec['kinematics']['counterbalance_fraction']
        assert not any(m.body_gravcomp)
        mujoco.mj_forward(m,d)
        for b in desc['bodies']:
            if not b['name'].startswith('section_wheel_'):continue
            assert len(b['geoms'])==24
            bid=m.body(b['name']).id
            assert m.body_mass[bid]==pytest.approx(.08,abs=1e-9)
            axle=m.geom(b['name']+'_axle').id
            for g in b['geoms']:
                gid=m.geom(g['name']).id
                # Adjacent-body contact filtering must not hide a solid
                # tyre intersecting the shaft that supposedly journals it.
                assert mujoco.mj_geomDistance(m,d,axle,gid,.1,None)>.0001
        for progress in np.linspace(0,1,25):
            d.qpos[:]=m.qpos0;resolve_sectional_configuration(m,d.qpos,desc['meta'],progress);mujoco.mj_forward(m,d)
            assert max(track_progress(d.site_xpos[m.site(s).id,1:],path)[1] for s in mechanism['roller_sites'])<2e-5
            for side in mechanism['counterbalance']['sides']:
                t=m.tendon(side['cable']).id
                assert d.ten_length[t]==pytest.approx(m.tendon_range[t,1],abs=2e-6)
        assert lowest_panel(m,d,desc)>1.8
        # The top roller has stopped rising while the lower panels still
        # continue overhead. Root Z is not falsely treated as task progress.
        poses=[inspection_pose(u,mechanism) for u in (.7,1.)]
        assert poses[0]['root_z']==pytest.approx(poses[1]['root_z'],abs=1e-7)
        assert poses[1]['root_y']>poses[0]['root_y']+.5


def test_closest_path_and_chord_solver_keep_panel_lengths():
    for spec in SPECS:
        path=track_dimensions(spec);end=path['vertical_end_z_m']+path['radius_m']*math.pi/2+.1
        mechanism={'path':path,'progress':{'closed_s_m':.05,'open_s_m':end}}
        for u in np.linspace(0,1,41):
            pose=inspection_pose(u,mechanism);nodes=np.asarray(pose['nodes_yz'])
            assert np.allclose(np.linalg.norm(np.diff(nodes,axis=0),axis=1),path['panel_height_m'],atol=1e-12)
            for p in nodes:assert track_progress(p,path)[1]<1e-12


@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_unlocked_nominal_sweep_clears_structure_and_released_keepers(exports,spec):
    m,d,desc=load(exports/'unlocked' if spec['lock'].get('engaged') else exports,spec)
    anchors={}
    for side in desc['meta']['sectional_track']['counterbalance']['sides']:
        names=[m.geom(i).name for i in range(m.ngeom) if m.geom(i).name.startswith(side['bottom_washer_prefix']+'_')]
        assert len(names)==12
        anchors[side['cable']]=[m.geom(name).id for name in names]
        assert all(m.geom_contype[g] and m.geom_conaffinity[g] for g in anchors[side['cable']])
    for progress in np.linspace(0,1,61):
        d.qpos[:]=m.qpos0;resolve_sectional_configuration(m,d.qpos,desc['meta'],progress);mujoco.mj_forward(m,d)
        bad=[(m.geom(c.geom1).name,m.geom(c.geom2).name,float(c.dist)) for c in d.contact if c.dist<-.00005]
        assert not bad,(spec['id'],progress,bad[:8])
        for side in desc['meta']['sectional_track']['counterbalance']['sides']:
            sign=np.sign(side['cable_plane_x_m'])
            # Track and moving bracket X bounds remain disjoint throughout
            # the curved travel; disabled native contact cannot hide a rod
            # passing through the web, as happened in the original model.
            for g in anchors[side['cable']]:
                rot=d.geom_xmat[g].reshape(3,3)
                center=d.geom_xpos[g]+rot@m.geom_aabb[g,:3]
                half=np.abs(rot)@m.geom_aabb[g,3:]
                assert sign*center[0]+half[0] < spec['leaf']['width']/2+.033
            endpoint=d.site_xpos[m.site(side['bottom_site']).id]
            axle=m.geom(side['bottom_axle']).id
            local=d.geom_xmat[axle].reshape(3,3).T@(endpoint-d.geom_xpos[axle])
            assert np.linalg.norm(local[:2])<1e-8 and abs(local[2])<m.geom_size[axle,1]


def test_original_outboard_cable_anchor_crossed_uncut_web_even_with_contact_disabled(exports):
    spec=next(s for s in SPECS if s['id']=='db0175_garage_sectional')
    path=exports/'doors'/spec['id'];tree=ET.parse(path/'door.xml')
    bottom=tree.find(f".//body[@name='section_{spec['kinematics']['n_sections']-1}']")
    sh=spec['leaf']['height']/spec['kinematics']['n_sections']
    ET.SubElement(bottom,'geom',name='original_noncolliding_anchor',type='cylinder',
        size='.004 .050',pos=f"{spec['leaf']['width']/2+.075} 0 {-sh}",
        quat='.7071067811865476 0 .7071067811865475 0',contype='0',conaffinity='0')
    private=path/'original-anchor-negative.xml';tree.write(private)
    m=mujoco.MjModel.from_xml_path(str(private));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    old=m.geom('original_noncolliding_anchor').id;web=m.geom('section_track_r_web_0').id
    assert mujoco.mj_geomDistance(m,d,old,web,.1,None)<-.03


@pytest.mark.parametrize('door_id',['db0148_garage_sectional','db0780_garage_sectional'])
def test_healthy_native_bottom_handle_opens_and_recloses_below_120N_without_pose_resets(exports,door_id):
    spec=next(s for s in SPECS if s['id']==door_id);m,d,desc=load(exports,spec);mechanism=desc['meta']['sectional_track'];path=mechanism['path']
    point=m.site('bottom_roller_mid').id;grip=m.site('lift_handle_grip').id;body=m.site_bodyid[grip]
    start=mechanism['progress']['closed_s_m'];end=mechanism['progress']['open_s_m'];peak=0.;depth=0.;error=0.
    for i in range(round(32/m.opt.timestep)):
        mujoco.mj_forward(m,d);now=i*m.opt.timestep
        u=min(1.,now/12) if now<=14 else max(0.,1-(now-14)/12)
        target,_=track_path(start+(end-start)*(10*u**3-15*u**4+6*u**5),path)
        jac=np.zeros((3,m.nv));rot=np.zeros_like(jac);mujoco.mj_jacSite(m,d,jac,rot,point)
        force=np.r_[0.,3500*(target-d.site_xpos[point,1:])-200*(jac@d.qvel)[1:]]
        force*=min(1.,120/max(np.linalg.norm(force),1e-12));assert np.linalg.norm(force)<=120+1e-9
        d.qfrc_applied[:]=0;mujoco.mj_applyFT(m,d,force,np.zeros(3),d.site_xpos[grip],body,d.qfrc_applied);mujoco.mj_step(m,d)
        depth=max(depth,max((-c.dist for c in d.contact),default=0.))
        if i%50==0:
            peak=max(peak,lowest_panel(m,d,desc))
            error=max(error,max(track_progress(d.site_xpos[m.site(s).id,1:],path)[1] for s in mechanism['roller_sites']))
    assert peak>1.8
    assert lowest_panel(m,d,desc)<.035
    assert error<.0045  # 3 mm radial running clearance plus bounded native compliance
    assert depth<.001
    assert not any(w.number for w in d.warning)


def test_failed_counterbalance_does_not_get_hidden_assistance(exports):
    spec=next(s for s in SPECS if s['id']=='db0806_garage_sectional');m,d,desc=load(exports,spec)
    assert not desc.get('spatial_springs') and np.max(m.jnt_stiffness)==0
    grip=m.site('lift_handle_grip').id
    for _ in range(1500):
        d.qfrc_applied[:]=0;mujoco.mj_applyFT(m,d,np.array([0.,0.,120.]),np.zeros(3),d.site_xpos[grip],m.site_bodyid[grip],d.qfrc_applied);mujoco.mj_step(m,d)
    assert d.site_xpos[m.site('bottom_roller_mid').id,2]<.08
    assert not any(w.number for w in d.warning)


def test_t_shapes_with_separate_locks_are_fixed_grips_not_required_turn_actions(exports):
    for spec in SPECS:
        if spec['operator']['model']!='pull_t_handle_garage':continue
        m,_,desc=load(exports,spec)
        assert desc['meta']['operator_joint'] is None
        assert desc['meta']['sectional_operator']['turn_required'] is False
        assert 't_handle_hinge' not in [m.joint(i).name for i in range(m.njnt)]


@pytest.mark.parametrize('door_id',['db0175_garage_sectional','db0433_garage_sectional'])
def test_native_motor_drives_trolley_and_actual_pinned_arm_with_released_lock_clearance(exports,door_id):
    spec=next(s for s in SPECS if s['id']==door_id);m,d,desc=load(exports,spec);mechanism=desc['meta']['sectional_track']
    drive=mechanism['drive'];aid=m.actuator(drive['actuator']).id;j=m.joint(drive['linkage']['trolley_joint']).id
    qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j];peak=0.;depth=0.;loop_error=0.
    assert m.actuator_trnid[aid,0]==j and tuple(m.actuator_ctrlrange[aid])==(-600.,600.)
    for i in range(round(32/m.opt.timestep)):
        now=i*m.opt.timestep;u=min(1.,now/12) if now<=14 else max(0.,1-(now-14)/12);progress=10*u**3-15*u**4+6*u**5
        pose=inspection_pose(progress,mechanism);d.ctrl[aid]=np.clip(3500*(pose['trolley_q']-d.qpos[qa])-200*d.qvel[va],-600,600)
        mujoco.mj_step(m,d);assert abs(d.actuator_force[aid])<=600+1e-9
        depth=max(depth,max((-c.dist for c in d.contact),default=0.))
        if i%50==0:peak=max(peak,lowest_panel(m,d,desc))
        for e in range(m.neq):
            if m.eq_type[e]!=mujoco.mjtEq.mjEQ_CONNECT:continue
            a,b=m.eq_obj1id[e],m.eq_obj2id[e]
            pa=d.xpos[a]+d.xmat[a].reshape(3,3)@m.eq_data[e,:3];pb=d.xpos[b]+d.xmat[b].reshape(3,3)@m.eq_data[e,3:6]
            loop_error=max(loop_error,np.linalg.norm(pa-pb))
    assert peak>1.8 and lowest_panel(m,d,desc)<.04
    assert depth<.001 and loop_error<.0002
    assert not any(w.number for w in d.warning)
