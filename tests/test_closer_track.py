"""Native track roller, solenoid release and force-path regression controls."""
import copy
import json
from contextlib import contextmanager
from pathlib import Path
import mujoco
import numpy as np
import pytest
from doorbench.spec import generate_all
from doorbench.build import export_door
from doorbench.geometry.closer_mounts import resolve_closer_configuration
from doorbench.closer_pinion import compile_pinion_closers,apply_pinion_closers
from doorbench.closer_track_hold import compile_track_holds,apply_track_holds

@pytest.fixture(scope='module')
def tracks(tmp_path_factory):
    root=tmp_path_factory.mktemp('native-track');rows=[]
    for source in generate_all():
        if source['closer']['model']!='magnetic_hold':continue
        spec=copy.deepcopy(source);spec['lock']['engaged']=False
        # The separate credential lock is explicitly unlocked in this closer
        # isolation fixture; its authored source and released assets are intact.
        ex=export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
        path=Path(ex['files']['mjcf']['full']);meta=json.loads(path.with_name('model.json').read_text())['meta']
        rows.append((source,path,meta))
    assert len(rows)==13
    assert sum(len(meta['closer_track_holds']) for _,_,meta in rows)==20
    return rows

@contextmanager
def forces(model,meta,power):
    pin=compile_pinion_closers(model,meta);holds=compile_track_holds(model,meta)
    previous=mujoco.get_mjcb_passive()
    def callback(m,d):
        if m is not model:return
        apply_pinion_closers(m,d,pin);apply_track_holds(m,d,holds,powered=power(d))
    try:
        mujoco.set_mjcb_passive(callback);yield holds
    finally:mujoco.set_mjcb_passive(previous)


def initialize(model,meta):
    d=mujoco.MjData(model)
    for row in meta['closer_track_holds']:
        d.qpos[model.jnt_qposadr[model.joint(row['leaf_joint']).id]]=row['nominal_hold_angle_rad']
    resolve_closer_configuration(model,d.qpos,meta)
    return d


def test_all20_holders_hold_and_release_through_native_contacts_all_tiers(tracks):
    for source,path,meta in tracks:
        for filename in ('door.xml','door_simple.xml','door_minimal.xml'):
            model=mujoco.MjModel.from_xml_path(str(path.with_name(filename)))
            for index,row in enumerate(meta['closer_track_holds']):
                d=initialize(model,meta);adr=model.jnt_qposadr[model.joint(row['leaf_joint']).id]
                def power(data):return {row['plunger_joint']:data.time<3.}
                with forces(model,meta,power) as rules:
                    hold=rules[index];held=None;contact=False;maxloop=0.
                    for k in range(round(8./model.opt.timestep)):
                        mujoco.mj_step(model,d)
                        if 2.<d.time<3.:
                            held=float(d.qpos[adr]);contact|=any({model.geom(c.geom1).name,model.geom(c.geom2).name}=={row['roller_geom'],row['cam_geom']} for c in d.contact)
                        if k%25==0:
                            eq=model.equality(meta['closer_mounts'][index]['connect']).id
                            check=mujoco.MjData(model);check.qpos[:]=d.qpos;mujoco.mj_kinematics(model,check)
                            b1,b2=model.eq_obj1id[eq],model.eq_obj2id[eq]
                            p1=check.xpos[b1]+check.xmat[b1].reshape(3,3)@model.eq_data[eq,:3]
                            p2=check.xpos[b2]+check.xmat[b2].reshape(3,3)@model.eq_data[eq,3:6]
                            maxloop=max(maxloop,np.linalg.norm(p1-p2))
                context=(source['id'],filename,row['leaf_joint'])
                assert not np.any(d.warning.number),context
                assert contact,context
                assert abs(held-row['nominal_hold_angle_rad'])<np.deg2rad(.5),(context,held)
                assert d.qpos[adr]<np.deg2rad(70),(context,float(d.qpos[adr]))
                assert d.qpos[hold.plunger_qpos]>.008,(context,float(d.qpos[hold.plunger_qpos]))
                assert maxloop<.001,(context,maxloop)


def test_real_opening_effort_enters_hold_and_button_or_manual_effort_releases(tracks):
    source,path,meta=next(x for x in tracks if x[0]['id']=='db0024_swing_single')
    for mode in ('open_then_hold','button','manual'):
        m=mujoco.MjModel.from_xml_path(str(path));d=initialize(m,meta);row=meta['closer_track_holds'][0]
        j=m.joint(row['leaf_joint']).id;a=m.jnt_qposadr[j];dof=m.jnt_dofadr[j];button=m.jnt_dofadr[m.joint(row['button_joint']).id]
        if mode=='open_then_hold':d.qpos[a]=.35;resolve_closer_configuration(m,d.qpos,meta)
        arrived=None;max_release=0.
        with forces(m,meta,lambda _:True) as holds:
            for k in range(round(10./m.opt.timestep)):
                d.qfrc_applied[:]=0.
                if mode=='open_then_hold' and (arrived is None or d.time<arrived+.4):d.qfrc_applied[dof]=100.-80*d.qvel[dof]
                if mode=='button' and d.time>3:d.qfrc_applied[button]=8.
                if mode=='manual' and d.time>3:d.qfrc_applied[dof]=-60.
                mujoco.mj_step(m,d)
                max_release=max(max_release,float(d.qpos[holds[0].plunger_qpos]))
                if arrived is None and d.qpos[a]>np.pi/2-.001:arrived=d.time
        assert not np.any(d.warning.number),mode
        if mode=='open_then_hold':
            assert arrived is not None
            assert abs(d.qpos[a]-np.pi/2)<np.deg2rad(.5)
            assert max_release>.005  # roller actually lifted the cam on entry
        else:
            assert abs(d.qpos[a])<np.deg2rad(1.),(mode,float(d.qpos[a]))
            if mode=='button':assert d.qpos[holds[0].button_qpos]>.003


def test_solenoid_only_adds_plunger_force_and_physical_switch_interrupts(tracks):
    _,path,meta=tracks[0];m=mujoco.MjModel.from_xml_path(str(path));rules=compile_track_holds(m,meta);r=rules[0]
    for powered,button,q in ((True,0.,0.),(True,0.,.005),(False,0.,0.),(True,.004,0.)):
        d=mujoco.MjData(m);d.qpos[r.plunger_qpos]=q;d.qpos[r.button_qpos]=button
        before={k:getattr(d,k).copy() for k in ('qpos','qvel','qacc','qfrc_applied','ctrl')};damping=m.dof_damping.copy()
        apply_track_holds(m,d,rules,powered=powered)
        for k,v in before.items():np.testing.assert_array_equal(getattr(d,k),v)
        np.testing.assert_array_equal(m.dof_damping,damping)
        expected=-r.force/(1+q/r.gap)**2 if powered and button<.003 else 0.
        assert np.isclose(d.qfrc_passive[r.plunger_dof],expected)
        assert np.count_nonzero(d.qfrc_passive)==int(expected!=0.)


def test_compiled_button_threshold_is_bound_and_validated(tracks):
    _,path,meta=tracks[0];m=mujoco.MjModel.from_xml_path(str(path));changed=copy.deepcopy(meta)
    changed['closer_track_holds'][0]['button_release_threshold_m']=.002
    r=compile_track_holds(m,changed)[0];d=mujoco.MjData(m);d.qpos[r.button_qpos]=.0025
    apply_track_holds(m,d,(r,),powered=True)
    assert not np.any(d.qfrc_passive)
    for invalid in (0.,-.001,.006,float('nan')):
        changed['closer_track_holds'][0]['button_release_threshold_m']=invalid
        with pytest.raises(ValueError):compile_track_holds(m,changed)


def test_dedicated_gate_rejects_blocked_release_and_preserves_original_model(tracks):
    from doorbench.closer_track_qa import run_closer_track_qa
    _,path,meta=tracks[0];m=mujoco.MjModel.from_xml_path(str(path));before=m.qpos0.copy();damping=m.dof_damping.copy()
    report=run_closer_track_qa(m,meta)
    assert report['ok'],report
    np.testing.assert_array_equal(m.qpos0,before);np.testing.assert_array_equal(m.dof_damping,damping)
    broken=mujoco.MjModel.from_xml_path(str(path));j=broken.joint(meta['closer_track_holds'][0]['plunger_joint']).id
    broken.jnt_range[j,1]=.0002
    report=run_closer_track_qa(broken,meta)
    assert not report['ok']
    assert any('release_failed' in f for f in report['failures'])


def test_contact_preview_follows_actual_cam_without_changing_boundary_or_input(tracks):
    from doorbench.closer_track_qa import TrackContactPreview
    for source,path,meta in tracks:
        m=mujoco.MjModel.from_xml_path(str(path));preview=TrackContactPreview(m,meta)
        original={key:getattr(m,key).copy() for key in ('jnt_range','jnt_limited','jnt_solref','jnt_solimp','dof_damping')}
        for row in meta['closer_track_holds']:
            ha=int(m.jnt_qposadr[m.joint(row['leaf_joint']).id]);pa=int(m.jnt_qposadr[m.joint(row['plunger_joint']).id])
            for degrees in (0,80,85,87,90):
                q=m.qpos0.copy();q[ha]=np.deg2rad(degrees);resolve_closer_configuration(m,q,meta);before=q.copy()
                report=preview.resolve(q)
                assert report['ok'],(source['id'],degrees,report)
                assert np.array_equal(q,before)
                result=np.asarray(report['qpos']);assert result[ha]==q[ha]
                assert report['cam_contact_depth_m']<.001
            # A directly requested physical release position remains the input;
            # inspection is not allowed to solve it back to a convenient pose.
            q=m.qpos0.copy();q[pa]=.006;resolve_closer_configuration(m,q,meta)
            report=preview.resolve(q,driven_joint=row['plunger_joint'])
            assert report['ok'],(source['id'],report)
            assert report['qpos'][pa]==q[pa]
        for key,value in original.items():assert np.array_equal(getattr(m,key),value)
    _,path,meta=next(x for x in tracks if x[0]['id']=='db0773_swing_single')
    m=mujoco.MjModel.from_xml_path(str(path));row=meta['closer_track_holds'][0]
    m.jnt_range[m.joint(row['plunger_joint']).id,1]=.0002
    q=m.qpos0.copy();q[m.jnt_qposadr[m.joint(row['leaf_joint']).id]]=np.deg2rad(87);resolve_closer_configuration(m,q,meta)
    assert not TrackContactPreview(m,meta).resolve(q)['ok']


def test_axle_is_seated_outside_vertical_tip_bore(tracks):
    for source,path,meta in tracks:
        m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
        for row in meta['closer_mounts']:
            prefix='' if row['leaf_body']=='leaf' else row['leaf_body']+'_'
            axle=m.geom(prefix+'closer_roller_axle').id;pin=m.geom(row['pivot_geom']).id
            assert mujoco.mj_geomDistance(m,d,axle,pin,.1,None)>.0009,source['id']
            shoes=[m.geom(name).id for name in row['shoe_geoms']]
            assert min(mujoco.mj_geomDistance(m,d,axle,shoe,.1,None) for shoe in shoes)<=0,source['id']
