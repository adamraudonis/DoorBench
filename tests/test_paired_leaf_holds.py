"""Prepared inactive-leaf hardware, physical load paths, and causal defects."""
import copy
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
import pytest

from doorbench.build import export_door,build_model
from doorbench.spec import generate_all
from doorbench.paired_hold_qa import run_paired_hold_qa
from doorbench.clearance import gate_model


FLUSH={149,222,334,341,413,454,534,577,604,682,702,707,714,733,792,832,846,929,944,963}
CANE={127,144,279,395,467,704,788,918}


@pytest.fixture(scope='module')
def doors(tmp_path_factory):
    root=tmp_path_factory.mktemp('inactive-leaf-holds');rows=[]
    for s in generate_all():
        if s['index']not in FLUSH|CANE:continue
        files=export_door(s,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json','urdf'))['files']
        path=root/'doors'/s['id'];model=json.loads((path/'model.json').read_text())
        rows.append((s,path,model,files))
    assert {s['index']for s,*_ in rows}==FLUSH|CANE
    return rows


@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_all_inactive_leaves_hold_release_retain_with_actual_surfaces(doors,tier):
    for spec,path,source,files in doors:
        m=mj.MjModel.from_xml_path(files['mjcf'][tier]);meta=source['meta']
        assert len(meta['paired_leaf_holds'])==(2 if spec['index']in FLUSH else 1)
        assert m.jnt_range[m.joint('leaf_b_hinge').id,1]>.8
        before=np.empty(mj.mj_sizeModel(m),dtype=np.uint8);mj.mj_saveModel(m,buffer=before)
        report=run_paired_hold_qa(m,meta)
        after=np.empty(mj.mj_sizeModel(m),dtype=np.uint8);mj.mj_saveModel(m,buffer=after)
        assert np.array_equal(before,after),'Service fixture mutated the input native model'
        report['source_sha256']={n:hashlib.sha256((path/n).read_bytes()).hexdigest()for n in('spec.json','model.json','door.xml')}
        report['tier']=tier;report['native_xml_sha256']=hashlib.sha256(open(files['mjcf'][tier],'rb').read()).hexdigest()
        (path/('inactive-proof-'+tier+'.json')).write_text(json.dumps(report,indent=2))
        assert report['ok'],(spec['id'],tier,report)
        assert sum(r['phase']=='retained_opening_load'for r in report['phases'])==2
        assert report['max_release_force_N']<=20 and report['max_pull_force_N']<=50
        assert report['max_active_fixture_torque_Nm']<=100
        assert report['max_bolt_joint_limit_force_N']<=.01
        assert report['native_warning_messages']==[]
        for r in meta['paired_leaf_holds']:
            assert r['requires_primary_open_rad']==(.20 if spec['index']in FLUSH else 0.)
            if spec['index']in CANE:
                approach=1 if spec['robot'].get('approach_side','-y')=='+y' else -1
                assert r['face']==(-approach if spec['robot']['robot_outside']else approach)
                assert r['accessible_from_robot']==(r['face']==approach)
        tree=ET.parse(files['urdf'][tier])
        assert all(j.attrib.get('type')=='prismatic'for j in tree.findall('joint')if j.attrib['name']in{r['joint']for r in meta['paired_leaf_holds']})


def test_all_added_hardware_has_no_initial_parent_filtered_overlap(doors):
    for spec,path,source,files in doors:
        m=gate_model(files['mjcf']['full']);d=mj.MjData(m);mj.mj_forward(m,d)
        pairs=[]
        for c in d.contact:
            names=[m.geom(g).name for g in c.geom]
            if c.dist<-.0001 and any(any(k in n for k in('_flush','_cane','service_pull'))for n in names):
                pairs.append((float(c.dist),names))
        assert not pairs,(spec['id'],pairs)
        declared=source['meta']['mass_reconciliation']['panels'];row=next(r for r in declared if r['body']=='leaf_b')
        assert row['geometry_backed_kg']>.3
        assert all(v>0 for v in row['geometry_backed_bodies_kg'].values())


def test_complete_operational_strokes_and_leaf_sweep_preserve_source_geometry(doors):
    from doorbench.geometry.closer_mounts import resolve_closer_configuration
    for spec,path,source,files in doors:
        meta=source['meta'];m=gate_model(files['mjcf']['full']);d=mj.MjData(m);violations=[]
        def inspect(label,q):
            d.qpos[:]=q
            if meta.get('closer_mounts'):resolve_closer_configuration(m,d.qpos,meta)
            mj.mj_kinematics(m,d);mj.mj_collision(m,d)
            for c in d.contact:
                names=[m.geom(g).name for g in c.geom]
                if c.dist<-.0001 and any(any(k in n for k in('_flush','_cane','service_pull'))for n in names):
                    violations.append((label,float(c.dist),names))
        for r in meta['paired_leaf_holds']:
            for value in np.linspace(0.,r['travel_m'],31):
                q=m.qpos0.copy();q[m.jnt_qposadr[m.joint(r['joint']).id]]=value
                inspect(r['joint']+':'+str(value),q)
        initial=m.qpos0.copy();initial[m.jnt_qposadr[m.joint('leaf_a_hinge').id]]=.8
        for r in meta['paired_leaf_holds']:initial[m.jnt_qposadr[m.joint(r['joint']).id]]=r['travel_m']
        for value in np.linspace(0.,m.jnt_range[m.joint('leaf_b_hinge').id,1],49):
            q=initial.copy();q[m.jnt_qposadr[m.joint('leaf_b_hinge').id]]=value;inspect('B_open:'+str(value),q)
        (path/'inactive-envelope.json').write_text(json.dumps({'ok':not violations,'violations':violations},indent=2))
        assert not violations,(spec['id'],violations[:20])


def _defect_xml(path,files,meta,defect):
    tree=ET.parse(files['mjcf']['full']);root=tree.getroot();record=meta['paired_leaf_holds'][0]
    if defect=='removed_receivers':
        names={g for r in meta['paired_leaf_holds']for g in r['keeper_geoms']}
        for g in root.iter('geom'):
            if g.get('name')in names:
                p=np.fromstring(g.attrib['pos'],sep=' ');p[0]+=3.;g.set('pos',' '.join(map(str,p)))
    elif defect=='filled_guide':
        # Recompile the modified source; never leave a native broadphase box
        # stale by mutating an already compiled primitive's transform/size.
        g=next(g for g in root.iter('geom')if g.get('name')==record['guide_geoms'][0])
        p=np.fromstring(g.get('pos'),sep=' ');p[0]+=.00775;g.set('pos',' '.join(map(str,p)));g.set('size','.020 .020 .004')
    elif defect=='range_only_hold':
        j=next(j for j in root.iter('joint')if j.get('name')=='leaf_b_hinge');j.set('range','0 .001')
    elif defect=='off_surface':
        s=next(s for s in root.iter('site')if s.get('name')==record['site']);p=np.fromstring(s.get('pos'),sep=' ');p[0]+=.03;s.set('pos',' '.join(map(str,p)))
    elif defect=='removed_stops':
        names={g for r in meta['paired_leaf_holds']for g in r['stop_geoms']}
        for g in root.iter('geom'):
            if g.get('name')in names:
                p=np.fromstring(g.attrib['pos'],sep=' ');p[0]+=3.;g.set('pos',' '.join(map(str,p)))
    out=path/(defect+'.xml');tree.write(out);return mj.MjModel.from_xml_path(str(out))


@pytest.mark.parametrize('defect',('removed_receivers','filled_guide','range_only_hold','off_surface','removed_stops'))
def test_fail_closed_for_missing_physical_hardware_or_fabricated_contact(doors,defect):
    s,path,source,files=next(row for row in doors if row[0]['index']==149)
    m=_defect_xml(path,files,source['meta'],defect);r=run_paired_hold_qa(m,source['meta'])
    (path/(defect+'-proof.json')).write_text(json.dumps(r,indent=2))
    assert not r['ok'],(defect,r)


def test_source_inside_face_is_preserved_for_counterfactual_approach(doors):
    s=copy.deepcopy(next(row[0]for row in doors if row[0]['index']==127));s['robot']['approach_side']='+y'
    model=build_model(s);r=model.meta['paired_leaf_holds'][0]
    assert r['face']==(-1 if s['robot']['robot_outside']else 1)
    assert r['accessible_from_robot']==(not s['robot']['robot_outside'])


def test_missing_second_flush_record_cannot_hide_an_installed_load_path(doors):
    _,_,source,files=next(row for row in doors if row[0]['index']==149)
    meta=copy.deepcopy(source['meta']);meta['paired_leaf_holds']=meta['paired_leaf_holds'][:1]
    r=run_paired_hold_qa(mj.MjModel.from_xml_path(files['mjcf']['full']),meta)
    assert not r['ok']and'distinct top/bottom'in r['failures'][0]


def test_zero_force_contact_candidates_cannot_prove_receiver_load(doors,monkeypatch):
    _,_,source,files=next(row for row in doors if row[0]['index']==149)
    def no_reaction(_m,_d,_i,wrench):wrench[:]=0.
    monkeypatch.setattr(mj,'mj_contactForce',no_reaction)
    result=run_paired_hold_qa(mj.MjModel.from_xml_path(files['mjcf']['full']),source['meta'])
    assert not result['ok']and'do not each carry'in result['failures'][0]
    assert result['phases'][-1]['receiver_contacts']


def test_flush_faces_receive_normal_finger_presses(doors):
    for spec,_,source,files in doors:
        if spec['index']not in FLUSH:continue
        m=mj.MjModel.from_xml_path(files['mjcf']['full']);d=mj.MjData(m);mj.mj_kinematics(m,d)
        for r in source['meta']['paired_leaf_holds']:
            axis=d.xaxis[m.joint(r['joint']).id]
            normal=d.site_xmat[m.site(r['site']).id].reshape(3,3)[:,2]
            reverse=d.site_xmat[m.site(r['engage_site']).id].reshape(3,3)[:,2]
            assert np.dot(axis,normal)==pytest.approx(-1.)
            assert np.dot(axis,reverse)==pytest.approx(1.)


def test_callback_warning_fails_even_without_native_counter(doors,monkeypatch):
    import doorbench.paired_hold_qa as q
    _,_,source,files=doors[0];m=mj.MjModel.from_xml_path(files['mjcf']['full'])
    previous=mj.get_mju_user_warning();passive=mj.get_mjcb_passive()
    def warning_only(*_):
        mj.get_mju_user_warning()('Synthetic callback-only failure')
        return {'ok':True,'failures':[]}
    monkeypatch.setattr(q,'_exercise',warning_only)
    result=q.run_paired_hold_qa(m,source['meta'])
    assert not result['ok']and result['native_warning_messages']==['Synthetic callback-only failure']
    assert mj.get_mju_user_warning()is previous and mj.get_mjcb_passive()is passive
