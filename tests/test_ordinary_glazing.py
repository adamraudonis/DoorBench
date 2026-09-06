"""Actual glazing, independent material/ray checks and causal source defects."""
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
import pytest
import trimesh

from doorbench import materials as M
from doorbench.build import build_model, export_door
from doorbench.geometry import common as C
from doorbench.geometry.pocket_hardware import cut_box_recess
from doorbench.glazing import construction, uses_ordinary_glazing
from doorbench.ir import Body, Model, ALL_TIERS, QUAT_ID, quat_to_mat
from doorbench.physics import derive
from doorbench.spec import generate_all


@pytest.fixture(scope='module')
def specs():return generate_all()


@pytest.fixture(scope='module')
def exported(specs,tmp_path_factory):
    root=tmp_path_factory.mktemp('ordinary-glazing');rows=[]
    for s in specs:
        if not uses_ordinary_glazing(s['leaf']):continue
        files=export_door(s,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))['files']
        path=root/'doors'/s['id'];source=json.loads((path/'model.json').read_text())
        rows.append((s,path,source,files))
    assert len(rows)==177
    assert sum(len(r[2]['meta']['ordinary_glazing_constructions'])for r in rows)==242
    return rows


def _rays(m,meta):
    """Two independent native rays through each visible aperture air space."""
    d=mj.MjData(m);mj.mj_kinematics(m,d);probes=[]
    for r in meta['ordinary_glazing_constructions']:
        body=m.body(r['body']).id;R=d.xmat[body].reshape(3,3)
        for p in r['panes']:
            contour=np.asarray(p['visible_polygon_xz'])
            # A strict convex interior point away from the symmetric centre
            # where several native mesh triangles share one ray/vertex seam.
            centre=.873*contour.mean(axis=0)+.127*contour[0]
            local=np.array([r['x0']+r['u']*centre[0],r['y_center'],r['z0']+centre[1]])
            glass=m.geom(r['prefix']+'_'+p['glass_name']).id
            for side in (-1,1):
                local_origin=local+[0,side*(p['glass_thickness_m']/2+.0004),0]
                origin=d.xpos[body]+R@local_origin;direction=R@np.array([0.,-side,0.])
                for other in range(m.ngeom):
                    if m.geom_bodyid[other]!=body or not m.geom_contype[other] or m.geom_type[other]!=mj.mjtGeom.mjGEOM_BOX:continue
                    q=d.geom_xmat[other].reshape(3,3).T@(origin-d.geom_xpos[other])
                    assert np.max(np.abs(q)-m.geom_size[other])>=-1e-8,('Solid stock occupies aperture air',m.geom(other).name)
                hit=np.array([-1],dtype=np.int32)
                distance=mj.mj_ray(m,d,origin,direction,None,True,-1,hit)
                assert hit[0]==glass,(r['body'],p['index'],m.geom(hit[0]).name if hit[0]>=0 else None)
                assert distance==pytest.approx(.0004,abs=3e-6),(r['body'],p['index'],distance)
                probes.append(distance)
    return probes


def _glass_inertia(record,density):
    """Independent planar polygon integration, extruded at declared thickness."""
    panels=[]
    for pane in record['panes']:
        a=np.asarray(pane['glass_polygon_xz']);b=np.roll(a,-1,axis=0)
        cross=a[:,0]*b[:,1]-b[:,0]*a[:,1];A=cross.sum()/2
        cx=((a[:,0]+b[:,0])*cross).sum()/(6*A)
        cz=((a[:,1]+b[:,1])*cross).sum()/(6*A)
        xx=((a[:,0]**2+a[:,0]*b[:,0]+b[:,0]**2)*cross).sum()/(12*A)-cx*cx
        zz=((a[:,1]**2+a[:,1]*b[:,1]+b[:,1]**2)*cross).sum()/(12*A)-cz*cz
        xz=((2*a[:,0]*a[:,1]+a[:,0]*b[:,1]+b[:,0]*a[:,1]+2*b[:,0]*b[:,1])*cross).sum()/(24*A)-cx*cz
        t=pane['glass_thickness_m'];mass=A*t*density
        centre=np.array([record['x0']+record['u']*cx,record['y_center'],record['z0']+cz])
        I=mass*np.diag([zz+t*t/12,xx+zz,xx+t*t/12]);I[0,2]=I[2,0]=-mass*xz*record['u']
        panels.append((mass,centre,I))
    mass=sum(m for m,c,I in panels);com=sum(m*c for m,c,I in panels)/mass
    inertia=sum(I+m*((c-com)@(c-com)*np.eye(3)-np.outer(c-com,c-com))for m,c,I in panels)
    return com,inertia


@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_every_native_pane_has_real_thickness_and_air_aperture(exported,tier):
    for s,path,source,files in exported:
        m=mj.MjModel.from_xml_path(files['mjcf'][tier]);meta=source['meta']
        distances=_rays(m,meta)
        assert distances
        for r in meta['ordinary_glazing_constructions']:
            for p in r['panes']:
                g=m.geom(r['prefix']+'_'+p['glass_name']).id
                assert m.geom_contype[g] and m.geom_conaffinity[g]
                if m.geom_type[g]==mj.mjtGeom.mjGEOM_BOX:
                    assert 2*m.geom_size[g,1]==pytest.approx(p['glass_thickness_m'],abs=1e-9)
        physics=json.loads((path/'spec.json').read_text())['physics']
        if tier=='full':
            assert sum(m.body_mass[1:])==pytest.approx(physics['mass']['total_kg'],abs=3e-5,rel=1e-6),s['id']
        # Unrelated legacy keypad/knocker/pet bodies are omitted by reduced
        # tiers. This construction retains each complete glazed leaf's mass.
        owners={r['body']for r in meta['ordinary_glazing_constructions']}
        for row in meta['mass_reconciliation']['panels']:
            if row['body']not in owners:continue
            assert m.body_mass[m.body(row['body']).id]==pytest.approx(row['leaf_body_kg'],abs=2e-6),s['id']
        for r in meta['ordinary_glazing_constructions']:
            assert m.body_mass[m.body(r['pane_body']).id]==pytest.approx(r['glass_kg'],abs=2e-6),s['id']
            assert m.body_mass[m.body(r['retainer_body']).id]==pytest.approx(r['retainer_kg'],abs=2e-6),s['id']
            body=m.body(r['pane_body']).id
            centre,inertia=_glass_inertia(r,M.MATERIALS[s['leaf']['glazing']['material']].density)
            R=quat_to_mat(m.body_iquat[body]);native=R@np.diag(m.body_inertia[body])@R.T
            np.testing.assert_allclose(m.body_ipos[body],centre,atol=1e-6,rtol=0,err_msg=s['id'])
            np.testing.assert_allclose(native,inertia,atol=2e-6,rtol=3e-6,err_msg=s['id'])


def test_profiles_partition_material_and_retain_stiles(specs):
    for s in specs:
        if not uses_ordinary_glazing(s['leaf']):continue
        leaf=s['leaf'];p=construction(leaf);parts=p['parts']
        stock=sum(g['volume_m3']for g in parts if g['semantic']=='leaf')
        glass=sum(g['volume_m3']for g in parts if g['semantic']=='glass')
        # Shoelace is independent of the prism volume implementation/exporter.
        holes=0.
        for pane in p['panes']:
            a=np.asarray(pane['rough_polygon_xz']);b=np.roll(a,-1,axis=0)
            holes+=abs((a[:,0]*b[:,1]-a[:,1]*b[:,0]).sum())/2
        assert stock==pytest.approx((leaf['width']*leaf['height']-holes)*leaf['thickness'],abs=1e-10)
        assert p['glass_kg']==pytest.approx(glass*M.MATERIALS[leaf['glazing']['material']].density)
        assert p['retainer_kg']>0
        assert all(x['mass_kg']>0 for x in parts)
        assert p['total_kg']==pytest.approx(sum(x['mass_kg']for x in parts))
        for pane in p['panes']:
            poly=np.asarray(pane['rough_polygon_xz'])
            assert poly[:,0].min()>0 and poly[:,0].max()<leaf['width']
            assert poly[:,1].min()>0 and poly[:,1].max()<leaf['height']


def test_geometry_and_derived_glass_mass_use_same_actual_panel_dimensions(exported):
    for s,path,source,files in exported:
        phys=json.loads((path/'spec.json').read_text())['physics']
        for r in source['meta']['ordinary_glazing_constructions']:
            row=next(v for v in phys['mass']['per_body']if v['body']==r['body'])
            assert row['glass_kg']==pytest.approx(r['glass_kg'],abs=1e-8),s['id']
            assert row['glazing_retainer_allowance_replaced_kg']==pytest.approx(r['retainer_kg']),s['id']
            assert phys['per_body_dynamics'][r['body']]['mass']['glass_kg']==pytest.approx(row['glass_kg'])


@pytest.mark.parametrize('defect',('hidden_slab','full_depth_glass','no_glass'))
def test_native_aperture_probes_reject_old_overlay_and_missing_pane(exported,defect):
    s,path,source,files=next(r for r in exported if r[0]['index']==413)
    tree=ET.parse(files['mjcf']['full']);root=tree.getroot();r=source['meta']['ordinary_glazing_constructions'][0]
    body=next(b for b in root.iter('body')if b.get('name')==r['body'])
    glass_body=next(b for b in root.iter('body')if b.get('name')==r['pane_body'])
    g=next(g for g in glass_body.findall('geom')if g.get('name')==r['pane_geoms'][0])
    if defect=='hidden_slab':
        size=np.fromstring(g.get('size'),sep=' ');size[1]=r['thickness_m']/2
        ET.SubElement(body,'geom',name='regression_hidden_slab',type='box',pos=g.get('pos'),size=' '.join(map(str,size)))
    elif defect=='full_depth_glass':
        size=np.fromstring(g.get('size'),sep=' ');size[1]=r['thickness_m']/2;g.set('size',' '.join(map(str,size)))
    else:glass_body.remove(g)
    target=path/(defect+'.xml');tree.write(target);m=mj.MjModel.from_xml_path(str(target))
    with pytest.raises((AssertionError,KeyError)): _rays(m,source['meta'])


def test_oval_and_porthole_are_curved_openings_with_convex_retained_stock(exported):
    for s,path,source,files in exported:
        if s['leaf']['glazing']['panel_style'] not in ('porthole','glass_oval','glass_fan'):continue
        for r in source['meta']['ordinary_glazing_constructions']:
            body=next(b for b in source['bodies']if b['name']==r['body'])
            glass_body=next(b for b in source['bodies']if b['name']==r['pane_body'])
            g=next(g for g in glass_body['geoms']if g['name']==r['pane_geoms'][0])
            assert g['type']=='mesh'
            assert len(r['panes'][0]['glass_polygon_xz'])>=33
            assert any(g['name'].startswith(r['prefix']+'_slab_curve_')for g in body['geoms'])
            # Curved stock is independently convex; a single concave native
            # mesh hull around the opening would restore the hidden barrier.
            assert all(g['collision'] for g in body['geoms']if g['name'].startswith(r['prefix']+'_slab_curve_'))


def test_mortise_skips_only_proven_disjoint_mesh_and_rejects_intersection():
    body=Body('test',None);mesh=trimesh.creation.icosphere(subdivisions=1,radius=.1)
    body.geoms=[C.mesh_geom('curved_stock','curved_stock',mesh,(1,0,0),QUAT_ID,'wood',600,True,ALL_TIERS,'leaf')]
    cut_box_recess(body,(-.1,-.1,-.1),(.1,.1,.1),'safe')
    assert len(body.geoms)==1 and body.geoms[0].name=='curved_stock'
    with pytest.raises(ValueError,match='Cannot mortise'):
        cut_box_recess(body,(.95,-.02,-.02),(1.05,.02,.02),'intersecting')


def test_solid_and_existing_framed_branches_remain_exact(specs,monkeypatch):
    # The new construction is forbidden for every unrelated leaf, including
    # all54 existing framed units, and their existing primitive recipe remains.
    import doorbench.geometry.glazing as G
    def unexpected(*args,**kwargs):raise AssertionError('Unrelated leaf entered ordinary glazing')
    monkeypatch.setattr(G,'add_glazing',unexpected)
    framed=0;solid=0
    for s in specs:
        if uses_ordinary_glazing(s['leaf']):continue
        m=Model(s['id']);b=Body('leaf',None);m.add_body(b)
        C.add_leaf_geoms(m,b,s,s['leaf'],1,0,0,None)
        if s['leaf']['slab']in M.FRAMED_GLASS_SLABS:
            p=M.framed_glass_profile(s['leaf']['slab'],s['leaf']['width'],s['leaf']['height'],s['leaf']['thickness'])
            assert len(b.geoms)==len(p['parts'])
            for g,part in zip(b.geoms,p['parts']):
                np.testing.assert_allclose(g.pos,part['pos'],atol=1e-12)
                assert g.size==tuple(part['size']) and g.density==M.MATERIALS[part['material']].density
            framed+=1
        elif not s['leaf'].get('glazing') and s['leaf'].get('panel_style')=='flush' and not (
                M.SLABS[s['leaf']['slab']].monolithic and M.SLABS[s['leaf']['slab']].core_material in ('glass_clear','mirror')):
            g=next(g for g in b.geoms if g.name=='leaf_slab')
            assert g.pos==(s['leaf']['width']/2,0,s['leaf']['height']/2)
            assert g.size==(s['leaf']['width']/2,s['leaf']['thickness']/2,s['leaf']['height']/2)
            solid+=1
    assert framed==54 and solid>100
