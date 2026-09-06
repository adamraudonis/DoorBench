"""Emit ordinary glazing as supported convex material pieces in every tier."""
from __future__ import annotations

import math
import hashlib
import numpy as np
import trimesh
from ..glazing import construction, area
from ..ir import QUAT_ID, Body, ALL_TIERS
from . import common as C
from .lock_stock import cut_stock


def add_glazing(model, body, leaf, *, spec, width, height, stock_width, x0, z0, u,
                y_center, prefix, material, tiers, hole=None):
    profile=construction(leaf,width,height,spec=spec)
    added=[]
    for p in profile['parts']:
        polygon=np.array(p['polygon_xz']);a=polygon.min(axis=0);b=polygon.max(axis=0)
        y0,y1=p['y_bounds'];name=f"{prefix}_{p['name']}"
        mat=material if p['material']=='stock' else C.mat_from_material(model,p['material'])
        pos=(x0+u*(a[0]+b[0])/2,y_center+(y0+y1)/2,z0+(a[1]+b[1])/2)
        if math.isclose(area(p['polygon_xz']),float(np.prod(b-a)),rel_tol=1e-9,abs_tol=1e-12):
            g=C.box(name,pos,((b[0]-a[0])/2,(y1-y0)/2,(b[1]-a[1])/2),mat,p['density'],
                True,True,tiers,p['semantic'],p['name'].replace('_',' ').title())
        else:
            vertices=[(u*(x-(a[0]+b[0])/2),y-(y0+y1)/2,z-(a[1]+b[1])/2)
                for y in (y0,y1) for x,z in polygon]
            mesh=trimesh.convex.convex_hull(np.array(vertices))
            digest=hashlib.sha256(np.asarray(mesh.vertices,dtype='<f8').tobytes()+np.asarray(mesh.faces,dtype='<i8').tobytes()).hexdigest()[:24]
            g=C.mesh_geom(name,'glazing_'+digest,mesh,pos,QUAT_ID,mat,p['density'],True,tiers,p['semantic'],p['name'].replace('_',' ').title())
            if not math.isclose(g.volume(),p['volume_m3'],rel_tol=1e-7,abs_tol=1e-12):
                raise ValueError(f'Glazing convex volume mismatch: {name}')
        body.geoms.append(g);added.append(g.name)
    if stock_width<width:
        xx=sorted((x0+u*stock_width,x0+u*width))
        cut_stock(body,(xx[0],y_center-leaf['thickness'],z0-.001),
            (xx[1],y_center+leaf['thickness'],z0+height+.001),'edge_column',names=set(added))
    if hole:
        hx,hz,hw,hh=hole
        # A pet opening is prepared in solid stock. Cutting an installed pane
        # would require a different glass fabrication and retaining frame.
        for p in profile['panes']:
            points=np.array(p['rough_polygon_xz']);points[:,0]=x0+u*points[:,0];points[:,1]+=z0
            if (min(points[:,0])<hx+hw/2 and max(points[:,0])>hx-hw/2
                    and min(points[:,1])<hz+hh/2 and max(points[:,1])>hz-hh/2):
                raise ValueError('Pet opening intersects an installed glazing unit')
        cut_stock(body,(hx-hw/2,y_center-leaf['thickness'],hz-hh/2),
            (hx+hw/2,y_center+leaf['thickness'],hz+hh/2),'pet_opening',names=set(added))
    record={k:v for k,v in profile.items() if k!='parts'}
    record.update(body=body.name,prefix=prefix,x0=x0,z0=z0,u=u,y_center=y_center,
        pane_geoms=[f"{prefix}_{p['glass_name']}"for p in profile['panes']],
        component_geoms=[g.name for g in body.geoms if g.name in added],
        subsequent_stock_preparation='Lock/spindle/pet/strike mortises may remove identified stock after this construction')
    model.meta.setdefault('ordinary_glazing_constructions',[]).append(record)


def finish_glazing(model, phys):
    """Retain exact glass/stop inertia after all actual stock preparations.

    Existing leaf-body catalogue calibration must not change glass density.
    Glass consumes the original material budget via an explicit transfer;
    retainers replace their own already derived allowance with the same BOM.
    Fixed parent-child bodies are actual rigid attachment, with no driven DOF.
    """
    for record in model.meta.get('ordinary_glazing_constructions',[]):
        if record.get('pane_body'):continue
        leaf=model.body(record['body']);prefix=record['prefix']
        panes=[g for g in leaf.geoms if g.name in record['pane_geoms']]
        if {g.name for g in panes}!=set(record['pane_geoms']):
            raise ValueError(f'{leaf.name}: subsequent hardware preparation cut an installed glass pane')
        retainers=[g for g in leaf.geoms if g.name.startswith(prefix+'_glazing_') and g.semantic=='glazing_retainer']
        glass_mass=sum(g.mass()for g in panes);retainer_mass=sum(g.mass()for g in retainers)
        mass=phys['mass'];row=next(r for r in mass['per_body']if r['body']==leaf.name)
        allowance=row['hardware_parts'].get('glazing_retainers',0.)
        if not math.isclose(glass_mass,row['glass_kg'],rel_tol=1e-8,abs_tol=1e-9):
            raise ValueError(f'{leaf.name}: actual glass mass differs from derived material budget')
        if not math.isclose(retainer_mass,allowance,rel_tol=1e-8,abs_tol=1e-9):
            raise ValueError(f'{leaf.name}: actual retaining material differs from its allowance')
        glass=Body(prefix+'_glazing_panes',leaf.name,tiers=ALL_TIERS,semantic='glass',label='Rigidly retained actual glass panes')
        glass.geoms=panes;model.add_body(glass)
        retain=Body(prefix+'_glazing_retainers',leaf.name,tiers=ALL_TIERS,semantic='mechanism',label='Fixed glazing stops and elastomer')
        retain.geoms=retainers;model.add_body(retain)
        moved={g.name for g in panes+retainers};leaf.geoms=[g for g in leaf.geoms if g.name not in moved]
        model.meta.setdefault('material_transfer_bodies',{})[glass.name]=leaf.name
        model.meta.setdefault('mechanism_mass_bodies',[]).append(retain.name)
        row['hardware_parts'].pop('glazing_retainers')
        row['hardware_kg']-=allowance;row['total_kg']-=allowance
        mass['hardware_parts']['glazing_retainers']-=allowance
        mass['hardware_kg']-=allowance;mass['total_kg']-=allowance
        row['glazing_retainer_allowance_replaced_kg']=allowance
        record.update(pane_body=glass.name,retainer_body=retain.name,
            actual_glass_kg=glass_mass,actual_retainer_kg=retainer_mass,
            inertia_scope='Glass retains actual density through material transfer; fixed retainers replace their exact prederived allowance with geometry BOM. Other leaf hardware keeps its existing catalogue calibration.')
