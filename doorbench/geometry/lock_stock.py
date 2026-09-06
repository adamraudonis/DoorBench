"""Original lock cartridges with actual stock cavities and open metal guides.

These simplified cartridges use their own documented envelopes. They are not
OEM lock internals or a strength rating. Stock is removed in every geometry
representation; parent-child contact filtering is never a substitute for it.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from ..ir import ALL_TIERS, QUAT_ID, quat_to_mat
from . import common as C


def geom_bounds(geom):
    R=quat_to_mat(geom.quat);p=np.asarray(geom.pos,float)
    if geom.type=='box':h=np.abs(R)@np.asarray(geom.size)
    elif geom.type in ('cylinder','capsule'):
        axis=R[:,2];radius,half=geom.size
        h=np.abs(axis)*half+(radius if geom.type=='capsule' else radius*np.sqrt(np.maximum(0,1-axis*axis)))
    elif geom.type=='sphere':h=np.full(3,geom.size[0])
    elif geom.type=='mesh':
        points=np.asarray(geom.mesh.vertices)@R.T+p
        return points.min(axis=0),points.max(axis=0)
    else:raise ValueError(f'Unsupported stock geometry {geom.name}: {geom.type}')
    return p-h,p+h


def cut_stock(body, lower, upper, suffix, *, names=None):
    """Exact box subtraction, preserving densities and proportional overrides."""
    lo,hi=np.asarray(lower,float),np.asarray(upper,float)
    if not np.isfinite([lo,hi]).all() or np.any(hi<=lo):raise ValueError('Invalid stock cavity bounds')
    result=[];removed=[];removed_volume=0.
    for g in body.geoms:
        selected=g.name in names if names is not None else g.semantic in ('leaf','glass')
        if not selected:result.append(g);continue
        a,b=geom_bounds(g);c,d=np.maximum(a,lo),np.minimum(b,hi)
        if np.any(d-c<=1e-9):result.append(g);continue
        R=quat_to_mat(g.quat)
        if g.type!='box' or not np.allclose(np.abs(R).sum(axis=0),1,atol=1e-9):
            raise ValueError(f'Lock cavity intersects unsupported non-box stock {g.name}')
        removed.append(g.name);removed_volume+=float(np.prod(d-c))
        work_a,work_b=a.copy(),b.copy();pieces=[]
        for axis in range(3):
            if c[axis]>work_a[axis]+1e-9:
                end=work_b.copy();end[axis]=c[axis];pieces.append((work_a.copy(),end))
            if d[axis]<work_b[axis]-1e-9:
                start=work_a.copy();start[axis]=d[axis];pieces.append((start,work_b.copy()))
            work_a[axis],work_b[axis]=c[axis],d[axis]
        volume=float(np.prod(b-a))
        for i,(aa,bb) in enumerate(pieces):
            mass=None if g.mass_override is None else g.mass_override*float(np.prod(bb-aa))/volume
            result.append(replace(g,name=f'{g.name}_{suffix}_{i}',pos=tuple((aa+bb)/2),size=tuple((bb-aa)/2),quat=QUAT_ID,mass_override=mass))
    body.geoms=result
    return {'lower':lo.tolist(),'upper':hi.tolist(),'removed_geoms':removed,'removed_geometry_volume_m3':removed_volume}


def expand_box_mesh_stock(body,names):
    """Represent an authored union of exact mesh boxes as the same primitives.

    This supports existing keypad case stock without leaving a solid convex
    hull over its new socket. Rounded or otherwise non-box CAD is rejected.
    """
    result=[];expanded=set()
    for g in body.geoms:
        if g.name not in names or g.type!='mesh':result.append(g);continue
        pieces=list(g.mesh.split(only_watertight=False));volume=sum(abs(float(p.volume)) for p in pieces)
        for i,part in enumerate(pieces):
            mesh=part.copy();mesh.vertices=np.asarray(part.vertices)@quat_to_mat(g.quat).T+g.pos
            a,b=mesh.bounds;v=float(np.prod(b-a))
            if not np.isclose(abs(mesh.volume),v,rtol=1e-7,atol=1e-12):raise ValueError(f'Cannot expand non-box stock mesh {g.name}')
            name=f'{g.name}_stock_{i}';expanded.add(name)
            result.append(replace(g,name=name,type='box',pos=tuple((a+b)/2),size=tuple((b-a)/2),quat=QUAT_ID,
                mesh=None,mesh_name=None,mass_override=None if g.mass_override is None else g.mass_override*v/volume))
    body.geoms=result
    return (names-{g.name for g in body.geoms if g.type=='mesh'})|expanded


def add_bolt_cartridge(model,leaf,body,*,name,u,t,edge,z,y=0.,faceplate=True):
    """Clear the complete bolt sweep and mount its open guide to real stock.

    Thin stock receives a local metal edge cartridge with padded face straps;
    no fictitious deep blind mortise is cut into a thin glass sheet.
    """
    lower=[];upper=[]
    for g in body.geoms:
        a,b=geom_bounds(g);lower.append(a);upper.append(b)
    a=np.min(lower,axis=0)+body.pos;b=np.max(upper,axis=0)+body.pos
    if body.joint:
        q=body.joint;initial_geometry=float(q.modeled_at or 0.)
        endpoints=[np.asarray(q.axis)*(value-initial_geometry) for value in q.range]
        a+=np.min(endpoints,axis=0);b+=np.max(endpoints,axis=0)
    clearance=.00075;wall=.0015
    inner_a=a-clearance;inner_b=b+clearance
    # The protruding nose remains outside the stock; the open guide ends flush
    # at the edge, with no transverse plate across the bolt's path.
    if u>0:inner_b[0]=edge+.0001
    else:inner_a[0]=edge-.0001
    outer_a=inner_a-wall;outer_b=inner_b+wall
    if u>0:outer_b[0]=edge
    else:outer_a[0]=edge
    cut=cut_stock(leaf,outer_a,outer_b,name+'_mortise')
    # Surface exit-device/night-lock housings are real stock too. Their old
    # solid case proxy cannot enclose a moving bolt merely because it shares
    # the parent leaf. Cut the full nose envelope, including beyond the edge.
    cases={g.name for g in leaf.geoms if g.type=='box' and any(tag in g.name for tag in ('_device_case','_night_latch_case'))}
    case_cut=cut_stock(leaf,a-clearance-wall,b+clearance+wall,name+'_case_bolt_passage',names=cases) if cases else None
    mat=C.mat_from_material(model,'stainless','mat_lock_cartridge')
    shell=[]
    for axis in (1,2):
        for side in (-1,1):
            aa=outer_a.copy();bb=outer_b.copy()
            if axis==2:aa[1]=inner_a[1];bb[1]=inner_b[1]
            if side<0:bb[axis]=inner_a[axis]
            else:aa[axis]=inner_b[axis]
            g=C.box(f'{name}_guide_{axis}_{side}',tuple((aa+bb)/2),tuple((bb-aa)/2),mat,7900,
                    tiers=body.tiers,semantic='lock',label='Open lock cartridge guide wall')
            leaf.geoms.append(g);shell.append(g.name)
    aa=outer_a.copy();bb=outer_b.copy();aa[1:]=inner_a[1:];bb[1:]=inner_b[1:]
    if u>0:bb[0]=inner_a[0]
    else:aa[0]=inner_b[0]
    g=C.box(name+'_guide_back',tuple((aa+bb)/2),tuple((bb-aa)/2),mat,7900,tiers=body.tiers,semantic='lock',label='Lock cartridge rear wall')
    leaf.geoms.append(g);shell.append(g.name)
    thin=bool(abs(y)<1e-9 and max(abs(outer_a[1]),abs(outer_b[1]))>t/2)
    mounts=[]
    if thin:
        # Ground edge notch in glass / prepared steel stock. Four clamping
        # straps contact intact stock above and below the notch through pads.
        rubber=C.mat_from_material(model,'rubber','mat_lock_mount_pad')
        # Keep clamping ears inboard of the frame rebate. The guide itself
        # reaches the leaf edge inside the strike preparation, but its ears
        # must not extend into the uncut stop above/below that preparation.
        clamp_a=outer_a[0];clamp_b=outer_b[0]
        if u>0:clamp_b=min(clamp_b,edge-.015)
        else:clamp_a=max(clamp_a,edge+.015)
        if clamp_b-clamp_a<.012:raise ValueError('Thin-stock cartridge lacks12 mm of supported inboard clamp land')
        cx=(clamp_a+clamp_b)/2;hx=(clamp_b-clamp_a)/2
        for face in (-1,1):
            for side in (-1,1):
                zz=(outer_a[2]-.008 if side<0 else outer_b[2]+.008)
                pad=C.box(f'{name}_mount_pad_{face}_{side}',(cx,face*(t/2+.0005),zz),(hx,.0005,.008),rubber,1100,
                    tiers=body.tiers,semantic='lock',label='Protective cartridge clamp pad on intact sheet')
                # Bridge from pad to case side; its lower edge is bonded to the
                # guide's top/bottom face, not floating outside the sheet.
                outer=max(t/2+.003,abs(outer_a[1]),abs(outer_b[1]))
                plate=C.box(f'{name}_mount_strap_{face}_{side}',(cx,face*(t/2+.001+outer)/2,zz),
                    (hx,(outer-(t/2+.001))/2,.008),mat,7900,tiers=body.tiers,semantic='lock',label='Metal edge-cartridge clamping strap')
                leaf.geoms.extend([pad,plate]);mounts.extend([pad.name,plate.name])
    # All cartridge masses come from their material geometry, attached directly
    # to the carried panel; the parent assembly includes them as leaf hardware.
    record={'name':name,'leaf_body':leaf.name,'bolt_body':body.name,'bolt_geoms':[g.name for g in body.geoms],'tiers':sorted(body.tiers),
            'swept_bolt_lower':a.tolist(),'swept_bolt_upper':b.tolist(),'clearance_m':clearance,
            'stock_cut':cut,'guide_geoms':shell,'mount_geoms':mounts,'thin_stock_edge_cartridge':thin,
            'scope':'Original simplified open metal guide and prepared stock; internal cam and strength are not certified.'}
    model.meta.setdefault('lock_stock',[]).append(record)
    if thin:
        _clear_knob_mounts(model,leaf,record,t)
    if case_cut:record['case_cut']=case_cut
    return record


def _clear_knob_mounts(model,leaf,record,thickness):
    """Stand thin-sheet knob assemblies on bored supports above cartridge stock.

    A mortise-sized edge cartridge can stand proud of a6 mm mesh panel. Its
    guide walls must not pass through the adjacent knob/rose. Only intersecting
    knob assemblies move; the supported spindle extends by the same distance.
    """
    from ..ir import quat_to_mat
    from .marine_dogs import bearing_y
    stock=[g for g in leaf.geoms if g.name in record['guide_geoms']+record['mount_geoms']]
    mounts=[]
    for operator in model.children(leaf.name):
        if not operator.joint or operator.joint.type!='hinge':continue
        for face,tag in ((-1,'n'),(1,'p')):
            knob=next((g for g in operator.geoms if g.type=='mesh' and g.name==operator.name+'_knob_'+tag),None)
            if knob is None:continue
            points=np.asarray(knob.mesh.vertices)@quat_to_mat(knob.quat).T+knob.pos
            radius=float(np.linalg.norm(points[:,[0,2]],axis=1).max())
            back=float((face*points[:,1]).min());front=float((face*points[:,1]).max())
            axis=np.asarray(operator.pos)[[0,2]];height=0.
            for geom in stock:
                a,b=geom_bounds(geom)
                separation=np.maximum(np.maximum(a[[0,2]]-axis,axis-b[[0,2]]),0.)
                if np.linalg.norm(separation)>=radius+.001:continue
                lo,hi=sorted((face*a[1],face*b[1]))
                if lo<front and hi>back-.001:height=max(height,hi+.001-back)
            if height<=1e-8:continue
            shifted=[]
            for geom in operator.geoms:
                if geom.name.endswith('_'+tag) and any(part in geom.name for part in ('_knob_','_cylinder_','_keyway_','_turn_button_')):
                    geom.pos=(geom.pos[0],geom.pos[1]+face*height,geom.pos[2]);shifted.append(geom.name)
            sites=[]
            for site in operator.sites:
                if site.name.endswith('_'+tag):
                    site.pos=(site.pos[0],site.pos[1]+face*height,site.pos[2]);sites.append(site.name)
            prefix=operator.name+'_cartridge_standoff_'+tag
            steel=C.mat_from_material(model,'stainless','mat_operator_standoff')
            before={g.name for g in leaf.geoms}
            # The rotating neck starts at the raised face. Leave an actual
            # axial running gap rather than a coincident bearing end face.
            spacer_length=height-.00075
            bearing_y(leaf,prefix,(operator.pos[0],face*(thickness/2+spacer_length/2),operator.pos[2]),steel,
                inner=.00675,outer=.012,half_length=spacer_length/2,semantic='mechanism')
            spacers=[g.name for g in leaf.geoms if g.name not in before]
            operator.geoms.append(C.cyl(prefix+'_shaft',(0,face*(thickness/2+height/2),0),.006,height/2+.001,
                steel,(0,1,0),7850,True,True,ALL_TIERS,'mechanism','Spindle extension through prepared standoff bore'))
            spindle_bore=cut_stock(leaf,
                (operator.pos[0]-.00675,-thickness/2-.001,operator.pos[2]-.00675),
                (operator.pos[0]+.00675,thickness/2+.001,operator.pos[2]+.00675),prefix+'_spindle_bore')
            mounts.append({'joint':operator.joint.name,'face':face,'standoff_m':height,
                'spacer_geoms':spacers,
                'moving_geoms':shifted,'sites':sites,'shaft_geom':prefix+'_shaft','spindle_bore':spindle_bore})
    if mounts:record.setdefault('operator_standoffs',[]).extend(mounts)
