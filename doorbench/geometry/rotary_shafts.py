"""Prepared conventional through-shafts and supported stationary journals.

This repairs installed stock/support geometry. Existing lock ranges, concealed
cam relations and input return laws are preserved, not certified by this step.
Independent entry locksets have their separate frozen geometry implementation.
"""
from __future__ import annotations
import copy,math
from dataclasses import replace
import numpy as np
import trimesh
from .. import hardware as H
from ..ir import Body,ALL_TIERS,QUAT_ID,quat_to_mat,quat_z_to
from . import common as C
from .lock_stock import cut_stock,geom_bounds
from .marine_dogs import bearing_y


def defer_rotary_shaft(model,body,op,u,t,faces):
    model.meta.setdefault('_deferred_rotary_shafts',[]).append(
        {'body':body.name,'operator':op.id,'u':u,'thickness':t,'faces':list(faces)})


def _stock_mass(leaf):
    return {semantic:sum(g.mass() for g in leaf.geoms if g.semantic==semantic) for semantic in ('leaf','glass')}


def _clip(points,axis,value,keep_lower):
    """Clip a convex vertex set by one axis plane; all edges are included."""
    signed=(points[:,axis]-value)*(1 if keep_lower else -1)
    inside=signed<=1e-12;kept=list(points[inside])
    for a,da in zip(points[inside],signed[inside]):
        for b,db in zip(points[~inside],signed[~inside]):
            kept.append(a+(b-a)*(-da)/(db-da))
    if len(kept)<4:return np.empty((0,3))
    verts=np.unique(np.round(kept,12),axis=0)
    if len(verts)<4 or np.linalg.matrix_rank(verts-verts[0],tol=1e-11)<3:return np.empty((0,3))
    return np.asarray(trimesh.convex.convex_hull(verts).vertices)


def _bore_stock(leaf,lower,upper,suffix):
    """Exact square bore in axis-aligned and rotated box stock.

    A rotated brace is divided into convex retained pieces, never one convex
    mesh whose hull would silently fill the prepared hole again in MuJoCo.
    """
    lo,hi=np.asarray(lower),np.asarray(upper);angled=[]
    for g in leaf.geoms:
        if g.semantic not in ('leaf','glass'):continue
        a,b=geom_bounds(g)
        if np.any(np.minimum(b,hi)-np.maximum(a,lo)<=1e-9):continue
        if ((g.type=='box' and not np.allclose(np.abs(quat_to_mat(g.quat)).sum(axis=0),1,atol=1e-9))
                or (g.type=='mesh' and g.mesh.is_convex)):angled.append(g)
    names={g.name for g in leaf.geoms if g.semantic in ('leaf','glass') and all(g is not other for other in angled)}
    result=cut_stock(leaf,lo,hi,suffix,names=names);result['rotated_box_preparations']=[]
    for g in angled:
        source_vertices=(trimesh.creation.box(extents=2*np.asarray(g.size)).vertices if g.type=='box' else g.mesh.vertices)
        vertices=np.asarray(source_vertices)@quat_to_mat(g.quat).T+g.pos
        work=vertices;pieces=[]
        for axis in range(3):
            if not len(work):break
            part=_clip(work,axis,lo[axis],True)
            if len(part):pieces.append(part)
            work=_clip(work,axis,lo[axis],False)
            if not len(work):break
            part=_clip(work,axis,hi[axis],False)
            if len(part):pieces.append(part)
            work=_clip(work,axis,hi[axis],True)
        if not len(work):continue  # AABB overlap alone is not a stock intersection.
        removed_volume=abs(float(trimesh.convex.convex_hull(work).volume))
        leaf.geoms.remove(g);volume=g.volume();outputs=[]
        for k,vertices in enumerate(pieces):
            mesh=trimesh.convex.convex_hull(vertices);v=abs(float(mesh.volume))
            key=C.MESH.key_for('rotary_brace_bore',source=g.name,vertices=np.asarray(mesh.vertices).round(12).tolist())
            part=replace(g,name=g.name+'_'+suffix+'_'+str(k),type='mesh',pos=(0.,0.,0.),quat=QUAT_ID,
                size=(),mesh_name=key,mesh=mesh,mass_override=None if g.mass_override is None else g.mass_override*v/volume)
            leaf.geoms.append(part);outputs.append(part.name)
        result['rotated_box_preparations'].append({'source':g.name,'geoms':outputs,'removed_geometry_volume_m3':removed_volume})
        result['removed_geometry_volume_m3']+=removed_volume;result['removed_geoms'].append(g.name)
    return result


def _account(model,spec,phys,leaf,body,op,removed,preparations):
    mass=phys['mass'];row=next(r for r in mass['per_body'] if r['body']==leaf.name)
    original=copy.deepcopy(row);source_op=H.OPERATORS[spec['operator']['model']]
    partial=source_op.kind in ('handleset','panic_touchbar','panic_crossbar')
    deduction=op.mass if partial else row['hardware_parts'].get('operator',0.)
    available=row['hardware_parts'].get('operator',0.)
    if deduction>available+1e-9:raise ValueError('Rotary component allowance exceeds remaining operator budget')
    for semantic,part in (('leaf','slab_kg'),('glass','glass_kg')):
        amount=removed[semantic]
        if not -1e-9<=amount<=row[part]+1e-9:raise ValueError('Rotary stock removal exceeds owning panel material')
        row[part]-=amount;mass[part]-=amount;row['total_kg']-=amount;mass['total_kg']-=amount
    row['hardware_parts']['operator']-=deduction;mass['hardware_parts']['operator']-=deduction
    for item in (row,mass):item['hardware_kg']-=deduction;item['total_kg']-=deduction
    primary=mass['per_body'][0];mass['dynamics_mass_kg']=primary.get('carried_mass_kg',primary['total_kg'])
    if row is primary:
        density=row['slab_kg']/(row['width']*row['height']);mass['slab_area_density_kg_m2']=density
        mass['reference_unit'].update(slab_kg=row['slab_kg'],glass_kg=row['glass_kg'],hardware_kg=row['hardware_kg'],
            total_kg=row['total_kg'],hardware_parts=copy.deepcopy(row['hardware_parts']),slab_area_density_kg_m2=density)
    if leaf.name in phys.get('per_body_dynamics',{}):
        phys['per_body_dynamics'][leaf.name]['mass'].update(copy.deepcopy(row),dynamics_mass_kg=row.get('carried_mass_kg',row['total_kg']))
    record={'leaf':leaf.name,'operator_body':body.name,'source_operator':source_op.id,'installed_rotary_operator':op.id,
        'original_row':original,'removed_material_kg':removed,'stock_preparations':preparations,
        'removed_operator_allowance_kg':deduction,'retained_operator_allowance_kg':row['hardware_parts']['operator'],
        'allocation_scope':'Passed rotary catalogue mass allocated from larger handleset/panic budget; documented allocation estimate, not a measured OEM subassembly.' if partial else 'Complete owning-panel operator allowance replaced by explicit installed trim/shaft/journal material BOM.'}
    model.meta.setdefault('rotary_shaft_accounting',[]).append(record)


def finish_rotary_shafts(model,spec,phys):
    """Run after cartridge/standoff construction, before inertia reconciliation."""
    pending=model.meta.pop('_deferred_rotary_shafts',[])
    for item in pending:
        body=next((b for b in model.bodies if b.name==item['body']),None)
        # Marine/vault rebuilds replace their former generic inputs completely.
        if body is None:continue
        old_shaft=next((g for g in body.geoms if g.name==body.name+'_spindle' and g.part_label=='Spindle'),None)
        if old_shaft is None:continue
        _prepare(model,spec,phys,body,H.OPERATORS[item['operator']],item)


def _prepare(model,spec,phys,body,op,item):
    leaf=model.body(body.parent);name=body.name;t=float(item['thickness']);u=float(item['u']);faces=item['faces']
    x,_,z=body.pos;steel=C.mat_from_material(model,'stainless','mat_rotary_shaft_support')
    mount=model.add_body(Body(name+'_shaft_support',leaf.name,body.pos,semantic='mechanism',
        label='Fixed prepared through-shaft journals and escutcheon stock'))
    before=_stock_mass(leaf);cuts=[]
    stock_bounds=[geom_bounds(g) for g in leaf.geoms if g.semantic in ('leaf','glass')]
    ymin=min(a[1] for a,b in stock_bounds)-.001;ymax=max(b[1] for a,b in stock_bounds)+.001
    cuts.append(_bore_stock(leaf,(x-.007,ymin,z-.007),(x+.007,ymax,z+.007),name+'_through_bore'))
    # Existing cartridge standoffs are kept as supported physical stock. Their
    # former separate extension is covered by the continuous new steel shaft.
    standoffs=[v for row in model.meta.get('lock_stock',[]) for v in row.get('operator_standoffs',[]) if v['joint']==body.joint.name]
    extensions={v['shaft_geom'] for v in standoffs}
    body.geoms=[g for g in body.geoms if g.name!=name+'_spindle' and g.name not in extensions]
    face_levels={f:t/2 for f in (-1.,1.)};piece_names={};profile_names=[];fixed_cases=[]
    for geom in tuple(body.geoms):
        if geom.type!='mesh' or geom.part_label not in ('Lever','Knob','T-handle','Cremone knob'):continue
        face=1. if geom.name.endswith('_p') else -1.;face_levels[face]=face*geom.pos[1]
        body.geoms.remove(geom);piece_names[geom.name]=[]
        for k,part in enumerate(geom.mesh.split(only_watertight=False)):
            lo,hi=part.bounds
            if lo[2]>=-1e-6 and hi[2]<=.010001 and max(hi[0]-lo[0],hi[1]-lo[1])>.040:continue
            if op.style_params.get('shape')=='safeguard' and np.allclose(hi-lo,(.06,.09,.04),atol=1e-9):
                case=replace(geom,name=geom.name+'_fixed_case',type='box',mesh=None,mesh_name=None,
                    pos=tuple(np.asarray(geom.pos)+quat_to_mat(geom.quat)@((lo+hi)/2)),size=tuple((hi-lo)/2),
                    density=C.M.MATERIALS[op.material].density,collision=True,tiers=ALL_TIERS,
                    part_label='Fixed cold-storage handle housing around prepared shaft journal')
                mount.geoms.append(case);fixed_cases.append((face,case.name));continue
            key=C.MESH.key_for('prepared_rotary_piece',source=geom.mesh_name,part=k)
            new=replace(geom,name=geom.name+'_piece_'+str(k),mesh=part,mesh_name=key,
                        density=C.M.MATERIALS[op.material].density,tiers=ALL_TIERS)
            body.geoms.append(new);piece_names[geom.name].append(new.name);profile_names.append(new.name)
    for g in body.geoms:
        g.tiers=ALL_TIERS
        if g.collision and any(tag in g.name for tag in ('_lever_col_','_hub_col_','_knob_col_','_knob_neck_','_t_col_','_cremone_col_')):g.density=0.
    body.tiers=ALL_TIERS
    # The old rim box is prepared into a real wall shell before drilling its
    # shaft opening. Its parent filtering/visual allowances cannot hide stock.
    cases=[g for g in leaf.geoms if g.name==name+'_rim_case']
    for g in cases:
        leaf.geoms.remove(g);g=replace(g,pos=tuple(np.asarray(g.pos)-body.pos),tiers=ALL_TIERS,density=C.M.MATERIALS['cast_iron'].density)
        mount.geoms.append(g);a,b=geom_bounds(g)
        cuts.append(cut_stock(mount,a+.002,b-.002,'hollow_case',names={g.name}))
        case_names={v.name for v in mount.geoms if v.name.startswith(g.name+'_hollow_case_')}
        cuts.append(cut_stock(mount,(-.007,a[1]-.001,-.007),(.007,b[1]+.001,.007),'spindle_opening',names=case_names))
    if cases:
        model.meta['clearance_allow']=[a for a in model.meta.get('clearance_allow',[]) if not (a[1]==name+'_rim_case' and a[0] in (name+'_knob_*',name+'_spindle'))]
    bearings=[];plates=[];spacer_names=[]
    for face in faces:
        tag='p' if face>0 else 'n';level=face_levels[face]
        installed=face in faces
        neck=.011 if op.kind in ('knob','keypad_deadbolt','cremone') else max(op.style_params.get('diameter',.019)*.65,.011)
        if op.style_params.get('square'):neck=max(neck,op.style_params.get('diameter',.019)*.65*math.sqrt(2))
        bore=((neck if installed else .006)+.0006)/math.cos(math.pi/12)
        rose=max(op.style_params.get('rose_diameter',.064)/2,bore+.004)
        if any(f==face for f,_ in fixed_cases):
            # Keep the actual 60×90 mm fixed case, with a central prepared
            # opening around a20 mm-radius journal and the rotating hub.
            rose=max(.020,bore+.004)
            names={n for f,n in fixed_cases if f==face};opening=rose+.001
            cuts.append(cut_stock(mount,(-opening,-t/2-.1,-opening),(opening,t/2+.1,opening),
                'prepared_housing',names=names))
        # Raised battens/diagonal stock require an actual journal seat. Keep
        # the captured mounting plane and prepare the surrounding face stock,
        # rather than burying a stationary rose inside an uncut wood brace.
        ya,yb=(level,ymax) if face>0 else (ymin,-level)
        if yb-ya>1e-9:
            cuts.append(_bore_stock(leaf,(x-rose-.001,ya,z-rose-.001),
                (x+rose+.001,yb,z+rose+.001),name+'_journal_seat_'+tag))
        gap=0.
        for old in standoffs:
            if old['face']!=face:continue
            original={g.name:g for g in leaf.geoms if g.name in old['spacer_geoms']}
            leaf.geoms=[g for g in leaf.geoms if g.name not in original]
            prefix=old['spacer_geoms'][0].rsplit('_',1)[0];length=old['standoff_m']-.00075
            first=len(mount.geoms)
            bearing_y(mount,prefix,(0,face*(t/2+length/2),0),steel,inner=.00675,outer=max(.020,bore+.002),half_length=length/2,semantic='mechanism')
            old['spacer_geoms']=[g.name for g in mount.geoms[first:]];spacer_names.extend(old['spacer_geoms']);gap=.00075
            old['shaft_geom']=name+'_spindle'
            old['moving_geoms']=[part for oldname in old['moving_geoms'] for part in piece_names.get(oldname,[oldname])]
        first=len(mount.geoms)
        bearing_y(mount,name+'_journal_'+tag,(0,face*(level+.004-gap/2),0),steel,
            inner=bore,outer=rose,half_length=.004+gap/2,semantic='mechanism')
        bearings.extend(g.name for g in mount.geoms[first:])
        for g in body.geoms:
            if g.name==name+'_hub_col_'+tag:g.size=(neck,.024);g.pos=(0,face*(level+.028),0)
        body.geoms.append(C.cyl(name+'_shaft_collar_'+tag,(0,face*(level+.0105),0),bore+.002,.0015,
            steel,(0,1,0),7850,True,True,ALL_TIERS,'mechanism','Retained shaft shoulder outside fixed journal'))
        plate=next((g for g in leaf.geoms if g.name==name+'_escutcheon_'+tag),None)
        if plate is not None:
            leaf.geoms.remove(plate);plate=replace(plate,pos=tuple(np.asarray(plate.pos)-body.pos),collision=True,
                density=C.M.MATERIALS[op.material].density,tiers=ALL_TIERS,mass_override=None)
            mount.geoms.append(plate);opening=rose+.001
            if min(plate.size[0],plate.size[2])<=opening:raise ValueError('Conventional escutcheon has no remaining journal stock')
            cuts.append(cut_stock(mount,(-opening,-t/2-.004,-opening),(opening,t/2+.004,opening),
                'prepared_journal',names={plate.name}))
            plates.extend(g.name for g in mount.geoms if g.name.startswith(plate.name+'_prepared_journal_'))
    # A handleset interior knob or panic exterior trim drives the concealed
    # follower through a one-face stub. It must not project through the other
    # face into that independently installed mechanism's housing. The finite
    # installed-face journal supplies its supported shaft span.
    low=-(face_levels[-1.]+.032) if -1. in faces else 0.
    high=face_levels[1.]+.032 if 1. in faces else 0.
    body.geoms.append(C.cyl(name+'_spindle',(0,(low+high)/2,0),.006,(high-low)/2,steel,(0,1,0),7850,
        True,True,ALL_TIERS,'mechanism','Continuous retained through-shaft in prepared stock' if len(faces)==2 else 'Supported one-face stub ending at latch centre plane'))
    # A point on a knob's rotation axis cannot supply its turning moment.
    # Bind useful actual surface patches; runtime may explicitly apply the
    # declared opposed pair rather than substituting free joint torque.
    opposed=[];input_surfaces={}
    for site in body.sites:
        if site.role!='grip' or site.name.rsplit('_',1)[-1] not in ('n','p'):continue
        tag=site.name.rsplit('_',1)[-1];face=1. if tag=='p' else -1.
        if op.kind in ('knob','keypad_deadbolt','cremone'):
            geom=next(g for g in body.geoms if g.name in (name+'_knob_col_'+tag,name+'_cremone_col_'+tag))
            radius=geom.size[0];normal=np.array([-.8,face*.6,0.]);other=np.array([.8,face*.6,0.])
            site.pos=tuple(np.asarray(geom.pos)+radius*normal)
            opposed.append(replace(site,name=site.name+'_opposed',pos=tuple(np.asarray(geom.pos)+radius*other),quat=tuple(quat_z_to(other))))
            input_surfaces[site.name]=geom.name;input_surfaces[site.name+'_opposed']=geom.name
        elif op.kind=='t_handle':
            geom=next(g for g in body.geoms if g.name==name+'_t_col_'+tag);normal=np.array([0.,face,0.])
            site.pos=(site.pos[0],geom.pos[1]+face*geom.size[1],0.);input_surfaces[site.name]=geom.name
        else:
            geom=next(g for g in body.geoms if g.name==name+'_lever_col_'+tag);normal=np.array([0.,face,0.])
            site.pos=(site.pos[0],geom.pos[1]+face*geom.size[0],0.);input_surfaces[site.name]=geom.name
        site.quat=tuple(quat_z_to(normal));site.size=.005
    body.sites.extend(opposed)
    body.mass_override=None
    for n in (body.name,mount.name):
        if n not in model.meta.setdefault('mechanism_mass_bodies',[]):model.meta['mechanism_mass_bodies'].append(n)
    if body.joint.name not in model.meta.setdefault('physical_inertia_joints',[]):model.meta['physical_inertia_joints'].append(body.joint.name)
    after=_stock_mass(leaf);removed={key:before[key]-after[key] for key in before}
    _account(model,spec,phys,leaf,body,op,removed,cuts)
    model.meta.setdefault('rotary_shafts',[]).append({'leaf':leaf.name,'body':body.name,'joint':body.joint.name,
        'shaft_geom':name+'_spindle','support_body':mount.name,'bearing_geoms':bearings,'plate_geoms':plates,
        'spacer_geoms':spacer_names,'handle_geometry':profile_names,'faces':faces,'shaft_radius_m':.006,
        'leaf_stock_geoms':[g.name for g in leaf.geoms if g.semantic in ('leaf','glass')],
        'fixed_parent_geoms':[g.name for g in leaf.geoms],
        'support_geoms':[g.name for g in mount.geoms],
        'input_surfaces':input_surfaces,'input_model':'opposed_surface_pair' if op.kind in ('knob','keypad_deadbolt','cremone') else 'single_surface_force',
        'input_sites_by_face':{tag:[s.name for s in body.sites if s.name in (name+'_grip_'+tag,name+'_grip_'+tag+'_opposed')] for tag in ('n','p')},
        'operator_force_cap_N':op.operable_force_limit,'operator_model':op.id,
        'nominal_operator_travel_rad':op.travel,'operator_dead_travel_rad':op.dead_travel,
        'stock_clearance_m':.001,'scope':'Prepared shaft/stock and connected fixed journals. Existing lock range, concealed cams and return law retained; no complete lockset or service-task certification.'})
