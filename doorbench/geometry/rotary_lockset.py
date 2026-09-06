"""Independent entry/privacy trim with a real exterior locking catch.

The two one-sided latch cams are ideal constraints; their concealed internal
contact profile is not claimed. Stub shafts, bearings, prepared stock and the
exterior catch are actual rigid geometry. This is original generic hardware,
not an OEM lock chassis or a simulation of key insertion/pin tumblers.
"""
from __future__ import annotations

import copy
import math
from dataclasses import replace
import numpy as np

from .. import hardware as H
from ..ir import Body, Joint, Site, ALL_TIERS, QUAT_ID, quat_z_to
from . import common as C
from .lock_stock import cut_stock
from .marine_dogs import bearing_y


LOCK_KINDS = frozenset(('privacy_button','keyed_cylinder','keypad_code','card_reader'))
OP_KINDS = frozenset(('lever','knob','keypad_lever','card_lever'))


def applicable(spec,op,faces):
    return (spec['family'] in ('swing_single','automatic_swing','pivot') and
            H.LOCKS[spec['lock']['model']].kind in LOCK_KINDS and
            op.kind in OP_KINDS and set(faces)=={-1.,1.} and
            not op.style_params.get('childproof_cover'))


def split_rotary_lockset(model,leaf,body,spec,phys,op,u,t,faces):
    """Split the installed trim; preserve the robot-face body/joint name."""
    if not applicable(spec,op,faces):return body
    name=body.name;x,_,z=body.pos
    approach=1. if spec['robot'].get('approach_side','-y')=='+y' else -1.
    outside=approach if spec['robot']['robot_outside'] else -approach
    steel=C.mat_from_material(model,'stainless','mat_entry_chassis')
    mount=model.add_body(Body(name+'_chassis',leaf.name,body.pos,semantic='mechanism',
                             label='Supported independent entry-trim shafts and catch housing'))
    original_sites=body.sites;original_geoms=[]
    for geom in body.geoms:
        if geom.type!='mesh' or not any(k in geom.name for k in ('_lever_','_knob_')):
            original_geoms.append(geom);continue
        # Keep the actual disconnected pieces. One convex hull around the
        # entire lever/rose fills the empty space behind the handle.
        for k,part in enumerate(geom.mesh.split(only_watertight=False)):
            lo,hi=part.bounds
            if lo[2]>-.000001 and hi[2]<=.010001 and max(hi[0]-lo[0],hi[1]-lo[1])>.040:
                continue  # replace solid rotating rose by fixed bored support
            key=C.MESH.key_for('independent_trim_piece',source=geom.mesh_name,part=k)
            original_geoms.append(replace(geom,name=geom.name+'_part_'+str(k),mesh_name=key,mesh=part))
    drivers={};backed=[mount.name];shaft_geoms=[];bearing_geoms=[]
    before=sum(g.mass() for g in leaf.geoms if g.semantic in ('leaf','glass'))
    cut=cut_stock(leaf,(x-.007,-t/2-.001,z-.007),(x+.007,t/2+.001,z+.007),name+'_split_shaft_bore')
    removed=before-sum(g.mass() for g in leaf.geoms if g.semantic in ('leaf','glass'))
    for face in (-1.,1.):
        tag='p' if face>0 else 'n'
        if face==approach:driver=body
        else:
            driver=model.add_body(Body(name+('_inside' if face==-outside else '_outside'),leaf.name,body.pos,
                joint=replace(body.joint,name=name+('_inside' if face==-outside else '_outside')+'_hinge'),
                semantic='operator',label=op.name+' — independent '+('inside' if face==-outside else 'outside')))
        driver.geoms=[g for g in original_geoms if g.name.endswith('_'+tag) or '_'+tag+'_part_' in g.name]
        # These are geometry-backed bodies: removing even decorative trim in
        # a reduced tier would silently alter its physical mass and inertia.
        for geom in driver.geoms:geom.tiers=ALL_TIERS
        driver.sites=[s for s in original_sites if s.name.endswith('_'+tag)]
        driver.joint.range=(0.,op.travel)
        driver.joint.notes='Independent inside egress' if face==-outside else 'Exterior trim arrested by native catch contact when engaged'
        driver.joint.robot_interactive=face==approach
        # Two separate stubs end 1 mm apart. Their full-length bores are real,
        # and the external necks join the existing lever/knob hubs.
        end=t/2+.032;start=.0005
        shaft=C.cyl(driver.name+'_split_spindle',(0,face*(start+end)/2,0),.006,(end-start)/2,
                    steel,(0,1,0),7850,True,True,ALL_TIERS,'mechanism','Independent retained stub shaft')
        driver.geoms.append(shaft);shaft_geoms.append(shaft.name)
        # Mesh and collision primitive represent the same handle material.
        # Retain the visible authored material; do not count its proxy twice.
        for g in driver.geoms:
            if g.collision and any(k in g.name for k in ('_lever_col_','_hub_col_','_knob_col_','_knob_neck_')):g.density=0.
        opposing_sites=[]
        for site in driver.sites:
            if op.kind=='knob':
                geom=next(g for g in driver.geoms if '_knob_col_' in g.name)
                radius=geom.size[0];normal=np.array([-u*.8,face*.6,0.])
                site.pos=tuple(np.asarray(geom.pos)+radius*normal)
                opposite=np.array([u*.8,face*.6,0.])
                opposing_sites.append(replace(site,name=site.name+'_opposed',
                    pos=tuple(np.asarray(geom.pos)+radius*opposite),quat=tuple(quat_z_to(opposite)),size=.005))
            else:
                geom=next(g for g in driver.geoms if '_lever_col_' in g.name)
                normal=np.array([0.,face,0.]);site.pos=(site.pos[0],geom.pos[1]+face*geom.size[0],0.)
            site.quat=tuple(quat_z_to(normal));site.size=.005
        driver.sites.extend(opposing_sites)
        neck_radius=.011 if op.kind=='knob' else max(op.style_params.get('diameter',.019)*.65,.011)
        if op.style_params.get('square'):neck_radius=max(neck_radius,op.style_params.get('diameter',.019)*.65*math.sqrt(2))
        bore=(neck_radius+.0006)/math.cos(math.pi/12)
        rose=max(op.style_params.get('rose_diameter',.064)/2,bore+.004)
        for g in driver.geoms:
            if '_hub_col_' in g.name:
                g.size=(neck_radius,.024);g.pos=(0,face*(t/2+.028),0)
        for lane,half in ((face*(t/2+.004),.004),):
            first=len(mount.geoms)
            bearing_y(mount,driver.name+'_bearing',(0,lane,0),steel,inner=bore,outer=rose,half_length=half,semantic='mechanism')
            bearing_geoms.extend(g.name for g in mount.geoms[first:])
        driver.geoms.append(C.cyl(driver.name+'_retainer',(0,face*(t/2+.0105),0),bore+.002,.0015,
                steel,(0,1,0),7850,True,True,ALL_TIERS,'mechanism','Retaining collar outside bearing shoulder'))
        drivers[face]=driver;backed.append(driver.name)
    exterior=drivers[outside];inside=drivers[-outside]
    # Exterior cam has a small radial arrest lug. A separate axial pin blocks
    # its downward sweep; withdrawing the pin clears the entire lever stroke.
    lane=outside*(t/2+.016)
    pin_radius=max(.034,op.style_params.get('diameter',0.)/2+.007 if op.kind=='knob' else 0.)
    inner=.012;outer=pin_radius+.0035
    lug=C.box(name+'_outside_catch_lug',(-u*(inner+outer)/2,lane,0),((outer-inner)/2,.002,.003),steel,7850,
              True,True,ALL_TIERS,'mechanism','Exterior spindle keyed locking lug')
    exterior.geoms.append(lug)
    # Connect the lug to its shaft through a keyed radial web.
    exterior.geoms.append(C.box(name+'_outside_cam_web',(-u*.006,lane,0),(.008,.002,.004),steel,7850,
                         True,True,ALL_TIERS,'mechanism','Keyed exterior cam web'))
    pinx=-u*pin_radius;pinz=-.0068;stroke=.008
    catch=model.add_body(Body(name+'_exterior_catch',leaf.name,(x+pinx,lane,z+pinz),
        joint=Joint(name+'_exterior_catch_slide','slide',(0,outside,0),range=(-.002,stroke+.002),
                    stiffness=180.,springref=-.005,damping=12.,frictionloss=.02,
                    initial=0. if spec['lock']['engaged'] else stroke,role='lock',robot_interactive=False,
                    label='Axial catch: native pin contact locks only exterior trim'),semantic='lock'))
    catch.geoms.append(C.cyl(name+'_catch_pin',(0,outside*.005,0),.003,.007,steel,(0,1,0),7850,
                            True,True,ALL_TIERS,'lock','Actual exterior cam arrest pin'))
    catch.geoms.append(C.cyl(name+'_catch_collar',(0,outside*.012,0),.005,.0015,steel,(0,1,0),7850,
                            True,True,ALL_TIERS,'lock','Catch stroke shoulder'))
    # The guide sits beyond the cam plane; its bridge is seated on the leaf.
    first=len(mount.geoms)
    bearing_y(mount,name+'_catch_guide',(pinx,lane+outside*.006,pinz),steel,
              inner=.0035,outer=.006,half_length=.003,semantic='mechanism')
    guides=[g.name for g in mount.geoms[first:]]
    mount.geoms.append(C.box(name+'_catch_support',(pinx-u*.010,outside*(t/2+.0225),pinz),
        (.004,.0225,.008),steel,7850,True,True,ALL_TIERS,'mechanism','Catch guide reaction bridge seated on leaf'))
    # Guide/collar faces physically stop both ends. Wider joint limits are
    # safety bounds only. The housing's open bore never intersects the pin.
    for end,offset in (('front',.0085),('rear',.0235)):
        for axis in (0,2):
            for sign in (-1,1):
                center=np.array([pinx,lane+outside*offset,pinz]);center[axis]+=sign*.00525
                half=[.0035,.002,.007];half[axis]=.00175
                if axis==0:half[2]=.007
                else:half[0]=.0035
                mount.geoms.append(C.box(f'{name}_catch_{end}_stop_{axis}_{sign}',tuple(center),tuple(half),
                    steel,7850,True,True,ALL_TIERS,'mechanism','Prepared square-bore catch end plate'))
    # Exact parent-child pairs are needed for the collar/guide reaction.
    collar=name+'_catch_collar';stop_names=[g.name for g in mount.geoms if g.name.startswith((name+'_catch_front_stop_',name+'_catch_rear_stop_'))]
    for stop in stop_names:model.meta.setdefault('native_contact_pairs',[]).append({'geom1':collar,'geom2':stop,
        'friction':(.08,.08,.002,.0001,.0001),'solref':(.002,1.),'solimp':(.99,.999,.0001)})
    # No joint-range lock or primary weld represents this catch.
    for g in exterior.geoms+catch.geoms:
        if g.collision:g.solref=(.002,1.);g.solimp=(.99,.999,.0001);g.friction=(.08,.002,.0001)
    backed.append(catch.name)
    model.meta.setdefault('mechanism_mass_bodies',[]).extend(backed)
    model.meta.setdefault('physical_inertia_joints',[]).extend([inside.joint.name,exterior.joint.name,catch.joint.name])
    model.meta.setdefault('inside_egress_inputs',[]).append(inside.joint.name)
    row={'leaf':leaf.name,'inside_joint':inside.joint.name,'outside_joint':exterior.joint.name,
         'inside_face':-outside,'outside_face':outside,'catch_joint':catch.joint.name,'catch_body':catch.name,
         'catch_geom':name+'_catch_pin','cam_geom':lug.name,'catch_stroke_m':stroke,'released_threshold_m':.0075,
         'catch_collar_geom':collar,
         'catch_radial_offset_m':pin_radius,
         'catch_force_cap_N':8.,'released_by_default':not spec['lock']['engaged'],
         'credential_available':bool(spec['lock']['robot_side_release']),'lock_kind':H.LOCKS[spec['lock']['model']].kind,
         'guide_geoms':guides,'support_geom':name+'_catch_support','shaft_geoms':shaft_geoms,'bearing_geoms':bearing_geoms,
         'operator_travel_rad':op.travel,'operator_force_cap_N':op.operable_force_limit,'native_stop_geoms':stop_names,
         'input_model':'opposed_surface_pair' if op.kind=='knob' else 'single_surface_force',
         'input_sites':{driver.joint.name:[site.name for site in driver.sites if site.role=='grip'] for driver in drivers.values()},
         'input_force_scope':'Per surface point; knob uses two equal/opposite tangential forces, total absolute cap twice the per-point cap. No free joint torque.',
         'scope':'Independent stub shafts and physical exterior catch. Concealed latch cams ideal and one-sided. Key insertion, tumbler coding and electronics are not simulated.'}
    model.meta.setdefault('rotary_locksets',[]).append(row)
    model.meta['native_timestep_s']=min(.00025,model.meta.get('native_timestep_s',.002))
    model.meta['native_arena_memory_mib']=max(64,model.meta.get('native_arena_memory_mib',16))
    _account_material(model,phys,leaf,removed,cut)
    return body


def _account_material(model,phys,leaf,removed,cut):
    """Replace the operator allowance with actual trim/catch BOM once."""
    mass=phys['mass'];rows=mass['per_body']
    matching=[r for r in rows if r['body']==leaf.name]
    if len(matching)!=1 or model.meta.get('rotary_material_accounting'):
        raise ValueError('Independent lockset requires one fresh leaf budget')
    row=matching[0];old=copy.deepcopy(row);hardware=row['hardware_parts'].get('operator',0.)
    if not 0<=removed<row['slab_kg']:raise ValueError('Invalid independent lockset stock deduction')
    row['hardware_parts']['operator']=0.;mass['hardware_parts']['operator']-=hardware
    for part in (row,mass):
        part['slab_kg']-=removed;part['hardware_kg']-=hardware;part['total_kg']-=removed+hardware
    mass['dynamics_mass_kg']=row['total_kg'];mass['slab_area_density_kg_m2']=row['slab_kg']/(row['width']*row['height'])
    mass['reference_unit'].update(slab_kg=row['slab_kg'],hardware_kg=row['hardware_kg'],total_kg=row['total_kg'],
        hardware_parts=copy.deepcopy(row['hardware_parts']),slab_area_density_kg_m2=mass['slab_area_density_kg_m2'])
    phys['per_body_dynamics'][leaf.name]['mass'].update(copy.deepcopy(row),dynamics_mass_kg=row['total_kg'])
    model.meta['rotary_material_accounting']={'original_row':old,'removed_stock_kg':removed,
        'removed_operator_allowance_kg':hardware,'cut':cut,'scope':'Existing handle meshes count once; real shafts, catch and supports use their own material BOM. Other lock/latch allowances retained.'}


def add_inside_cam(model,primary_joint):
    """Each cam constrains the full stroke independently: max, never sum."""
    from ..ir import Tendon
    for row in model.meta.get('rotary_locksets',[]):
        if primary_joint not in (row['inside_joint'],row['outside_joint']):continue
        other=row['outside_joint'] if primary_joint==row['inside_joint'] else row['inside_joint']
        for td in tuple(model.tendons):
            if not any(j==primary_joint for j,_ in td.sites):continue
            terms=[(other if j==primary_joint else j,c) for j,c in td.sites]
            follower=Tendon(td.name+'_independent_trim',terms,td.range,td.stiffness,td.damping,ALL_TIERS,
                'Independent ideal concealed cam: bolt follows greater trim stroke, never their sum')
            follower.kind='fixed';model.tendons.append(follower)
            row.update(latch_joint=next(j for j,_ in terms if j!=other),cam_tendons=[td.name,follower.name])
