"""Original marine dog mounts, with bored bearings and retained through shafts.

The shaft/packing/bushing/cleat load path follows NAVSHIPS 316-0042, not an
imported manufacturer's CAD model. Dimensions below are explicit engineering
choices. Rigid metal contacts do not certify gasket compression or pressure
capacity, and the geometry does not model water leakage or packing friction.
"""
from __future__ import annotations

import math
from dataclasses import replace
import trimesh

from ..ir import ALL_TIERS, QUAT_ID, Body, Site, SpatialSpring
from . import common as C
from .pocket_hardware import cut_box_recess


def bearing_y(body, name, center, material, *, inner=.0076, outer=.016,
              half_length=.006, semantic='lock'):
    """Twelve convex ring sectors around a genuinely empty Y-axis bore."""
    for k in range(12):
        angles=[2*math.pi*k/12,2*math.pi*(k+1)/12]
        points=[(r*math.cos(a),y,r*math.sin(a))
                for y in (-half_length,half_length) for r in (inner,outer) for a in angles]
        mesh=trimesh.convex.convex_hull(points)
        key=f'marine_bearing_y_{round(inner*1e6)}_{round(outer*1e6)}_{round(half_length*1e6)}_{k}'
        body.geoms.append(C.mesh_geom(f'{name}_{k}',key,mesh,center,QUAT_ID,material,
            7850,True,ALL_TIERS,semantic,'Bored shaft bearing / packing gland'))


def mount_dog(model, leaf, dog, *, thickness, edge_dir, swing_sign, material):
    """Give a rotating dog a through-leaf shaft, bearings, collars and dog web.

    Existing visible levers and their collision capsules represent one piece:
    the latter carry no duplicate material mass. Bearing rings are fixed to the
    leaf; their shoulders seat on real stock around the machined shaft aperture.
    """
    name=dog.name;x,_,z=dog.pos;t=float(thickness)
    relocated=[]
    for geom in leaf.geoms:
        if '_rivet_' in geom.name and math.hypot(geom.pos[0]-x,geom.pos[2]-z)<.027:
            geom.pos=(geom.pos[0],geom.pos[1],z+.035)
            relocated.append(geom.name)
    cut_box_recess(leaf,(x-.008,-t/2-.001,z-.008),
        (x+.008,t/2+.001,z+.008),name+'_shaft_aperture')
    # The 14 mm spindle has 0.6 mm radial clearance in the faceted bearing.
    dog.geoms.append(C.cyl(name+'_spindle',(0,0,0),.007,t/2+.064,material,
        (0,1,0),7850,True,True,ALL_TIERS,'lock','Retained through-leaf dog spindle'))
    mount=model.add_body(Body(name+'_mount',leaf.name,(x,0,z),semantic='lock',
        label='Welded dog sleeve and packing bearings'))
    for side in (-1,1):
        bearing_y(mount,f'{name}_bearing_{side}',(0,side*(t/2+.006),0),material)
        dog.geoms.append(C.cyl(f'{name}_shaft_collar_{side}',(0,side*(t/2+.016),0),
            .011,.003,material,(0,1,0),7850,True,True,ALL_TIERS,'lock','Axial spindle retaining collar'))
    wy=t/2+.034
    dog.geoms.append(C.box(name+'_dog_web',(edge_dir*.041,-swing_sign*wy,0),
        (.041,.009,.012),material,7850,True,True,ALL_TIERS,'lock','Keyed spindle-to-dog steel web'))
    for geom in dog.geoms:
        if geom.name.endswith('_lever_col'):
            geom.density=0.
            geom.pos=(geom.pos[0],-(t/2+.070),geom.pos[2])
        elif geom.name.endswith('_lever'):
            # The rotating rose sits outside the bearing and retained collar.
            geom.pos=(geom.pos[0],-(t/2+.020),geom.pos[2])
    for site in dog.sites:
        if site.role=='grip':
            site.pos=(site.pos[0],-(t/2+.070),site.pos[2])
    lever=next((g for g in dog.geoms if g.name.endswith('_lever')),None)
    if lever:
        # Both faces operate the same retained spindle, as declared by this
        # family. This is a real second grip, not a remote site in empty space.
        dog.geoms.append(replace(lever,name=name+'_lever_far',
            pos=(0,t/2+.020,0),quat=tuple(C.q_face(1,edge_dir))))
        col=next(g for g in dog.geoms if g.name.endswith('_lever_col'))
        dog.geoms.append(replace(col,name=name+'_lever_col_far',pos=(col.pos[0],t/2+.070,0)))
        dog.sites.append(Site(name+'_grip_p',(-edge_dir*.18,t/2+.070,0),QUAT_ID,.012,'grip'))
        _retain_individual_dog(model, dog, mount, t, edge_dir, material)
    model.meta.setdefault('mechanism_mass_bodies',[]).extend([name,mount.name])
    # The all-contact inspection model retains all bore sectors and probes
    # deliberately latched leaf poses. Preserve those contacts in its arena.
    model.meta['native_arena_memory_mib']=max(32,model.meta.get('native_arena_memory_mib',16))
    model.meta.setdefault('marine_dog_mounts',[]).append({
        'body':name,'joint':dog.joint.name,'leaf':leaf.name,'mount_body':mount.name,
        'spindle':name+'_spindle','bearing_prefix':name+'_bearing_',
        'shaft_radius_m':.007,'bearing_bore_radius_m':.0076,
        'shaft_aperture_half_width_m':.008,
        'relocated_mount_fasteners':relocated,
        'cleat_load_path':'flange / outboard bridge / rear base / dog wedge',
        'scope':'Rigid shaft, bearings and cleat; fluid sealing and pressure strength unverified'})


def _retain_individual_dog(model, dog, mount, thickness, edge_dir, material):
    """An explicit over-centre spring retains either end without added friction.

    Original engineering dimensions, informed by NAVSHIPS 316-0042's spring-
    retained end positions, not an OEM individual-dog reproduction. The spring
    passes in front of the spindle end, not through it at the centre position.
    A linear elastic tendon carries the load; hook strength/fatigue is untested.
    """
    name=dog.name;t=thickness
    # Rotation around local Y maps the crank's X into -Z. The fixed anchor
    # opposes the crank at half travel, making that position unstable and
    # supplying real restoring torque toward both finite angular stops.
    axis_y=dog.joint.axis[1];r=.035;fixed_r=.043
    theta=-axis_y*math.pi/4
    fixed_x=edge_dir*fixed_r*math.cos(theta)
    fixed_z=edge_dir*fixed_r*math.sin(theta)
    plane=-(t/2+.099);crank_plane=-(t/2+.077)
    dog.geoms.append(C.cyl(name+'_retention_spindle',(0,-(t/2+.073),0),.007,.011,
        material,(0,1,0),7850,True,True,ALL_TIERS,'lock','Keyed spindle extension for spring crank'))
    dog.geoms.append(C.box(name+'_retention_crank',(-edge_dir*r/2,crank_plane,0),
        (r/2,.004,.007),material,7850,True,True,ALL_TIERS,'lock','Over-centre spring crank'))
    dog.geoms.append(C.cyl(name+'_retention_pin',(-edge_dir*r,(plane+crank_plane)/2,0),
        .004,(crank_plane-plane)/2+.003,material,(0,1,0),7850,True,True,ALL_TIERS,'lock','Retained spring hook pin'))
    # Carry the anchor around the complete wedge sweep. A straight standoff
    # through fixed_x/fixed_z would pierce the rotating dog on near-face dogs.
    root_z=math.copysign(.140,fixed_z);bridge_y=-(t/2+.063)
    mount.geoms.append(C.box(name+'_retention_mount',(fixed_x,-(t/2+.063/2),root_z),
        (.006,.063/2,.008),material,7850,True,True,ALL_TIERS,'lock','Spring bracket welded to leaf beyond wedge sweep'))
    mount.geoms.append(C.box(name+'_retention_bridge',(fixed_x,bridge_y,(root_z+fixed_z)/2),
        (.006,.006,abs(root_z-fixed_z)/2+.008),material,7850,True,True,ALL_TIERS,'lock','Spring anchor bridge in front of rotating wedge'))
    mount.geoms.append(C.cyl(name+'_retention_anchor_pin',(fixed_x,(bridge_y+plane)/2,fixed_z),
        .004,(bridge_y-plane)/2+.003,material,(0,1,0),7850,True,True,ALL_TIERS,'lock','Retained spring anchor pin'))
    dog.sites.append(Site(name+'_retention_tip',(-edge_dir*r,plane,0),QUAT_ID,.004,'spring_anchor'))
    mount.sites.append(Site(name+'_retention_anchor',(fixed_x,plane,fixed_z),QUAT_ID,.004,'spring_anchor'))
    model.spatial_springs.append(SpatialSpring(name+'_retention_spring',
        (name+'_retention_tip',name+'_retention_anchor'),10000.,.050,damping=.5,width=.006,
        label='Original over-centre extension spring retaining individual dog end positions'))
    model.meta.setdefault('marine_dog_retention',[]).append({'body':name,'joint':dog.joint.name,
        'spring':name+'_retention_spring','stiffness_N_per_m':10000.,'free_length_m':.050,
        'crank_radius_m':r,'fixed_anchor_radius_m':fixed_r,'unstable_angle_rad':math.pi/4,
        'source':'https://maritime.org/doc/doors/index.php',
        'scope':'Original spring-retention design; dimensions and spring rating are engineering choices, not OEM measurements'})


def connect_cleat_bases(world):
    """Close the old 20 mm unsupported gap behind each dog without fouling it.

    Only the rear base widens. Widening the outboard bridge inward would put
    steel into the rotating wedge's swept volume and jam an otherwise released
    dog; the bridge stays beyond the wedge tip.
    """
    for geom in world.geoms:
        if geom.name.startswith('cleat_') and geom.name.endswith('_base'):
            bridge=next(g for g in world.geoms if g.name==geom.name[:-5]+'_bridge')
            sign=1 if bridge.pos[0]>geom.pos[0] else -1
            # Extend only outboard. Retain the original inboard face so the
            # nearby rod crank cannot strike the rear base when undogging.
            geom.pos=(geom.pos[0]+sign*.015,geom.pos[1],geom.pos[2])
            geom.size=(.035,geom.size[1],geom.size[2])
