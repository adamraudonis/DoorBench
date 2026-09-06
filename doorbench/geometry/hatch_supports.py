"""Original frame-mounted telescopic hatch stays and accessible hatch pulls.

The gas model is a mildly progressive axial spring, not an OEM gas curve. The
optional hold-open pin carries load by contact through a hole in the sliding
member; pulling the knob clears it. No latch equality or environment freeze.
"""
from __future__ import annotations

import math
import numpy as np

from ..ir import ALL_TIERS, Body, Equality, Joint, Model, QUAT_ID, Site, quat_from_axis_angle
from . import common as C
from .pocket_hardware import cut_box_recess


def _eye(body,name,material):
    # Four convex walls surround a 12 mm square bore for the 8 mm pin.
    for axis in (1,2):
        for side in (-1,1):
            p=[0.,0.,0.];h=[.012,.006,.006]
            p[axis]=side*.008;h[axis]=.002
            body.geoms.append(C.box(f'{name}_{axis}_{side}',tuple(p),tuple(h),material,7900,
                semantic='hinge',label='Open stay pivot eye'))


def stay_pose(angle, dimensions):
    a=np.asarray(dimensions['frame_anchor'],float)
    pivot=np.asarray(dimensions['hatch_pivot'],float)
    b=np.asarray(dimensions['lid_anchor'],float)
    co,si=math.cos(angle),math.sin(angle)
    rot=np.array([[1,0,0],[0,co,si],[0,-si,co]])
    end=pivot+rot@(b-pivot)
    v=end-a
    return {'hinge':math.atan2(-v[1],v[2])-dimensions['rest_angle'],
            'slide':float(np.linalg.norm(v))-dimensions['closed_length']}


def resolve_hatch_configuration(model, qpos, meta):
    """Inspection-only exact loop closure; the simulator uses native constraints."""
    record=meta.get('hatch_support')
    if not record:return qpos
    angle=float(qpos[model.jnt_qposadr[model.joint('hatch_hinge').id]])
    for item in [record]+([record['gas_assist']] if record.get('gas_assist') else []):
        values=stay_pose(angle,item['dimensions'])
        for name,key in ((item['hinge_joint'],'hinge'),(item['slide_joint'],'slide')):
            qpos[model.jnt_qposadr[model.joint(name).id]]=values[key]
    return qpos


def add_hatch_support(model, world, lid, spec, hinge_out):
    gas=spec['closer']['model']=='gas_strut'
    locking=spec['kinematics'].get('stop')=='prop_arm'
    if not (gas or locking):return
    _add_stay(model,world,lid,spec,hinge_out,gas=gas and not locking,locking=locking)
    if gas and locking:
        primary=model.meta['hatch_support']
        before_world=len(world.geoms);before_lid=len(lid.geoms)
        secondary=Model(model.name)
        _add_stay(secondary,world,lid,spec,hinge_out,gas=True,locking=False,side=-1)
        # The second assembly uses the same tested construction with a distinct
        # namespace; no single body belongs to both independent native loops.
        def rename(value):
            if isinstance(value,str):
                if value.startswith('hatch_stay'):return value.replace('hatch_stay','hatch_gas_stay',1)
                if value.startswith('stay_'):return 'gas_'+value
                return value
            if isinstance(value,list):return [rename(v) for v in value]
            if isinstance(value,dict):return {k:rename(v) for k,v in value.items()}
            return value
        for body in secondary.bodies:
            body.name=rename(body.name)
            if body.parent and body.parent.startswith('hatch_stay'):body.parent=rename(body.parent)
            if body.joint:body.joint.name=rename(body.joint.name)
            for item in body.geoms+body.sites:item.name=rename(item.name)
            model.add_body(body)
        for geom in world.geoms[before_world:]+lid.geoms[before_lid:]:geom.name=rename(geom.name)
        for eq in secondary.equalities:
            eq.name=rename(eq.name);eq.a=rename(eq.a);model.equalities.append(eq)
        model.meta['mechanism_mass_bodies'].extend(rename(secondary.meta['mechanism_mass_bodies']))
        primary['gas_assist']=rename(secondary.meta['hatch_support']);model.meta['hatch_support']=primary


def _add_stay(model,world,lid,spec,hinge_out,*,gas,locking,side=1):
    width=spec['leaf']['width'];height=spec['opening']['height'];thickness=spec['leaf']['thickness']
    pivot=np.asarray(lid.pos)+[0,hinge_out,0]
    x=side*(width/2-.085)
    a=pivot+np.array([x,-.15*height,-.08*height])
    b=np.asarray(lid.pos)+[x,-.65*height,-thickness/2-.034]
    v=b-a;length=float(np.linalg.norm(v));theta=math.atan2(-v[1],v[2])
    maximum=math.radians(spec['kinematics']['max_open_deg'])
    dims={'frame_anchor':a.tolist(),'lid_anchor':b.tolist(),'hatch_pivot':pivot.tolist(),
          'rest_angle':theta,'closed_length':length}
    samples=[stay_pose(q,dims) for q in np.linspace(0,maximum,201)]
    stroke=samples[-1]['slide'];case_length=length-.010;rod_length=stroke+.055
    if stroke<=0 or rod_length>=length-.025:raise ValueError('Hatch stay cannot retain telescopic overlap')
    steel=C.mat_from_material(model,'stainless','mat_hatch_stay')
    # A downstand is physically continuous with the curb, with its pin inboard
    # of the opening. Both anchors remain below the lid in the closed state.
    edge=side*(spec['opening']['width']/2+.025)
    top=lid.pos[2]-.012
    world.geoms.append(C.box('stay_frame_downstand',(edge,a[1],(top+a[2])/2),
        (.008,.028,(top-a[2])/2+.012),steel,7900,semantic='frame',label='Stay bracket fixed to curb'))
    world.geoms.append(C.box('stay_frame_shelf',((edge+x)/2,a[1],a[2]-.024),
        (abs(edge-x)/2+.008,.026,.006),steel,7900,semantic='frame',label='Stay bracket foot'))
    for side in (-1,1):
        world.geoms.append(C.box(f'stay_frame_cheek_{side}',(x+side*.026,a[1],a[2]-.006),
            (.005,.020,.020),steel,7900,semantic='hinge',label='Frame pivot clevis'))
    world.geoms.append(C.cyl('stay_frame_pin',tuple(a),.004,.035,steel,(1,0,0),7900,
        True,True,ALL_TIERS,'hinge','Frame pivot pin'))
    # The case starts beyond its bearing; no solid cylinder overlapping a rod.
    case=Body('hatch_stay_case',None,tuple(a),tuple(quat_from_axis_angle((1,0,0),theta)),
        None,[],[],ALL_TIERS,'mechanism','Telescopic stay case')
    case.joint=Joint('hatch_stay_hinge','hinge',(1,0,0),(0,0,0),
        (min(s['hinge'] for s in samples)-.05,max(s['hinge'] for s in samples)+.05),
        damping=.1,armature=.001,role='mechanism',robot_interactive=False)
    _eye(case,'stay_frame_eye',steel)
    case.geoms.append(C.box('stay_case_eye_neck',(0,0,.014),(.009,.009,.005),steel,7900,
        semantic='mechanism',label='Case connection to pivot eye'))
    pin_z=length-.031
    for axis in (() if gas else (0,1)):
        for side in (-1,1):
            intervals=[(.016,case_length)]
            if locking and axis==0 and side==1:
                intervals=[(.016,pin_z-.007),(pin_z+.007,case_length)]
            for i,(lo,hi) in enumerate(intervals):
                pos=[0.,0.,(lo+hi)/2];half=[.010,.010,(hi-lo)/2]
                pos[axis]=side*.013;half[axis]=.003
                case.geoms.append(C.box(f'stay_case_{axis}_{side}_{i}',tuple(pos),tuple(half),steel,7900,
                    semantic='mechanism',label='Hollow stay guide wall'))
    if gas:
        # Convex radial strips represent a genuinely hollow cylinder, leaving
        # the polished round piston rod clear of every collision surface.
        for i in range(12):
            angle=i*math.tau/12;rad=.0135
            case.geoms.append(C.box(f'stay_gas_cylinder_{i}',(rad*math.cos(angle),rad*math.sin(angle),(.016+case_length)/2),
                (.0015,.00365,(case_length-.016)/2),steel,7900,semantic='closer',label='Gas cylinder wall',
                quat=tuple(quat_from_axis_angle((0,0,1),angle))))
    # End stop collar with a real central opening, representing piston travel.
    for side in (-1,1):
        case.geoms.append(C.box(f'stay_end_stop_{side}',(0,side*.008,length-.037),(.010,.002,.003),
            steel,7900,semantic='mechanism',label='Retained rod guide/end stop'))
    model.add_body(case)
    rod=Body('hatch_stay_rod',case.name,(0,0,length),QUAT_ID,None,[],[],ALL_TIERS,'mechanism','Sliding stay member')
    force=float(spec['closer'].get('gas_force_N',spec['kinematics'].get('gas_force_N',250))) if gas else 0.
    stiffness=.10*force/stroke if gas else 0.
    rod.joint=Joint('hatch_stay_slide','slide',(0,0,1),(0,0,0),(0,stroke+.002),
        damping=25. if gas else 2.,frictionloss=1.,stiffness=stiffness,
        springref=stroke+force/stiffness if gas else 0.,armature=.01,role='mechanism',robot_interactive=False)
    _eye(rod,'stay_lid_eye',steel)
    rod.geoms.append(C.box('stay_rod_eye_neck',(0,0,-.014),(.006,.006,.005),steel,7900,
        semantic='mechanism',label='Rod connection to pivot eye'))
    slot_z=pin_z-length-stroke
    intervals=[(-rod_length,-.016)]
    if locking:intervals=[(-rod_length,slot_z-.0051),(slot_z+.0051,-.016)]
    for i,(lo,hi) in enumerate(intervals):
        if gas:
            rod.geoms.append(C.cyl(f'stay_gas_piston_rod_{i}',(0,0,(lo+hi)/2),.006,(hi-lo)/2,steel,(0,0,1),7900,
                True,True,ALL_TIERS,'closer','Unperforated polished gas spring rod'))
        else:
            rod.geoms.append(C.box(f'stay_rod_{i}',(0,0,(lo+hi)/2),(.006,.006,(hi-lo)/2),steel,7900,
                semantic='mechanism',label='Sliding member with transverse locking slot'))
    if locking:
        # Retain two side rails around the transverse X bore; the pin passes
        # through open air, not through an uncut solid collision proxy.
        for side in (-1,1):
            rod.geoms.append(C.box(f'stay_slot_rail_{side}',(0,side*.0055,slot_z),(.006,.0005,.0051),
                steel,7900,semantic='mechanism',label='Locking slot side wall'))
    rod.geoms.append(C.box('stay_rod_retainer',(0,0,-rod_length+.006),(.009,.009,.006),steel,7900,
        semantic='mechanism',label='Captured lower rod stop'))
    for geom in rod.geoms:
        geom.solref=(.004,1.);geom.solimp=(.95,.999,.0001)
    rod.sites.append(Site('stay_lid_pin',(0,0,0),QUAT_ID,.004,'closer_anchor'))
    model.add_body(rod)
    local=b-np.asarray(lid.pos)
    for side in (-1,1):
        lid.geoms.append(C.box(f'stay_lid_cheek_{side}',tuple(local+[side*.023,0,.017]),
            (.005,.020,.022),steel,7900,semantic='hinge',label='Stay clevis screwed to lid underside'))
    lid.geoms.append(C.cyl('stay_lid_pin',tuple(local),.004,.033,steel,(1,0,0),7900,
        True,True,ALL_TIERS,'hinge','Lid clevis pin'))
    model.equalities.append(Equality('connect','hatch_stay_connection',rod.name,lid.name,
        anchor=(0,0,0),tiers=ALL_TIERS,label='Stay end pinned to lid clevis',solref=(.004,1.),solimp=(.99,.999,.0001)))
    record={'schema_version':1,'kind':'telescopic_pin_lock' if locking else 'telescopic_gas_spring',
        'dimensions':dims,'hinge_joint':case.joint.name,'slide_joint':rod.joint.name,
        'nominal_angle_rad':maximum,'stroke_m':stroke,'gas_force_at_extension_N':force,
        'gas_model':'Orientation-independent original spring model: force declines 10% over extension; no OEM curve.',
        'support_release_joint':None,'support_release_site':None,
        'native_constraint':'hatch_stay_connection'}
    if locking:
        pin=Body('hatch_stay_lock_pin',case.name,(.022,0,pin_z),QUAT_ID,None,[],[],ALL_TIERS,'operator','Stay release knob')
        pin.joint=Joint('hatch_stay_release','slide',(1,0,0),(0,0,0),(0.,.016),
            damping=3.,stiffness=600.,springref=0.,armature=.002,role='operator',
            initial=.016,modeled_at=.016,label='Stay knob (+ = withdraw; support lid before releasing)')
        pin.geoms.append(C.cyl('stay_lock_pin',(-.003,0,0),.0045,.012,steel,(1,0,0),7900,
            True,True,ALL_TIERS,'lock','Spring-engaged transverse stay pin'))
        pin.geoms[-1].solref=(.004,1.)
        pin.geoms[-1].solimp=(.95,.999,.0001)
        pin.geoms.append(C.cyl('stay_release_knob',(.022,0,0),.013,.007,steel,(1,0,0),7900,
            True,True,ALL_TIERS,'operator','Pull release knob'))
        pin.sites.append(Site('stay_release_grip',(.029,0,0),QUAT_ID,.008,'grip'))
        model.add_body(pin)
        record.update({'support_release_joint':pin.joint.name,'support_release_site':'stay_release_grip',
            'release_position_m':.016,'engaged_position_m':0.,
            'sequence':['Lift lid to full travel; spring pin enters stay slot.',
                        'Support lid load, pull stay_release_grip outward 16 mm, then lower lid.',
                        'Release knob after leaving the locking slot; pin rides on sliding member.']})
    model.meta['hatch_support']=record
    model.meta.setdefault('mechanism_mass_bodies',[]).extend([case.name,rod.name]+(['hatch_stay_lock_pin'] if locking else []))
    model.meta['mechanical_export_support']={'mjcf':'native point constraint, axial spring and contact locking pin',
        'urdf':'requires loop closure and native contact spring support',
        'usd':'unsupported closed-loop and contact-lock dynamics; static interchange only'}


def add_hatch_pull(model,lid,spec,operator):
    """Real ring aperture/mortise, or a rigid D pull, on the approached face."""
    if operator.kind not in ('ring_pull','pull'):
        raise ValueError('Hatch operator must explicitly name a lifting ring or D pull')
    thickness=spec['leaf']['thickness'];height=spec['opening']['height']
    face=-1. if spec['family']=='hatch_ceiling' else 1.
    y=-height*.75;z=face*thickness/2
    steel=C.mat_from_material(model,operator.material,'mat_hatch_pull')
    if operator.kind=='ring_pull':
        # Thin steel plate needs a through-cut stamped cup which projects onto
        # the opposite face; a 14 mm blind mortise cannot fit in 6 mm plate.
        depth=.014
        low=[-.052,y-.045,min(z,z-face*depth)];high=[.052,y+.075,max(z,z-face*depth)]
        cut_box_recess(lid,low,high,'ring_mortise')
        # Cup back is recessed, its rim does not fill the finger aperture.
        lid.geoms.append(C.box('ring_cup_back',(0,y+.015,z-face*(depth-.001)),(.052,.060,.001),
            steel,7900,semantic='operator',label='Mortised ring cup back'))
        for side in (-1,1):
            lid.geoms.append(C.box(f'ring_cup_side_{side}',(side*.051,y+.015,z-face*depth/2),(.001,.060,depth/2),
                steel,7900,semantic='operator',label='Ring cup wall'))
            lid.geoms.append(C.box(f'ring_cup_end_{side}',(0,y+.015+side*.059,z-face*depth/2),(.050,.001,depth/2),
                steel,7900,semantic='operator',label='Ring cup wall'))
        ring=Body('ring',lid.name,(0,y,z-face*.006),QUAT_ID,None,[],[],ALL_TIERS,'operator','Hatch lifting ring')
        ring.joint=Joint('ring_hinge','hinge',(face,0,0),(0,0,0),(0,math.pi/2),damping=.015,
            frictionloss=.01,role='operator',label='Lifting ring (+ = flip out)')
        for side in (-1,1):
            ring.geoms.append(C.cyl(f'ring_side_{side}',(side*.035,.028,0),.005,.028,steel,(0,1,0),7900,
                True,True,ALL_TIERS,'operator','Ring side'))
        ring.geoms.append(C.cyl('ring_grip_bar',(0,.056,0),.005,.035,steel,(1,0,0),7900,
            True,True,ALL_TIERS,'operator','Ring grip'))
        ring.sites.append(Site('grip_ring',(0,.056,0),QUAT_ID,.007,'grip'))
        model.add_body(ring);model.meta['operator_joint']='ring_hinge'
    else:
        # Horizontal bar with 32 mm finger clearance; no remapped stale sites.
        for side in (-1,1):
            lid.geoms.append(C.cyl(f'hatch_pull_foot_{side}',(side*.060,y,z+face*.017),.009,.017,
                steel,(0,0,1),7900,True,True,ALL_TIERS,'operator','D pull mounting leg'))
        lid.geoms.append(C.cyl('hatch_pull_bar',(0,y,z+face*.041),.009,.060,steel,(1,0,0),7900,
            True,True,ALL_TIERS,'operator','Hatch D pull'))
        lid.sites.append(Site('hatch_pull_grip',(0,y,z+face*.041),QUAT_ID,.01,'grip'))
        model.meta['operator_joint']=None
    model.meta['hatch_hand_access']={'face':'underside' if face<0 else 'top',
        'opening_contact':'grip_ring' if operator.kind=='ring_pull' else 'hatch_pull_grip',
        'closed_height_m':lid.pos[2]+z,'requires_elevation_aid':face<0,
        'scope':'Physical hand contact; reach from floor is not guaranteed for ceiling hatches.'}
