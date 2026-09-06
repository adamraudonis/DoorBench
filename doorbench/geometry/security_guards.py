"""Original contact-operated security guards, informed by Ives 481/482 layouts.

No OEM CAD is incorporated. Jointed steel chain links approximate welded-link
articulation; contact with a real headed pin and slotted keeper carries the
load. Neither the leaf range nor an environment-toggled weld supplies security.
"""
from __future__ import annotations

import math
import numpy as np

from ..ir import ALL_TIERS, Body, Joint, QUAT_ID, Site, mat_to_quat, quat_from_axis_angle
from . import common as C


def _passive(model, body):
    model.add_body(body)
    model.meta.setdefault('mechanism_mass_bodies', []).append(body.name)
    if body.joint:
        model.meta.setdefault('physical_inertia_joints', []).append(body.joint.name)


def _capsule(body, name, a, b, radius, mat, semantic='lock'):
    from ..ir import Geom, quat_z_to
    a,b=np.asarray(a,float),np.asarray(b,float)
    body.geoms.append(Geom(name,'capsule',(radius,float(np.linalg.norm(b-a))/2),
        tuple((a+b)/2),tuple(quat_z_to(b-a)),mat,True,True,7900.,
        tiers=ALL_TIERS,semantic=semantic,part_label='Steel chain link',solref=(.001,1)))


def _mount(model,world,leaf,spec,u,hx,x_edge,t,z,kind):
    f=1. if spec['robot']['robot_outside'] else -1.
    if (1. if spec['robot']['is_push'] else -1.) != f:
        raise ValueError('Inside security guard requires the inward swing side')
    steel=C.mat_from_material(model,'stainless','mat_security_guard')
    edge=hx+x_edge;frame_x=u*(spec['opening']['width']/2+.021)
    surface=float(model.meta.get('wall_y',0))+f*spec['opening']['wall_thickness']/2
    level=f*(t/2+.028)
    if kind=='chain':
        # A frame plate cannot sit behind applied casing. Mount wholly on the
        # trim's actual face, and retain that load-bearing stock in every tier.
        # It also remains a collider when a released chain falls against it.
        trim=[g for g in world.geoms if g.type=='box' and g.name.startswith('casing_')
              and abs(g.pos[2]-z)<g.size[2] and f*g.pos[1]>0
              and abs(g.pos[0]-frame_x)<g.size[0]+.014]
        if trim:
            casing=min(trim,key=lambda g:abs(g.pos[0]-frame_x))
            if casing.size[0]<.014:raise ValueError('Chain anchor plate does not fit jamb casing')
            frame_x=float(np.clip(frame_x,casing.pos[0]-casing.size[0]+.014,
                                  casing.pos[0]+casing.size[0]-.014))
            surface=casing.pos[1]+f*casing.size[1]
            level=f*max(f*level,f*surface+.021)
            casing.collision=True;casing.tiers=ALL_TIERS
            casing.part_label='Load-bearing jamb casing supporting chain anchor'
    world.geoms.append(C.box(f'{leaf.name}_{kind}_frame_plate',(frame_x,surface+f*.002,z),
        (.014,.002,.034 if kind=='guard' else .025),steel,7900,semantic='lock',label='Security guard plate fixed to jamb'))
    lo,hi=sorted((surface+f*.004,level))
    world.geoms.append(C.box(f'{leaf.name}_{kind}_frame_bracket',(frame_x,(lo+hi)/2,z+(.030 if kind=='guard' else .017)),
        (.008,(hi-lo)/2,.003 if kind=='guard' else .004),steel,7900,semantic='lock',label='Frame anchor standoff'))
    return steel,f,frame_x,level,edge


def add_chain_guard(model,world,leaf,spec,u,hx,x_edge,t,z):
    steel,f,frame_x,level,edge=_mount(model,world,leaf,spec,u,hx,x_edge,t,z,'chain')
    name=leaf.name+'_chain';face=f*t/2
    # Bore the actual fixed anchor shelf; a rotating shaft cannot occupy solid
    # bracket stock. The enlarged front edge surrounds the complete bore.
    bracket=next(g for g in world.geoms if g.name==name+'_frame_bracket')
    bracket.pos=(bracket.pos[0],bracket.pos[1]+f*.003,bracket.pos[2])
    bracket.size=(bracket.size[0],bracket.size[1]+.003,bracket.size[2])
    from .lock_stock import cut_stock
    cut_stock(world,(frame_x-.0035,level-.0035,z+.012),(frame_x+.0035,level+.0035,z+.022),
              name+'_anchor_bore',names={bracket.name})
    # Plate A is on the leaf, while the permanent chain anchor is on the frame.
    # The headed end runs in a narrow slot; only the enlarged inboard opening
    # admits the head. All dimensions below are authored, not an OEM replica.
    near=x_edge-u*.034;far=near-u*.070;mid=(near+far)/2
    leaf.geoms.append(C.box(name+'_keeper_back',(mid,face+f*.002,z),
        (.054,.002,.014),steel,7900,semantic='lock',label='Slotted chain keeper back fixed to leaf'))
    for side in (-1,1):
        leaf.geoms.append(C.box(name+f'_keeper_wall_{side}',(mid,face+f*.008,z+side*.012),
            (.054,.004,.002),steel,7900,semantic='lock',label='Chain keeper side wall'))
    # Three front intervals: the narrow slot reaches a 14 mm keyhole opening.
    n0,n1=sorted((near+u*.010,far-u*.010));h0,h1=far-.007,far+.007
    for tag,lo,hi,hole in [('a',n0,h0,.0025),('keyhole',h0,h1,.007),('b',h1,n1,.0025)]:
        if hi<=lo:continue
        for side in (-1,1):
            height=(.014-hole)/2
            leaf.geoms.append(C.box(name+f'_keeper_lip_{tag}_{side}',((lo+hi)/2,face+f*.013,z+side*(hole+height)),
                ((hi-lo)/2,.0015,height),steel,7900,semantic='lock',label='Head-retaining slot lip with enlarged release opening'))
    for tag,x in [('near',near+u*.012),('far',far-u*.012)]:
        leaf.geoms.append(C.box(name+'_keeper_end_'+tag,(x,face+f*.008,z),
            (.002,.006,.014),steel,7900,semantic='lock',label='Physical chain slot travel end'))
    engaged=bool(spec['lock']['engaged'])
    # The neck bridges the keeper lip to an exposed grasp cube. The freely
    # turning last link remains a collider throughout release and reinsertion.
    head_offset=.014
    anchor=np.array([frame_x,level,z]);tip=np.array([hx+near,face+f*(.008+head_offset),z])
    if not engaged:tip=anchor+np.array([-u*.008,f*.010,-.130])
    radial=tip-anchor;horizontal=float(np.linalg.norm(radial[:2]));length=.148
    # Both the keyhole insertion and outward withdrawal must be reachable by
    # the terminal pivot. A plate centred farther out on wide casing can look
    # well supported while making the installed chain too short to release.
    release_reach=max(float(np.linalg.norm(np.array([hx+far,face+f*(.008+head_offset+out),z])-anchor))
                      for out in (0.,.025))
    if release_reach>=length-.001:
        raise ValueError('Installed chain cannot reach its release keyhole with terminal clearance')
    # Eight equal links in a downward V give an explicit slack configuration.
    if horizontal<1e-9:raise ValueError('Chain endpoint has no defined hanging plane')
    direction=np.r_[radial[:2]/horizontal,0.];normal=np.cross(direction,[0,0,1.])
    segment=length/8
    # Symmetric V, solved for its sag when endpoints have different elevations.
    half=length/2;dz=radial[2]
    if np.linalg.norm(radial)>=length-.004:raise ValueError('Chain does not retain installation slack')
    # Equal-length two legs; midpoint is in the plane perpendicular to the chord.
    chord=radial/np.linalg.norm(radial);down=np.cross(normal,chord)
    if down[2]>0:down=-down
    midpoint=(anchor+tip)/2+(down if engaged else -down)*math.sqrt(half*half-(np.linalg.norm(radial)/2)**2)
    points=[anchor+(midpoint-anchor)*k/4 for k in range(4)]+[midpoint+(tip-midpoint)*k/4 for k in range(5)]
    yaw=Body(name+'_swivel',None,tuple(anchor),QUAT_ID,None,[],[],ALL_TIERS,'lock','Permanent frame chain swivel')
    yaw.joint=Joint(name+'_yaw','hinge',(0,0,1),range=None,damping=.00005,role='mechanism',robot_interactive=False)
    yaw.geoms.append(C.cyl(name+'_anchor_pin',(0,0,.0125),.003,.0085,steel,(0,0,1),7900,
        False,True,ALL_TIERS,'lock','Fixed-position rotating chain anchor pin'))
    # The frame eye and first link must be perpendicular interlocking loops.
    # A fixed world-Y eye was coplanar with the first loop in these installs.
    eye=[-direction*.005+[0,0,-.005],direction*.005+[0,0,-.005],
         direction*.005+[0,0,.005],-direction*.005+[0,0,.005]]
    for k in range(4):_capsule(yaw,name+f'_anchor_eye_{k}',eye[k],eye[(k+1)%4],.001,steel)
    _passive(model,yaw);parent=yaw.name;previous=np.eye(3);bodies=[yaw.name];joints=[yaw.joint.name];adjacent_pairs=[]
    for i,(a,b) in enumerate(zip(points,points[1:])):
        ez=(b-a)/segment;ex=normal;ey=np.cross(ez,ex);rotation=np.column_stack([ex,ey,ez])
        local=rotation if i==0 else previous.T@rotation
        body=Body(name+f'_link_{i}',parent,(0,0,0) if i==0 else (0,0,segment),tuple(mat_to_quat(local)),
            None,[],[],ALL_TIERS,'lock','Articulated welded steel chain link')
        body.joint=Joint(body.name+'_pitch','hinge',(1,0,0),range=None,damping=.00002,
            role='mechanism',robot_interactive=False)
        # Closed wire loops, alternating planes. The previous loop extends
        # past the pivot around this loop's first crossbar; the bore has real
        # clearance for the 3 mm wire, including the initially folded links.
        across=np.array([1.,0,0]) if i%2==0 else np.array([0.,1.,0])
        halfwidth=.008 if i==7 else .005
        end=segment+.007 if i==7 else segment+.006
        pts=[-across*halfwidth+[0,0,.0015],across*halfwidth+[0,0,.0015],
             across*halfwidth+[0,0,end],-across*halfwidth+[0,0,end]]
        for k in range(4):_capsule(body,body.name+f'_wire_{k}',pts[k],pts[(k+1)%4],.0015,steel)
        previous_wires=[name+f'_link_{i-1}_wire_{k}' for k in range(4)] if i else [name+f'_anchor_eye_{k}' for k in range(4)]
        # MuJoCo normally filters parent/child contact. These real wire/eye
        # interfaces must instead constrain bending and carry measured loads.
        pairs=[{'geom1':a,'geom2':body.name+f'_wire_{k}',
              'solref':[.0002,1.],'solimp':[.95,.95,.0001]}
             for a in previous_wires for k in range(4)]
        model.meta.setdefault('native_contact_pairs',[]).extend(pairs)
        adjacent_pairs.extend({'geom1':p['geom1'],'geom2':p['geom2']} for p in pairs)
        _passive(model,body);parent=body.name;previous=rotation;bodies.append(body.name);joints.append(body.joint.name)
    # Three intersecting gimbal axes let the grasp cube and headed neck align
    # independently of the chain's hanging plane. The ball/eye and bored ring
    # fit within the existing hollow cube; no geometry floats without inertia.
    orient=Body(name+'_tip_swivel',parent,(0,0,segment),tuple(mat_to_quat(previous.T)),None,[],[],ALL_TIERS,'lock','Chain end swivel')
    orient.joint=Joint(name+'_tip_yaw','hinge',(0,0,1),range=None,damping=.00001,role='mechanism',robot_interactive=False)
    orient.geoms.append(C.sphere(name+'_tip_eye',(0,0,0),.001,steel,7900,True,ALL_TIERS,'lock','Inner swivel ball'))
    _passive(model,orient);bodies.append(orient.name);joints.append(orient.joint.name)
    roll=Body(name+'_tip_roll',orient.name,(0,0,0),QUAT_ID,None,[],[],ALL_TIERS,'lock','Bored terminal swivel ring')
    roll.joint=Joint(name+'_tip_roll_hinge','hinge',(0,1,0),range=None,damping=.00001,role='mechanism',robot_interactive=False)
    for axis in (0,2):
        for side in (-1,1):
            p=[0.,0.,0.];h=[.0017,.00025,.0017];p[axis]=side*.001425;h[axis]=.000275
            roll.geoms.append(C.box(name+f'_tip_roll_ring_{axis}_{side}',tuple(p),tuple(h),steel,7900,
                semantic='lock',label='Bored ring around swivel ball'))
    _passive(model,roll);bodies.append(roll.name);joints.append(roll.joint.name)
    head=Body(name+'_head',roll.name,(0,0,0),QUAT_ID,None,[],[],ALL_TIERS,'lock','Removable headed chain end')
    head.joint=Joint(name+'_tip_pitch','hinge',(1,0,0),range=None,damping=.00001,role='mechanism',robot_interactive=False)
    head.geoms.append(C.sphere(name+'_head_ball',(0,-f*head_offset,0),.004,steel,7900,True,ALL_TIERS,'lock','Head retained behind slot lips'))
    # Preserve this small steel head's authored contact stiffness against the
    # much softer generic jamb setting. Priority changes solver parameter
    # mixing, not collision detection; head/frame contacts remain measured.
    head.geoms[-1].contact_priority=1
    head.geoms.append(C.cyl(name+'_head_neck',(0,-f*.009,0),.0018,.006,steel,(0,1,0),7900,
        True,True,ALL_TIERS,'lock','Narrow neck through slot'))
    # The swivel eye occupies an actual cavity within the visible finger grip.
    for axis in range(3):
        for side in (-1,1):
            p=[0.,0.,0.];h=[.0045,.0045,.0045];p[axis]=side*.0035;h[axis]=.001
            head.geoms.append(C.box(name+f'_finger_grip_{axis}_{side}',tuple(p),tuple(h),steel,7900,
                semantic='lock',label='Hollow chain-end finger grip'))
    # Press the end face while sliding toward the keyhole. A side-face point
    # would give the large sliding force a needless moment about the swivel.
    site=name+'_release_grip';head.sites.append(Site(site,(u*.0045,0,0),QUAT_ID,.003,'grip'))
    model.meta.setdefault('site_wrench_limits_Nm',{})[site]=.02
    _passive(model,head);bodies.append(head.name);joints.append(head.joint.name)
    # The last link and headed end are adjacent physical members of one
    # compound three-axis swivel. Its intermediate coordinate bodies must not
    # make that actual bearing pair appear to be unrelated colliders.
    model.contact_excludes.append((name+'_link_7',head.name))
    record={'kind':'chain','leaf_body':leaf.name,'engaged_initial':engaged,
        'accessible_from_robot':not spec['robot']['robot_outside'],'release_site':site,
        'frame_anchor_world':anchor.tolist(),'keeper_geoms':[g.name for g in leaf.geoms if g.name.startswith(name+'_keeper')],
        'chain_bodies':bodies,'guard_joints':joints,'chain_length_m':length,
        'keyhole_terminal_reach_m':release_reach,
        'adjacent_wire_contact_pairs':adjacent_pairs,
        'head_geom':name+'_head_ball','head_diameter_m':.008,'neck_geom':name+'_head_neck','neck_diameter_m':.0036,
        'slot_width_m':.005,'keyhole_width_m':.014,
        'handoff_tolerances_m':{'keyhole_lateral':(.014-.008)/2-.0001,
            'slot_vertical':(.005-.0036)/2-.0001,'seated_position':.003},
        'contact_solver_scope':{'priority_geoms':[],'priority':1,
            'solref_s':[.0002,1.],'solimp':[.95,.95,.0001],
            'scope':'Authored stiff steel chain, head and keeper contacts, including measured jamb contact; prevents equal-priority mixing with softer default wood. No collision exclusion or strength certification.'},
        'head_center_local':[0,-f*head_offset,0],'grip_site_local':[u*.0045,0,0],
        'seated_head_center_leaf':[near,face+f*.008,z],
        'keyhole_center_leaf':[far,face+f*.008,z],
        'release_sequence':[{'action':'close_leaf'},{'action':'slide_to_keyhole','head_target_leaf':[far,face+f*.008,z]},
            {'action':'withdraw_head','direction':[0,f,0],'distance_m':.025}],
        'source':'https://us.allegion.com/content/dam/allegion-us-2/web-files/ives/installation-documents/Ives_481_Chain_Door_Guard_Installation_Instructions_107824.pdf',
        'scope':'Original articulated-chain approximation with contact-operated headed keeper; no strength rating or tool-bypass certification.'}
    model.meta.setdefault('security_guards',[]).append(record)
    for body in model.bodies:
        for geom in body.geoms:
            if geom.name.startswith(name):
                geom.solref=(.0002,1.);geom.solimp=(.95,.95,.0001)
                geom.contact_priority=1
                record['contact_solver_scope']['priority_geoms'].append(geom.name)
    record['contact_solver_scope']['priority_geoms'].sort()
    model.meta['native_timestep_s']=min(model.meta.get('native_timestep_s',.000025),.000025)
    model.meta['native_integrator']='implicit'
    return record


def add_swing_guard(model,world,leaf,spec,u,hx,x_edge,t,z):
    steel,f,frame_x,level,edge=_mount(model,world,leaf,spec,u,hx,x_edge,t,z,'guard')
    name=leaf.name+'_guard';origin=np.array([frame_x,level,z])
    # Two separate coaxial pivots leave the centre throat open. The original
    # slotted arm follows the cast-bar layout; no pin crosses its keeper throat.
    ex=np.array([-u,0,0]);ey=np.array([0,f,0]);ez=np.cross(ex,ey);rotation=np.column_stack([ex,ey,ez])
    for side in (-1,1):
        world.geoms.append(C.cyl(name+f'_pivot_pin_{side}',tuple(origin+[0,0,side*.023]),
            .003,.008,steel,(0,0,1),7900,True,True,ALL_TIERS,'hinge','Separate guard hinge pin'))
        world.geoms.append(C.box(name+f'_pivot_support_{side}',tuple(origin+[0,-f*.009,side*.030]),
            (.007,.011,.003),steel,7900,semantic='lock',label='Guard hinge bracket fixed to frame'))
    bar=Body(name+'_bar',None,tuple(origin),tuple(mat_to_quat(rotation)),None,[],[],ALL_TIERS,'lock','Pivoting slotted security bar')
    bar.joint=Joint(name+'_swing','hinge',(0,0,1),(0,0,0),(0,math.pi),damping=.002,
        frictionloss=.002,role='lock',robot_interactive=True,
        initial=0 if spec['lock']['engaged'] else math.pi,label='Guard swing (0 engaged, pi parked)')
    for side in (-1,1):
        # Convex open bearing rings fit around each independent vertical pin.
        for axis in (0,1):
            for sign in (-1,1):
                p=[0.,0.,side*.021];h=[.008,.008,.004];p[axis]=sign*.006;h[axis]=.002
                bar.geoms.append(C.box(name+f'_bearing_{side}_{axis}_{sign}',tuple(p),tuple(h),steel,7900,
                    semantic='lock',label='Guard pivot eye'))
        _capsule(bar,name+f'_root_{side}',[.008,0,side*.021],[.017,0,side*.009],.003,steel)
        _capsule(bar,name+f'_rail_{side}',[.017,0,side*.009],[.113,0,side*.009],.003,steel)
    _capsule(bar,name+'_closed_end',[.113,0,-.009],[.113,0,.009],.003,steel)
    bar.geoms.append(C.sphere(name+'_finger_end',(.121,0,0),.007,steel,7900,True,ALL_TIERS,'lock','Guard finger grip'))
    bar.sites.append(Site(name+'_release_grip',(.128,0,0),QUAT_ID,.007,'grip'))
    _passive(model,bar)
    # Curved holder fixed to the leaf: its high-root section guides the bar
    # while the ball at its return tip is retained by the narrower slot.
    base=np.array([x_edge-u*.038,f*(t/2+.003),z])
    ball=np.array([frame_x-hx,level,z])
    bend=np.array([base[0],level+f*.020,z])
    leaf.geoms.append(C.box(name+'_keeper_plate',tuple(base+[0,-f*.001,-.002]),(.013,.002,.026),steel,7900,
        semantic='lock',label='Curved guard holder mounting plate'))
    _capsule(leaf,name+'_keeper_root',base,bend,.0035,steel)
    _capsule(leaf,name+'_keeper_return',bend,ball,.0035,steel)
    leaf.geoms.append(C.sphere(name+'_keeper_ball',tuple(ball),.008,steel,7900,True,ALL_TIERS,'lock','Ball retained by guard slot'))
    record={'kind':'swing_bar_guard','leaf_body':leaf.name,'engaged_initial':bool(spec['lock']['engaged']),
        'accessible_from_robot':not spec['robot']['robot_outside'],'release_site':name+'_release_grip',
        'guard_joint':bar.joint.name,'guard_joints':[bar.joint.name],'frame_anchor_world':origin.tolist(),
        'keeper_geoms':[g.name for g in leaf.geoms if g.name.startswith(name+'_keeper')],
        'head_geom':name+'_keeper_ball','slot_height_m':.012,'head_diameter_m':.016,'arm_length_m':.12065,
        'release_sequence':[{'action':'close_leaf'},{'action':'swing_bar','joint':bar.joint.name,'target':math.pi}],
        'scope':'Original slotted-bar and curved-holder approximation; no OEM strength rating or tool-bypass certification.',
        'source':'https://www.iveshardware.com/en/products/latches-catches-and-bolts/chain-and-bar-door-guards.html'}
    model.meta.setdefault('security_guards',[]).append(record)
    for body in model.bodies:
        for geom in body.geoms:
            if geom.name.startswith(name):geom.solref=(.0003,1.);geom.solimp=(.95,.95,.0001)
    model.meta['native_timestep_s']=min(model.meta.get('native_timestep_s',.0001),.0001)
    return record
