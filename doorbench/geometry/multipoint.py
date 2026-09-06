"""Original lift-lever boltwork with contact-driven lost motion and cam slots.

The operating sequence follows Yale's lift-then-key instructions. This is an
original generic mortise assembly, not a reproduction of Yale internal CAD.
"""
from __future__ import annotations

import math
import numpy as np
from ..ir import ALL_TIERS, Body, Joint, Site, QUAT_ID
from . import common as C
from .pocket_hardware import cut_box_recess

SOURCE='https://www.yalehome.com/nz/en/campaigns/4-point-locks'


def add_multipoint(model,leaf,world,spec,*,u,v,x_edge,hz,zb,height,t,handle_joint,opening_width,pair):
    """Lever lifts auxiliary points; the separate key bolt blocks their bar.

    Only the conventional central cylinder uses an ideal internal screw/cam
    coupling. Lever pin, lost-motion cage, retaining friction, two diagonal
    follower slots and the key/bar interlock all transmit load by contact.
    """
    if not handle_joint:raise ValueError('Multipoint boltwork needs a lever spindle')
    name=leaf.name+'_multipoint';stroke=.020;radius=.031
    handle=next(b for b in model.bodies if b.joint and b.joint.name==handle_joint)
    # At this angle the real pin/cage reaches the 20 mm bar stop. Beyond
    # that angle the handle cannot rotate without deforming its hardware.
    lever_end=math.asin((stroke+.0003)/radius)
    handle.joint.range=(-lever_end,lever_end)
    handle.joint.springref=0.;handle.joint.stiffness=2.;handle.joint.frictionloss=.04
    handle.joint.damping=.15;handle.joint.armature=0.
    model.meta.setdefault('physical_inertia_joints',[]).append(handle_joint)
    model.meta.setdefault('mechanism_mass_bodies',[]).append(handle.name)
    model.meta.setdefault('gravity_balanced_operators',[]).append(handle_joint)
    handle.joint.label='Lift to extend locking points; depress to retract after unlocking'
    # This lever's longer depression must still end at the spring latch's
    # real throw. The unilateral tendon permits lift without bolt extension
    # beyond its closed stop; its ratio follows the installed lever travel.
    for tendon in model.tendons:
        if any(joint==handle_joint for joint,_ in tendon.sites):
            bolt_name=next((joint for joint,_ in tendon.sites if joint!=handle_joint),None)
            bolt=next((b for b in model.bodies if b.joint and b.joint.name==bolt_name and b.joint.role=='latch'),None)
            if bolt:
                tendon.sites=[(joint,-bolt.joint.range[1]/handle.joint.range[1] if joint==handle_joint else coefficient)
                               for joint,coefficient in tendon.sites]
                for record in model.meta.get('lock_stock',[]):
                    if record['bolt_body']==bolt.name and record.get('handle_coupling',{}).get('joint')==handle_joint:
                        record['handle_coupling']['bolt_m_per_joint_unit']=bolt.joint.range[1]/handle.joint.range[1]
    steel=C.mat_from_material(model,'stainless','mat_multipoint_steel')
    zdb=hz+.14;upper=min(zdb+.5,zb+height-.12);lower=max(zdb-.55,zb+.15)
    # Real internal mortise preserves both outer door skins.
    xs=sorted((x_edge-u*.135,x_edge+u*.002))
    removed=cut_box_recess(leaf,(xs[0],-.016,lower-.065),(xs[1],.016,upper+.045),name+'_case')
    bar=Body(name+'_drivebar',leaf.name,(0,0,0),semantic='lock',label='Lift-lock drive bar')
    engaged=bool(spec['lock']['engaged'])
    bar.joint=Joint(name+'_drivebar_slide','slide',(0,0,-1),range=(0,stroke),damping=8.,
        frictionloss=8.,armature=0.,role='mechanism',robot_interactive=False,
        initial=0. if engaged else stroke,modeled_at=0.,label='0 = points extended; 20 mm = withdrawn')
    # The vertical spine connects both end cams and the middle locking tongue.
    bar.geoms.append(C.box(name+'_spine',(x_edge-u*.118,.010,(lower+upper)/2),
        (.003,.002,(upper-lower)/2+.014),steel,7850,True,True,ALL_TIERS,'lock','Continuous steel drive-bar spine'))
    pin_x=handle.pos[0]-u*radius
    handle.geoms.append(C.cyl(name+'_lever_drive_pin',(-u*radius,.004,0),.003,.006,steel,(0,1,0),7850,True,True,ALL_TIERS,'mechanism','Lever pin within lost-motion cage'))
    # Pin traverses this open cage freely on spring return to neutral. It
    # meets the bottom face on depression and top face during lifting.
    cage=[]
    for tag,z in (('lower',hz-.0043),('upper',hz+stroke+.0043)):
        nm=name+'_cage_'+tag;cage.append(nm)
        bar.geoms.append(C.box(nm,(pin_x,.004,z),(.012,.005,.001),steel,7850,True,True,ALL_TIERS,'lock','Lost-motion drive cage'))
        bridge_end=pin_x-u*.010
        bridge_x=(bridge_end+x_edge-u*.118)/2
        bar.geoms.append(C.box(nm+'_bridge',(bridge_x,.010,z),
            (abs(bridge_end-(x_edge-u*.118))/2+.001,.002,.001),steel,7850,True,True,ALL_TIERS,'lock','Cage welded to drive bar'))
    # A transverse tongue on the central bolt enters this small open-sided
    # window. When retracted the tongue clears its left edge completely.
    for tag,sign in (('lower',-1),('upper',1)):
        bar.geoms.append(C.box(name+'_key_window_'+tag,(x_edge-u*.0295,.012,zdb+sign*.0045),
            (.0095,.002,.001),steel,7850,True,True,ALL_TIERS,'lock','Key-bolt tongue interlock window'))
        bar.geoms.append(C.box(name+'_key_bridge_'+tag,(x_edge-u*.0695,.010,zdb+sign*.05),
            (.0515,.002,.002),steel,7850,True,True,ALL_TIERS,'lock','Interlock arm clear of the withdrawn tongue'))
    bar.geoms.append(C.box(name+'_key_tower',(x_edge-u*.0195,.012,zdb),
        (.0015,.002,.052),steel,7850,True,True,ALL_TIERS,'lock','Window support joined to both bar arms'))
    inside=1. if spec['robot']['robot_outside'] else -1.
    # The interior turn remains mechanically present even in an outside
    # locked scenario; approach-side permissions govern access to it.
    central,pockets,eq=C.add_deadbolt(model,leaf,spec,u,v,x_edge,zdb,t,stroke,engaged,inside,
        math.pi/2,.3,name=leaf.name+'_deadbolt',tiers=ALL_TIERS,keyed_side=-inside)
    # add_deadbolt authors its geometry at the initial (possibly withdrawn)
    # configuration; this welded tongue uses that same reference convention.
    offset=0. if engaged else stroke
    central.geoms.append(C.box(name+'_key_tongue',(-u*(.030+offset),.012,0),
        (.008,.002,.003),steel,7850,True,True,ALL_TIERS,'lock','Transverse bolt tongue locks the drive bar'))
    from .lock_stock import cut_stock
    tongue_x=sorted((x_edge-u*.059,x_edge-u*.021))
    guide_names={g.name for g in leaf.geoms if g.name.startswith(leaf.name+'_deadbolt_guide_')}
    cut_stock(leaf,(tongue_x[0],.00925,zdb-.00375),(tongue_x[1],.01475,zdb+.00375),
              name+'_tongue_slot',names=guide_names)
    stock=next(r for r in model.meta['lock_stock'] if r['bolt_body']==central.name)
    stock['guide_geoms']=[g.name for g in leaf.geoms if any(g.name==n or g.name.startswith(n+'_') for n in guide_names)]
    stock['bolt_geoms']=[g.name for g in central.geoms]
    stock['auxiliary_tongue_slot']={'lower':[tongue_x[0],.00925,zdb-.00375],
                                   'upper':[tongue_x[1],.01475,zdb+.00375]}
    turn=next(b for b in model.bodies if b.name==leaf.name+'_deadbolt_thumbturn')
    if not turn.sites:
        turn.sites.append(Site(turn.name+'_grip_'+('p' if inside>0 else 'n'),(0,inside*.032,.010),QUAT_ID,.006,'grip',ALL_TIERS))
    model.equalities.extend(eq)
    if not pair:C.add_strike_plate(world.geoms,leaf.name+'_deadbolt_strike',u*opening_width/2,u,0,zdb,.0095,.0155,steel)
    auxiliary=[]
    for tag,z in (('upper',upper),('lower',lower)):
        nm=name+'_'+tag
        bolt=Body(nm,leaf.name,(x_edge,0,z),semantic='lock',label='Lever-driven auxiliary deadbolt')
        bolt.joint=Joint(nm+'_slide','slide',(-u,0,0),range=(0,stroke),damping=5.,frictionloss=.5,
            armature=0.,role='lock',robot_interactive=False,initial=0. if engaged else stroke,modeled_at=0.,
            label='Auxiliary bolt (0 = extended, 20 mm = withdrawn)')
        bolt.geoms.append(C.box(nm+'_bolt',(-u*.020,-.008,0),(.040,.003,.0125),steel,7850,True,True,ALL_TIERS,'lock','Steel bolt and rear cam carrier'))
        # Two rails form a real diagonal slot; a cylindrical drive-bar pin
        # loads alternate faces during extension and withdrawal.
        tangent=(u/math.sqrt(2),0,-1/math.sqrt(2))
        center=(-u*.050+u*stroke/2,.004,-stroke/2)
        cam=[]
        for sign in (-1,1):
            gn=nm+'_cam_'+str(sign);cam.append(gn)
            bolt.geoms.append(C.obox(gn,center,tangent,(0,1,0),0,sign*.0048,0,
                stroke/math.sqrt(2)+.006,.0015,.003,steel,True,ALL_TIERS,'lock','Diagonal cam-slot rail'))
            bolt.geoms[-1].friction=(.08,.001,.0001)
        # A rear web joins slot rails to the bolt below the pin's sweep.
        bolt.geoms.append(C.box(nm+'_cam_back',(-u*.040,-.003,-stroke/2),(.026,.002,.022),steel,7850,True,True,ALL_TIERS,'lock','Cam-slot rear web'))
        pin=name+'_'+tag+'_follower'
        bar.geoms.append(C.cyl(pin,(x_edge-u*.050,.004,z),.003,.004,steel,(0,1,0),7850,True,True,ALL_TIERS,'lock','Drive-bar follower pin'))
        bar.geoms.append(C.box(pin+'_arm',(x_edge-u*.084,.011,z),(.037,.002,.003),steel,7850,True,True,ALL_TIERS,'lock','Follower arm welded to continuous drive bar'))
        model.add_body(bolt)
        model.meta.setdefault('mechanism_mass_bodies',[]).append(bolt.name)
        model.meta.setdefault('physical_inertia_joints',[]).append(bolt.joint.name)
        pockets.append({'z':z,'y':0.,'h':.033,'w':.026,'depth':stroke+.004,'ramp':False})
        if not pair:C.add_strike_plate(world.geoms,nm+'_strike',u*opening_width/2,u,0,z,.013,.0165,steel)
        auxiliary.append({'joint':bolt.joint.name,'body':bolt.name,'bolt_geom':nm+'_bolt','follower_geom':pin,'cam_geoms':cam})
    model.add_body(bar)
    model.meta.setdefault('mechanism_mass_bodies',[]).append(bar.name)
    model.meta.setdefault('physical_inertia_joints',[]).append(bar.joint.name)
    model.meta.setdefault('multipoint_locks',[]).append({'leaf_body':leaf.name,'leaf_joint':leaf.joint.name,
        'lever_joint':handle_joint,'drivebar_joint':bar.joint.name,'central_bolt_joint':central.joint.name,
        'thumbturn_joint':turn.joint.name,'thumbturn_side':inside,'auxiliary':auxiliary,
        'lever_pin_geom':name+'_lever_drive_pin','cage_geoms':cage,'stroke_m':stroke,
        'key_window_geoms':[name+'_key_window_lower',name+'_key_window_upper'],
        'mortised_stock':removed,'reference':SOURCE,
        'scope':'Original internal contact-driven cam/drive-bar assembly; ideal cylinder coupling and prismatic/bearing guides. Key insertion and cylinder tumblers are not modeled.'})
    model.meta['native_timestep_s']=min(.0005,model.meta.get('native_timestep_s',.002))
    return pockets
