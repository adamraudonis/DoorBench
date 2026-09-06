"""A contact-driven Suffolk thumb lever, gravity bar and post-mounted keeper.

Original generic dimensions, informed by public Suffolk installation guides.
The thumb tang genuinely contacts the bar; there is no joint equality.
"""
from __future__ import annotations

from dataclasses import replace
import math
import numpy as np

from ..ir import Body, Joint, Site, ALL_TIERS, QUAT_ID, quat_z_to, quat_from_axis_angle
from . import common as C
from .gate_hardware import _screw


def _cut_slot(geoms, prefix, x0, x1, z0, z1):
    """Cut a through-Y rectangle from axis-aligned boxes, including proxies."""
    result=[]
    for g in geoms:
        if g.type != 'box' or g.semantic != 'leaf' or not np.allclose(g.quat,QUAT_ID):
            result.append(g); continue
        gx0,gx1=g.pos[0]-g.size[0],g.pos[0]+g.size[0]
        gz0,gz1=g.pos[2]-g.size[2],g.pos[2]+g.size[2]
        ix0,ix1=max(x0,gx0),min(x1,gx1)
        iz0,iz1=max(z0,gz0),min(z1,gz1)
        if ix1<=ix0 or iz1<=iz0:
            result.append(g); continue
        pieces=[(gx0,ix0,gz0,gz1),(ix1,gx1,gz0,gz1),
                (ix0,ix1,gz0,iz0),(ix0,ix1,iz1,gz1)]
        for k,(a,b,c,d) in enumerate(pieces):
            if b-a>1e-6 and d-c>1e-6:
                fraction=(b-a)*(d-c)/((gx1-gx0)*(gz1-gz0))
                result.append(replace(g,name=f'{g.name}_{prefix}_{k}',
                    pos=((a+b)/2,g.pos[1],(c+d)/2),size=((b-a)/2,g.size[1],(d-c)/2),
                    mass_override=None if g.mass_override is None else g.mass_override*fraction))
    geoms[:]=result


def add_suffolk_latch(model,world,leaf_body,spec,*,u,v,hx,x_edge,
                       leaf_bottom,leaf_height,leaf_name='leaf'):
    """Return selected accessible operator and actual contact height.

    The bar and keeper are on the opening-side face. From the opposite face a
    thumb press raises the bar through native contact; from the bar face it can
    be lifted directly. Fixed pulls, not the moving release, move the gate.
    """
    name=leaf_name
    ps=C.frame_jamb_thickness(spec)
    is_gate=spec['opening']['frame']['kind'] in ('gate_posts','pressure_frame')
    frame_surface=v*ps/2 if is_gate else float(model.meta.get('wall_y',0.))+v*spec['opening']['wall_thickness']/2
    frame_prefix='post_latch' if is_gate else 'jamb_strike'
    mat=C.mat_from_material(model,'wrought_iron','mat_suffolk')
    steel=C.mat_from_material(model,'stainless','mat_suffolk_pin')
    support=next((g for g in leaf_body.geoms if g.name==f'{name}_stile_l'),None)
    if support is None:
        support=next(g for g in leaf_body.geoms if g.name==f'{name}_slab')
    t=2*support.size[1]
    grip_z=min(float(spec['operator']['height']),leaf_bottom+leaf_height-.15)
    # Keep the measured contact height explicit; root normalizes short gates.
    zt=grip_z-.004
    xt=x_edge-u*.105
    xp=x_edge-u*.225
    yt=-v*(t/2+.008)
    yb=frame_surface+v*.028
    reach=abs(yb-yt)
    zb=zt+.009
    sx=u*spec['opening']['width']/2
    tip=abs((sx+u*.020)-(hx+xp))
    slot=(xt-.010,xt+.010,zt-.010,zt+.030)
    _cut_slot(leaf_body.geoms,'thumb_slot',*slot)
    support_names=[g.name for g in leaf_body.geoms if g.name==support.name or g.name.startswith(support.name+'_thumb_slot_')]
    attachments=[]
    def box(geoms,n,p,s,semantic='latch',label=''):
        geoms.append(C.box(n,p,s,mat,7850,True,True,ALL_TIERS,semantic,label or n))
    # A real backing board spans the sparse pickets to the continuous stile.
    # Its slot remains open through both plates and all underlying leaf boxes.
    for face in (-1,1):
        plate=[]
        n=f'{name}_suffolk_plate_{face}'
        # Leave the latch-edge strip clear for the real frame stop moulding.
        xmid=x_edge-u*.135
        plate.append(C.box(n,(xmid,face*(t/2+.002),zt-.085),(.117,.002,.135),
                           mat,7850,True,True,ALL_TIERS,'leaf','Suffolk structural backing plate'))
        _cut_slot(plate,'slot',*slot)
        for g in plate:
            g.semantic='latch';leaf_body.geoms.append(g)
        # A declared group means contact with at least one actual cut piece.
        attachments.append({'first':[g.name for g in plate],'second':support_names,'label':'backing plate to actual leaf structure'})
        for dz in (-.18,.015):
            x=x_edge-u*.023
            _screw(leaf_body.geoms,f'{n}_screw_{int(dz*1000)}',
                   (x,face*(t/2+.002),zt+dz),face,steel)
    plates=[g.name for g in leaf_body.geoms if g.name.startswith(f'{name}_suffolk_plate_') and '_screw_' not in g.name]

    thumb=Body(f'{name}_thumb',leaf_body.name,(xt,yt,zt),QUAT_ID,
               tiers=ALL_TIERS,semantic='operator',label='Suffolk thumb lever')
    thumb.joint=Joint(f'{name}_thumb_hinge','hinge',(v,0,0),(0,0,0),(0.,.30),
                      damping=.002,frictionloss=.0003,armature=1e-6,role='operator',
                      robot_interactive=v>0,label='Press down; tang raises the gravity bar by contact')
    box(thumb.geoms,f'{name}_thumb_geom',(0,-v*.033,0),(.015,.022,.004),'operator','Thumb depression pad')
    box(thumb.geoms,f'{name}_thumb_lifter',(0,v*(reach+.012)/2,0),
        (.004,(reach+.012)/2,.003),'latch','Continuous through-door thumb tang')
    box(thumb.geoms,f'{name}_thumb_neck',(0,-v*.007,0),(.008,.010,.004),'operator','Pad to pivot connection')
    thumb.geoms.append(C.cyl(f'{name}_thumb_shaft',(0,0,0),.004,.028,steel,(1,0,0),7850,
                             True,True,ALL_TIERS,'operator','Thumb pivot shaft'))
    thumb.sites.append(Site(f'{name}_thumb_push',(0,-v*.035,.004),QUAT_ID,.008,'push'))
    model.add_body(thumb)
    # Keep both bearings outside the 30 mm thumb pad throughout rotation;
    # the extended shaft passes through their bores with 2 mm radial clearance.
    for side in (-1,1):
        for axis in (1,2):
            for sign in (-1,1):
                p=[xt+side*.022,yt,zt];p[axis]+=sign*.009
                half=[.004,.012,.012];half[axis]=.003
                box(leaf_body.geoms,f'{name}_thumb_bearing_{side}_{axis}_{sign}',tuple(p),tuple(half),label='Open thumb shaft bearing')

    bar=Body(f'{name}_latch_bar',leaf_body.name,(xp,yb,zb),QUAT_ID,
             tiers=ALL_TIERS,semantic='latch',label='Contact-lifted gravity latch bar')
    bar.joint=Joint(f'{name}_latch_bar_hinge','hinge',(0,-u,0),(0,0,0),(-.025,.35),
                    damping=.002,frictionloss=.0003,armature=1e-6,role='latch',
                    robot_interactive=v<0,label='Lift bar directly, or via thumb tang; gravity return')
    box(bar.geoms,f'{name}_latch_bar_geom',(u*(tip+.009)/2,0,0),
        ((tip-.009)/2,.005,.005),label='Gravity bar')
    bar.geoms[-1].friction=(.18,.005,.0001)
    for side in (-1,1):
        box(bar.geoms,f'{name}_latch_bar_eye_x_{side}',(u*side*.009,0,0),(.003,.005,.012),label='Open bar pivot eye')
        box(bar.geoms,f'{name}_latch_bar_eye_z_{side}',(0,0,side*.009),(.006,.005,.003),label='Open bar pivot eye')
    bar.sites.append(Site(f'{name}_latch_bar_grip',(u*.17,0,.005),QUAT_ID,.008,'grip'))
    model.add_body(bar)
    # Boss and retaining washer support a real pin through the open bar eye.
    surf=v*(t/2+.004)
    end=yb-v*.007
    leaf_body.geoms.append(C.cyl(f'{name}_latch_bar_boss',(xp,(surf+end)/2,zb),.012,abs(end-surf)/2,
        mat,(0,1,0),7850,True,True,ALL_TIERS,'latch','Bar pivot standoff'))
    leaf_body.geoms.append(C.cyl(f'{name}_latch_bar_pin',(xp,(surf+yb+v*.010)/2,zb),.004,abs(yb+v*.010-surf)/2,
        steel,(0,1,0),7850,True,True,ALL_TIERS,'latch','Bar pivot pin'))
    leaf_body.geoms.append(C.cyl(f'{name}_latch_bar_washer',(xp,yb+v*.009,zb),.010,.002,
        steel,(0,1,0),7850,True,True,ALL_TIERS,'latch','Bar retaining washer'))
    attachments.append({'first':[f'{name}_latch_bar_boss'],'second':plates,'label':'bar support to backing plate'})

    # An open-top catch has a shallow, physically reachable lift requirement.
    # Keep the catch wholly on the post. Offset hinge axes can carry the leaf
    # edge a few millimetres toward the post early in its opening arc.
    kx=sx+u*.025
    keeper=[]
    for side in (-1,1):
        n=f'{name}_latch_bar_keeper_{side}'
        box(world.geoms,n,(kx,yb+side*.010,zb-.002),(.015,.003,.010),label='Gravity bar keeper wall')
        keeper.append(n)
    box(world.geoms,f'{name}_latch_bar_keeper_floor',(kx,yb,zb-.012),(.015,.013,.003),label='Keeper floor')
    box(world.geoms,f'{name}_latch_bar_keeper_plate',(sx+u*.025,frame_surface+v*.003,zb-.010),
        (.025,.003,.028),label='Post-mounted keeper plate')
    box(world.geoms,f'{name}_latch_bar_keeper_bracket',(kx,frame_surface+v*.018,zb-.014),
        (.020,.012,.005),label='Keeper standoff bracket')
    for k,dz in enumerate((-.026,.010)):
        _screw(world.geoms,f'{name}_latch_bar_keeper_screw_{k}',
               (sx+u*.025,frame_surface+v*.004,zb+dz),v,steel)
    for side in (-1,1):
        drop,run=.016,.032
        phi=math.atan2(-side*drop,run)
        normal=np.array([0,side*drop,run])/math.hypot(drop,run)
        middle=np.array([kx,yb+side*.029,zb])-.002*normal
        g=C.box(f'{name}_latch_bar_ramp_{side}',tuple(middle),(.015,math.hypot(run,drop)/2,.002),
                mat,7850,True,True,ALL_TIERS,'latch','Gravity bar closing ramp',
                quat=quat_from_axis_angle((1,0,0),phi),friction=(.12,.005,.0001))
        world.geoms.append(g)
    attachments.extend([
        {'first':[f'{name}_latch_bar_keeper_plate'],'second':[frame_prefix],'label':'keeper plate to actual frame'},
        {'first':[f'{name}_latch_bar_keeper_bracket'],'second':[f'{name}_latch_bar_keeper_plate'],'label':'keeper support'},
        {'first':[f'{name}_latch_bar_keeper_floor'],'second':[f'{name}_latch_bar_keeper_bracket'],'label':'keeper floor support'}])
    pulls=[]
    for face in (-1,1):
        n=f'{name}_suffolk_pull_{face}'
        y=face*(t/2+.055)
        z=zt-.115
        leaf_body.geoms.append(C.cyl(n,(xt,y,z),.009,.060,mat,(0,0,1),7850,True,True,ALL_TIERS,'operator','Fixed Suffolk pull'))
        for k,dz in enumerate((-.060,.060)):
            arm=f'{n}_arm_{k}'
            a,b=sorted((face*(t/2+.004),y))
            leaf_body.geoms.append(C.cyl(arm,(xt,(a+b)/2,z+dz),.007,(b-a)/2,mat,(0,1,0),7850,True,True,ALL_TIERS,'operator','Fixed pull mounting arm'))
            attachments.extend([{'first':[arm],'second':plates,'label':'pull arm to backing plate'},
                                {'first':[arm],'second':[n],'label':'fixed pull connection'}])
        site=f'{name}_grip_n' if face<0 else f'{name}_grip_p'
        leaf_body.sites.append(Site(site,(xt,y+face*.009,z),tuple(quat_z_to((0,face,0))),.008,'grip'))
        pulls.append(site)
    selected=thumb.joint.name if v>0 else bar.joint.name
    release=f'{name}_thumb_push' if v>0 else f'{name}_latch_bar_grip'
    model.meta.setdefault('gate_hardware',[]).append({
        'schema':'doorbench.gate-hardware.v1','kind':'contact_suffolk','operator_joint':selected,
        'release_site':release,'thumb_joint':thumb.joint.name,'bar_joint':bar.joint.name,
        'thumb_site':f'{name}_thumb_push','bar_site':f'{name}_latch_bar_grip',
        'thumb_face':-v,'bar_face':v,'pull_sites':pulls,'attachments':attachments,
        'keeper_geoms':keeper,'tang_geom':f'{name}_thumb_lifter','bar_geom':f'{name}_latch_bar_geom',
        'slot_bounds_xz_m':list(slot),'plate_geoms':plates,'contact_driven':True,'self_latching':True})
    model.meta['gate_hardware'][-1]['independent_blocking_lock']=bool(
        spec.get('lock',{}).get('engaged',False) and spec['lock']['model']!='jam_stuck')
    model.meta['gate_hardware'][-1]['friction_jam']=spec.get('lock',{}).get('model')=='jam_stuck'
    model.meta.setdefault('notes',[]).append('Suffolk thumb tang raises the gravity bar through native contact; no joint equality. Bearing joints are ideal; geometry/friction are engineering approximations.')
    return {'operator_joint':selected,'grip_height':grip_z if v>0 else zb+.005}
