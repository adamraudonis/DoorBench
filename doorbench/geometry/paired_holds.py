"""Prepared inactive-leaf bolts with actual fixed receivers and moving rods.

Original generic hardware, informed by Ives FB358 and National N165-902
installation classes. Prismatic bearings and frictional position retention are
idealized; no OEM internals, fastener pullout or strength rating is claimed.
"""
from __future__ import annotations

import copy
import numpy as np

from ..ir import ALL_TIERS, Body, Joint, Site, quat_z_to, quat_to_mat
from . import common as C
from .lock_stock import cut_stock, geom_bounds
from .paired_hardware import _vertical_collar

FLUSH_SOURCE='https://allegion.ca/content/dam/allegion-us-2/web-files/ives/installation-documents/Ives_FB358_Manual_Flush_Bolt_Installation_Template_101738.pdf'
CANE_SOURCE='https://nationalhardwarestorage.blob.core.windows.net/documents/nh_td_836_n165-902.pdf'


def _backed(model,body):
    model.add_body(body)
    model.meta.setdefault('mechanism_mass_bodies',[]).append(body.name)
    if body.joint:model.meta.setdefault('physical_inertia_joints',[]).append(body.joint.name)


def _pair(model,a,b):
    model.meta.setdefault('native_contact_pairs',[]).append({'geom1':a,'geom2':b,
        'solref':[.001,1.],'solimp':[.95,.99,.0001]})


def _ring_z(body,name,center,inner,outer,half,material,label):
    x,y,z=center;names=[];thickness=outer-inner
    for side in(-1,1):
        for axis in(0,1):
            p=[x,y,z];p[axis]+=side*(inner+thickness/2)
            h=[inner,outer,half] if axis==0 else [inner,thickness/2,half]
            if axis==0:h[0]=thickness/2
            g=C.box(name+f'_{axis}_{side}',tuple(p),tuple(h),material,7900,
                    tiers=ALL_TIERS,semantic='lock',label=label)
            body.geoms.append(g);names.append(g.name)
    return names


def _receiver(world,name,x,y,plane,direction,radius,material):
    """A real lined square socket prepared into floor or head stock."""
    inner=radius+.00075;outer=inner+.003;depth=.027
    end=plane+direction*depth;lo,hi=sorted((plane,end))
    names={g.name for g in world.geoms if g.type=='box' and g.semantic in('floor','frame','wall')}
    cut=cut_stock(world,(x-outer,y-outer,lo),(x+outer,y+outer,hi),name+'_socket',names=names)
    if not cut['removed_geoms']:raise ValueError('Paired bolt receiver is not embedded in actual fixed stock')
    keepers=_ring_z(world,name,(x,y,(lo+hi)/2),inner,outer,(hi-lo)/2,material,'Embedded prepared inactive-leaf bolt receiver')
    cap=C.box(name+'_end',(x,y,end+direction*.001), (outer,outer,.001),material,7900,
              tiers=ALL_TIERS,semantic='lock',label='Closed end of embedded bolt socket')
    # Prepare the back-cap thickness too, keeping its outside face against
    # fixed stock instead of embedding an uncut steel plate in the substrate.
    caplo,caphigh=geom_bounds(cap)
    cut_stock(world,caplo,caphigh,name+'_end_bore',names={g.name for g in world.geoms if g.type=='box' and g.semantic in('floor','frame','wall')})
    world.geoms.append(cap);keepers.append(cap.name)
    return keepers,cut


def _record(model,body,*,kind,leaf,primary,stroke,threshold,rod,site,grip,guides,keepers,stops,face,source):
    row={'kind':kind,'leaf_body':leaf.name,'leaf_joint':leaf.joint.name,
         'primary_body':primary.name,'primary_joint':primary.joint.name,
         'joint':body.joint.name,'body':body.name,'site':site,'rod_geom':rod,'grip_geom':grip,
         'guide_geoms':guides,'keeper_geoms':keepers,'stop_geoms':stops,
         'travel_m':stroke,'withdrawn_threshold_m':threshold,'engaged_initial':True,
         'guide_clearance_m':.00075,'force_cap_N':20.,'face':face,
         'accessible_from_robot':bool(body.joint.robot_interactive),
         'requires_primary_open_rad':.20 if kind=='flush_bolt' else 0.,
         'source':source,
         'scope':'Actual rod, prepared guides and embedded receiver; ideal prismatic bearing and frictional retention. No credential, reach, fastener strength or one-handed usability certification.'}
    model.meta.setdefault('paired_leaf_holds',[]).append(row)
    return row


def _material_cut(model,phys,leaf,before):
    after={kind:sum(g.mass() for g in leaf.geoms if g.semantic==kind) for kind in('leaf','glass')}
    removed={kind:before[kind]-after[kind] for kind in before}
    mass=phys['mass'];row=next(r for r in mass['per_body'] if r['body']==leaf.name)
    old=copy.deepcopy(row)
    for semantic,key in(('leaf','slab_kg'),('glass','glass_kg')):
        amount=removed[semantic]
        if not -1e-10<=amount<row[key]+1e-10:raise ValueError('Invalid inactive bolt stock deduction')
        row[key]-=amount;mass[key]-=amount
        row['total_kg']-=amount;mass['total_kg']-=amount
    phys['per_body_dynamics'][leaf.name]['mass'].update(copy.deepcopy(row),dynamics_mass_kg=row['total_kg'])
    model.meta.setdefault('paired_hold_material_accounting',[]).append({'body':leaf.name,
        'original_row':old,'removed_slab_kg':removed['leaf'],'removed_glass_kg':removed['glass'],
        'scope':'Exact removed routed stock. The source had no inactive-bolt allowance; new moving hardware adds its actual BOM.'})


def _flush(model,world,leaf,primary,spec,material,x_edge,u,z_edge,plane,sign):
    """A recessed edge slide directly connected to a vertical bolt shaft."""
    name=leaf.name+('_flush_top' if sign>0 else '_flush_bottom')
    stroke=.035;radius=.006;x=x_edge-u*.019
    tip=plane+sign*.015;rod_end=tip-sign*.180;zmid=(tip+rod_end)/2
    # The 171 x25 x35 mm housing is recessed into the meeting stile. A small
    # enclosed shaft continuation takes up the retracted rod behind the face.
    za,zb=sorted((z_edge-sign*.004,z_edge-sign*.175))
    xa,xb=sorted((x_edge,x_edge-u*.035))
    cut_stock(leaf,(xa,-.0125,za),(xb,.0125,zb),name+'_housing')
    rz0,rz1=sorted((tip,rod_end-sign*stroke))
    cut_stock(leaf,(x-radius-.00075,-radius-.00075,rz0-.001),
              (x+radius+.00075,radius+.00075,rz1+.001),name+'_shaft')
    mount=Body(name+'_housing',leaf.name,semantic='mechanism',label='Prepared flush-bolt edge housing')
    # Recessed finger opening with a real empty window in its faceplate.
    op_z=z_edge-sign*.092;face_x=x_edge-u*.001
    for side in(-1,1):
        mount.geoms.append(C.box(name+f'_face_side_{side}',(face_x,side*.01025,(za+zb)/2),
            (.001,.00225,(zb-za)/2),material,7900,tiers=ALL_TIERS,semantic='lock',label='Flush-bolt faceplate beside open finger window'))
    window=.040
    for tag,lo,hi in(('low',za,op_z-window),('high',op_z+window,zb)):
        if hi>lo:mount.geoms.append(C.box(name+'_face_'+tag,(face_x,0,(lo+hi)/2),
            (.001,.008,(hi-lo)/2),material,7900,tiers=ALL_TIERS,semantic='lock',label='Flush-bolt faceplate end'))
    back_x=x_edge-u*.034
    mount.geoms.append(C.box(name+'_back',(back_x,0,(za+zb)/2),(.001,.0125,(zb-za)/2),material,7900,
        tiers=ALL_TIERS,semantic='lock',label='Flush-bolt rear housing supported by routed stile'))
    for side in(-1,1):
        mount.geoms.append(C.box(name+f'_case_side_{side}',((xa+xb)/2,side*.0115,(za+zb)/2),
            ((xb-xa)/2,.001,(zb-za)/2),material,7900,tiers=ALL_TIERS,semantic='lock',label='Flush-bolt side housing'))
    guides=[]
    for k,z in enumerate((z_edge-sign*.022,z_edge-sign*.145)):
        guides+=_ring_z(mount,name+f'_guide_{k}',(x,0,z),radius+.00075,.0105,.004,material,'Bored flush-bolt guide fixed to stile')
    # A collar travels between two finite annular stop plates; the slightly
    # wider native range is a safety bound, never the operational stop.
    collar_z=z_edge-sign*.040
    stops=[]
    for tag,z in(('engaged',collar_z+sign*.0045),('withdrawn',collar_z-sign*(stroke+.0045))):
        stops+=_ring_z(mount,name+'_stop_'+tag,(x,0,z),radius+.00075,.011,.002,material,'Physical flush-bolt stroke shoulder')
    rod=Body(name,leaf.name,(x,0,0),semantic='lock',label='Independent flush bolt')
    rod.joint=Joint(name+'_slide','slide',(0,0,-sign),range=(-.003,stroke+.003),
        damping=12.,frictionloss=3.,initial=0.,modeled_at=0.,role='lock',robot_interactive=True,
        label='Flush bolt (withdraw after opening active leaf)',notes='Meeting-edge finger window requires active leaf open; no hidden actuator access')
    rod.geoms.append(C.cyl(name+'_rod',(0,0,zmid),radius,.090,material,(0,0,1),7900,True,True,ALL_TIERS,'lock','Steel flush-bolt shaft'))
    rod.geoms.append(C.cyl(name+'_collar',(0,0,collar_z),.009,.002,material,(0,0,1),7900,True,True,ALL_TIERS,'lock','Bonded shaft collar for finite end stops'))
    # The finger knob is flush behind the edge when closed; it cannot be
    # mistaken for a pull available through the 3 mm meeting-leaf seam.
    knob_x=u*.012;knob_z=op_z+sign*stroke/2
    rod.geoms.append(C.cyl(name+'_finger_stem',(u*.006,0,knob_z),.003,.006,material,(1,0,0),7900,True,True,ALL_TIERS,'operator','Finger tab directly welded to flush rod'))
    rod.geoms.append(C.sphere(name+'_finger_knob',(knob_x,0,knob_z),.006,material,7900,True,ALL_TIERS,'operator','Recessed flush-bolt finger knob'))
    site=name+'_grip';rod.sites.append(Site(site,(knob_x+u*.006,0,knob_z),tuple(quat_z_to((u,0,0))),.005,'grip',ALL_TIERS))
    _backed(model,mount);_backed(model,rod)
    keepers,_=_receiver(world,name+'_receiver',leaf.pos[0]+x,0,plane,sign,radius,material)
    for g in guides+keepers:_pair(model,name+'_rod',g)
    for g in stops:_pair(model,name+'_collar',g)
    return _record(model,rod,kind='flush_bolt',leaf=leaf,primary=primary,stroke=stroke,
        threshold=abs(tip-z_edge)+.004,rod=name+'_rod',site=site,grip=name+'_finger_knob',
        guides=guides,keepers=keepers,stops=stops,face=0,source=FLUSH_SOURCE)


def _stock_surface(leaf,x,z,face):
    """Exact ray intersections with stock boxes, not their rotated AABBs."""
    values=[]
    for g in leaf.geoms:
        if g.semantic not in('leaf','glass') or g.type!='box':continue
        R=quat_to_mat(g.quat);origin=R.T@(np.array([x,0,z])-g.pos);direction=R.T@np.array([0,face,0])
        low=-float('inf');high=float('inf')
        for i,h in enumerate(g.size):
            if abs(direction[i])<1e-10:
                if abs(origin[i])>h+1e-9:break
            else:
                a,b=sorted(((-h-origin[i])/direction[i],(h-origin[i])/direction[i]));low=max(low,a);high=min(high,b)
        else:
            if high>=low:values.append(high)
    if not values:raise ValueError('Cane guide has no supporting leaf stock')
    return max(values)


def _cane(model,world,leaf,primary,spec,material,x_edge,u,thickness):
    name=leaf.name+'_cane';radius=.00615;stroke=.080;x=x_edge-u*.060
    approach=1 if spec['robot'].get('approach_side','-y')=='+y' else -1
    face=-approach if spec['robot']['robot_outside'] else approach
    # Guides may mount on flat slab or a raised brace, but both must reach
    # the same shaft line and have a supported finite mounting footprint.
    guide_z=[];surfaces=[]
    for desired in(.130,.245):
        for z in np.linspace(desired-.015,desired+.015,31):
            samples=[_stock_surface(leaf,x+dx,float(z)+dz,face)for dx in(-.009,0,.009)for dz in(-.006,0,.006)]
            if max(samples)-min(samples)<1e-7:
                guide_z.append(float(z));surfaces.append(max(samples));break
        else:raise ValueError('No finite flat stock footprint for cane-bolt guide')
    # Keep the complete rod and bent grip outside the real plank braces.
    stock_outer=max(face*(geom_bounds(g)[1][1] if face>0 else geom_bounds(g)[0][1])
                    for g in leaf.geoms if g.semantic in('leaf','glass'))
    shaft_level=max(stock_outer+.010,max(surfaces)+.012)
    mount=Body(name+'_mount',leaf.name,semantic='mechanism',label='Cane bolt guides fixed to actual plank surfaces')
    guides=[]
    for k,(z,surface) in enumerate(zip(guide_z,surfaces)):
        guides+=_vertical_collar(mount,name+f'_guide_{k}',x,face*surface,z,face,material,
            standoff=shaft_level-surface,radius=radius,half_length=.006)
    rod=Body(name,leaf.name,(x,face*shaft_level,0),semantic='lock',label='Independent cane bolt')
    rod.joint=Joint(name+'_slide','slide',(0,0,1),range=(-.003,stroke+.003),damping=12.,frictionloss=5.,
        initial=0.,modeled_at=0.,role='lock',robot_interactive=face==approach,
        label='Inside cane bolt (lift clear of floor)',notes='Inside-face manual control; no exterior release')
    tip=-.025;top=tip+.317
    rod.geoms.append(C.cyl(name+'_rod',(0,0,(tip+top)/2),radius,.317/2,material,(0,0,1),7900,True,True,ALL_TIERS,'lock','12.3 mm cane shaft entering floor socket'))
    rod.geoms.append(C.cyl(name+'_bent_grip',(-u*.03875,0,top),radius,.03875,material,(1,0,0),7900,True,True,ALL_TIERS,'operator','77.5 mm bent cane grip bonded to shaft'))
    stops=[]
    for tag,z in(('engaged',guide_z[1]+.0085),('withdrawn',guide_z[0]-.0085-stroke)):
        collar=name+'_collar_'+tag
        rod.geoms.append(C.cyl(collar,(0,0,z),.009,.002,material,(0,0,1),7900,True,True,ALL_TIERS,'lock','Cane shaft travel collar'))
        for guide in guides:_pair(model,collar,guide)
        stops.extend(guides)
    site=name+'_grip';rod.sites.append(Site(site,(-u*.055,face*radius,top),tuple(quat_z_to((0,face,0))),.005,'grip',ALL_TIERS))
    _backed(model,mount);_backed(model,rod)
    keepers,_=_receiver(world,name+'_receiver',leaf.pos[0]+x,face*shaft_level,0.,-1,radius,material)
    for g in guides+keepers:_pair(model,name+'_rod',g)
    return _record(model,rod,kind='cane_bolt',leaf=leaf,primary=primary,stroke=stroke,threshold=.040,
        rod=name+'_rod',site=site,grip=name+'_bent_grip',guides=guides,keepers=keepers,
        stops=sorted(set(stops)),face=face,source=CANE_SOURCE)


def add_inactive_holds(model,world,leaf,primary,spec,phys):
    choice=spec['leaf']['inactive_leaf']['lock']
    if choice not in('flush_bolts','cane_bolt'):raise ValueError('Unsupported inactive-leaf holding hardware')
    stock=[g for g in leaf.geoms if g.semantic in('leaf','glass')]
    bounds=[geom_bounds(g) for g in stock]
    x_edge=min(a[0] for a,b in bounds);z0=min(a[2] for a,b in bounds);z1=max(b[2] for a,b in bounds)
    before={kind:sum(g.mass()for g in leaf.geoms if g.semantic==kind)for kind in('leaf','glass')}
    material=C.mat_from_material(model,'stainless','mat_inactive_leaf_bolts')
    if choice=='flush_bolts':
        _flush(model,world,leaf,primary,spec,material,x_edge,-1,z1,spec['opening']['height'],1)
        _flush(model,world,leaf,primary,spec,material,x_edge,-1,z0,0.,-1)
    else:_cane(model,world,leaf,primary,spec,material,x_edge,-1,spec['leaf']['thickness'])
    _material_cut(model,phys,leaf,before)
    for body in model.bodies:
        for geom in body.geoms:
            if geom.name.startswith(leaf.name+'_flush') or geom.name.startswith(leaf.name+'_cane'):
                geom.solref=(.001,1.);geom.solimp=(.95,.99,.0001)
    model.meta['native_timestep_s']=min(.0005,model.meta.get('native_timestep_s',.002))
    leaf.joint.label='Inactive leaf (physical bolts hold; withdraw before opening)'
    leaf.joint.notes='Full native hinge range; holding comes from actual floor/head bolt contact'
