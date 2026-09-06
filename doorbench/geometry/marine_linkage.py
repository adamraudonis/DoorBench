"""Original supported handwheel reduction and pin-linked marine dog mechanism.

Four independent dog equalities are replaced by four physical parallelogram
linkages. A supported 6:1 spur pair is represented by one explicitly ideal gear
constraint. The rods, pins, bearings, clearances and masses remain native.
This is not an OEM model or a pressure/sealing/gear-strength certification.
"""
from __future__ import annotations
import math
import numpy as np

from ..ir import Body, Joint, Equality, ALL_TIERS, quat_from_axis_angle
from .. import materials as M, hardware as H
from . import common as C
from .marine_dogs import bearing_y
from .pocket_hardware import cut_box_recess


def _gear(body,name,y,r,teeth,material,phase=0.):
    # Separate tooth solids document actual meshing dimensions. Native contact
    # does not double-constrain the same teeth already linked by ideal gearing.
    body.geoms.append(C.cyl(name+'_web',(0,y,0),r-.0025,.005,material,(0,1,0),
        7850,False,True,ALL_TIERS,'mechanism','Keyed gear web; ideal rigid tooth relation'))
    for k in range(teeth):
        a=2*math.pi*k/teeth+phase
        body.geoms.append(C.box(name+f'_tooth_{k}',(r*math.cos(a),y,r*math.sin(a)),
            (.0025,.005,math.pi*r/teeth*.38),material,7850,False,True,ALL_TIERS,
            'mechanism','Spur tooth; no compliance/wear/strength model',
            quat=tuple(quat_from_axis_angle((0,-1,0),a))))


def _crank(body,name,offset,y,material):
    x,z=offset
    radius=math.hypot(x,z)
    body.geoms.append(C.box(name+'_arm',(x/2,y-.0045,z/2),
        (radius/2,.0015,.005),material,7850,True,True,ALL_TIERS,'mechanism','Keyed flat steel dog crank',
        quat=tuple(quat_from_axis_angle((0,-1,0),math.atan2(z,x)))))
    body.geoms.append(C.cyl(name+'_pin',(x,y-.0015,z),.0025,.0045,material,(0,1,0),
        7850,True,True,ALL_TIERS,'mechanism','Retained connecting-rod pin'))
    body.geoms.append(C.cyl(name+'_pin_collar',(x,y+.0035,z),.0045,.0005,material,(0,1,0),
        7850,True,True,ALL_TIERS,'mechanism','Connecting-rod pin retaining collar'))


def add_marine_wheel_linkage(model,spec):
    """Install supported reduction and genuine rod loops on a four-dog wheel leaf."""
    if not spec['kinematics'].get('wheel_dogging'):
        raise ValueError('Wheel linkage requires a wheel-dogged marine door')
    leaf=model.body('leaf');wheel=model.body('wheel')
    dogs=[model.body(f'dog_{k}') for k in range(4)]
    u=float(model.meta['u']);t=float(spec['leaf']['thickness'])
    steel=C.mat_from_material(model,'stainless','mat_marine_transmission')
    x,_,z=wheel.pos;output_pos=np.array([x,0.,z+.14])
    travel=float(wheel.joint.range[1]);maximum=travel/6
    if abs(maximum-math.pi/2)>.001:
        raise ValueError('Marine reduction expects the authored 1.5-turn wheel')
    # The output shaft counter-rotates, so positive scalar progress still
    # releases every dog. Their geometric positions and locking wedges remain.
    output=model.add_body(Body('marine_reduction_output',leaf.name,tuple(output_pos),
        joint=Joint('marine_reduction_hinge','hinge',(0,u,0),range=(0.,maximum),
            damping=.05,frictionloss=.05,robot_interactive=False,role='mechanism'),
        semantic='mechanism',label='Supported 120-tooth output gear and connecting cranks'))
    wheel.geoms=[g for g in wheel.geoms if g.name!='wheel_spindle']
    wheel.mass_override=None
    for geom in wheel.geoms:
        if geom.name.startswith('wheel_wheel_col_'):
            geom.density=0.
        elif geom.type=='mesh':
            geom.density=M.MATERIALS[H.OPERATORS[spec['operator']['model']].material].density
        if geom.pos[1]>0:
            geom.pos=(geom.pos[0],geom.pos[1]+.13,geom.pos[2])
        elif geom.pos[1]<0:
            geom.pos=(geom.pos[0],geom.pos[1]-.020,geom.pos[2])
    for site in wheel.sites:
        if site.pos[1]>0:site.pos=(site.pos[0],site.pos[1]+.13,site.pos[2])
        elif site.pos[1]<0:site.pos=(site.pos[0],site.pos[1]-.020,site.pos[2])
    # One actual spindle reaches both handwheels. The rear stand-off places
    # the handwheel beyond the rod planes, and a rigid bearing frame supports it.
    wheel.geoms.append(C.cyl('wheel_spindle',(0,.07,0),.007,t/2+.15,steel,(0,1,0),
        7850,True,True,ALL_TIERS,'mechanism','Through spindle keyed to both handwheels and input pinion'))
    output.geoms.append(C.cyl('marine_output_spindle',(0,.054,0),.007,t/2+.076,
        steel,(0,1,0),7850,True,True,ALL_TIERS,'mechanism','Keyed output spindle'))
    gear_y=t/2+.025
    _gear(wheel,'marine_pinion',gear_y,.02,20,steel)
    _gear(output,'marine_output_gear',gear_y,.12,120,steel,phase=math.pi/120)
    mount=model.add_body(Body('marine_reduction_mount',leaf.name,(x,0,z),
        semantic='mechanism',label='Welded bearing frame carrying wheel and reduction shafts'))
    for offset in (0.,.14):
        cut_box_recess(leaf,(x-.008,-t/2-.001,z+offset-.008),
            (x+.008,t/2+.001,z+offset+.008),'marine_shaft_'+str(offset))
        for side in (-1,1):
            bearing_y(mount,f'marine_shaft_{offset}_{side}',(0,side*(t/2+.006),offset),steel)
        bearing_y(mount,f'marine_shaft_{offset}_rear',(0,t/2+.115,offset),steel)
        carrier=wheel if offset==0 else output
        for yy in (-(t/2+.018),t/2+.018,t/2+.126):
            carrier.geoms.append(C.cyl(f'{carrier.name}_shaft_retainer_{yy}',(0,yy,0),.011,.003,
                steel,(0,1,0),7850,True,True,ALL_TIERS,'mechanism','Axial shaft retaining collar'))
    # Rear bearing frame above two lower stand-offs. All four rods lie in
    # front of this frame; the stand-offs are below their innermost sweep.
    for side in (-1,1):
        mount.geoms.append(C.box(f'marine_mount_standoff_{side}',(side*.024,(t/2+.115)/2,-.14),
            (.008,(t/2+.115)/2,.012),steel,7850,semantic='mechanism',label='Reduction bearing frame welded to leaf'))
        mount.geoms.append(C.box(f'marine_mount_rear_post_{side}',(side*.024,t/2+.115,.0025),
            (.009,.006,.1545),steel,7850,semantic='mechanism',label='Rear bearing frame rail'))
    for zz in (0.,.14):
        for side in (-1,1):
            mount.geoms.append(C.box(f'marine_mount_bridge_{zz}_{side}',(side*.019,t/2+.115,zz),
                (.008,.006,.010),steel,7850,semantic='mechanism',label='Bearing outer rim welded to rail'))
    old_names={f'wheel_dog_{k}' for k in range(4)}
    model.equalities=[e for e in model.equalities if e.name not in old_names]
    model.equalities.append(Equality('joint','marine_spur_ratio',output.joint.name,wheel.joint.name,
        (0,1/6,0,0,0),tiers=ALL_TIERS,label='Ideal keyed 20:120 external spur reduction; tooth compliance not modeled',
        solref=(.002,1.),solimp=(.999,.9999,.0001)))
    rods=[];cranks=[]
    for k,dog in enumerate(dogs):
        dog.joint.axis=(0,u,0);dog.joint.range=(0.,maximum)
        delta=np.asarray(dog.pos)-output_pos
        angle=math.atan2(delta[2],delta[0])-u*math.pi/4
        crank=np.array([.035*math.cos(angle),.035*math.sin(angle)])
        lane=t/2+.045+k*.010
        shaft=next(g for g in dog.geoms if g.name==dog.name+'_spindle')
        near=-(t/2+.064);rear=lane+.008
        shaft.pos=(0,(near+rear)/2,0);shaft.size=(.007,(rear-near)/2)
        _crank(output,f'marine_crank_{k}',crank,lane,steel)
        _crank(dog,f'marine_dog_crank_{k}',crank,lane,steel)
        # The fixed offset of each pin from its shaft is identical at both
        # ends. The rod remains parallel to the line joining those shafts.
        rod=model.add_body(Body(f'marine_rod_{k}',output.name,(crank[0],lane,crank[1]),
            joint=Joint(f'marine_rod_{k}_hinge','hinge',(0,1,0),range=(-math.pi,math.pi),
                damping=.002,frictionloss=.002,robot_interactive=False,role='mechanism'),
            semantic='mechanism',label='Rigid connecting rod with two bored pin eyes'))
        length=np.linalg.norm(delta)
        rod.geoms.append(C.box(f'marine_rod_{k}_bar',tuple(delta/2),
            (length/2-.005,.0015,.006),steel,7850,True,True,ALL_TIERS,
            'mechanism','3 mm flat steel connecting rod transmitting dog load',
            quat=tuple(quat_from_axis_angle((0,-1,0),math.atan2(delta[2],delta[0])))))
        bearing_y(rod,f'marine_rod_{k}_eye_a',(0,0,0),steel,inner=.0029,outer=.006,half_length=.0015)
        bearing_y(rod,f'marine_rod_{k}_eye_b',tuple(delta),steel,inner=.0029,outer=.006,half_length=.0015)
        model.equalities.append(Equality('connect',f'marine_rod_{k}_pin',rod.name,dog.name,
            anchor=tuple(delta),tiers=ALL_TIERS,label='Real rod eye pinned to dog crank',
            solref=(.002,1.),solimp=(.999,.9999,.0001)))
        rods.append(rod);cranks.append({'dog_body':dog.name,'rod_body':rod.name,
            'crank_offset_xz_m':crank.tolist(),'rod_vector_m':delta.tolist(),'plane_y_m':lane})
    additions=[wheel,output,mount,*rods]
    backed=model.meta.setdefault('mechanism_mass_bodies',[])
    backed.extend(b.name for b in additions if b.name not in backed)
    model.meta.setdefault('physical_inertia_joints',[]).extend(
        b.joint.name for b in [wheel,output,*dogs,*rods] if b.joint.name not in model.meta.get('physical_inertia_joints',[]))
    model.meta['marine_dog_linkage']={'input_joint':wheel.joint.name,'output_joint':output.joint.name,
        'ratio':1/6,'dog_joints':[d.joint.name for d in dogs],
        'rod_joints':[b.joint.name for b in rods],'rod_ratio':-u,
        'gear_equality':'marine_spur_ratio','connect_equalities':[f'marine_rod_{k}_pin' for k in range(4)],
        'cranks':cranks,'input_range_rad':[0.,travel],'output_range_rad':[0.,maximum],
        'scope':'Actual pinned rods and supported shafts; ideal rigid gear ratio. No gasket pressure or strength certification.'}
    model.meta['native_arena_memory_mib']=max(64,model.meta.get('native_arena_memory_mib',16))
    return model.meta['marine_dog_linkage']


def resolve_marine_configuration(model,qpos,meta):
    """Exact parallelogram branch for inspection only; never step native physics."""
    row=meta.get('marine_dog_linkage')
    if not row:return
    def addr(name):return model.jnt_qposadr[model.joint(name).id]
    q=float(qpos[addr(row['input_joint'])])*row['ratio']
    qpos[addr(row['output_joint'])]=q
    for name in row['dog_joints']:qpos[addr(name)]=q
    for name in row['rod_joints']:qpos[addr(name)]=q*row['rod_ratio']
