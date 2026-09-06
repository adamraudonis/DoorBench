"""Generic contact ratchet and separate solenoid-released turnstile index bolt.

The topology is informed by Alvarado's documented reversing pawl, lock arms,
solenoids and fixed mechanism base. Dimensions and force ratings are authored
engineering parameters, not recovered manufacturer CAD or certification.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
import trimesh
from ..ir import Body,Joint,ALL_TIERS,QUAT_ID,quat_z_to,quat_rotate,quat_mul
from . import common as C


def add_turnstile_locks(model,world,rotor,spec,*,full_height=False):
    mat=C.mat_from_material(model,'steel_galvanized','mat_rotor_lock')
    axis=np.asarray(rotor.joint.axis,float);quat=quat_z_to(axis)
    R=np.column_stack([quat_rotate(quat,e) for e in np.eye(3)])
    offset=float(spec['leaf']['height'])+.055 if full_height else -.080
    origin=np.asarray(rotor.pos)+axis*offset
    frame=Body('turnstile_mechanism_frame',None,(0.,0.,0.),QUAT_ID,tiers=ALL_TIERS,semantic='mechanism')
    frame.static=True
    wheel=Body('turnstile_ratchet_wheel',rotor.name,tuple(axis*offset),tuple(quat),tiers=ALL_TIERS,semantic='mechanism')
    model.add_body(frame);model.add_body(wheel)
    def append(body,geom):
        if body is frame:
            geom.pos=tuple(origin+R@np.asarray(geom.pos));geom.quat=tuple(quat_mul(quat,geom.quat))
        body.geoms.append(geom)
    def box(body,name,pos,half,*,collision=True):
        geom=C.box(name,pos,half,mat,7850,collision,True,ALL_TIERS,'mechanism',name.replace('_',' '),friction=(.05,.0001,.00001))
        geom.solref=(.002,1.);geom.solimp=(.99,.999,.0001);append(body,geom);return name
    def cylinder(body,name,pos,radius,half,*,collision=True):
        geom=C.cyl(name,pos,radius,half,mat,(0,0,1),7850,collision,True,ALL_TIERS,'mechanism',name.replace('_',' '))
        geom.friction=(.05,.0001,.00001);geom.solref=(.002,1.);geom.solimp=(.99,.999,.0001);append(body,geom);return name
    def mesh(body,name,vertices):
        shape=trimesh.convex.convex_hull(np.asarray(vertices,float));key=name+'_'+hashlib.sha256(shape.vertices.tobytes()).hexdigest()[:12]
        geom=C.mesh_geom(name,key,shape,(0,0,0),QUAT_ID,mat,7850,True,ALL_TIERS,'mechanism',name.replace('_',' '))
        geom.friction=(.04,.0001,.00001);geom.solref=(.002,1.);geom.solimp=(.99,.999,.0001);body.geoms.append(geom);return name
    if full_height:
        # The rotor's upper column terminates below a real journal. A slender
        # coaxial shaft continues through the fixed bearing and gear bores.
        column=next(g for g in rotor.geoms if g.name=='rotor_column')
        low=column.pos[2]-column.size[1];top=float(spec['leaf']['height'])+.005
        column.pos=(0.,0.,(low+top)/2);column.size=(column.size[0],(top-low)/2)
        cylinder(rotor,'turnstile_upper_journal',(0,0,top+.025),.018,.055)
    # A split fixed base surrounds the existing coaxial journal. The native
    # rotor hinge remains the bearing model; no intersecting solid bearing.
    backing=[]
    bore=.0185
    for sign in (-1,1):
        backing.append(box(frame,f'turnstile_base_x_{sign}',(sign*(.160+bore)/2,0,-.040),((.160-bore)/2,.140,.004)))
        backing.append(box(frame,f'turnstile_base_y_{sign}',(0,sign*(.140+bore)/2,-.040),(bore,(.140-bore)/2,.004)))
    mounts=[]
    if full_height:
        for sign in (-1,1):mounts.append(box(frame,f'turnstile_roof_mount_{sign}',(sign*.120,.105,.018),(.008,.010,.062)))
        anchor='cage_roof'
    else:
        anchor='tripod_bearing_back'
        # Two stays meet the existing bearing's actual backplate.
        for sign in (-1,1):mounts.append(box(frame,f'turnstile_bearing_mount_{sign}',(sign*.031,0,-.092),(.010,.010,.048)))
    def ring(name,outer,z,half):
        for k in range(36):
            a,b=2*math.pi*k/36,2*math.pi*(k+1)/36
            points=[(r*math.cos(t),r*math.sin(t)) for r,t in ((.018,a),(outer,a),(outer,b),(.018,b))]
            mesh(wheel,f'{name}_{k}',[(x,y,zz) for x,y in points for zz in (z-half,z+half)])
    ring('turnstile_ratchet_hub',.060,0,.004)
    box(wheel,'turnstile_gear_key',(.018,0,.0125),(.002,.002,.0175),collision=False)
    teeth=[];pawl_name=None
    if spec['kinematics'].get('one_way',False):
        count=36;step=2*math.pi/count;phase=.035
        for k in range(count):
            angle=k*step+phase
            points=[(.060*math.cos(angle),.060*math.sin(angle)),(.067*math.cos(angle),.067*math.sin(angle)),(.060*math.cos(angle+step),.060*math.sin(angle+step))]
            teeth.append(mesh(wheel,f'turnstile_ratchet_tooth_{k}',[(x,y,z) for x,y in points for z in (-.004,.004)]))
        pivot=(-.060,-.085,0.)
        # Static frame wrappers are flattened by exporters. These fixed-axis
        # moving bodies therefore carry their composed world rest frame.
        pawl=Body('turnstile_reverse_pawl',None,tuple(origin+R@np.asarray(pivot)),tuple(quat),tiers=ALL_TIERS,semantic='mechanism')
        pawl.joint=Joint('turnstile_reverse_pawl_hinge','hinge',(0,0,1),(0,0,0),(-.03,.25),
            stiffness=6.,springref=-.035,damping=.04,limit_solref=(.002,1.),role='mechanism',robot_interactive=False,
            label='Spring-loaded physical reverse pawl')
        # The tip lies just inside the tooth crest. The pivot is to its inside
        # so reverse tangential force seats the pawl against its lower stop;
        # the gentle forward flank lifts it outward against its spring.
        box(pawl,'turnstile_reverse_pawl_heel',(-.008,.012,0),(.010,.005,.005))
        box(pawl,'turnstile_reverse_pawl_bar',(-.016,.046,0),(.002,.038,.005))
        box(pawl,'turnstile_reverse_pawl_nose',(-.010,.085,0),(.008,.0008,.005))
        tip='turnstile_reverse_pawl_nose'
        # Annular eye clears the fixed load stop through the full lift range.
        # The former square eye corners swept into that stop at large lift.
        for k in range(16):
            a,b=2*math.pi*k/16,2*math.pi*(k+1)/16
            points=[(r*math.cos(t),r*math.sin(t)) for r,t in ((.004,a),(.006,a),(.006,b),(.004,b))]
            mesh(pawl,f'turnstile_pawl_eye_{k}',[(x,y,z) for x,y in points for z in (-.005,.005)])
        box(pawl,'turnstile_pawl_eye_bridge',(-.004,.008,0),(.004,.004,.005))
        cylinder(frame,'turnstile_pawl_pivot',(pivot[0],pivot[1],-.020),.0035,.029,collision=False)
        cylinder(frame,'turnstile_pawl_retainer',(pivot[0],pivot[1],.008),.006,.002,collision=False)
        box(frame,'turnstile_pawl_load_stop',(-.056,-.073,-.0165),(.002,.006,.0235))
        model.add_body(pawl);pawl_name=pawl.joint.name
        model.meta.setdefault('physical_inertia_joints',[]).append(pawl_name)
    # An independent slot wheel has one credential stop per passage sector.
    # It is separate from the one-way pawl, permitting bidirectional sources.
    slots=int(spec['kinematics']['wings']);half_slot=.075;outer=.077;inner=.056;index_geoms=[]
    ring('turnstile_index_hub',inner,.025,.005)
    for slot in range(slots):
        start=slot*2*math.pi/slots+half_slot;end=(slot+1)*2*math.pi/slots-half_slot
        angles=np.linspace(start,end,max(4,math.ceil((end-start)/.10))+1)
        for k,(a,b) in enumerate(zip(angles[:-1],angles[1:])):
            points=[(r*math.cos(t),r*math.sin(t)) for r,t in ((inner,a),(outer,a),(outer,b),(inner,b))]
            index_geoms.append(mesh(wheel,f'turnstile_index_sector_{slot}_{k}',[(x,y,z) for x,y in points for z in (.020,.030)]))
    # The fail-secure bolt is the solenoid armature itself. Power pulls this
    # actual rigid member out of a physical slot, never disables a rotor limit.
    bolt=Body('turnstile_credential_bolt',None,tuple(origin+R@np.asarray((.081,0,.025))),tuple(quat),tiers=ALL_TIERS,semantic='lock')
    locked=bool(spec['kinematics'].get('locked_until_credential'))
    bolt.joint=Joint('turnstile_credential_release','slide',(1,0,0),(0,0,0),(0.,.022),
        stiffness=1000.,springref=0.,damping=8.,limit_solref=(.002,1.),role='lock',robot_interactive=False,
        label='Power retracts physical fail-secure index bolt')
    bolt_geom=box(bolt,'turnstile_credential_bolt_geom',(0,0,0),(.015,.004,.006))
    box(bolt,'turnstile_credential_armature',(.034,0,0),(.020,.003,.004))
    supports=[]
    for sign in (-1,1):
        supports.append(box(frame,f'turnstile_bolt_guide_y_{sign}',(.097,sign*.0065,.025),(.004,.002,.009)))
        supports.append(box(frame,f'turnstile_bolt_guide_z_{sign}',(.097,0,.025+sign*.009),(.004,.0045,.0025)))
        supports.append(box(frame,f'turnstile_coil_y_{sign}',(.123,sign*.014,.025),(.034,.006,.016)))
        supports.append(box(frame,f'turnstile_coil_z_{sign}',(.123,0,.025+sign*.0115),(.034,.008,.005)))
        supports.append(box(frame,f'turnstile_coil_base_{sign}',(.122,sign*.018,-.005),(.025,.006,.031)))
    box(frame,'turnstile_coil_end',(.160,0,.025),(.003,.020,.016))
    frame.geoms[-1].solref=(.0005,1.)
    model.add_body(bolt)
    model.meta.setdefault('mechanism_mass_bodies',[]).extend([wheel.name,bolt.name]+(['turnstile_reverse_pawl'] if pawl_name else []))
    model.meta.setdefault('physical_inertia_joints',[]).append(bolt.joint.name)
    rotor.joint.range=None;rotor.joint.ratchet_one_way=False
    rotor.joint.label='Rotor constrained by native pawl and credential bolt contacts'
    row={'schema':'doorbench.turnstile-lock.v1','rotor_joint':rotor.joint.name,'pawl_joint':pawl_name,
         'one_way':bool(spec['kinematics'].get('one_way',False)),'ratchet_teeth':teeth,'index_geoms':index_geoms,
         'pawl_tip_geom':'turnstile_reverse_pawl_nose' if pawl_name else None,'pawl_stop_geom':'turnstile_pawl_load_stop' if pawl_name else None,
         'bolt_joint':bolt.joint.name,'bolt_geom':bolt_geom,'coil_force_at_seat_N':30.,'gap_scale_m':.040,'stroke_m':.022,
         'powered_by_default':not locked,'credential_locked_by_default':locked,'sector_angle_rad':2*math.pi/slots,
         'mount_geoms':mounts,'frame_anchor_geom':anchor,'base_geoms':backing,'solenoid_support_geoms':supports,
         'fixed_support_geoms':[g.name for g in frame.geoms],
         'mechanism_frame_pos':origin.tolist(),'mechanism_frame_quat':list(quat),
         'journal_geom':'turnstile_upper_journal' if full_height else 'hub_boss',
         'scope':'Generic sprung pivot pawl and physical index slot/solenoid bolt; ideal electrical and magnetic input; no OEM/egress certification'}
    model.meta['turnstile_locks']=row
    model.meta['native_timestep_s']=min(float(model.meta.get('native_timestep_s',.002)),.00025)
    return row
