"""Generic indexed horizontal-arm release, separate from credential rotation.

Three sprung catches retain three hinged arms. A single fixed release shoe
reaches only the indexed horizontal catch. Its powered solenoid holds the
spring-driven shoe clear; loss of power releases that catch through contact.
These are authored dimensions, not an OEM CAD reconstruction.
"""
from __future__ import annotations
import math
import hashlib
import numpy as np
import trimesh
from ..ir import Body,Joint,Geom,Site,ALL_TIERS,QUAT_ID,quat_from_axis_angle,quat_to_mat,quat_mul,quat_z_to
from . import common as C


def add_tripod_drop_arm(model,world,rotor,spec):
    if not spec['kinematics'].get('drop_arm'):return None
    steel=C.mat_from_material(model,'stainless','mat_drop_steel')
    axis=np.asarray(rotor.joint.axis,float);origin=np.asarray(rotor.pos,float)
    # Place the independent credential assembly on the back side of the hub;
    # its original right-hand coil end otherwise occupies the vertical drop
    # corridor. All gear, pawl, bolt and fixed support frames rotate together.
    phase=quat_from_axis_angle(axis,math.pi);phase_R=quat_to_mat(phase)
    existing={b.name:b for b in model.bodies}
    for name in ('turnstile_credential_bolt','turnstile_reverse_pawl'):
        if name in existing:
            b=existing[name];b.pos=tuple(origin+phase_R@(np.asarray(b.pos)-origin));b.quat=tuple(quat_mul(phase,b.quat))
    for g in existing['turnstile_mechanism_frame'].geoms:
        g.pos=tuple(origin+phase_R@(np.asarray(g.pos)-origin));g.quat=tuple(quat_mul(phase,g.quat))
    wheel=existing['turnstile_ratchet_wheel'];wheel.quat=tuple(quat_mul(phase,wheel.quat))
    locks=model.meta['turnstile_locks'];locks['mechanism_frame_quat']=list(quat_mul(phase,locks['mechanism_frame_quat']));locks['layout_phase_rad']=math.pi
    # The drop tube passes vertically in front of the rotor. Move the complete
    # credential unit and its coaxial bearing rearward rather than cutting away
    # a load-bearing plate or disabling its contacts. Extend the real journal
    # by the same amount; the bearing cantilever still meets the cabinet back.
    rearward=.100;shift=-axis*rearward
    for name in ('turnstile_credential_bolt','turnstile_reverse_pawl'):
        if name in existing:existing[name].pos=tuple(np.asarray(existing[name].pos)+shift)
    for g in existing['turnstile_mechanism_frame'].geoms:g.pos=tuple(np.asarray(g.pos)+shift)
    wheel.pos=tuple(np.asarray(wheel.pos)+shift)
    for g in world.geoms:
        if g.name.startswith('tripod_bearing_') or g.name=='tripod_support_beam':g.pos=tuple(np.asarray(g.pos)+shift)
    journal=next(g for g in rotor.geoms if g.name=='hub_boss')
    journal.pos=tuple(np.asarray(journal.pos)+shift/2);journal.size=(journal.size[0],journal.size[1]+rearward/2)
    rotor.geoms.remove(journal)
    journal_body=Body('turnstile_drop_journal',rotor.name,(0,0,0),QUAT_ID,tiers=ALL_TIERS,semantic='mechanism')
    journal_body.geoms.append(journal);model.add_body(journal_body)
    locks['mechanism_frame_pos']=list(np.asarray(locks['mechanism_frame_pos'])+shift)
    locks['rearward_layout_offset_m']=rearward
    carrier=Body('turnstile_drop_carrier',rotor.name,(0,0,0),QUAT_ID,tiers=ALL_TIERS,semantic='mechanism')
    fixed=Body('turnstile_drop_fixed',None,(0,0,0),QUAT_ID,tiers=ALL_TIERS,semantic='mechanism',static=True)
    model.add_body(carrier);model.add_body(fixed)
    def box(body,name,pos,half,quat=QUAT_ID):
        g=C.box(name,pos,half,steel,7900,True,True,ALL_TIERS,'mechanism',name.replace('_',' '),quat=quat,friction=(.04,.0001,.00001))
        g.solref=(.0005,1.);g.solimp=(.99,.999,.0001);body.geoms.append(g);return name
    def prism(body,name,vertices):
        shape=trimesh.convex.convex_hull(np.asarray(vertices,float));key=name+'_'+hashlib.sha256(shape.vertices.tobytes()).hexdigest()[:12]
        g=C.mesh_geom(name,key,shape,(0,0,0),QUAT_ID,steel,7900,True,ALL_TIERS,'mechanism',name.replace('_',' '))
        g.friction=(.04,.0001,.00001);g.solref=(.0005,1.);g.solimp=(.99,.999,.0001);body.geoms.append(g);return name
    def transformed_box(name,local,half,rotation,base):
        return box(carrier,name,tuple(base+quat_to_mat(rotation)@local),half,rotation)
    rotor.geoms=[g for g in rotor.geoms if not g.name.startswith('arm_')]
    arm_sites=[s for s in rotor.sites if s.name=='arm_push'];rotor.sites=[s for s in rotor.sites if s.name!='arm_push']
    rows=[];moving=[carrier.name,journal_body.name];transfers={}
    for k in range(3):
        rotation=quat_from_axis_angle(axis,2*math.pi*k/3);R=quat_to_mat(rotation);base=R@np.array([.065,0,0])
        arm=Body(f'turnstile_drop_arm_{k}',rotor.name,tuple(base),tuple(rotation),tiers=ALL_TIERS,semantic='mechanism')
        arm.joint=Joint(f'turnstile_arm_fold_{k}','hinge',(0,1,0),range=(-.02,math.radians(100)),damping=.025,frictionloss=.01,
            role='mechanism',robot_interactive=False,label='Manually resettable gravity drop arm')
        length=float(spec['leaf']['width'])-.065;mass=math.pi*(.019**2-.0175**2)*length*7900
        arm.geoms.append(Geom(f'arm_{k}_col','capsule',(.019,(length-.038)/2),(length/2,0,0),tuple(quat_z_to((1,0,0))),steel,
            True,True,7900,mass,(.6,.005,.0001),tiers=ALL_TIERS,semantic='operator',part_label='38 mm stainless drop arm with closed end caps'))
        parts=Body(f'turnstile_drop_arm_hardware_{k}',arm.name,(0,0,0),QUAT_ID,tiers=ALL_TIERS,semantic='mechanism')
        # A closed eye surrounds the actual transverse hinge pin.
        for n in range(16):
            a,b=2*math.pi*n/16,2*math.pi*(n+1)/16
            points=[(r*math.cos(t),r*math.sin(t)) for r,t in ((.0045,a),(.008,a),(.008,b),(.0045,b))]
            prism(parts,f'turnstile_drop_eye_{k}_{n}',[(x,y,z) for x,z in points for y in (-.019,.019)])
        box(parts,f'turnstile_drop_eye_bridge_{k}',(.015,0,0),(.009,.012,.006))
        # The raised toe permits a release shoe on the clear, forward side of
        # the swept arm plane. The load-carrying web is rigidly joined to tube.
        box(parts,f'turnstile_drop_web_{k}',(.022,0,.05),(.006,.006,.050))
        toe=box(parts,f'turnstile_drop_toe_{k}',(.044,0,.100),(.016,.006,.004))
        stop=box(parts,f'turnstile_drop_fold_stop_{k}',(.012,0,-.005),(.006,.010,.004))
        for side in (-1,1):
            transformed_box(f'turnstile_drop_hinge_ear_{k}_{side}',np.array([0,side*.024,0]),(.012,.004,.016),rotation,base)
            transformed_box(f'turnstile_drop_hinge_root_{k}_{side}',np.array([-.020,side*.024,0]),(.012,.004,.012),rotation,base)
        pin=C.cyl(f'turnstile_drop_hinge_pin_{k}',tuple(base),.004,.032,steel,tuple(R@np.array([0,1,0])),7900,True,True,ALL_TIERS,'hinge','Drop-arm hinge pin')
        carrier.geoms.append(pin)
        closed_stop=transformed_box(f'turnstile_drop_stop_{k}',np.array([-.015,0,-.012]),(.006,.012,.010),rotation,base)
        # Native parent/child filtering otherwise suppresses the real arm-stop
        # contact. Request this pair explicitly, never disable all filtering.
        model.meta.setdefault('native_contact_pairs',[]).append({'geom1':stop,'geom2':closed_stop,
            'solref':[.0005,1.],'solimp':[.99,.999,.0001],'friction':[.04,.04,.0001,.00001,.00001]})
        catch=Body(f'turnstile_drop_catch_{k}',rotor.name,tuple(base+R@np.array([.044,0,.088])),tuple(rotation),tiers=ALL_TIERS,semantic='mechanism')
        catch.joint=Joint(f'turnstile_drop_catch_slide_{k}','slide',(0,-1,0),range=(0,.028),stiffness=500.,springref=0.,damping=3.,
            role='mechanism',robot_interactive=False,label='Spring-return contact catch; rising arm cams it out for reset')
        # Flat top carries downward load; diagonal underside cams the catch
        # aside when the operator raises the fallen arm back into engagement.
        catch_geom=prism(catch,f'turnstile_drop_catch_wedge_{k}',[(x,y,z) for x in (-.012,.052) for y,z in ((-.008,-.008),(-.008,.008),(.008,.008))])
        tail=box(catch,f'turnstile_drop_catch_tail_{k}',(-.080,.008,.042),(.009,.006,.003))
        box(catch,f'turnstile_drop_catch_outer_riser_{k}',(.048,0,.015),(.004,.004,.013))
        box(catch,f'turnstile_drop_catch_tail_beam_{k}',(-.016,0,.025),(.064,.004,.003))
        box(catch,f'turnstile_drop_catch_tail_web_{k}',(-.080,0,.034),(.006,.004,.011))
        box(catch,f'turnstile_drop_catch_stem_{k}',(0,-.030,0),(.003,.030,.003))
        flange=box(catch,f'turnstile_drop_catch_return_flange_{k}',(0,-.061,0),(.008,.004,.008))
        # A real, hub-fixed guide surrounds the stem below the toe; its walls
        # have half-millimetre clearance from the moving 6 mm stem.
        for s in (-1,1):
            guides=[transformed_box(f'turnstile_drop_catch_guide_x_{k}_{s}',np.array([.044+s*.0055,-.047,.088]),(.002,.010,.0055),rotation,base),
                    transformed_box(f'turnstile_drop_catch_guide_z_{k}_{s}',np.array([.044,-.047,.088+s*.0055]),(.0035,.010,.002),rotation,base)]
            for guide in guides:
                model.meta['native_contact_pairs'].append({'geom1':flange,'geom2':guide,
                    'solref':[.0005,1.],'solimp':[.99,.999,.0001],'friction':[.04,.04,.0001,.00001,.00001]})
        # The mounting web ends below the running stem; the four guide walls
        # continue around the bore. A solid web through that bore would be
        # hidden by native parent/child collision filtering.
        transformed_box(f'turnstile_drop_catch_guide_root_{k}',np.array([.012,-.037,.04425]),(.038,.004,.04025),rotation,base)
        if k==0:
            for site in arm_sites:
                site.pos=tuple(np.asarray(site.pos)-base);arm.sites.append(site)
        arm.sites.append(Site(f'turnstile_drop_reset_grip_{k}',(min(.38,length-.035),0,-.019),tuple(quat_z_to((0,0,-1))),.012,'grip'))
        model.add_body(arm);model.add_body(parts);model.add_body(catch);moving.extend([parts.name,catch.name]);transfers[arm.name]=rotor.name
        rows.append({'arm_joint':arm.joint.name,'arm_body':arm.name,'catch_joint':catch.joint.name,'catch_body':catch.name,
                     'catch_geom':catch_geom,'toe_geom':toe,'tail_geom':tail,'fold_stop_geoms':[stop,closed_stop],
                     'reset_site':f'turnstile_drop_reset_grip_{k}','indexed_rotor_angle_rad':(-2*math.pi*k/3)%(2*math.pi),
                     'tube_mass_kg':mass,'tube_length_m':length})
    # Fixed spring-driven release shoe. The solenoid supplies force only to
    # this physical plunger, never to an arm or rotor coordinate.
    px=float(origin[0]+.065+.044-.080);py=float(origin[1]);pz=float(origin[2]+.130)
    home=np.array([px,py+.040,pz]);cam=Body('turnstile_drop_release',None,tuple(home),QUAT_ID,tiers=ALL_TIERS,semantic='mechanism')
    cam.joint=Joint('turnstile_drop_release_slide','slide',(0,-1,0),range=(0,.036),stiffness=800.,springref=.080,damping=20.,
        role='mechanism',robot_interactive=False,label='Power holds drop-release shoe clear; power loss lets spring release indexed catch')
    nose=box(cam,'turnstile_drop_release_nose',(0,-.008,0),(.008,.004,.002))
    box(cam,'turnstile_drop_release_stem',(0,.022,0),(.004,.034,.003))
    box(cam,'turnstile_drop_release_flange',(0,.046,0),(.009,.004,.008))
    for sign in (-1,1):
        box(fixed,f'turnstile_drop_coil_x_{sign}',(px+sign*.0145,py+.072,pz),(.005,.030,.0165))
        box(fixed,f'turnstile_drop_coil_z_{sign}',(px,py+.072,pz+sign*.0125),(.0095,.030,.004))
        box(fixed,f'turnstile_drop_front_stop_x_{sign}',(px+sign*.00725,py+.042,pz),(.00275,.004,.012))
        box(fixed,f'turnstile_drop_front_stop_z_{sign}',(px,py+.042,pz+sign*.0085),(.0045,.004,.0035))
    box(fixed,'turnstile_drop_coil_end',(px,py+.100,pz),(.0195,.004,.0165))
    # Cabinet-fixed cantilever stays entirely ahead of the rotating arm plane.
    cabinet=next(g for g in world.geoms if g.name=='cabinet_end_p');cx,cy,_=cabinet.pos;top=1.17
    box(fixed,'turnstile_drop_cabinet_stay',(cx,cy,1.075),(.012,.012,.095))
    box(fixed,'turnstile_drop_cross_beam',((cx+px)/2,cy,top),((px-cx)/2+.012,.012,.012))
    box(fixed,'turnstile_drop_forward_beam',(px,(cy+py+.11)/2,top),(.012,(cy-py-.11)/2+.012,.012))
    box(fixed,'turnstile_drop_coil_stay',(px,py+.110,(top+pz)/2),(.012,.012,(top-pz)/2))
    model.add_body(cam);moving.append(cam.name)
    model.meta.setdefault('mechanism_mass_bodies',[]).extend(moving)
    model.meta.setdefault('material_transfer_bodies',{}).update(transfers)
    model.meta.setdefault('physical_inertia_joints',[]).extend([cam.joint.name]+[n for r in rows for n in (r['arm_joint'],r['catch_joint'])])
    record={'schema':'doorbench.turnstile-drop-arm.v1','rotor_joint':rotor.joint.name,'arms':rows,
        'release_joint':cam.joint.name,'release_nose_geom':nose,'powered_by_default':True,
        'coil_force_at_seat_N':100.,'gap_scale_m':.100,'release_stroke_m':.036,
        'fixed_support_geoms':[g.name for g in fixed.geoms],'anchor_geom':'cabinet_end_p',
        'journal_body':journal_body.name,'journal_geom':journal.name,'journal_mass_kg':journal.mass(),
        'mass_status':'Tube material transfers from the rotor slab budget; pins, catches, solenoid and brackets retain separate geometry-backed mass',
        'scope':'Generic indexed horizontal-arm catch and manual cam-reset mechanism; no OEM, fire, strength or egress certification'}
    model.meta['turnstile_drop_arm']=record
    return record
