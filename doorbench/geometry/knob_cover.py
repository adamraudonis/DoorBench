"""Original freely rotating knob cover with two real finger apertures.

The access-hole class is informed by Safety 1st's Grip 'n Twist instructions.
Dimensions, retained bearing and shell segmentation are original generic CAD;
this is not a child-resistance or product certification.
"""
from __future__ import annotations

import math
import numpy as np
import trimesh

from ..ir import Body,Joint,Site,ALL_TIERS,QUAT_ID,quat_z_to
from . import common as C
from . import meshes as MESH

SOURCE='https://safety1st.com/products/home-safeguarding-set-80-piece-hs265'


def add_knob_cover_operator(model,leaf,spec,op,u,x,z,t,faces,locked_backlash,name):
    metal=C.mat_from_material(model,'brass','mat_covered_knob')
    plastic=C.mat_from_material(model,'pvc','mat_knob_cover')
    travel=op.travel if locked_backlash is None else max(locked_backlash,.01)
    operator=Body(name,leaf.name,(x,0.,z),semantic='operator',label='Knob inside a freely rotating cover')
    operator.joint=Joint(name+'_hinge','hinge',(0.,-u,0.),range=(0.,travel),damping=.02,
        frictionloss=.02+.02*op.mass,stiffness=op.spring_rate,
        springref=-op.spring_torque_preload/op.spring_rate,armature=2e-5,
        role='operator',label='Turn the inner knob through the cover openings')
    radius=.027;center=.0438
    operator.geoms.append(C.cyl(name+'_spindle',(0.,0.,0.),.006,t/2+.006,metal,(0,1,0),7850,False,True,ALL_TIERS,'mechanism','Retained knob spindle'))
    rows=[]
    for face in faces:
        tag='p' if face>0 else 'n'
        for sector in range(24):
            angles=[sector*math.tau/24,(sector+1)*math.tau/24]
            points=[(r*math.cos(a),face*depth,r*math.sin(a))
                    for depth in (t/2,t/2+.008) for r in (.01175,.032) for a in angles]
            mesh=trimesh.convex.convex_hull(np.asarray(points))
            key=MESH.key_for('covered_knob_rose_sector',thickness=t,face=face,sector=sector)
            leaf.geoms.append(C.mesh_geom(name+f'_fixed_rose_{tag}_{sector}',key,mesh,
                (x,0.,z),QUAT_ID,metal,7100,True,ALL_TIERS,'operator','Fixed mounting rose with open spindle bore'))
        operator.geoms.append(C.cyl(name+'_neck_'+tag,(0.,face*(t/2+.017),0.),.011,.010,metal,(0,1,0),7100,True,True,ALL_TIERS,'operator','Knob neck through cover collar'))
        knob=name+'_knob_'+tag
        operator.geoms.append(C.sphere(knob,(0.,face*(t/2+center),0.),radius,metal,3000,True,ALL_TIERS,'operator','Inner knob'))
        operator.geoms[-1].friction=(.9,.005,.0001)
        cover_name=name+'_cover_'+tag
        cover=Body(cover_name,leaf.name,(x,0.,z),semantic='mechanism',label='Independent knob-cover shell')
        cover.joint=Joint(cover_name+'_hinge','hinge',(0.,-u,0.),range=None,damping=.0001,
            frictionloss=.0002,armature=0.,role='mechanism',robot_interactive=False,
            label='Free cover rotation; does not turn the knob')
        # Piecewise rounded shell. Separate convex sectors keep its cavity and
        # apertures open in native collision, unlike one filled convex hull.
        profile=[(.010,.014),(.016,.032),(.022,.041),(.025,.041),(.063,.041),(.068,.037),(.078,.022),(.082,.008)]
        shell=[]
        for band,((a,ra),(b,rb)) in enumerate(zip(profile,profile[1:])):
            for sector in range(36):
                low=(sector-.5)*math.tau/36;high=(sector+.5)*math.tau/36
                mid=sector*math.tau/36
                aperture=abs(math.atan2(math.sin(mid),math.cos(mid)))<math.radians(40)+1e-9 or abs(abs(math.atan2(math.sin(mid),math.cos(mid)))-math.pi)<math.radians(40)+1e-9
                if band==3 and aperture:continue
                points=[(r*math.cos(theta),face*(t/2+depth),r*math.sin(theta))
                        for depth,outer in ((a,ra),(b,rb)) for r in (outer-.002,outer)
                        for theta in (low,high)]
                mesh=trimesh.convex.convex_hull(np.asarray(points))
                key=MESH.key_for('knob_cover_sector',thickness=t,face=face,band=band,sector=sector)
                geom=cover_name+f'_shell_{band}_{sector}'
                cover.geoms.append(C.mesh_geom(geom,key,mesh,(0,0,0),QUAT_ID,plastic,1250,True,ALL_TIERS,'mechanism','Hollow cover wall'))
                cover.geoms[-1].friction=(.15,.001,.0001)
                shell.append(geom)
        # Two fingers start halfway before the operator's target angle so
        # their whole required turn remains inside the opposing apertures.
        sites=[]
        for finger in range(2):
            theta=-u*op.travel/2+finger*math.pi
            normal=np.array([math.cos(theta),0.,math.sin(theta)])
            point=radius*normal;point[1]=face*(t/2+center)
            site=name+('_grip_'+tag if finger==0 else '_opposed_grip_'+tag)
            operator.sites.append(Site(site,tuple(point),tuple(quat_z_to(normal)),.006,'grip',ALL_TIERS))
            sites.append(site)
        cover.sites.append(Site(cover_name+'_surface',(0.,face*(t/2+center),.041),tuple(quat_z_to((0,0,1))),.006,'touch',ALL_TIERS))
        model.add_body(cover)
        model.meta.setdefault('mechanism_mass_bodies',[]).append(cover.name)
        model.meta.setdefault('physical_inertia_joints',[]).append(cover.joint.name)
        rows.append({'face':face,'cover_body':cover.name,'cover_joint':cover.joint.name,
            'shell_geoms':shell,'cover_site':cover.sites[0].name,'knob_geom':knob,
            'grip_sites':sites,'knob_radius_m':radius,'finger_radius_m':.006,
            'opening_half_angle_rad':math.radians(45),'knob_center_local':[0.,face*(t/2+center),0.]})
    model.add_body(operator)
    model.meta.setdefault('knob_covers',[]).append({'kind':'free_shell_with_finger_apertures',
        'operator_joint':operator.joint.name,'operator_body':operator.name,'leaf_body':leaf.name,
        'required_turn_rad':op.travel,'faces':rows,'reference':SOURCE,
        'scope':'Independent retained shell, real opposing apertures and inner knob contact. Ideal spindle/bearing joints; no child-resistance certification.'})
    return operator
