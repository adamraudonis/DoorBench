"""Original prepared joining and inactive-leaf hardware, with contact keepers.

Installation classes follow Ives Dutch/flush bolts and ordinary cane bolts;
the generic dimensions and ideal prismatic bearings are not OEM internals.
"""
from __future__ import annotations

from ..ir import ALL_TIERS, Body, Joint, Site, quat_z_to
from . import common as C


DUTCH_SOURCE='https://allegion.ca/content/dam/allegion-us-2/web-files/ives/installation-documents/Ives_054_Dutch_Door_Bolt_Installation_Instructions_107772.pdf'


def _vertical_collar(body,name,x,y_surface,z,face,material,*,standoff=.012,radius=.006,
                     gap=.00075,half_length=.006):
    """Four solid walls around an open shaft bore, bonded to the leaf face."""
    inner=radius+gap;wall=.002;outer=inner+wall;names=[]
    # Side walls run from the actual mounting surface to the outer cover.
    height=standoff+outer
    for side in (-1,1):
        g=C.box(name+f'_side_{side}',(x+side*(inner+wall/2),y_surface+face*height/2,z),
                (wall/2,height/2,half_length),material,7900,tiers=ALL_TIERS,
                semantic='lock',label='Prepared bolt guide side fixed to leaf')
        body.geoms.append(g);names.append(g.name)
    for tag,low,high in (('back',0.,standoff-inner),('cover',standoff+inner,height)):
        g=C.box(name+'_'+tag,(x,y_surface+face*(low+high)/2,z),
                (inner,(high-low)/2,half_length),material,7900,tiers=ALL_TIERS,
                semantic='lock',label='Prepared bolt guide with open shaft bore')
        body.geoms.append(g);names.append(g.name)
    return names


def add_dutch_join_bolt(model,upper,lower,spec,*,x_edge,u,split,thickness):
    face=1 if spec['robot']['robot_outside'] else -1
    x=x_edge-u*.060;surface=face*thickness/2;standoff=.012
    metal=C.mat_from_material(model,'stainless','mat_dutch_joining_bolt')
    mount=Body('join_bolt_mount',upper.name,semantic='mechanism',label='Dutch bolt prepared guides')
    keeper=Body('join_keeper_mount',lower.name,semantic='mechanism',label='Dutch bolt keeper on lower leaf')
    guides=[]
    for k,z in enumerate((split+.020,split+.06445)):
        guides.extend(_vertical_collar(mount,f'join_bolt_guide_{k}',x,surface,z,face,metal))
    keepers=_vertical_collar(keeper,'join_keeper',x,surface,split-.009,face,metal,half_length=.004)
    # The shaft remains in both prepared guides throughout withdrawal. The
    # rear finger lever stays beyond the upper guide through its whole stroke.
    bolt=Body('join_bolt',upper.name,(x,surface+face*standoff,split),semantic='lock',label='Dutch joining bolt')
    bolt.joint=Joint('join_bolt_slide','slide',(0,0,1),range=(0.,.030),
        damping=2.,frictionloss=3.,limit_solref=(.001,1.),
        initial=0. if spec['kinematics'].get('joining_bolt_engaged') else .030,modeled_at=0.,
        role='lock',robot_interactive=face<0,label='Joining bolt (lift to separate leaves)')
    bolt.geoms.append(C.cyl('join_bolt_rod',(0,0,.041),.006,.055,metal,(0,0,1),7900,
        True,True,ALL_TIERS,'lock','Joining rod through actual lower-leaf keeper'))
    bolt.geoms.append(C.cyl('join_bolt_lever',(0,face*.015,.090),.004,.015,metal,(0,1,0),7900,
        True,True,ALL_TIERS,'operator','Finger lever bonded to joining rod'))
    bolt.geoms.append(C.sphere('join_bolt_knob',(0,face*.030,.090),.006,metal,7900,
        True,ALL_TIERS,'operator','Exposed joining-bolt finger knob'))
    bolt.sites.append(Site('join_bolt_grip',(0,face*.036,.090),tuple(quat_z_to((0,face,0))),.005,'grip',ALL_TIERS))
    for body in (mount,keeper,bolt):
        for geom in body.geoms:
            geom.solref=(.001,1.);geom.solimp=(.95,.99,.0001)
        model.add_body(body)
        model.meta.setdefault('mechanism_mass_bodies',[]).append(body.name)
    model.meta.setdefault('physical_inertia_joints',[]).append(bolt.joint.name)
    model.meta['native_timestep_s']=min(.0005,model.meta.get('native_timestep_s',.002))
    # Explicit shaft/guide pairs retain contacts despite welded parent filtering.
    for geom in guides+keepers:
        model.meta.setdefault('native_contact_pairs',[]).append({'geom1':'join_bolt_rod','geom2':geom,
            'solref':[.001,1.], 'solimp':[.95,.99,.0001]})
    record={'joint':bolt.joint.name,'site':'join_bolt_grip','body':bolt.name,
        'upper_body':upper.name,'lower_body':lower.name,
        'upper_joint':upper.joint.name,'lower_joint':lower.joint.name,
        'face':face,'accessible_from_robot':face<0,
        'engaged_initial':bool(spec['kinematics'].get('joining_bolt_engaged')),
        'travel_m':.030,'withdrawn_threshold_m':.025,'rod_geom':'join_bolt_rod',
        'grip_geom':'join_bolt_knob','guide_geoms':guides,'keeper_geoms':keepers,
        'guide_clearance_m':.00075,'rod_radius_m':.006,'tip_z_joined_m':split-.014,
        'keeper_z_bounds_m':[split-.013,split-.005],
        'guide_spacing_m':.04445,'edge_offset_m':.060,
        'source':DUTCH_SOURCE,
        'scope':'Inside-face joining rod and prepared guides/keeper. Ideal prismatic bearing with frictional position retention; no bolt-strength or single-hand manipulation certification.'}
    model.meta['dutch_joining_bolt']=record
    return record
