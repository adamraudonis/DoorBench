"""Supported fixed pulls for independently opening a Dutch upper leaf."""
from ..ir import Body,Site,ALL_TIERS,quat_z_to
from . import common as C


def add_dutch_upper_pulls(model,upper,spec,*,x_edge,u,split,thickness):
    # Original welded bar geometry:31 mm finger gap behind a14 mm grip,
    # two stems, and face-mounted25 mm diameter fixing pads. Every tier
    # retains the complete load path; the joint bolt is not a substitute pull.
    steel=C.mat_from_material(model,'stainless','mat_dutch_upper_pull')
    height=split+.24;x=x_edge-u*.15;sites=[]
    for face,tag in ((-1,'n'),(1,'p')):
        name='dutch_upper_pull_'+tag
        body=model.add_body(Body(name,upper.name,(x,face*thickness/2,height),
            semantic='operator',label='Fixed upper-leaf pull with two supported stems'))
        body.geoms.append(C.cyl(name+'_bar',(0,face*.038,0),.007,.080,steel,(0,0,1),7850,
            True,True,ALL_TIERS,'operator','14 mm fixed grip'))
        for i,z in enumerate((-.080,.080)):
            body.geoms.append(C.cyl(name+f'_pad_{i}',(0,face*.002,z),.0125,.002,steel,(0,1,0),7850,
                True,True,ALL_TIERS,'operator','Face fixing pad'))
            body.geoms.append(C.cyl(name+f'_stem_{i}',(0,face*.021,z),.006,.017,steel,(0,1,0),7850,
                True,True,ALL_TIERS,'operator','Stem joins fixing pad to grip'))
        site=name+'_grip_'+tag;sites.append(site)
        body.sites.append(Site(site,(0,face*.045,0),tuple(quat_z_to((0,face,0))),.01,'grip'))
        model.meta.setdefault('mechanism_mass_bodies',[]).append(body.name)
    model.meta['dutch_upper_controls']={'joint':upper.joint.name,'sites':sites,
        'finger_gap_m':.031,'kind':'fixed_supported_two_stem_pull',
        'scope':'Original fixed grip and mounting geometry; not strength or embodied reach certification'}
