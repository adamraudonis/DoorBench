"""Generic spring-return wall switches with prepared plates and real press sites.

The stem/guide and mounting plate are modeled; electrical switching is an
explicit displacement threshold, not a detailed contact-block simulation.
"""
from ..ir import Body,Joint,Site,ALL_TIERS,QUAT_ID,quat_z_to
from . import common as C


def add_wall_button(model,world,spec,*,name,x,height,face,radius,travel,colour,
                    joint_role='lock',site_name=None,plate_half=(.04,.06)):
    wall_y=float(model.meta.get('wall_y',0));plane=wall_y+face*spec['opening']['wall_thickness']/2
    metal=C.mat_from_material(model,'stainless',f'mat_{name}_plate')
    plastic=C.mat_rgba(model,f'mat_{name}',colour,.4)
    b=Body(name,world.name,(x,plane+face*.012,height),semantic='sensor',tiers=ALL_TIERS)
    b.joint=Joint(name+'_slide','slide',(0,-face,0),range=(0,travel),damping=1.,
        stiffness=1500.,springref=-travel,limit_solref=(.001,1.),
        role=joint_role,label=name.replace('_',' ').title()+' (press)')
    # Resolve the millimetre stroke stop: the generic 5 ms joint limit lets a
    # sudden 20 N press drive this small cap through its prepared plate.
    model.meta['native_timestep_s']=min(.0005,model.meta.get('native_timestep_s',.002))
    b.geoms.append(C.cyl(name+'_geom',(0,face*.004,0),radius,.004,plastic,(0,1,0),1000,
                         True,True,ALL_TIERS,joint_role,'Spring-return push cap'))
    b.geoms.append(C.cyl(name+'_stem',(0,-face*.0025,0),.004,.0045,metal,(0,1,0),7900,
                         True,True,ALL_TIERS,'mechanism','Stem in prepared plate guide'))
    site=site_name or name+'_push_'+('n' if face<0 else 'p')
    b.sites.append(Site(site,(0,face*.008,0),tuple(quat_z_to((0,face,0))),.008,'push'))
    model.add_body(b)
    # The plate contacts the authored wall face. A 9.5 mm square guide clears
    # the 8 mm stem through its complete stroke; stock is never a solid box
    # behind a moving shaft. The cap clears the plate at full depression.
    half_x,half_z=plate_half;bore=.00475
    guides=[]
    for axis,extent in ((0,half_x),(2,half_z)):
        for sign in (-1,1):
            p=[x,plane+face*.00375,height];size=[half_x,.00375,half_z]
            p[axis]+=sign*(extent+bore)/2;size[axis]=(extent-bore)/2
            size[2 if axis==0 else 0]=half_z if axis==0 else bore
            part=f'{name}_plate_{axis}_{sign}'
            world.geoms.append(C.box(part,p,size,metal,7900,True,True,ALL_TIERS,'sensor','Prepared wall switch plate'))
            guides.append(part)
    model.meta.setdefault('wall_switches',[]).append({'kind':name,'joint':b.joint.name,'body':b.name,
        'face':face,'site':site,'accessible_from_robot':face<0,'travel_m':travel,'cap_geom':name+'_geom',
        'stem_geom':name+'_stem','plate_geoms':guides,'wall_face_y_m':plane,'stem_radial_gap_m':.00075,
        'scope':'Spring-return cap/stem with prepared mounting plate; ideal guide joint and electrical threshold.'})
    model.meta.setdefault('mechanism_mass_bodies',[]).append(b.name)
    return b
