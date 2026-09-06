"""Original guided roller-hand-chain and ideal 4:1 spur-gear transmission.

This is generic engineered geometry, not an OEM CAD copy. Real material links
circulate on contact-selected pocket wheels. The native gear equality models
ideal rigid meshing; its displayed teeth do not duplicate that constraint with
contact. All shafts, bearings, axial cheeks and wall mounts are explicit.
"""
from __future__ import annotations
import math
import numpy as np
from ..ir import Body,Joint,Geom,Site,Equality,ALL_TIERS,quat_z_to,quat_from_axis_angle
from . import common as C


def _capsule(name,a,b,r,material,*,density=7850.,label='Rounded steel chain pin'):
    a=np.asarray(a,float);b=np.asarray(b,float);v=b-a
    return Geom(name,'capsule',(r,float(np.linalg.norm(v))/2),tuple((a+b)/2),tuple(quat_z_to(v)),material,
        True,True,density,friction=(.12,.001,.0001),solref=(.001,1.),solimp=(.999,.99999,.0001),semantic='mechanism',part_label=label)


def _contact(geom):
    geom.friction=(.12,.001,.0001);geom.solref=(.001,1.);geom.solimp=(.999,.99999,.0001)
    return geom


def hand_chain_dimensions(top_z,pitch=.04,teeth=16):
    r=pitch/(2*math.sin(math.pi/teeth));steps=math.ceil((top_z-.45)/pitch);low=top_z-steps*pitch
    points=[(r*math.cos(a),top_z+r*math.sin(a))for a in np.linspace(math.pi,0,teeth//2+1)]
    points.extend((r,top_z-i*pitch)for i in range(1,steps+1))
    points.extend((r*math.cos(a),low+r*math.sin(a))for a in np.linspace(0,-math.pi,teeth//2+1)[1:])
    points.extend((-r,low+i*pitch)for i in range(1,steps+1))
    vectors=np.asarray(points[:-1])-np.asarray(points[1:]);angles=np.arctan2(vectors[:,0],vectors[:,1])
    return {'pitch_m':pitch,'wheel_teeth':teeth,'pitch_radius_m':r,'apothem_m':r*math.cos(math.pi/teeth),
        'top_z_m':top_z,'idler_z_m':low,'link_count':len(angles),'points_yz':points,'angles_rad':angles.tolist()}


def _wheel(body,prefix,x,params,metal):
    """Cast aluminium web with real steel rounded teeth, all collidable."""
    apo=params['apothem_m'];teeth=params['wheel_teeth'];delta=2*math.pi/teeth
    body.geoms.append(_contact(C.cyl(prefix+'_web',(x,0,0),apo-.012,.008,metal,(1,0,0),2700,
        semantic='mechanism',label='Solid cast-aluminium pocket-wheel web')))
    for k in range(teeth):
        a=math.pi-(k+.5)*delta;p=np.array([apo*math.cos(a),apo*math.sin(a)])
        body.geoms.append(_capsule(prefix+f'_tooth_{k}',(x,*(p*.87)),(x,*p),.003,metal,label='Rounded steel pocket-wheel tooth'))
        body.geoms.append(_capsule(prefix+f'_pin_{k}',(x-.0025,*p),(x+.0025,*p),.003,metal))


def _wheel_housing(world,prefix,x,y,z,upper,params,steel,wall_back):
    from .garage_tiltup import _bearing_eye
    r=params['pitch_radius_m'];aa=np.linspace(0,math.pi,49)if upper else np.linspace(math.pi,2*math.pi,49)
    for edge in (-1,1):
        rr=r+(.008 if edge>0 else -.009)
        for side in (-1,1):
            for k,(a,b)in enumerate(zip(aa[:-1],aa[1:])):
                world.geoms.append(_capsule(f'{prefix}_radial_{edge}_{side}_{k}',(x+side*.010,y+rr*math.cos(a),z+rr*math.sin(a)),
                    (x+side*.010,y+rr*math.cos(b),z+rr*math.sin(b)),.0015,steel,label='Housing race retaining chain roller ends in sprocket pockets'))
                if k%8==0:
                    mid=(a+b)/2
                    world.geoms.append(_capsule(f'{prefix}_race_mount_{edge}_{side}_{k}',
                        (x+side*.010,y+rr*math.cos(mid),z+rr*math.sin(mid)),
                        (x+side*.015,y+rr*math.cos(mid),z+rr*math.sin(mid)),.0012,steel,
                        label='Welded race standoff joining the axial housing cheek'))
    for side in (-1,1):
        for k,(a,b)in enumerate(zip(aa[:-1],aa[1:])):
            mid=(a+b)/2
            world.geoms.append(_contact(C.box(f'{prefix}_cheek_{side}_{k}',(x+side*.015,y+r*math.cos(mid),z+r*math.sin(mid)),
                (.002,.023,r*(b-a)*.55),steel,7850,semantic='frame',label='Axial chain-wheel housing cheek',
                quat=tuple(quat_from_axis_angle((1,0,0),mid)))))
        _bearing_eye(world,f'{prefix}_bearing_{side}',(x+side*.035,y,z),steel,inner=.0075,outer=.018,half_length=.007)
    # Brackets sit outboard of the circulating material, then join the
    # bearing upper rim. Every fixed load path reaches the actual side wall.
    world.geoms.append(C.box(prefix+'_wall_mount',(x+.075,(wall_back+y)/2,z),(.015,(y-wall_back)/2,.03),steel,7850,
        semantic='frame',label='Pocket-wheel bearing bracket bolted to side wall'))
    world.geoms.append(C.box(prefix+'_bearing_bridge',(x+.050,y,z+.016),(.025,.012,.006),steel,7850,
        semantic='frame',label='Bearing upper-rim connection to wall bracket'))
    # Axial cheek sectors join a rigid radial housing crown; its outboard
    # bridge reaches the bearing mount without obstructing the wheel.
    crown=z+(1 if upper else -1)*(r+.024)
    world.geoms.append(C.box(prefix+'_housing_crown',(x,y,crown),(.017,.032,.003),steel,7850,semantic='frame',label='Joined chain housing crown'))
    world.geoms.append(C.box(prefix+'_housing_mount',(x+.060,y,(crown+z+.03)/2),(.007,.018,abs(crown-(z+.03))/2+.004),steel,7850,semantic='frame',label='Housing-to-bearing bracket connector beyond the shaft end'))
    world.geoms.append(C.box(prefix+'_housing_bridge',(x+.038,y,crown),(.026,.018,.004),steel,7850,semantic='frame',label='Housing crown outboard bridge'))
    bearing_rim=z+(1 if upper else -1)*.014
    world.geoms.append(C.box(prefix+'_inboard_bearing_post',(x-.035,y,(crown+bearing_rim)/2),
        (.006,.012,abs(crown-bearing_rim)/2+.004),steel,7850,semantic='frame',label='Inboard bearing rim joined to housing crown'))
    world.geoms.append(C.box(prefix+'_inboard_housing_bridge',(x-.024,y,crown),(.017,.018,.004),steel,7850,
        semantic='frame',label='Inboard bearing post connection to joined housing cheeks'))


def _gear(body,prefix,x,r,teeth,steel):
    body.geoms.append(C.cyl(prefix+'_web',(x,0,0),r-.008,.004,steel,(1,0,0),7850,collision=False,
        semantic='mechanism',label='Steel spur-gear web; native ideal ratio constraint carries tooth load'))
    for k in range(teeth):
        a=2*math.pi*k/teeth
        body.geoms.append(C.box(prefix+f'_tooth_{k}',(x,r*math.cos(a),r*math.sin(a)),(.004,.006,math.pi*r/teeth*.35),steel,7850,
            collision=False,semantic='mechanism',label='Spur tooth visualization; ideal geared-joint constraint is authoritative',
            quat=tuple(quat_from_axis_angle((1,0,0),a))))


def add_chain_hoist(model,spec,curtain,barrel,world,steel):
    """Build the physical input; caller promotes metadata only after native QA."""
    from .garage_tiltup import _bearing_eye
    width=curtain['width_m'];y=curtain['barrel_y_m'];z=curtain['barrel_z_m'];gear_x=width/2+.31;chain_x=gear_x+.10
    params=hand_chain_dimensions(z-.15);top=params['top_z_m'];low=params['idler_z_m'];wall_back=spec['opening']['wall_thickness']/2
    metal=C.mat_from_material(model,'aluminum_dark','mat_hand_chain_wheel')
    # Through-shaft variant needs a bored stationary counterbalance anchor.
    world.geoms[:]=[g for g in world.geoms if g.name!='curtain_torsion_anchor']
    _bearing_eye(world,'curtain_torsion_anchor',(width/2+.19,y,z),steel,inner=.0125,outer=.035,half_length=.025)
    # The existing bracket stopped the manual shaft. A through-drive shaft
    # needs a genuine open bore/recess, with upper/lower webs still anchored.
    world.geoms[:]=[g for g in world.geoms if g.name!='curtain_barrel_bracket_r']
    rear=y-.015
    world.geoms.append(C.box('curtain_barrel_bracket_r_web',(width/2+.20,(wall_back+rear)/2,z),(.025,(rear-wall_back)/2,.08),steel,7850,semantic='frame',label='Through-shaft bracket front web'))
    for sign,tag in ((-1,'lower'),(1,'upper')):
        world.geoms.append(C.box('curtain_barrel_bracket_r_'+tag,(width/2+.20,y-.0075,z+sign*.0475),(.025,.0075,.0325),steel,7850,semantic='frame',label='Bracket web bordering actual shaft-clearance recess'))
    output=model.add_body(Body('hoist_output_gear',barrel.name,semantic='mechanism',label='Keyed 48-tooth barrel gear and shaft extension'))
    _gear(output,'hoist_output',gear_x,.12,48,steel)
    output.geoms.append(C.cyl('hoist_barrel_extension',(width/2+.265,0,0),.012,.115,steel,(1,0,0),7850,
        semantic='hinge',label='Keyed steel output shaft extending the existing barrel axle'))
    _bearing_eye(world,'hoist_output_bearing',(width/2+.35,y,z),steel,inner=.0125,outer=.028,half_length=.012)
    world.geoms.append(C.box('hoist_output_wall_mount',(width/2+.47,(wall_back+y)/2,z),(.015,(y-wall_back)/2,.045),steel,7850,semantic='frame',label='Barrel gear end bearing wall bracket beyond the hand-chain plane'))
    world.geoms.append(C.box('hoist_output_bearing_bridge',(width/2+.42,y,z+.023),(.065,.018,.006),steel,7850,semantic='frame',label='Output bearing connection above the chain-wheel sweep'))
    input_body=model.add_body(Body('hoist_input',None,(gear_x,y,top),joint=Joint('hoist_input_hinge','hinge',(-1,0,0),range=None,
        damping=.02,frictionloss=.05,role='mechanism',robot_interactive=False),semantic='mechanism',label='Hand pocket wheel and keyed 12-tooth input pinion'))
    _gear(input_body,'hoist_input',0,.03,12,steel);_wheel(input_body,'hoist_hand_wheel',.10,params,metal)
    input_body.geoms.append(C.cyl('hoist_input_shaft',(.05,0,0),.007,.092,steel,(1,0,0),7850,semantic='hinge',label='Pinion and hand-wheel supported keyed shaft'))
    idler=model.add_body(Body('hoist_return_idler',None,(chain_x,y,low),joint=Joint('hoist_idler_hinge','hinge',(-1,0,0),range=None,
        damping=.02,role='mechanism',robot_interactive=False),semantic='mechanism',label='Supported lower hand-chain return idler'))
    _wheel(idler,'hoist_idler',0,params,metal);idler.geoms.append(C.cyl('hoist_idler_shaft',(0,0,0),.007,.045,steel,(1,0,0),7850,semantic='hinge',label='Steel return-idler shaft in two bearings'))
    for prefix,zz,upper in [('hoist_upper',top,True),('hoist_lower',low,False)]:
        _wheel_housing(world,prefix,chain_x,y,zz,upper,params,steel,wall_back)
    bodies=[output.name,input_body.name,idler.name];sites=[];nodes=params['points_yz'];angles=params['angles_rad'];pitch=params['pitch_m'];parent=None
    for k,angle in enumerate(angles):
        initial=parent is None
        link=model.add_body(Body(f'hoist_chain_link_{k}',parent,
            (chain_x,y+nodes[0][0],nodes[0][1])if initial else (0,0,-pitch),tuple(quat_from_axis_angle((-1,0,0),angle if initial else angle-angles[k-1])),
            joint=Joint('hoist_chain_free','free',range=None,role='mechanism',robot_interactive=False)if initial else
                  Joint(f'hoist_chain_pin_{k}','hinge',(-1,0,0),range=(-1.3,1.3),damping=.0004,role='mechanism',robot_interactive=False),
            semantic='mechanism',label=f'Circulating steel roller-chain material link {k+1}'))
        for side in (-1,1):
            link.geoms.append(_contact(C.box(f'hoist_link_{k}_plate_{side}',(side*.009,0,-pitch/2),(.002,.004,pitch/2),steel,7850,
                semantic='mechanism',label='Steel roller-chain side plate')))
        link.geoms.append(_capsule(f'hoist_link_{k}_roller',(-.008,0,0),(.008,0,0),.004,steel,label='Rounded-end steel chain roller'))
        name=f'hoist_chain_grip_{k}';link.sites.extend([Site(name,(0,0,-pitch/2),role='grip'),Site(f'hoist_chain_node_{k}',(0,0,0),role='mechanism')])
        sites.append(name);bodies.append(link.name);parent=link.name
    model.bodies[-1].sites.append(Site('hoist_chain_loop_end',(0,0,-pitch),role='mechanism'))
    model.equalities.extend([
        Equality(kind='connect',name='hoist_chain_loop',a=parent,b='hoist_chain_link_0',anchor=(0,0,-pitch),tiers=ALL_TIERS,
            solref=(.001,1.),solimp=(.999,.99999,.0001),label='Physical final chain pin closes the circulating material loop'),
        Equality(kind='joint',name='hoist_gear_ratio',a=barrel.joint.name,b='hoist_input_hinge',polycoeff=(0,-.25,0,0,0),tiers=ALL_TIERS,
            solref=(.002,1.),solimp=(.99,.999,.001),label='Ideal external 48:12 spur-gear pair, output = -input / 4')])
    model.contact_excludes.append(('hoist_chain_link_0',parent))  # Shared final/first pin neighbors only.
    model.meta.setdefault('mechanism_mass_bodies',[]).extend(bodies)
    model.meta.setdefault('physical_inertia_joints',[]).extend(
        model.body(name).joint.name for name in bodies if model.body(name).joint is not None)
    return {'schema_version':1,'kind':'guided_roller_hand_chain_with_ideal_4_to_1_spur_gears','parameters':params,
        'input_joint':'hoist_input_hinge','output_joint':barrel.joint.name,'output_per_input':-.25,
        'free_root_joint':'hoist_chain_free','free_root_body':'hoist_chain_link_0','material_bodies':bodies[3:],'material_grip_sites':sites,
        'wheel_center_m':[chain_x,y,top],'opening_pull_strand_y_sign':-1,'closing_pull_strand_y_sign':1,
        'hand_force_limit_N':120.,'nominal_regrasp_height_m':1.2,
        'scope':'Original guided roller-chain operator. Material links and pocket-wheel contacts are native; external spur gears use an ideal torque-transmitting ratio constraint, not tooth compliance. Not OEM-rated CAD.',
        'state_requirement':'Native full qpos/body-pose recording; never interpolate the free root or material chain from a scalar drum angle.'}
