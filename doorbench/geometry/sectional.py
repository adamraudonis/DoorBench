"""Original standard-lift sectional mechanism with native rollers and cables.

Each section is a rigid hinged panel. Paired captured rollers constrain its
vertical, quarter-circle and overhead travel; no pose projection runs during
native stepping. The two root slides represent the planar axle assembly's
remaining coordinates, not a powered vertical-lift surrogate.

Extension springs use real 2:1 routed cables. Their generic force curves are
calibrated to the authored closed-door counterbalance fraction, not to an OEM
rating or to guaranteed success. Zero/weak fractions remain zero/weak.
"""
from __future__ import annotations

import math
from functools import lru_cache
import xml.etree.ElementTree as ET
import numpy as np

from ..ir import Body, Geom, Joint, Site, Equality, SpatialCable, SpatialSpring, ALL_TIERS, QUAT_ID, quat_z_to
from .. import hardware as H
from . import common as C


def track_dimensions(spec):
    height=float(spec['leaf']['height']);thickness=float(spec['leaf']['thickness'])
    radius=.305
    y=float(spec['opening']['wall_thickness'])/2+thickness/2+.055
    vertical_end=max(height+.10,float(spec['opening']['height'])+.05)
    return {'offset_y_m':y,'vertical_end_z_m':vertical_end,'radius_m':radius,
            'horizontal_end_y_m':y+radius+height+.55,
            'opening_width_m':float(spec['opening']['width']),
            'panel_height_m':height/int(spec['kinematics'].get('n_sections',4)),
            'panel_count':int(spec['kinematics'].get('n_sections',4)),
            'closed_top_z_m':height+.05,'closed_bottom_z_m':.05}


def track_path(s, path):
    """Track centre and unit tangent in world YZ, parameterized by arc length."""
    y=path['offset_y_m'];end=path['vertical_end_z_m'];r=path['radius_m']
    if s<=end:return np.array([y,float(s)]),np.array([0.,1.])
    angle=(s-end)/r
    if angle<=math.pi/2:
        return np.array([y+r*(1-math.cos(angle)),end+r*math.sin(angle)]),np.array([math.sin(angle),math.cos(angle)])
    return np.array([y+r+s-end-r*math.pi/2,end+r]),np.array([1.,0.])


def track_progress(yz, path):
    """Closest actual track parameter, with a distance residual in metres."""
    point=np.asarray(yz,float);y=path['offset_y_m'];z=path['vertical_end_z_m'];r=path['radius_m']
    candidates=[max(0.,min(z,float(point[1])))]
    angle=np.clip(math.atan2(point[1]-z,y+r-point[0]),0.,math.pi/2)
    candidates.append(z+r*angle)
    horizontal=np.clip(point[0]-(y+r),0.,path['horizontal_end_y_m']-(y+r))
    candidates.append(z+r*math.pi/2+horizontal)
    errors=[float(np.linalg.norm(point-track_path(s,path)[0])) for s in candidates]
    i=int(np.argmin(errors));return float(candidates[i]),errors[i]


def inspection_pose(progress, mechanism):
    """Exact rigid chord solution for inspection only, never a native controller."""
    path=mechanism['path'];sh=path['panel_height_m'];count=path['panel_count']
    bounds=mechanism['progress'];s=bounds['closed_s_m']+np.clip(progress,0,1)*(bounds['open_s_m']-bounds['closed_s_m'])
    nodes=[track_path(s,path)[0]]
    # Successive roller centres are one panel chord apart, not one arc apart.
    for _ in range(count):
        low=s;high=s+sh*1.8
        for __ in range(48):
            mid=(low+high)/2
            if np.linalg.norm(track_path(mid,path)[0]-nodes[-1])<sh:low=mid
            else:high=mid
        s=(low+high)/2;nodes.append(track_path(s,path)[0])
    nodes=np.asarray(nodes[::-1]);delta=nodes[:-1]-nodes[1:]
    angles=np.arctan2(delta[:,0],delta[:,1])
    pose={'nodes_yz':nodes.tolist(),'root_y':float(nodes[0,0]-path['offset_y_m']),
            'root_z':float(nodes[0,1]-path['closed_top_z_m']),
            'panel_angles':np.diff(np.r_[0.,angles]).tolist()}
    drive=mechanism.get('drive',{})
    if drive.get('linkage'):
        linkage=drive['linkage'];a=angles[0];local=np.asarray(linkage['panel_anchor_yz'])
        anchor=nodes[0]+np.array([[math.cos(a),math.sin(a)],[-math.sin(a),math.cos(a)]])@local
        dz=anchor[1]-linkage['trolley_z_m'];dy=-math.sqrt(max(0.,linkage['arm_length_m']**2-dz**2))
        pose['trolley_q']=float(anchor[0]-dy-linkage['closed_trolley_y_m'])
        pose['opener_arm_q']=float(math.atan2(dz,dy)-linkage['closed_arm_angle_rad'])
    return pose


def resolve_sectional_configuration(model, qpos, meta, progress=None):
    """Prescribe nominal inspection poses; caller forwards afterward.

    The source progress is explicit because root Z saturates while sections
    continue travelling overhead. Cable carriages are solved on a private
    MjData, leaving the live native state untouched apart from requested qpos.
    """
    mechanism=meta.get('sectional_track')
    if not mechanism:return qpos
    if progress is None:
        j=model.joint(mechanism['root_z_joint']).id
        dz=mechanism['path']['vertical_end_z_m']+mechanism['path']['radius_m']-mechanism['path']['closed_top_z_m']
        progress=float(qpos[model.jnt_qposadr[j]])/max(dz,1e-9)
    pose=inspection_pose(progress,mechanism)
    for name,value in [(mechanism['root_y_joint'],pose['root_y']),(mechanism['root_z_joint'],pose['root_z']),*zip(mechanism['panel_joints'],pose['panel_angles'])]:
        qpos[model.jnt_qposadr[model.joint(name).id]]=value
    if mechanism['drive'].get('linkage'):
        for key,name in [('trolley_q',mechanism['drive']['linkage']['trolley_joint']),('opener_arm_q',mechanism['drive']['linkage']['arm_joint'])]:
            qpos[model.jnt_qposadr[model.joint(name).id]]=pose[key]
    import mujoco
    scratch=mujoco.MjData(model);scratch.qpos[:]=qpos
    for cable in mechanism['counterbalance']['sides']:
        jid=model.joint(cable['spring_joint']).id;qa=model.jnt_qposadr[jid];tid=model.tendon(cable['cable']).id
        lo,hi=model.jnt_range[jid]
        for _ in range(36):
            mid=(lo+hi)/2;scratch.qpos[qa]=mid;mujoco.mj_forward(model,scratch)
            if scratch.ten_length[tid]<model.tendon_range[tid,1]:lo=mid
            else:hi=mid
        qpos[qa]=(lo+hi)/2;scratch.qpos[qa]=qpos[qa]
    return qpos


def _segment(name, a, b, radius, material, semantic='track', mass=None):
    a=np.asarray(a,float);b=np.asarray(b,float);delta=b-a
    return Geom(name,'capsule',(radius,float(np.linalg.norm(delta))/2),tuple((a+b)/2),
        tuple(quat_z_to(delta)),material,True,True,7850.,mass,(.08,.001,.0001),
        tiers=ALL_TIERS,semantic=semantic,part_label='Continuous roller-track flange' if semantic=='track' else 'Steel support member')


def _closed_cable_length(points):
    """Use the same native cylinder-wrap convention as the exported cable.

    This small mesh-free model only evaluates the closed route. It does not
    step, calibrate assistance, or prescribe subsequent section motion.
    """
    import mujoco
    root=ET.Element('mujoco');world=ET.SubElement(root,'worldbody');tendon=ET.SubElement(root,'tendon')
    route=ET.SubElement(tendon,'spatial',name='route')
    for i,point in enumerate(points):
        if len(point)==2:
            center,side=point
            ET.SubElement(world,'geom',name=f'g{i}',type='cylinder',pos=' '.join(map(str,center)),size='.05 .012',quat='.7071067811865476 0 .7071067811865475 0',contype='0',conaffinity='0')
            ET.SubElement(world,'site',name=f's{i}',pos=' '.join(map(str,side)))
            ET.SubElement(route,'geom',geom=f'g{i}',sidesite=f's{i}')
        else:
            ET.SubElement(world,'site',name=f'p{i}',pos=' '.join(map(str,point)))
            ET.SubElement(route,'site',site=f'p{i}')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return float(data.ten_length[0])


def _cable_eye_washer(body, name, center, material):
    """Retain a looped cable on the existing bottom axle, with a real bore."""
    import trimesh
    for segment in range(12):
        angles=[2*math.pi*segment/12,2*math.pi*(segment+1)/12]
        points=[(x,r*math.cos(a),r*math.sin(a))
                for x in (-.002,.002) for r in (.0042,.008) for a in angles]
        mesh=trimesh.convex.convex_hull(points)
        body.geoms.append(C.mesh_geom(f'{name}_{segment}',f'section_cable_washer_{segment}',mesh,
            center,QUAT_ID,material,7850,True,ALL_TIERS,'hinge',
            'Bored cable-eye retaining washer on the bottom roller axle'))


@lru_cache(maxsize=1)
def _roller_sectors():
    """A 50 mm roller with an 8.4 mm bore; convex pieces preserve the hole."""
    import trimesh
    result=[]
    for segment in range(24):
        angles=[2*math.pi*segment/24,2*math.pi*(segment+1)/24]
        points=[(x,r*math.cos(a),r*math.sin(a))
                for x in (-.012,.012) for r in (.0042,.025) for a in angles]
        result.append(trimesh.convex.convex_hull(points))
    return tuple(result)


def _add_bottom_astragal(model, panel, width, thickness, panel_height, face_offset):
    """Bottom U-seal captured in an aluminium twin-slot retainer.

    The existing bottom roller is 50 mm above the closed floor and the panel
    edge is 27 mm below that axle. This missing 23 mm weatherseal used to let
    the whole door fall onto a numerical slide limit. The hollow 1.5 mm EPDM
    extrusion now meets the floor after a 0.5 mm installation clearance.
    Its native contact is a rigid profile with solver compliance, not a
    finite-element rubber constitutive model or a certified weather rating.
    """
    import trimesh
    rubber=C.mat_from_material(model,'rubber','mat_sectional_astragal')
    aluminium=C.mat_from_material(model,'aluminum','mat_sectional_retainer')
    seal=Body('section_bottom_astragal',panel.name,(0,face_offset,-panel_height),QUAT_ID,
        semantic='seal',label='Attached bottom weatherseal and aluminium retainer')
    # All dimensions below are relative to the bottom roller centre.
    a=thickness*.4;b=.020;top=-.0295;wall=.0015
    seal.geoms.append(C.box('section_astragal_retainer_plate',(0,0,-.02775),
        (width/2,thickness/2,.00075),aluminium,2700,False,True,ALL_TIERS,'seal',
        '1.5 mm aluminium retainer fastened against the bottom panel edge'))
    for sign in (-1,1):
        yc=sign*a
        seal.geoms.append(C.box(f'section_astragal_t_bead_{sign}',(0,yc,top),
            (width/2,.0025,.0005),rubber,1150,False,True,ALL_TIERS,'seal','Captured T edge of EPDM extrusion'))
        for side in (-1,1):
            seal.geoms.append(C.box(f'section_astragal_channel_wall_{sign}_{side}',(0,yc+side*.003,-.0295),
                (width/2,.0005,.001),aluminium,2700,False,True,ALL_TIERS,'seal','Retainer slot wall'))
            seal.geoms.append(C.box(f'section_astragal_channel_lip_{sign}_{side}',(0,yc+side*.00175,-.031),
                (width/2,.00125,.0005),aluminium,2700,False,True,ALL_TIERS,'seal','Retainer lip around the seal stem'))
    for i in range(12):
        angles=(math.pi*i/12,math.pi*(i+1)/12)
        points=[(x,ay*math.cos(theta),top-bz*math.sin(theta))
                for x in (-width/2,width/2) for ay,bz in ((a,b),(a-wall,b-wall)) for theta in angles]
        mesh=trimesh.convex.convex_hull(points)
        geom=C.mesh_geom(f'section_astragal_u_{i}',f'section_astragal_{width:.6f}_{thickness:.6f}_{i}',
            mesh,(0,0,0),QUAT_ID,rubber,1150,True,ALL_TIERS,'seal',
            'Hollow 1.5 mm EPDM U-seal; faceted contact cross-section')
        geom.friction=(.9,.001,.0001)
        seal.geoms.append(geom)
    # Visible fastener shafts join the continuous retainer to the bottom skin.
    for i,x in enumerate(np.linspace(-width/2+.05,width/2-.05,max(3,math.ceil(width/.45)))):
        seal.geoms.append(C.cyl(f'section_astragal_fastener_{i}',(x,0,-.023),.0015,.0055,
            aluminium,(0,0,1),2700,False,True,ALL_TIERS,'seal','Retainer screw shank into the bottom panel'))
    model.add_body(seal)
    model.meta['sectional_bottom_seal']={'body':seal.name,'floor_geom':'floor',
        'contact_geoms':[f'section_astragal_u_{i}' for i in range(12)],
        'closed_floor_gap_m':.0005,'profile_wall_m':wall,'material':'EPDM rubber',
        'mass_kg':seal.inertial()[0],
        'scope':'Original twin-T U-astragal topology; rigid profile with native contact compliance, not measured rubber deformation',
        'reference':'https://cdn.clopay.com/public/Weather%20Seal%202010%20Instructions.pdf'}
    return seal


def build_sectional(spec, phys, model):
    leaf=spec['leaf'];width=leaf['width'];height=leaf['height'];thickness=leaf['thickness'];kin=spec['kinematics']
    path=track_dimensions(spec);y=path['offset_y_m'];turn=path['vertical_end_z_m'];radius=path['radius_m'];sh=path['panel_height_m'];count=path['panel_count']
    world=C.add_floor_and_wall(model,spec,wall_half_width=max(3.,width/2+1.),wall_height=turn+radius+.65)
    steel=C.mat_from_material(model,'steel_galvanized','mat_sectional_track');rubber=C.mat_from_material(model,'black_matte_metal','mat_sectional_wheel')
    end=turn+radius*math.pi/2+path['horizontal_end_y_m']-y-radius
    knots=np.r_[np.linspace(0,turn,16),turn+np.linspace(0,radius*math.pi/2,61)[1:],np.linspace(turn+radius*math.pi/2,end,16)[1:]]
    for sign,tag in ((-1,'l'),(1,'r')):
        x=sign*(width/2+.045)
        for edge in (-1,1):
            for k,(a,b) in enumerate(zip(knots[:-1],knots[1:])):
                pa,ta=track_path(a,path);pb,tb=track_path(b,path)
                pa+=edge*.031*np.array([ta[1],-ta[0]]);pb+=edge*.031*np.array([tb[1],-tb[0]])
                world.geoms.append(_segment(f'section_track_{tag}_{edge}_{k}',(x,*pa),(x,*pb),.003,steel))
        # Web behind the tyre and discrete brackets carrying the track into
        # jamb/rear posts. Channel cross-section is genuinely open inward.
        for k,(a,b) in enumerate(zip(knots[:-1],knots[1:])):
            pa,ta=track_path(a,path);pb,tb=track_path(b,path)
            if abs(pa[0]-pb[0])<1e-12 and sign<0 and spec['lock']['model'] in ('garage_slide_lock','keyed_cylinder'):
                # A side bolt passes through a real slot in the outer web.
                # The keeper sits beyond the tyre's axial width, so rollers
                # never have to pass through a keeper loop inside the track.
                _slotted_web(world,f'section_track_{tag}_web_{k}',x+sign*.018,y,pa[1],pb[1],steel)
                continue
            center=(x+sign*.018,*((pa+pb)/2));delta=np.r_[0.,pb-pa]
            world.geoms.append(C.obox(f'section_track_{tag}_web_{k}',center,delta,(sign,0,0),0,0,0,
                float(np.linalg.norm(delta))/2,.031,.003,steel,True,ALL_TIERS,'track','Full outer web of roller C channel'))
        for z in np.linspace(.25,turn-.15,4):
            world.geoms.append(C.box(f'section_jamb_bracket_{tag}_{z:.2f}',(sign*(width/2+.115),(y+spec['opening']['wall_thickness']/2)/2,z),
                (.052,(y-spec['opening']['wall_thickness']/2)/2,.006),steel,7850,semantic='frame',label='Track jamb anchor outside tyre width'))
        rear=path['horizontal_end_y_m']-.08
        world.geoms.append(C.box(f'section_rear_post_{tag}',(sign*(width/2+.16),rear,(turn+radius)/2),(.025,.025,(turn+radius)/2),steel,7850,semantic='frame',label='Rear track support post'))
        world.geoms.append(C.box(f'section_rear_bracket_{tag}',(sign*(width/2+.10),rear,turn+radius+.065),(.085,.025,.010),steel,7850,semantic='frame',label='Rear overhead track hanger'))
    cy=Body('section_carriage_y',None,(0,y,height+.05),QUAT_ID,semantic='mechanism',label='Planar top axle rearward coordinate')
    cy.joint=Joint('door_rear_slide','slide',(0,1,0),range=(-.02,height+1.),damping=.15,role='mechanism',robot_interactive=False)
    # An intermediate coordinate needs positive native inertia. This explicit
    # 1 g numerical reserve is charged to existing hardware, not advertised as
    # a separate BOM part or a physical bearing which fails to move vertically.
    cy.extra_mass=.001;model.add_body(cy)
    cz=Body('section_carriage_z',cy.name,semantic='mechanism',label='Planar top axle vertical coordinate')
    cz.joint=Joint('door_slide','slide',(0,0,1),range=(-.02,turn+radius-height-.05+.02),damping=.15,role='primary',robot_interactive=False,
        label='Top roller height; use sectional track progress for complete opening')
    cz.geoms.append(C.cyl('section_top_axle',(0,0,0),.004,width/2+.045,steel,(1,0,0),7850,
        False,True,ALL_TIERS,'hinge','Top axle connecting panel bearings'))
    model.add_body(cz)
    mechanism_masses=[cz.name];panels=[];rollers=[];parent=cz.name
    face_offset=-(thickness/2+.005)
    for i in range(count):
        panel=Body(f'section_{i}',parent,(0,0,0 if i==0 else -sh),QUAT_ID,semantic='leaf',label=f'Articulated garage section {i+1} from top')
        panel.joint=Joint(f'section_{i}_hinge','hinge',(-1,0,0),range=(-.10,1.65) if i==0 else (-1.65,.10),damping=.20,role='mechanism',robot_interactive=False)
        model.add_body(panel);panels.append(panel);parent=panel.name
        sub={**leaf,'height':sh-.006}
        row=next(p for p in phys['mass']['per_body'] if p['body']==panel.name)
        C.add_leaf_geoms(model,panel,spec,sub,1.,-width/2,-sh-.027,{'mass':{'slab_kg':row['slab_kg']}},name_prefix=panel.name,y_center=face_offset)
        site=f'roller_center_{i}';panel.sites.append(Site(site,(0,0,0),role='mechanism'));rollers.append(site)
        if i==0:panel.sites.append(Site('opener_attachment',(0,0,0),role='powered_drive'))
        ends=[('head',0)] + ([('foot',-sh)] if i==count-1 else [])
        for endtag,z in ends:
            for sign,tag in ((-1,'l'),(1,'r')):
                x=sign*(width/2+.045)
                wheel=Body(f'section_wheel_{i}_{endtag}_{tag}',panel.name,(x,0,z),QUAT_ID,semantic='mechanism',label='Bearing-mounted track roller')
                wheel.joint=Joint(wheel.name+'_spin','hinge',(1,0,0),range=None,damping=.0001,role='mechanism',robot_interactive=False)
                for segment,mesh in enumerate(_roller_sectors()):
                    wheel.geoms.append(C.mesh_geom(f'{wheel.name}_tread_{segment}',f'section_bored_roller_{segment}',
                        mesh,(0,0,0),QUAT_ID,rubber,1100,True,ALL_TIERS,'track',
                        '50 mm track roller with an actual axle bore; 24-sector contact approximation',mass=.08/24))
                model.add_body(wheel);mechanism_masses.append(wheel.name)
                panel.geoms.append(C.cyl(wheel.name+'_axle',(sign*(width/2+.01),0,z),.004,.045,steel,(1,0,0),7850,False,True,ALL_TIERS,'hinge','Roller axle fixed to panel bracket'))
                bracket_z=z+(.035 if endtag=='foot' else -.035)
                panel.geoms.append(C.box(wheel.name+'_bracket',(sign*(width/2-.035),face_offset+thickness/2+.004,bracket_z),(.030,.005,.040),steel,7850,False,True,ALL_TIERS,'hinge','Panel-mounted roller bracket, extending into its section'))
        if i>0:
            # Actual hinge axis is the roller line, behind both panel faces.
            for x in (-width/2+.12,0.,width/2-.12):
                panel.geoms.append(C.cyl(f'section_hinge_{i}_{x:.2f}',(x,0,0),.006,.025,steel,(1,0,0),7850,False,True,ALL_TIERS,'hinge','Inter-section hinge pin'))
    bottom=panels[-1];bottom.sites.extend([Site(f'roller_center_{count}',(0,0,-sh),role='mechanism'),Site('bottom_roller_mid',(0,0,-sh),role='mechanism')]);rollers.append(f'roller_center_{count}')
    astragal=_add_bottom_astragal(model,bottom,width,thickness,sh,face_offset)
    mechanism_masses.append(astragal.name)
    # A low lift grip is the real manual contact. An exterior T handle may
    # additionally operate locks; the controller never pulls a top roller.
    lift=Body('section_lift_handle',bottom.name,(0,face_offset,0),QUAT_ID,semantic='operator',label='Bottom lift handle')
    C.add_pull(model,lift,H.OPERATORS['pull_lift_garage'],1.,0.,-sh+.13,thickness,-1.,name='lift_handle')
    lift.sites.append(Site('lift_handle_grip',lift.sites[-1].pos,role='grip'));model.add_body(lift)
    if H.OPERATORS[spec['operator']['model']].kind!='pull':
        mechanism_masses.append(lift.name)  # additional physical lift grip beside the catalogue T handle/motor
    opener=kin.get('opener','none_manual');engaged=opener.endswith('_engaged')
    cb=float(kin.get('counterbalance_fraction',0.));sides=[]
    for sign,tag in ((-1,'l'),(1,'r')):
        # Standard-lift cables run inboard of the roller channel. An outboard
        # plane requires a different bottom bracket; extending this axle
        # through the uncut outer track web is not such a bracket.
        x=sign*(width/2+.020);zc=turn+radius+.12;start_y=y+.82;rear_y=start_y+max(3.,height+1.)
        fixed_center=(x,y+.05,zc);fixed_side=(x,y-.01,zc+.06);via=(x,y+.27,zc+.05);anchor=(x,y+.27,zc+.15)
        fixed_name=f'section_fixed_pulley_{tag}'
        world.geoms.append(C.cyl(fixed_name,fixed_center,.05,.012,steel,(1,0,0),7850,False,True,ALL_TIERS,'mechanism','Fixed cable sheave'))
        world.sites.extend([Site(fixed_name+'_side',fixed_side),Site(f'section_cable_via_{tag}',via),Site(f'section_cable_anchor_{tag}',anchor),Site(f'section_spring_anchor_{tag}',(x,rear_y,zc+.10))])
        # Fixed sheave bracket and continuous side beam make the load path
        # inspectable; wrap geoms are excluded from solid cable collision by
        # the tendon formulation itself, not an interpenetrating mesh proxy.
        world.geoms.append(C.box(f'section_spring_beam_{tag}',(x,(start_y+rear_y)/2,zc+.19),(.020,(rear_y-start_y)/2+.10,.015),steel,7850,semantic='frame',label='Spring track and rear anchor beam'))
        world.geoms.append(C.box(fixed_name+'_mount',(x,y+.05,zc+.08),(.020,.015,.035),steel,7850,semantic='frame',label='Sheave bracket on header'))
        world.geoms.append(C.box(f'section_spring_rear_post_{tag}',(x,rear_y,(zc+.19)/2),(.020,.020,(zc+.19)/2),steel,7850,semantic='frame',label='Rear spring-anchor support'))
        carrier=Body(f'section_spring_carriage_{tag}',None,(x,start_y,zc+.10),QUAT_ID,semantic='mechanism',label='Travelling 2:1 spring pulley')
        carrier.joint=Joint(f'section_spring_slide_{tag}','slide',(0,1,0),range=(-.01,height*.75+.2),damping=1.,role='mechanism',robot_interactive=False)
        moving_name=f'section_moving_pulley_{tag}'
        carrier.geoms.append(C.cyl(moving_name,(0,0,0),.05,.012,steel,(1,0,0),7850,False,True,ALL_TIERS,'mechanism','Spring pulley',mass=.20))
        carrier.sites.extend([Site(moving_name+'_side',(0,.06,0)),Site(f'section_spring_tip_{tag}',(0,0,0))])
        model.add_body(carrier);mechanism_masses.append(carrier.name)
        bottom_name=f'section_cable_bottom_{tag}';bottom.sites.append(Site(bottom_name,(x,0,-sh),role='mechanism'))
        # The existing bottom axle already reaches from its panel bracket to
        # the roller. A looped cable fits between that bracket and the tyre;
        # its bored retaining washer ends 5 mm before the tyre's inner face.
        # Do not duplicate the axle with the old 100 mm noncolliding rod,
        # which passed straight through the channel's outer web.
        _cable_eye_washer(bottom,bottom_name+'_anchor',(sign*(width/2+.026),0,-sh),steel)
        length=_closed_cable_length([(x,y,.05),(fixed_center,fixed_side),via,((x,start_y,zc+.10),(x,start_y+.06,zc+.10)),anchor])
        cable_name=f'section_lift_cable_{tag}'
        model.spatial_cables.append(SpatialCable(cable_name,({'site':bottom_name},{'geom':fixed_name,'sidesite':fixed_name+'_side'},{'site':f'section_cable_via_{tag}'},{'geom':moving_name,'sidesite':moving_name+'_side'},{'site':f'section_cable_anchor_{tag}'}),length,label='Tension-only lift cable with fixed and moving sheaves'))
        # Each 2:1 spring pulls at twice its cable tension; two sides together
        # supply cb*m*g in the closed vertical run. Preserve failed springs.
        mass=float(phys['mass']['total_kg'])+cz.inertial()[0]+.16*(count+1)+.40+astragal.inertial()[0]+(lift.inertial()[0] if lift.name in mechanism_masses else 0.)
        preload=cb*mass*9.81;k=preload/(.64*height) if cb else 0.
        rest=rear_y-start_y-.64*height
        if cb:model.spatial_springs.append(SpatialSpring(f'section_extension_{tag}',(f'section_spring_tip_{tag}',f'section_spring_anchor_{tag}'),k,rest,damping=3.,width=.006,label='Physical extension counterbalance spring'))
        sides.append({'cable':cable_name,'spring_joint':carrier.joint.name,'spring':f'section_extension_{tag}' if cb else None,'closed_spring_force_N':preload,'spring_stiffness_N_m':k,'rest_length_m':rest,
            'bottom_site':bottom_name,'bottom_axle':f'section_wheel_{count-1}_foot_{tag}_axle',
            'bottom_washer_prefix':bottom_name+'_anchor','cable_plane_x_m':x,
            'mount_scope':'Standard-lift cable loop retained on existing inboard bottom roller axle; no outboard web penetration'})
    drive={'mode':'powered' if engaged else 'manual','opener':opener,'attachment_site':'opener_attachment','max_force_N':float(kin.get('actuator',{}).get('max_force_N',600.)),
        'manual_max_force_N':120.,'activation':'wall_button' if engaged else None,'force_axis_world':[0.,1.,0.],
        'force_scope':'Original capped motor acting on rail trolley; not an OEM rating'}
    if opener!='none_manual':
        drive['linkage']=_add_opener(model,world,panels[0],path,steel,rubber,engaged)
        mechanism_masses.extend(drive['linkage']['mechanism_mass_bodies'])
        if engaged:
            drive['actuator']='sectional_opener_motor'
            model.meta.setdefault('actuators',[]).append({'name':drive['actuator'],'kind':'motor',
                'joint':'opener_trolley_slide','gear':1.,'ctrlrange':[-drive['max_force_N'],drive['max_force_N']],
                'role':'sectional_drive'})
    _add_operator_and_lock(model,spec,phys,panels,world,path,face_offset)
    world.sites.extend([Site('approach_point',(0,-2.,0),role='approach'),Site('goal_point',(0,2.,0),role='goal'),Site('door_plane_center',(0,0,spec['opening']['height']/2),role='pass_plane')])
    model.meta.setdefault('mechanism_mass_bodies',[]).extend(mechanism_masses)
    model.meta.update({'primary_joint':'door_slide','operator_joint':model.meta.get('operator_joint'),'handle_height':.18,'counterbalance_fraction':cb,
        'sectional_track':{'schema_version':1,'kind':'articulated_roller_track','root_y_joint':'door_rear_slide','root_z_joint':'door_slide','panel_joints':[p.joint.name for p in panels],
          'roller_sites':rollers,'manual_grip_site':'lift_handle_grip','powered_drive_site':'opener_trolley_drive' if engaged else None,'path':path,
          'progress':{'site':'bottom_roller_mid','closed_s_m':.05,'open_s_m':turn+radius*math.pi/2+.10},
          'drive':drive,
          'counterbalance':{'fraction':cb,'state':kin.get('counterbalance_state'),'mass_basis_kg':mass,'sides':sides,'scope':'Native extension springs and routed 2:1 tension-only lift cables; generic force curve, no success guarantee'},
          'references':['https://www.clopaydoor.com/residential/support','https://www.clopaydoor.com/commercial-support-center/technical-library-commercial-builder/tracks-springs-commercial']},
        'mechanical_export_support':{'mjcf':'Native articulated panels, roller contacts and routed tension-only cables','urdf':'Requires roller/contact and routed cable simulation plugin','usd':'unsupported routed cable dynamics; static interchange only'}})
    return panels[0]


def _add_opener(model,world,panel,path,steel,case_material,engaged):
    """Trolley on a supported rail and a real pin-ended panel connection.

    A disengaged opener leaves the arm attached to the freely sliding trolley;
    the drive clutch supplies no force. The belt/chain transmission's internal
    motor/gear dynamics are outside this original bounded-force model.
    """
    from .garage_tiltup import _bearing_eye
    y=path['offset_y_m'];height=path['closed_top_z_m'];z=path['vertical_end_z_m']+path['radius_m']+.25
    anchor=np.array([.060,-.050]);length=.90;dz=height+anchor[1]-z
    dy=-math.sqrt(length*length-dz*dz);closed_y=y+anchor[0]-dy
    motor_y=path['horizontal_end_y_m']+1.10
    for side in (-1,1):
        world.geoms.append(C.box(f'section_opener_rail_side_{side}',(side*.037,(y+motor_y)/2,z+.025),(.004,(motor_y-y)/2,.033),steel,7850,semantic='track',label='Opener trolley rail side'))
        world.geoms.append(C.box(f'section_opener_rail_flange_{side}',(side*.032,(y+motor_y)/2,z+.024),(.009,(motor_y-y)/2,.004),steel,7850,semantic='track',label='Opener trolley rail lower flange'))
    world.geoms.append(C.box('section_opener_rail_web',(0,(y+motor_y)/2,z+.062),(.041,(motor_y-y)/2,.004),steel,7850,semantic='track',label='Opener rail upper web'))
    world.geoms.append(C.box('section_opener_unit',(0,motor_y,z+.02),(.14,.18,.09),case_material,500,semantic='mechanism',label='Garage door motor and transmission housing'))
    # Rear support spans the two side structure posts; no floating ceiling mount.
    span=path.get('opening_width_m',2.5)/2+.20
    world.geoms.append(C.box('section_opener_rear_crossbeam',(0,motor_y,z+.15),(span,.025,.025),steel,7850,semantic='frame',label='Motor support crossbeam'))
    for side in (-1,1):world.geoms.append(C.box(f'section_opener_support_post_{side}',(side*span,motor_y,(z+.15)/2),(.025,.025,(z+.15)/2),steel,7850,semantic='frame',label='Motor support post'))
    world.geoms.append(C.box('section_opener_front_mount',(0,y-.06,z+.01),(.085,.060,.070),steel,7850,semantic='frame',label='Header rail mounting plate'))
    trolley=Body('opener_trolley',None,(0,closed_y,z),QUAT_ID,semantic='mechanism',label='Rail-captured opener trolley')
    trolley.joint=Joint('opener_trolley_slide','slide',(0,1,0),range=(-.02,motor_y-closed_y-.25),damping=2.,frictionloss=1.,role='mechanism',robot_interactive=False)
    trolley.geoms.append(C.box('opener_trolley_bearing',(0,0,.043),(.029,.060,.012),steel,7850,True,True,ALL_TIERS,'mechanism','Trolley bearing block inside channel',mass=.45))
    for side in (-1,1):trolley.geoms.append(C.box(f'opener_trolley_fork_{side}',(side*.015,0,.014),(.004,.012,.026),steel,7850,True,True,ALL_TIERS,'hinge','Open fork around arm eye'))
    trolley.sites.append(Site('opener_trolley_drive',(0,0,.035),role='powered_drive'))
    trolley.geoms.append(C.cyl('opener_trolley_pin',(0,0,0),.005,.026,steel,(1,0,0),7850,True,True,ALL_TIERS,'hinge','Trolley connecting-arm pin'))
    model.add_body(trolley)
    arm=Body('opener_connecting_arm',trolley.name,semantic='mechanism',label='Pin-ended opener connecting arm')
    arm.joint=Joint('opener_connecting_arm_hinge','hinge',(1,0,0),range=(-1.0,1.0),damping=.15,role='mechanism',robot_interactive=False)
    vector=np.array([0.,dy,dz]);arm.geoms.append(C.cyl('opener_connecting_arm_rod',tuple(vector/2),.007,(length-.048)/2,steel,vector,7850,True,True,ALL_TIERS,'mechanism','Steel connecting-arm rod'))
    _bearing_eye(arm,'opener_arm_root_eye',(0,0,0),steel);_bearing_eye(arm,'opener_arm_tip_eye',tuple(vector),steel)
    model.add_body(arm)
    panel.geoms.append(C.box('opener_panel_bracket',(0,.0075,anchor[1]),(.032,.0075,.025),steel,7850,True,True,ALL_TIERS,'hinge','Rear-face panel opener bracket'))
    for side in (-1,1):panel.geoms.append(C.box(f'opener_panel_fork_{side}',(side*.020,.0375,anchor[1]),(.005,.0225,.010),steel,7850,True,True,ALL_TIERS,'hinge','Open fork around connecting-arm eye'))
    panel.geoms.append(C.cyl('opener_panel_pin',(0,*anchor),.005,.026,steel,(1,0,0),7850,True,True,ALL_TIERS,'hinge','Panel connecting-arm pin'))
    model.equalities.append(Equality('connect','opener_arm_connection',arm.name,panel.name,anchor=tuple(vector),tiers=ALL_TIERS,
        solref=(.002,1),solimp=(.99,.999,.001),label='Connecting-arm eye pinned to top panel'))
    return {'trolley_joint':trolley.joint.name,'arm_joint':arm.joint.name,'panel_anchor_yz':anchor.tolist(),'arm_length_m':length,
        'closed_trolley_y_m':closed_y,'trolley_z_m':z,'closed_arm_angle_rad':math.atan2(dz,dy),
        'mechanism_mass_bodies':[trolley.name,arm.name],'clutch_engaged':engaged,
        'scope':'Rail-supported trolley transmits bounded drive force through native pinned connecting arm; internal transmission efficiency is not modeled'}


def _slotted_web(world,name,x,y,z0,z1,steel):
    low=max(z0,.641);high=min(z1,.659)
    if low>=high:
        world.geoms.append(C.box(name,(x,y,(z0+z1)/2),(.003,.031,(z1-z0)/2),steel,7850,semantic='track',label='Outer C-channel web'))
        return
    for tag,a,b in [('below',z0,low),('above',high,z1)]:
        if b-a>1e-9:world.geoms.append(C.box(name+'_'+tag,(x,y,(a+b)/2),(.003,.031,(b-a)/2),steel,7850,semantic='track',label='Web below/above side-bolt slot'))
    for tag,a,b in [('front',-.031,.002),('rear',.018,.031)]:
        world.geoms.append(C.box(name+'_'+tag,(x,y+(a+b)/2,(low+high)/2),(.003,(b-a)/2,(high-low)/2),steel,7850,semantic='track',label='Side-bolt slot edge'))


def _add_operator_and_lock(model,spec,phys,panels,world,path,face_offset):
    """Panel-local fittings retain their actual side and mounting height."""
    sh=path['panel_height_m'];height=spec['leaf']['height'];count=len(panels)
    def owner(z):
        index=int(np.clip(math.floor((height+.05-z)/sh),0,count-1))
        return panels[index],height+.05-index*sh
    op=H.OPERATORS[spec['operator']['model']]
    if op.kind=='t_handle':
        hz=min(spec['operator']['height'],height-.10);panel,top=owner(hz)
        material=C.mat_from_material(model,op.material,f'mat_op_{op.material}')
        hb=model.add_body(Body('t_handle',panel.name,(0,face_offset,hz-top),QUAT_ID,
            semantic='operator',label='Fixed T-shaped lifting pull'))
        face=-spec['leaf']['thickness']/2;standoff=.035
        hb.geoms.append(C.cyl('t_handle_mount_n',(0,face-.004,0),.025,.004,material,(0,1,0),7850,
            True,True,ALL_TIERS,'operator','Fixed pull mounting foot on leaf surface'))
        hb.geoms.append(C.cyl('t_handle_stem_n',(0,face-(.008+standoff)/2,0),.007,(standoff-.008)/2,material,(0,1,0),7850,
            True,True,ALL_TIERS,'operator','Fixed T-pull stem'))
        hb.geoms.append(C.cyl('t_handle_bar_n',(0,face-standoff,0),.008,.055,material,(1,0,0),7850,
            True,True,ALL_TIERS,'operator','110 mm T grip with 27 mm finger clearance'))
        hb.sites.append(Site('t_handle_grip_n',(-.0385,face-standoff-.008,0),
            tuple(quat_z_to((0,-1,0))),.01,'grip'))
        # These variants specify either no lock or a separate side bolt/hasp.
        # Their T shape is a fixed lifting grip, not a functionless rotary
        # action or an invented linkage to an unrelated rear lock.
        model.meta['operator_joint']=None
        model.meta['sectional_operator']={'kind':'fixed_t_pull','turn_required':False,'separate_lock_joint':None,
            'bar_standoff_m':standoff,'finger_clearance_m':standoff-.008,
            'scope':'Original fixed pull with no cosmetic key cylinder; source mounting height retained'}
    kind=spec['lock']['model']
    if kind not in ('garage_slide_lock','padlock','keyed_cylinder'):return
    from .garage_locks import add_tiltup_lock
    z=.18 if kind=='padlock' else .65;panel,top=owner(z)
    body_start=len(model.bodies);world_start=len(world.geoms);geom_start=len(panel.geoms)
    if kind=='padlock':
        add_tiltup_lock(model,panel,world,spec,top)
    else:
        steel=C.mat_from_material(model,'steel_galvanized','mat_garage_lock');width=spec['leaf']['width'];surface=spec['leaf']['thickness']/2
        bolt,_=C.add_barrel_bolt(model,panel,'garage_slide_lock',(-width/2,surface,z-top),(-1,0,0),(0,1,0),.23,.012,.125,
            bool(spec['lock'].get('engaged')),steel,protrusion=.110,standoff=.015,tiers=ALL_TIERS,role='lock',rod_semantic='lock',
            joint_name='garage_slide_lock_slide',grip_site='slide_lock_grip')
        C.add_keeper_loop(world.geoms,'garage_lock_keeper',(-width/2-.090,surface,z),(-width/2-.090,surface+.015,z),
            (-1,0,0),(0,1,0),.006,steel,ALL_TIERS,base=.030,bar=.005,bar_len=.014)
        world.geoms.append(C.box('garage_lock_jamb_bracket',(-width/2-.18,surface-.007,z),(.10,.007,.040),steel,7850,semantic='lock',label='Side-bolt keeper bracket beyond roller width'))
        if spec['lock'].get('engaged') and not spec['lock'].get('robot_side_release'):bolt.joint.range=(0.,.001)
        model.meta['garage_lock_hardware']={'kind':'rear_slide_bolt','joint':bolt.joint.name,'grip_site':'slide_lock_grip','engaged_q':0.,'released_q':.125,
            'keeper_prefix':'garage_lock_keeper','side':'garage_interior','track_slot_dimensions_m':[.016,.018]}
    if model.meta.get('sectional_operator'):
        model.meta['sectional_operator']['separate_lock_joint']=model.meta['garage_lock_hardware']['joint']
    offset=face_offset+(.080 if kind=='padlock' else 0.)
    for geom in panel.geoms[geom_start:]:geom.pos=(geom.pos[0],geom.pos[1]+offset,geom.pos[2])
    for body in model.bodies[body_start:]:
        if body.parent==panel.name:body.pos=(body.pos[0],body.pos[1]+offset,body.pos[2])
    for geom in world.geoms[world_start:]:geom.pos=(geom.pos[0],geom.pos[1]+path['offset_y_m']+offset,geom.pos[2])
    steel=C.mat_from_material(model,'steel_galvanized','mat_garage_lock');width=spec['leaf']['width'];back=spec['opening']['wall_thickness']/2
    mount_y=path['offset_y_m']+offset+spec['leaf']['thickness']/2-.007
    sx=1 if kind=='padlock' else -1
    world.geoms.append(C.box('section_lock_wall_standoff',(sx*(width/2+.20),(back+mount_y)/2,z),(.020,(mount_y-back)/2,.030),steel,7850,semantic='frame',label='Keeper load path to masonry outside roller envelope'))
    if kind=='padlock':
        panel.geoms.append(C.box('section_hasp_standoff',(width/2-.14,face_offset+spec['leaf']['thickness']/2+.040,z-top),(.040,.040,.045),steel,7850,True,True,ALL_TIERS,'lock','Rear hasp standoff clear of roller path'))
    if kind=='keyed_cylinder':
        model.meta['garage_lock_hardware']['kind']='keyed_slide_bolt'
        model.meta['garage_lock_hardware']['key_operation']='Cylinder credential is an external task input; pin tumblers are not simulated'
        material=C.mat_from_material(model,'brass','mat_sectional_key_cylinder')
        panel.geoms.append(C.cyl('section_key_cylinder',(-spec['leaf']['width']/2+.08,face_offset-spec['leaf']['thickness']/2-.008,z-top),.015,.010,material,(0,1,0),8500,True,True,ALL_TIERS,'lock','Key cylinder controlling side bolt'))
    if spec['lock'].get('engaged') and not spec['lock'].get('robot_side_release'):
        model.meta['locked']=True
