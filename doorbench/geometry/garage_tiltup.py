"""Original retractable up-and-over linkage: top rollers and two lifting arms.

Topology follows Garador's retractable installation drawing, not a vendor CAD
copy. Dimensions are explicit DoorBench engineering choices. A horizontal
roller carriage plus a hinged rigid panel and two fixed-length side arms form
one closed-loop DOF. Native connect constraints carry the load; the analytical
resolver is only for prescribed-pose inspection, never a simulation update.
"""
from __future__ import annotations
import math
import numpy as np
from ..ir import Body, Joint, Site, Equality, SpatialSpring, ALL_TIERS, QUAT_ID
from .. import hardware as H
from . import common as C


def _bearing_eye(body, name, center, material, inner=.0065, outer=.024, half_length=.008):
    """Convex wedge ring: an actual open bore in both visual and collision meshes."""
    import trimesh
    for segment in range(12):
        angles = [2*math.pi*segment/12, 2*math.pi*(segment+1)/12]
        points = [(x,r*math.cos(a),r*math.sin(a))
                  for x in (-half_length,half_length) for r in (inner,outer) for a in angles]
        mesh = trimesh.convex.convex_hull(points)
        key=f'tilt_bearing_eye_{round(inner*1e6)}_{round(outer*1e6)}_{round(half_length*1e6)}_{segment}'
        body.geoms.append(C.mesh_geom(f'{name}_{segment}',key,mesh,
            center,QUAT_ID,material,7850,True,ALL_TIERS,'hinge','Open-bore lifting-arm eye'))


def linkage_dimensions(spec):
    height = float(spec['leaf']['height'])
    top = height - .01
    attach_height = .15 * height
    pivot_height = .60 * height
    # A 150 mm rearward door bracket gives the closed lifting arm appreciable
    # offset from a vertical toggle. Its complete branch is tested under
    # native spring/gravity loads, not only prescribed configurations.
    offset = .150
    return {'top_height_m': top, 'attach_below_top_m': top - attach_height,
            'attach_rear_offset_m': offset, 'pivot_height_m': pivot_height,
            'arm_length_m': math.hypot(offset, attach_height - pivot_height)}


def linkage_pose(angle, dimensions):
    """Return the positive-rearward assembly branch, in metres/radians."""
    zt, length, offset, zp, radius = (dimensions[k] for k in
        ('top_height_m','attach_below_top_m','attach_rear_offset_m','pivot_height_m','arm_length_m'))
    dz = zt - offset * math.sin(angle) - length * math.cos(angle) - zp
    radicand = radius * radius - dz * dz
    if radicand < -1e-10:
        raise ValueError('Tilt-up angle is beyond the lifting-arm reach')
    y = math.sqrt(max(0., radicand))
    slide = y - offset * math.cos(angle) + length * math.sin(angle)
    initial = math.atan2(zt - length - zp, offset)
    arm = math.atan2(dz, y) - initial
    return {'carriage_m': slide, 'arm_rad': arm, 'attachment_y_m': y,
            'attachment_z_m': zp + dz}


def resolve_garage_configuration(model, qpos, metadata):
    """In-place exact configuration for exported inspection/replay helpers.

    Retains the primary hinge and every unrelated joint. Never invoke during
    mj_step: the native point constraints, not this resolver, govern dynamics.
    """
    mechanism = metadata.get('garage_tiltup_linkage')
    if not mechanism:
        return qpos
    primary = model.joint(mechanism['primary_joint']).id
    pose = linkage_pose(float(qpos[model.jnt_qposadr[primary]]), mechanism['dimensions'])
    for name, value in [(mechanism['carriage_joint'], pose['carriage_m'])] + [
            (name, pose['arm_rad']) for name in mechanism['arm_joints']]:
        qpos[model.jnt_qposadr[model.joint(name).id]] = value
    return qpos


def projected_static_resistance(model, data, metadata):
    """Project native passive loads onto one radian of primary opening.

    The caller must have forwarded the measured native configuration. No data
    or model state is changed, and actuator/applied forces are excluded. The
    exact assembly-branch tangent includes carriage translation and BOTH arm
    rotations, so spatial springs are counted through native qfrc_passive.
    Coulomb friction is separate; spring preload must not be added again.
    """
    mechanism = metadata.get('garage_tiltup_linkage')
    if not mechanism:
        return None
    primary = model.joint(mechanism['primary_joint']).id
    angle = float(data.qpos[model.jnt_qposadr[primary]])
    dims = mechanism['dimensions']
    pose = linkage_pose(angle,dims)
    dz = pose['attachment_z_m'] - dims['pivot_height_m']
    dz_prime = (-dims['attach_rear_offset_m']*math.cos(angle)
                + dims['attach_below_top_m']*math.sin(angle))
    y = pose['attachment_y_m']
    if y <= 1e-9:
        raise ValueError('Tilt-up assembly tangent is singular')
    arm_prime = dz_prime/y
    carriage_prime = (-dz*dz_prime/y + dims['attach_rear_offset_m']*math.sin(angle)
                      + dims['attach_below_top_m']*math.cos(angle))
    derivatives = {mechanism['primary_joint']:1., mechanism['carriage_joint']:carriage_prime,
                   **{name:arm_prime for name in mechanism['arm_joints']}}
    tangent = np.zeros(model.nv)
    for name, value in derivatives.items():
        tangent[model.jnt_dofadr[model.joint(name).id]] = value
    return {'static_resistance':float(tangent @ (data.qfrc_bias-data.qfrc_passive)),
            'frictionloss':float(np.abs(tangent) @ model.dof_frictionloss),
            'joint_derivatives':derivatives, 'tangent_dof':tangent.tolist(),
            'primary_q_rad':angle,
            'scope':'Native passive load and friction projected onto dq/dprimary; no actuator or applied force.'}


def build_tiltup(spec, phys, model):
    leaf, opening, kin = spec['leaf'], spec['opening'], spec['kinematics']
    width, height, thickness = (float(leaf[k]) for k in ('width','height','thickness'))
    wo, ho = opening['width'], opening['height']
    dims = linkage_dimensions(spec); zt = dims['top_height_m']; zp = dims['pivot_height_m']
    length, rear = dims['attach_below_top_m'], dims['attach_rear_offset_m']
    maximum = math.radians(float(kin.get('max_open_deg',88)))
    final = linkage_pose(maximum,dims)
    sweep = [linkage_pose(q,dims) for q in np.linspace(0,maximum,1001)]
    travel = max(p['carriage_m'] for p in sweep)
    arm_minimum = min(p['arm_rad'] for p in sweep)
    # Mechanism pockets lie outside the nominal passage, behind the front
    # reveal. The visible front trim closes the pockets beside the shut panel.
    rough = .18
    world = C.add_floor_and_wall(model,spec,wall_half_width=max(3.,width/2+1.),
        wall_height=ho+1.2,hole=(-wo/2-rough,wo/2+rough,0.,ho))
    frame = C.mat_from_material(model,opening['frame']['material'],'mat_frame')
    steel = C.mat_from_material(model,'steel_galvanized','mat_track')
    roller_mat = C.mat_from_material(model,'black_matte_metal','mat_roller')
    for sign, tag in [(-1,'l'),(1,'r')]:
        wall_edge = wo/2+rough
        world.geoms.append(C.box('jamb_'+tag,(sign*(wall_edge-.03),0,ho/2),
            (.03,opening['wall_thickness']/2,ho/2),frame,400,semantic='frame',label='Garage side jamb'))
        world.geoms.append(C.box('front_reveal_'+tag,(sign*(width/2+.10),-.12,ho/2),
            (.095,.018,ho/2),frame,400,semantic='frame',label='Front reveal covering mechanism pocket'))
    carriage = Body('door_carriage',None,(0,0,zt),QUAT_ID,None,[],[],ALL_TIERS,'mechanism','Top roller carriage')
    carriage.joint = Joint('door_carriage_slide','slide',(0,1,0),(0,0,0),(0.,travel+.01),
        damping=2.,frictionloss=1.,armature=.1,role='mechanism',robot_interactive=False,
        label='Top rollers travel rearward on horizontal tracks')
    # The two roller-bearing eyes need a real rigid connection. A rearward
    # offset RHS crossmember clears both the wall header and the rotating
    # panel; its stock mass remains in the carriage's native inertia.
    beam_y=opening['wall_thickness']/2+.08
    beam_z=.060
    half_span=width/2+.050
    for sign in (-1,1):
        carriage.geoms.append(C.box(f'tilt_carriage_tube_side_{sign}',(0,beam_y+sign*.014,beam_z),
            (half_span,.001,.020),steel,7850,True,True,ALL_TIERS,'mechanism','2 mm RHS crossmember side wall'))
        carriage.geoms.append(C.box(f'tilt_carriage_tube_cap_{sign}',(0,beam_y,beam_z+sign*.019),
            (half_span,.013,.001),steel,7850,True,True,ALL_TIERS,'mechanism','2 mm RHS crossmember wall'))
        x=sign*(width/2+.042)
        carriage.geoms.append(C.box(f'tilt_carriage_offset_{sign}',(x,(.008+beam_y)/2,0),
            (.008,(beam_y-.008)/2,.005),steel,7850,True,True,ALL_TIERS,'mechanism','Bearing-eye rearward support strap'))
        carriage.geoms.append(C.box(f'tilt_carriage_elbow_{sign}',(sign*(width/2+.031),beam_y,0),
            (.019,.006,.005),steel,7850,True,True,ALL_TIERS,'mechanism','Inboard offset beneath track flange'))
        carriage.geoms.append(C.box(f'tilt_carriage_upright_{sign}',(sign*(width/2+.020),beam_y,.025),
            (.008,.006,.030),steel,7850,True,True,ALL_TIERS,'mechanism','Offset support welded into crossmember'))
    model.add_body(carriage)
    panel = Body('door','door_carriage',(0,0,0),QUAT_ID,None,[],[],ALL_TIERS,'leaf','Rigid up-and-over panel')
    panel.joint = Joint('door_hinge','hinge',(-1,0,0),(0,0,0),(0.,maximum),
        damping=6.+.03*phys['mass']['total_kg'],frictionloss=phys['hinge']['coulomb_torque_Nm']+1.,
        armature=.5,role='primary',label='Up-and-over rotation about the travelling top rollers')
    model.add_body(panel)
    C.add_leaf_geoms(model,panel,spec,leaf,1.,-width/2,.01-zt,phys,name_prefix='door')
    spring_fraction=float(kin.get('counterbalance_fraction',0.))
    spring_anchor_height=.35*height
    spring_arm_fraction=.4
    def spring_length(angle):
        phi=linkage_pose(angle,dims)['arm_rad']
        y=spring_arm_fraction*(rear*math.cos(phi)-(zt-length-zp)*math.sin(phi))
        z=spring_arm_fraction*(rear*math.sin(phi)+(zt-length-zp)*math.cos(phi))
        return math.hypot(y,z-spring_anchor_height)
    spring_closed=spring_length(0.);spring_open=spring_length(maximum)
    spring_rest=.7*min(spring_length(q) for q in np.linspace(0,maximum,101))
    # Two springs release the specified fraction of the panel's gravitational
    # potential gain. This is an energy calibration, not an exact force curve.
    spring_k=spring_fraction*phys['mass']['total_kg']*9.81*(height/2)*(1-math.cos(maximum))/(
        (spring_closed-spring_rest)**2-(spring_open-spring_rest)**2)
    arms=[]
    for sign,tag in [(-1,'l'),(1,'r')]:
        x=sign*(width/2+.065)
        track_length=travel+.14
        # C channel is open toward the panel. Treads are tangent to the lower
        # flange; the ideal slider represents the wheel/bearing constraint.
        for side in (-1,1):
            # Five millimetres of running clearance above the tread prevents
            # opposing zero-gap contacts from pinching a wheel under load.
            flange_z = -.024 if side < 0 else .029
            world.geoms.append(C.box(f'tilt_track_{tag}_{side}',(x,travel/2,zt+flange_z),
                (.025,track_length/2,.004),steel,7850,semantic='track',label='Horizontal roller-track flange'))
        world.geoms.append(C.box(f'tilt_track_{tag}_web',(x+sign*.029,travel/2,zt+.0025),
            (.004,track_length/2,.0305),steel,7850,semantic='track',label='Roller-track outer web'))
        for y,anchor_tag in [(.0,'front'),(travel,'rear')]:
            world.geoms.append(C.box(f'tilt_track_support_{tag}_{anchor_tag}',
                (sign*(width/2+.155),y,zt+.04),(.10,.025,.012),steel,7850,
                semantic='frame',label='Track support from side beam'))
        world.geoms.append(C.box(f'tilt_side_beam_{tag}',(sign*(width/2+.245),travel/2,zt+.04),
            (.025,(travel+.15)/2,.025),frame,600,semantic='frame',label='Garage side beam carrying track'))
        world.geoms.append(C.box(f'tilt_rear_post_{tag}',(sign*(width/2+.245),travel,(zt+.04)/2),
            (.025,.025,(zt+.04)/2),frame,600,semantic='frame',label='Rear track support post'))
        wheel = Body(f'tilt_top_wheel_{tag}',carriage.name,(x,0,0),QUAT_ID,None,[],[],ALL_TIERS,
            'mechanism','Freely rotating top roller')
        wheel.joint = Joint(f'tilt_top_wheel_{tag}_spin','hinge',(1,0,0),(0,0,0),None,
            damping=.0001,frictionloss=.0001,armature=1e-6,role='mechanism',robot_interactive=False,
            label='Roller turns on its panel axle')
        wheel.geoms.append(C.cyl(f'tilt_top_roller_{tag}',(0,0,0),.020,.012,roller_mat,(1,0,0),1200,
            True,True,ALL_TIERS,'hinge','Top roller'))
        model.add_body(wheel)
        panel.geoms.append(C.cyl(f'tilt_top_axle_{tag}',(sign*(width/2+.018),0,0),.006,.035,steel,(1,0,0),
            7850,True,True,ALL_TIERS,'hinge','Panel-to-roller axle'))
        _bearing_eye(carriage,f'tilt_top_bearing_{tag}',(sign*(width/2+.042),0,0),steel,outer=.012)
        # Fixed arm pivot is bolted through a standoff into the side jamb.
        world.geoms.append(C.box(f'tilt_arm_mount_{tag}',(sign*(width/2+.13),0,zp),
            (.035,.03,.045),steel,7850,semantic='hinge',label='Lifting-arm fixed mounting bracket'))
        world.geoms.append(C.cyl(f'tilt_arm_fixed_pin_{tag}',(x+sign*.008,0,zp),.005,.024,steel,(1,0,0),
            7850,True,True,ALL_TIERS,'hinge','Fixed pin through the open arm eye'))
        arm=Body(f'tilt_arm_{tag}',None,(x,0,zp),QUAT_ID,None,[],[],ALL_TIERS,'mechanism','Side lifting arm')
        arm.joint=Joint(f'tilt_arm_{tag}_hinge','hinge',(1,0,0),(0,0,0),(arm_minimum-.03,final['arm_rad']+.1),
            damping=1.,frictionloss=.2,armature=.05,
            role='mechanism',robot_interactive=False,label='Spring-assisted side lifting arm')
        vector=np.array([0.,rear,zt-length-zp]);radius=float(np.linalg.norm(vector))
        arm.geoms.append(C.cyl(f'tilt_arm_{tag}_rod',tuple(vector/2),.012,(radius-.048)/2,steel,vector,7850,
            True,True,ALL_TIERS,'mechanism','Fixed-length lifting arm'))
        _bearing_eye(arm,f'tilt_arm_{tag}_root_eye',(0,0,0),steel)
        _bearing_eye(arm,f'tilt_arm_{tag}_tip_eye',tuple(vector),steel)
        spring_x=sign*.045
        anchor_name=f'tilt_spring_anchor_{tag}'
        tip_name=f'tilt_spring_arm_{tag}'
        world.sites.append(Site(anchor_name,(x+spring_x,0,zp+spring_anchor_height),QUAT_ID,.005,'spring_anchor'))
        arm.sites.append(Site(tip_name,tuple(spring_arm_fraction*vector+np.array([spring_x,0,0])),QUAT_ID,.005,'spring_anchor'))
        world.geoms.append(C.box(f'tilt_spring_mount_{tag}',(sign*(width/2+.15),0,zp+spring_anchor_height),
            (.045,.015,.015),steel,7850,semantic='hinge',label='Extension spring anchor on side jamb'))
        arm.geoms.append(C.cyl(f'tilt_spring_arm_pin_{tag}',tuple(spring_arm_fraction*vector+np.array([spring_x/2,0,0])),
            .005,.0275,steel,(1,0,0),7850,True,True,ALL_TIERS,'hinge','Spring hook pin on lifting arm'))
        if spring_fraction>0:
            model.spatial_springs.append(SpatialSpring(f'tilt_extension_spring_{tag}',
                (anchor_name,tip_name),spring_k,spring_rest,damping=8.,width=.007,
                label='Extension counterbalance spring between jamb and lifting arm'))
        model.add_body(arm);arms.append(arm.joint.name)
        panel.geoms.append(C.box(f'tilt_panel_bracket_{tag}',(sign*(width/2),rear/2,-length),
            (.035,rear/2,.018),steel,7850,semantic='hinge',label='Door bracket for lifting-arm pin'))
        panel.geoms.append(C.cyl(f'tilt_panel_pin_{tag}',(x-sign*.01,rear,-length),.005,.025,steel,(1,0,0),7850,
            True,True,ALL_TIERS,'hinge','Lifting-arm door pin'))
        model.equalities.append(Equality('connect',f'tilt_arm_{tag}_connection',arm.name,panel.name,
            anchor=tuple(vector),tiers=ALL_TIERS,label='Lifting-arm tip pinned to the door bracket',
            solref=(.002,1),solimp=(.99,.999,.001)))
        # The wheel bearing is represented by its ideal axle joint. Arm eyes
        # have actual holes and need no collision exclusions or QA waivers.
        model.meta.setdefault('clearance_allow',[]).extend([
            [f'tilt_top_axle_{tag}',f'tilt_top_roller_{tag}','Wheel bearing on axle']])
    opm=H.OPERATORS[spec['operator']['model']];hz=spec['operator']['height']
    if opm.kind=='t_handle':
        handle=C.add_rotary_operator(model,panel,spec,phys,opm,1.,1.,0.,hz-zt,thickness,[-1.],None,name='t_handle')
        model.meta['operator_joint']=handle.joint.name
    else:
        C.add_pull(model,panel,opm,1.,0.,hz-zt,thickness,-1.,name='lift_handle')
        model.meta['operator_joint']=None
    if (spec['lock'].get('engaged') and not spec['lock'].get('robot_side_release')
            and spec['lock']['model'] not in ('garage_slide_lock','padlock')):
        panel.joint.range=(0.,.01)
    from .garage_locks import add_tiltup_lock
    add_tiltup_lock(model,panel,world,spec,zt)
    # The lifting arms retain their geometry-backed steel mass, separately
    # from the slab/catalogue budget. Calibrate the stored energy against
    # their actual COM height change as well as the panel; a rigid-door mass
    # multiplier would incorrectly make the arms rise through H/2.
    arm_potential=0.
    for tag in ('l','r'):
        arm=model.body(f'tilt_arm_{tag}');arm_mass,arm_com,_=arm.inertial()
        phi=final['arm_rad']
        dz=arm_com[1]*math.sin(phi)+arm_com[2]*(math.cos(phi)-1)
        arm_potential+=arm_mass*9.81*dz
    panel_potential=phys['mass']['total_kg']*9.81*(height/2)*(1-math.cos(maximum))
    spring_k=spring_fraction*(panel_potential+arm_potential)/((spring_closed-spring_rest)**2-(spring_open-spring_rest)**2)
    for spring in model.spatial_springs:
        if spring.name.startswith('tilt_extension_spring_'):spring.stiffness=spring_k
    world.sites.extend([Site('approach_point',(0,-2.,0),QUAT_ID,.05,'approach'),
        Site('goal_point',(0,2.,0),QUAT_ID,.05,'goal'),Site('door_plane_center',(0,0,ho/2),QUAT_ID,.02,'pass_plane')])
    model.meta.update({'primary_joint':'door_hinge','handle_height':hz,'counterbalance_fraction':spring_fraction,
        'garage_tiltup_linkage':{'schema_version':1,'kind':'retractable_top_roller_side_arm',
          'primary_joint':'door_hinge','carriage_joint':'door_carriage_slide','arm_joints':arms,
          'dimensions':dims,'nominal_range_rad':[0.,maximum],'track_travel_m':travel,
          'counterbalance_model':'Two native spatial extension springs, energy-scaled to the authored fraction; no manufacturer force curve.',
          'spring_stiffness_N_m':spring_k,'spring_rest_length_m':spring_rest,
          'spring_closed_length_m':spring_closed,'spring_open_length_m':spring_open,
          'counterbalance_potential_J':{'panel_and_catalogue_hardware':panel_potential,'geometry_backed_lifting_arms':arm_potential},
          'reference':'https://www.garador.co.uk/media/eawdrr4j/retractable.pdf'}})
    model.meta['mechanical_export_support']={'mjcf':'native point constraints and spatial springs',
        'urdf':'requires loop-closure and endpoint-force plugin',
        'usd':'unsupported closed-loop/spatial spring dynamics; static interchange only'}
    model.meta.setdefault('mechanism_mass_bodies',[]).extend(['door_carriage',
        'tilt_top_wheel_l','tilt_top_wheel_r','tilt_arm_l','tilt_arm_r'])
    return panel
