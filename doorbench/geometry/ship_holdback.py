"""Original spring-return hook and striker for an open marine door.

The hook/catch topology is informed by A. L. Hansen's 29 door holder. This
station, self-seating jaw and material sizes are original engineering, not
OEM CAD or a rated marine restraint. Native capture, hand-free retention and
release require the independent ship_holdback_qa service gate.
"""
from __future__ import annotations
import math
import numpy as np
import trimesh
from ..ir import Body,Joint,Site,SpatialSpring,Geom,ALL_TIERS,QUAT_ID,mat_to_quat,quat_from_axis_angle,quat_to_mat,quat_z_to
from . import common as C
from .marine_dogs import bearing_y

REFERENCE='https://alhansen.com/products/29-hook-catch-door-holder'


def _prism(body,name,polygon,material,half_width=.004):
    """One convex load-bearing steel piece; callers preserve the hook opening."""
    mesh=trimesh.convex.convex_hull([(x,y,z)for y in (-half_width,half_width)for x,z in polygon])
    body.geoms.append(C.mesh_geom(name,name,mesh,(0,0,0),QUAT_ID,material,7850,True,ALL_TIERS,
        'holdback','Steel hook jaw / web with explicit open striker pocket'))


def add_ship_holdback(model,spec):
    """Add the physical assembly; construction itself does not certify its QA."""
    leaf=model.body('leaf');w=float(spec['leaf']['width']);h=float(spec['leaf']['height']);t=float(spec['leaf']['thickness'])
    u=float(model.meta['u']);axis=np.asarray(leaf.joint.axis,float);hinge=np.asarray(leaf.pos)+np.asarray(leaf.joint.pos)
    angle=float(leaf.joint.range[1])-.080;rotation=quat_to_mat(quat_from_axis_angle(axis,angle))
    # Individual six/eight-dog layouts need the station between actual dog
    # rows: a blanket quarter-height post hit their returned lower handles.
    # Four-dog wheel layouts keep the holder below the central transmission.
    dog_heights=sorted({float(model.body(row['body']).pos[2]) for row in model.meta['marine_dog_mounts']})
    holder_height=(dog_heights[0]+dog_heights[1])/2 if len(dog_heights)>2 else h*.25
    local=np.array([u*(.004+w/2),0.,holder_height]);normal=rotation@np.array([0.,1.,0.])
    provisional=hinge+rotation@(local-np.asarray(leaf.joint.pos))
    closing=-np.cross(axis,provisional-hinge);closing[2]=0;closing/=np.linalg.norm(closing)
    side=1. if np.dot(normal,-closing)>0 else -1.;local[1]=side*(t/2+.060)
    eye=hinge+rotation@(local-np.asarray(leaf.joint.pos))
    closing=-np.cross(axis,eye-hinge);closing[2]=0;closing/=np.linalg.norm(closing)
    lateral=np.cross((0.,0.,1.),closing);basis=np.column_stack((closing,lateral,(0.,0.,1.)))
    eye_local=np.array([.088,0.,.020]);origin=eye-basis@eye_local
    steel=C.mat_from_material(model,'steel_galvanized','mat_ship_holdback')
    rubber=C.mat_from_material(model,'rubber','mat_ship_holdback_bumper')
    # Floor anchorage is outboard of the clear aperture, on the hinge side.
    station=Body('ship_holdback_station',None,tuple(origin),tuple(mat_to_quat(basis)),semantic='frame',
        label='Braced floor-mounted hook station outside the doorway',static=True)
    station.geoms.append(C.box('ship_holdback_base',(0,0,-origin[2]+.008),(.065,.055,.008),steel,7850,
        True,True,ALL_TIERS,'frame','Floor anchor plate'))
    for sx in (-1,1):
        for sy in (-1,1):
            station.geoms.append(C.cyl(f'ship_holdback_anchor_{sx}_{sy}',
                (sx*.045,sy*.035,-origin[2]-.020),.004,.040,steel,(0,0,1),7850,
                True,True,ALL_TIERS,'frame','M8 floor anchor through the plate into the structural floor'))
            station.geoms.append(C.cyl(f'ship_holdback_anchor_head_{sx}_{sy}',
                (sx*.045,sy*.035,-origin[2]+.019),.007,.003,steel,(0,0,1),7850,
                True,True,ALL_TIERS,'frame','Anchor head bearing directly on the base plate'))
    # Four walls of a hollow40×50mm post, rather than a solid steel bar.
    top=-.065;bottom=-origin[2]+.016;zc=(top+bottom)/2;hh=(top-bottom)/2
    for s in (-1,1):
        station.geoms.append(C.box(f'ship_holdback_post_x_{s}',(s*.0185,0,zc),(.0015,.025,hh),steel,7850,True,True,ALL_TIERS,'frame','3 mm steel tube wall'))
        station.geoms.append(C.box(f'ship_holdback_post_y_{s}',(0,s*.0235,zc),(.017,.0015,hh),steel,7850,True,True,ALL_TIERS,'frame','3 mm steel tube wall'))
    station.geoms.append(C.box('ship_holdback_post_cap',(0,0,-.0625),(.020,.025,.0025),steel,7850,
        True,True,ALL_TIERS,'frame','Post cap supporting the pivot cheeks'))
    for s in (-1,1):
        # The shaft aperture is removed stock, including the negative-Z half
        # of each cheek. Fixed-body collision filtering cannot supply a bore.
        station.geoms.append(C.box(f'ship_holdback_cheek_{s}_base',
            (0,s*.0155,-.0348),(.020,.0035,.0292),steel,7850,True,True,ALL_TIERS,'frame','Welded pivot cheek below the machined shaft aperture'))
        for edge in (-1,1):
            station.geoms.append(C.box(f'ship_holdback_cheek_{s}_side_{edge}',
                (edge*.0128,s*.0155,-.0028),(.0072,.0035,.0028),steel,7850,True,True,ALL_TIERS,'frame','Cheek stock beside the empty 11.2 mm shaft aperture'))
        bearing_y(station,f'ship_holdback_bearing_{s}',(0,s*.0155,0),steel,inner=.0054,outer=.012,half_length=.004,semantic='holdback')
        bearing_y(station,f'ship_holdback_thrust_washer_{s}',(0,s*.0085,0),steel,
            inner=.0056,outer=.009,half_length=.0025,semantic='holdback')
    # Rear striker stop and hook shoulder both transmit load into this post.
    station.geoms.append(C.box('ship_holdback_stop_carrier',(.041,0,-.045),(.021,.033,.015),steel,7850,True,True,ALL_TIERS,'frame','Striker-stop carrier welded to post'))
    for s in (-1,1):
        station.geoms.append(C.box(f'ship_holdback_bumper_standoff_{s}',(.058,s*.025,-.013),(.004,.008,.017),steel,7850,
            True,True,ALL_TIERS,'frame','Stop carrier passing beside the hook sweep'))
        station.geoms.append(C.box(f'ship_holdback_open_bumper_{s}',(.059,s*.025,.018),(.005,.008,.014),rubber,1150,
            True,True,ALL_TIERS,'holdback','Physical striker stop clear of the moving hook'))
    _prism(station,'ship_holdback_hook_shoulder',[(.022,-.030),(.042,-.030),(.042,.0109),(.022,.0030)],steel,half_width=.007)
    station.sites.append(Site('ship_holdback_spring_anchor',(.02,.024,-.060),role='mechanism'))
    station.geoms.append(C.box('ship_holdback_spring_bracket',(.02,.022,-.060),(.006,.007,.006),steel,7850,True,True,ALL_TIERS,'frame','Fixed spring anchor bracket'))
    model.add_body(station)
    model.meta.setdefault('native_fixed_body_names',[]).append(station.name)
    hook=Body('ship_holdback_hook',station.name,semantic='holdback',label='Spring-return hook with self-seating closing-load face')
    hook.joint=Joint('ship_holdback_release','hinge',(0,-1,0),range=(-.080,1.10),damping=.025,frictionloss=.015,
        role='lock',label='Lift hook to release the open-door striker')
    # A bored rotating root and real axle are individually visible.
    bearing_y(hook,'ship_holdback_hook_eye',(0,0,0),steel,inner=.0054,outer=.014,half_length=.005,semantic='holdback')
    station.geoms.append(C.cyl('ship_holdback_pivot_pin',(0,0,0),.005,.023,steel,(0,1,0),7850,
        True,True,ALL_TIERS,'holdback','Fixed retained pivot shaft'))
    for s in (-1,1):station.geoms.append(C.cyl(f'ship_holdback_pin_head_{s}',(0,s*.024,0),.008,.002,steel,(0,1,0),7850,
        True,True,ALL_TIERS,'holdback','Axial retaining head'))
    _prism(hook,'ship_holdback_arm',[(.010,-.001),(.010,.013),(.087,.053),(.135,.065),(.139,.050),(.085,.034)],steel)
    _prism(hook,'ship_holdback_jaw',[(.106,.006),(.106,.034),(.139,.050)],steel)
    # The welded grip is near the distal web, above the striker pocket. Its
    # roughly 102 mm lever arm permits a 25 N light-hand input; a short 49 mm
    # lever required uncomfortable transient effort on several variants.
    handle_x,handle_z=.090,.047
    hook.geoms.append(C.cyl('ship_holdback_release_handle',(handle_x,-.032,handle_z),.009,.029,steel,(0,1,0),7850,
        True,True,ALL_TIERS,'operator','Physical release handle clear of the capture pocket'))
    # The site is on the grip's lower cylindrical surface, with an outward
    # normal. An upward/inward finger force has a useful opening moment; the
    # former centreline point was inside the solid handle, not on a surface.
    outward=np.array([handle_z,0.,-handle_x]);outward/=np.linalg.norm(outward)
    grip=np.array([handle_x,-.045,handle_z])+.009*outward
    hook.sites.extend([Site('ship_holdback_release_grip',tuple(grip),tuple(quat_z_to(outward)),role='grip'),
        Site('ship_holdback_spring_tip',(.035,.024,.015),role='mechanism')])
    hook.geoms.append(C.cyl('ship_holdback_spring_pin',(.035,.014,.015),.003,.014,steel,(0,1,0),7850,
        True,True,ALL_TIERS,'holdback','Spring hook pin connected to the rotating web'))
    model.add_body(hook)
    model.spatial_springs.append(SpatialSpring('ship_holdback_return',('ship_holdback_spring_tip','ship_holdback_spring_anchor'),
        300.,.045,damping=.5,width=.002,label='Original extension spring returning the unloaded retaining hook'))
    striker=Body('ship_holdback_striker',leaf.name,semantic='holdback',label='Striker crossbar welded through two leaf mounting ears')
    bar_axis=rotation.T@lateral
    # A rounded-end solid bar retains the 12 mm diameter and 90 mm total
    # envelope. Its native capsule contact has one consistent normal at the
    # bumper; a flat-ended cylinder produced opposed duplicate contact
    # normals at a sub-micrometre near-touch in MuJoCo 3.12.0.
    striker.geoms.append(Geom('ship_holdback_striker_bar', 'capsule', (.006,.039),
        tuple(local), tuple(quat_z_to(bar_axis)), steel, density=7850,
        tiers=ALL_TIERS, semantic='holdback', part_label='12 mm rounded-end striker bar held between steel ears'))
    for s in (-1,1):
        end=local+s*.039*bar_axis;foot=end.copy();foot[1]=side*(t/2+.004)
        delta=end-foot
        striker.geoms.append(C.cyl(f'ship_holdback_striker_ear_{s}',tuple((end+foot)/2),.007,float(np.linalg.norm(delta))/2,
            steel,delta,7850,True,True,ALL_TIERS,'holdback','Striker ear continuously joining its mounting plate'))
        striker.geoms.append(C.box(f'ship_holdback_striker_plate_{s}',tuple(foot),(.018,.004,.020),steel,7850,
            True,True,ALL_TIERS,'holdback','Leaf welded mounting plate'))
    striker.sites.append(Site('ship_holdback_striker_center',tuple(local),role='mechanism'));model.add_body(striker)
    # Rigid steel bearing faces use a millimetre-scale contact response rather
    # than the general soft door gasket default. This is solver compliance,
    # not an elastic-stress or proof-load rating; all contacts remain active.
    for body in (station,hook,striker):
        for geom in body.geoms:
            if geom.semantic == 'holdback':
                geom.semantic='lock'
            if geom.material == steel:
                geom.solref=(.004,1.)
                geom.solimp=(.99,.999,.0005)
    model.meta.setdefault('mechanism_mass_bodies',[]).extend([hook.name,striker.name])
    # This directly pivoted hook has no gearbox or reflected motor inertia.
    model.meta.setdefault('physical_inertia_joints',[]).append(hook.joint.name)
    # The light steel hook can return onto its shoulder during a single coarse
    # step. A 2 ms minimal-tier trial exceeded the unchanged 1 mm gate; 1 ms
    # resolves the same source's capture/return without changing forces or mass.
    model.meta['native_timestep_s']=min(.001,model.meta.get('native_timestep_s',.001))
    model.meta['ship_holdback']={'schema_version':1,'status':'physical_spring_return_hook','leaf_joint':leaf.joint.name,
        'hook_joint':hook.joint.name,'release_site':'ship_holdback_release_grip','striker_site':'ship_holdback_striker_center',
        'hook_body':hook.name,'station_body':station.name,'striker_geom':'ship_holdback_striker_bar',
        'load_face_geom':'ship_holdback_jaw','load_shoulder_geom':'ship_holdback_hook_shoulder',
        'load_shoulder_moving_geom':'ship_holdback_arm',
        'opening_stop_geoms':['ship_holdback_open_bumper_-1','ship_holdback_open_bumper_1'],
        'inspection_hook_range_rad':[0.,1.10],
        'spring':'ship_holdback_return','nominal_capture_angle_rad':angle,'full_open_angle_rad':leaf.joint.range[1],
        'station_origin_world':origin.tolist(),'station_rotation_world':basis.tolist(),
        'striker_height_leaf_m':holder_height,
        'manual_force_limit_N':120.,'reference':REFERENCE,
        'scope':'Original floor-mounted spring-return hook and striker. No OEM load, weather, fatigue or human-task rating.'}
    return hook


def first_ship_holdback_stop_angle(model,meta):
    """Find the physical full-open bumper, without moving live simulation data.

    Inspection only: dogs are released and the holdback is lifted. The native
    service gate separately reaches this stop under actual capped hand forces.
    The primary joint's farther safety limit must not be swept through it.
    """
    import mujoco
    from .marine_linkage import resolve_marine_configuration
    hb=meta['ship_holdback'];leaf=model.joint(hb['leaf_joint']).id
    la=int(model.jnt_qposadr[leaf]);hook=model.joint(hb['hook_joint']).id
    striker=model.geom(hb['striker_geom']).id
    stops=[model.geom(name).id for name in hb['opening_stop_geoms']]
    data=mujoco.MjData(model);base=model.qpos0.copy()
    base[model.jnt_qposadr[hook]]=hb['inspection_hook_range_rad'][1]
    linkage=meta.get('marine_dog_linkage')
    if linkage:
        j=model.joint(linkage['input_joint']).id
        base[model.jnt_qposadr[j]]=model.jnt_range[j,1]
        resolve_marine_configuration(model,base,meta)
    else:
        for row in meta['marine_dog_mounts']:
            j=model.joint(row['joint']).id;base[model.jnt_qposadr[j]]=model.jnt_range[j,1]
    def gap(angle):
        data.qpos[:]=base;data.qpos[la]=angle;mujoco.mj_kinematics(model,data)
        return min(float(mujoco.mj_geomDistance(model,data,striker,g,.5,None)) for g in stops)
    lower=0.;upper=float(model.jnt_range[leaf,1]);previous=0.
    if gap(0.)<=0:raise ValueError('Holdback opening bumper intersects the closed striker')
    for angle in np.linspace(0.,upper,193)[1:]:
        if gap(angle)<=0:
            lower,upper=previous,float(angle);break
        previous=float(angle)
    else:raise ValueError('Physical marine opening bumper not found before the safety limit')
    for _ in range(40):
        mid=(lower+upper)/2
        if gap(mid)>0:lower=mid
        else:upper=mid
    return {'ok':True,'angle_rad':lower,'gap_m':gap(lower),
        'scope':'First rigid striker-to-bumper contact for geometric inspection; no native state initialization or force proof.'}
