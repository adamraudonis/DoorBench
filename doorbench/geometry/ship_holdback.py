"""Original spring-return hook and striker for an open marine door.

The hook/catch topology is informed by A. L. Hansen's 29 door holder. This
station, self-seating jaw and material sizes are original engineering, not
OEM CAD or a rated marine restraint. The prototype is not installed by the
production builder until native capture, hand-free retention and release pass.
"""
from __future__ import annotations
import math
import numpy as np
import trimesh
from ..ir import Body,Joint,Site,SpatialSpring,ALL_TIERS,QUAT_ID,mat_to_quat,quat_from_axis_angle,quat_to_mat
from . import common as C
from .marine_dogs import bearing_y

REFERENCE='https://alhansen.com/products/29-hook-catch-door-holder'


def _prism(body,name,polygon,material,half_width=.004):
    """One convex load-bearing steel piece; callers preserve the hook opening."""
    mesh=trimesh.convex.convex_hull([(x,y,z)for y in (-half_width,half_width)for x,z in polygon])
    body.geoms.append(C.mesh_geom(name,name,mesh,(0,0,0),QUAT_ID,material,7850,True,ALL_TIERS,
        'holdback','Steel hook jaw / web with explicit open striker pocket'))


def add_ship_holdback(model,spec):
    """Add a prototype; the caller must retain the mechanical-incomplete flag."""
    leaf=model.body('leaf');w=float(spec['leaf']['width']);h=float(spec['leaf']['height']);t=float(spec['leaf']['thickness'])
    u=float(model.meta['u']);axis=np.asarray(leaf.joint.axis,float);hinge=np.asarray(leaf.pos)+np.asarray(leaf.joint.pos)
    angle=float(leaf.joint.range[1])-.080;rotation=quat_to_mat(quat_from_axis_angle(axis,angle))
    # Mid-width at quarter height avoids edge dogs and the central wheel.
    local=np.array([u*(.004+w/2),0.,h*.25]);normal=rotation@np.array([0.,1.,0.])
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
    # Four walls of a hollow40×50mm post, rather than a solid steel bar.
    top=-.065;bottom=-origin[2]+.016;zc=(top+bottom)/2;hh=(top-bottom)/2
    for s in (-1,1):
        station.geoms.append(C.box(f'ship_holdback_post_x_{s}',(s*.0185,0,zc),(.0015,.025,hh),steel,7850,True,True,ALL_TIERS,'frame','3 mm steel tube wall'))
        station.geoms.append(C.box(f'ship_holdback_post_y_{s}',(0,s*.0235,zc),(.017,.0015,hh),steel,7850,True,True,ALL_TIERS,'frame','3 mm steel tube wall'))
    station.geoms.append(C.box('ship_holdback_post_cap',(0,0,-.0625),(.020,.025,.0025),steel,7850,
        True,True,ALL_TIERS,'frame','Post cap supporting the pivot cheeks'))
    for s in (-1,1):
        station.geoms.append(C.box(f'ship_holdback_cheek_{s}',(0,s*.0155,-.032),(.020,.0035,.032),steel,7850,True,True,ALL_TIERS,'frame','Welded pivot cheek'))
        bearing_y(station,f'ship_holdback_bearing_{s}',(0,s*.0155,0),steel,inner=.0054,outer=.012,half_length=.004,semantic='holdback')
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
    hook.geoms.append(C.cyl('ship_holdback_release_handle',(.040,-.032,.027),.009,.027,steel,(0,1,0),7850,
        True,True,ALL_TIERS,'operator','Physical release handle clear of the capture pocket'))
    hook.sites.extend([Site('ship_holdback_release_grip',(.040,-.056,.027),role='grip'),
        Site('ship_holdback_spring_tip',(.035,.024,.015),role='mechanism')])
    hook.geoms.append(C.cyl('ship_holdback_spring_pin',(.035,.014,.015),.003,.014,steel,(0,1,0),7850,
        True,True,ALL_TIERS,'holdback','Spring hook pin connected to the rotating web'))
    model.add_body(hook)
    model.spatial_springs.append(SpatialSpring('ship_holdback_return',('ship_holdback_spring_tip','ship_holdback_spring_anchor'),
        300.,.045,damping=.5,width=.002,label='Original extension spring returning the unloaded retaining hook'))
    striker=Body('ship_holdback_striker',leaf.name,semantic='holdback',label='Striker crossbar welded through two leaf mounting ears')
    bar_axis=rotation.T@lateral
    striker.geoms.append(C.cyl('ship_holdback_striker_bar',tuple(local),.006,.045,steel,bar_axis,7850,
        True,True,ALL_TIERS,'holdback','12 mm striker bar held between steel ears'))
    for s in (-1,1):
        end=local+s*.039*bar_axis;foot=end.copy();foot[1]=side*(t/2+.004)
        delta=end-foot
        striker.geoms.append(C.cyl(f'ship_holdback_striker_ear_{s}',tuple((end+foot)/2),.007,float(np.linalg.norm(delta))/2,
            steel,delta,7850,True,True,ALL_TIERS,'holdback','Striker ear continuously joining its mounting plate'))
        striker.geoms.append(C.box(f'ship_holdback_striker_plate_{s}',tuple(foot),(.018,.004,.020),steel,7850,
            True,True,ALL_TIERS,'holdback','Leaf welded mounting plate'))
    striker.sites.append(Site('ship_holdback_striker_center',tuple(local),role='mechanism'));model.add_body(striker)
    model.meta.setdefault('mechanism_mass_bodies',[]).extend([hook.name,striker.name])
    model.meta['ship_holdback']={'schema_version':1,'status':'prototype_unverified','leaf_joint':leaf.joint.name,
        'hook_joint':hook.joint.name,'release_site':'ship_holdback_release_grip','striker_site':'ship_holdback_striker_center',
        'hook_body':hook.name,'station_body':station.name,'striker_geom':'ship_holdback_striker_bar',
        'load_face_geom':'ship_holdback_jaw','load_shoulder_geom':'ship_holdback_hook_shoulder',
        'spring':'ship_holdback_return','nominal_capture_angle_rad':angle,'full_open_angle_rad':leaf.joint.range[1],
        'station_origin_world':origin.tolist(),'station_rotation_world':basis.tolist(),
        'manual_force_limit_N':120.,'reference':REFERENCE,
        'scope':'Original floor-mounted spring-return hook and striker. No OEM load, weather, fatigue or human-task rating.'}
    return hook
