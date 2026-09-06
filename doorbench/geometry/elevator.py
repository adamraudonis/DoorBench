"""Original landing-door suspension and contact-operated retiring-cam interlocks.

This is a stationary, level car fixture. It models a hook behind a leaf-mounted
bar, a release roller and a force-driven cam. The guide bearings and electrical
contacts are idealized explicitly; this is not an OEM or safety certification.
"""
from ..ir import Body, Joint, ALL_TIERS
from . import common as C


def rebuild_elevator(model, spec, leaves):
    world=model.body('world_env')
    # The installed hook/bar replaces the old abstract interlock weld. Leaving
    # both in place pins the leaf even after actual cam release, and makes a
    # removed-hook negative falsely appear secure.
    leaf_names={leaf.name for leaf in leaves}
    replaced={row['name'] for row in model.meta.get('breakable_welds',[])
              if row.get('body') in leaf_names and row.get('lock_model')=='interlock'}
    model.equalities=[eq for eq in model.equalities if eq.name not in replaced]
    model.meta['breakable_welds']=[row for row in model.meta.get('breakable_welds',[]) if row['name'] not in replaced]
    W,H,t=(spec['leaf'][k] for k in ('width','height','thickness'))
    Ho=spec['opening']['height'];travel=spec['kinematics']['travel_m']
    wall=spec['opening']['wall_thickness']/2
    lane=wall+.075
    steel=C.mat_from_material(model,'stainless','mat_elevator_mechanics')
    supports=model.meta['sliding_track_supports']
    if len(leaves)==2:
        for leaf in leaves:
            leaf.pos=(leaf.pos[0]+leaf.joint.axis[0]*.002,leaf.pos[1],leaf.pos[2])
    removed={s['rail'] for s in supports}|{'sill','car_floor','car_wall_l','car_wall_r','car_back'}
    world.geoms=[g for g in world.geoms if g.name not in removed]
    endpoints=[b.pos[0]+b.joint.axis[0]*q+edge for b in leaves for q in (0.,travel) for edge in (-W/2,W/2)]
    xmin,xmax=min(endpoints)-.20,max(endpoints)+.20
    center,half=(xmin+xmax)/2,(xmax-xmin)/2
    world.geoms.append(C.box('car_floor',(center,1.23,-.01),(half,1.23,.01),steel,7900,True,True,ALL_TIERS,'floor','Stationary level car floor'))
    for x,tag in ((xmin,'l'),(xmax,'r')):
        world.geoms.append(C.box('car_wall_'+tag,(x,1.23,Ho/2+.1),(.02,1.23,Ho/2+.1),steel,7900,True,True,ALL_TIERS,'wall','Car wall beyond complete panel storage envelope'))
    world.geoms.append(C.box('car_back',(center,2.46,Ho/2+.1),(half,.02,Ho/2+.1),steel,7900,True,True,ALL_TIERS,'wall','Stationary car back wall'))
    rows=[]
    for leaf,support in zip(leaves,supports):
        name=leaf.name;xc=leaf.pos[0];direction=leaf.joint.axis[0]
        leaf.pos=(xc,lane,leaf.pos[2]);leaf.joint.range=(-.010,travel+.010)
        leaf.joint.notes='Full travel; a native hook/bar contact carries the interlock load.'
        support.update(nominal_range=[0.,travel],rail_coverage_mode='trolley_sweep',
                       rollers=[],end_stops=[],floor_guides_required=True,floor_guides=[])
        moving=Body(name+'_hanger_assembly',name,tiers=ALL_TIERS,semantic='mechanism',label='Two supported hangers and locking bar')
        model.add_body(moving);model.meta.setdefault('mechanism_mass_bodies',[]).append(moving.name)
        radius=.04;inset=min(.12,W*.22);wheel_z=Ho+.20
        run_low=min(xc,xc+direction*travel)-W/2+inset-radius-.014
        run_high=max(xc,xc+direction*travel)+W/2-inset+radius+.014
        rx,rh=(run_low+run_high)/2,(run_high-run_low)/2
        rail=support['rail'];ry=lane+.005;rail_top=wheel_z-radius
        world.geoms.append(C.box(rail,(rx,ry,rail_top-.015),(rh,.009,.015),steel,7900,True,True,ALL_TIERS,'track','Continuous steel hanger running rail'))
        for k in range(5):
            x=run_low+.03+(run_high-run_low-.06)*k/4
            world.geoms.append(C.box(f'{name}_rail_support_{k}',(x,(wall+ry-.009)/2,rail_top-.015),
                (.012,(ry-.009-wall)/2,.007),steel,7900,True,True,ALL_TIERS,'track','Header-to-rail mounting bracket'))
        for k,x in enumerate((-W/2+inset,W/2-inset)):
            wheel=f'{name}_hanger_wheel_{k}'
            moving.geoms.append(C.cyl(wheel,(x,.005,wheel_z),radius,.010,steel,(0,1,0),7900,True,True,ALL_TIERS,'track','Rigid rolling proxy on an ideal hanger bearing'))
            moving.geoms.append(C.cyl(f'{name}_hanger_axle_{k}',(x,.015,wheel_z),.006,.017,steel,(0,1,0),7900,True,True,ALL_TIERS,'track','Hanger wheel axle'))
            moving.geoms.append(C.box(f'{name}_hanger_strap_{k}',(x,.025,(H-.04+wheel_z)/2),
                (.017,.004,(wheel_z-(H-.04))/2+.014),steel,7900,True,True,ALL_TIERS,'track','Leaf-mounted hanger strap'))
            moving.geoms.append(C.box(f'{name}_hanger_pad_{k}',(x,.017,H),(.025,.004,.055),steel,7900,True,True,ALL_TIERS,'track','Hanger mounting pad on actual panel stock'))
            support['rollers'].append(wheel)
        for tag,x in (('low',run_low+.006),('high',run_high-.006)):
            stop=f'{name}_hanger_stop_{tag}'
            world.geoms.append(C.box(stop,(x,ry,wheel_z-.0075),(.008,.011,.0375),steel,7900,True,True,ALL_TIERS,'track','Rail-mounted terminal wheel stop'))
            support['end_stops'].append(stop)
        # Continuous sill jaws hold the panel throughout travel; each jaw
        # meets a floor plate and leaves 1 mm on the actual steel skin.
        jaws=[];feet=[]
        for sign,tag in ((-1,'front'),(1,'rear')):
            y=lane+sign*(t/2+.001+.004)
            jaw=f'{name}_sill_jaw_{tag}';foot=f'{name}_sill_foot_{tag}'
            world.geoms.append(C.box(jaw,(rx,y,.0235),(rh,.004,.0185),steel,7900,True,True,ALL_TIERS,'track','Continuous panel guide jaw'))
            world.geoms.append(C.box(foot,(rx,y,.0025),(rh,.007,.0025),steel,7900,True,True,ALL_TIERS,'track','Floor-mounted sill guide foot'))
            jaws.append(jaw);feet.append(foot)
        support['floor_guides']=[{'jaws':jaws,'feet':feet}]
        support['guide_leaf_geoms']=[g.name for g in leaf.geoms if g.collision and g.semantic in ('leaf','glass')]
        rows.append(_interlock(model,world,moving,name,xc,lane,direction,H,Ho,steel))
    for actuator in model.meta.get('actuators',[]):
        if actuator['joint'] in {b.joint.name for b in leaves}:
            # A finite closing preload seats the leaf before the hook drops;
            # an exact-zero position demand can stop short under dry friction.
            force=max(abs(v) for v in actuator['forcerange'])
            actuator.update(kind='motor',gear=1.,ctrlrange=[-force,force],role='elevator_drive',
                            position_control={'kp':400.,'kv':60.,'targets_m':[-.020,travel+.020]})
    model.meta['elevator_interlocks']={'schema_version':1,'leaves':rows,
        'car_at_landing':True,'car_storage_envelope_x_m':[xmin,xmax],
        'scope':'Stationary level car fixture; geometric hanger/sill support and native hook/bar/cam contacts. Ideal slide/hinge bearings, ideal closed/locked electrical switches, ideal paired-leaf transmission. No moving-car or regulatory certification.',
        'reference':'https://cjanderson.com/lr-use-with-retiring-cam/'}
    model.meta['native_timestep_s']=min(.00025,model.meta.get('native_timestep_s',.002))


def _interlock(model,world,moving,name,xc,lane,direction,H,Ho,steel):
    prefix=name+'_interlock';px=xc+direction*.06;py=lane+.09;pz=H+.40
    # The leaf bar approaches a hook face 4 mm away. Its opening force acts
    # below the hook pivot and therefore seats the hook instead of lifting it.
    bar=prefix+'_bar'
    moving.geoms.append(C.box(bar,(-direction*.018,.09,H+.38),(.008,.009,.010),steel,7900,True,True,ALL_TIERS,'lock','Leaf-mounted locking bar'))
    moving.geoms.append(C.box(prefix+'_bar_upright',(-direction*.025,.09,H+.165),(.008,.008,.225),steel,7900,True,True,ALL_TIERS,'lock','Locking bar upright behind fixed bearing'))
    moving.geoms.append(C.box(prefix+'_bar_mount',(-direction*.015,.023,H-.015),(.025,.01,.035),steel,7900,True,True,ALL_TIERS,'lock','Bar bracket fixed to panel skin'))
    moving.geoms.append(C.box(prefix+'_bar_bridge',(-direction*.022,.0625,H-.015),(.008,.0355,.010),steel,7900,True,True,ALL_TIERS,'lock','Locking bar bridge below fixed bearing'))
    hook=Body(prefix+'_hook',None,(px,py,pz),tiers=ALL_TIERS,semantic='lock',label='Gravity-return landing door hook')
    hook.joint=Joint(prefix+'_hinge','hinge',(0,direction,0),range=(-.10,1.10),stiffness=.3,springref=-.3,damping=.015,frictionloss=.004,
                     role='lock',robot_interactive=False,limit_solref=(.001,1.))
    hook.geoms.extend([
        C.box(prefix+'_hook_arm',(-direction*.03,0,.006),(.036,.007,.006),steel,7900,True,True,ALL_TIERS,'lock','Hook arm'),
        C.box(prefix+'_hook_nose',(-direction*.06,0,-.014),(.006,.008,.008),steel,7900,True,True,ALL_TIERS,'lock','Load-bearing hook nose'),
        C.box(prefix+'_hook_web',(-direction*.06,0,-.003),(.006,.007,.003),steel,7900,True,True,ALL_TIERS,'lock','Continuous hook web between nose and arm'),
        C.cyl(prefix+'_shaft',(0,.003,0),.006,.058,steel,(0,1,0),7900,True,True,ALL_TIERS,'lock','Hook spindle through bearing and release arm'),
        C.box(prefix+'_roller_arm',(direction*.033,.028,0),(.039,.006,.006),steel,7900,True,True,ALL_TIERS,'lock','Coaxial release roller arm'),
        C.cyl(prefix+'_roller',(direction*.065,.028,0),.015,.008,steel,(0,1,0),7900,True,True,ALL_TIERS,'lock','Release roller')])
    # Four bearing cheeks leave a genuine 13 mm square bore for the 12 mm
    # spindle. Their back bracket attaches to the wall above the leaf path.
    bearing=[]
    for axis in (0,2):
        for sign in (-1,1):
            p=[px,py-.045,pz];size=[.017,.008,.017];p[axis]+=sign*(.017+.0065)/2;size[axis]=(.017-.0065)/2
            size[2 if axis==0 else 0]=.017 if axis==0 else .0065
            n=f'{prefix}_bearing_{axis}_{sign}';bearing.append(n)
            world.geoms.append(C.box(n,p,size,steel,7900,True,True,ALL_TIERS,'lock','Prepared hook spindle bearing'))
    wall=lane-.075
    world.geoms.append(C.box(prefix+'_mount',(px,(wall+py-.053)/2,pz+.023),(.025,(py-.053-wall)/2,.006),steel,7900,True,True,ALL_TIERS,'lock','Wall-to-interlock mounting bridge'))
    world.geoms.append(C.box(prefix+'_mount_drop',(px,py-.045,pz+.0165),(.025,.008,.0065),steel,7900,True,True,ALL_TIERS,'lock','Bearing mounting web above spindle bore'))
    # Opening load seats the hook against this fixed stop through its
    # positive-X arm. The spindle bearing alone cannot resist that torque.
    stop=prefix+'_closed_stop'
    world.geoms.append(C.box(stop,(px+direction*.03,py+.028,pz+.0155),(.012,.009,.0095),steel,7900,True,True,ALL_TIERS,'lock','Frame-connected closed hook stop'))
    world.geoms.append(C.box(prefix+'_stop_mount',(px+direction*.03,(wall+py+.034)/2,pz+.025),
        (.012,(py+.034-wall)/2,.005),steel,7900,True,True,ALL_TIERS,'lock','Wall-mounted hook stop bridge above moving bar'))
    world.geoms[-2].solref=(.001,1.);world.geoms[-2].contact_priority=1
    cam=Body(prefix+'_cam',None,(px+direction*.061,py+.028,pz+.023),tiers=ALL_TIERS,semantic='mechanism',label='Force-driven retiring cam')
    cam.joint=Joint(prefix+'_cam_slide','slide',(0,0,-1),range=(-.010,.071),stiffness=250.,springref=-.008,damping=2.,frictionloss=.2,
                    role='mechanism',robot_interactive=False,limit_solref=(.001,1.))
    cam.geoms.extend([
        C.box(prefix+'_cam_shoe',(0,0,0),(.010,.010,.003),steel,7900,True,True,ALL_TIERS,'mechanism','Retiring cam contacting release roller'),
        C.cyl(prefix+'_cam_stem',(0,0,.09),.004,.09,steel,(0,0,1),7900,True,True,ALL_TIERS,'mechanism','Cam stem through prepared guide'),
        C.cyl(prefix+'_cam_return_collar',(0,0,.081),.009,.003,steel,(0,0,1),7900,True,True,ALL_TIERS,'mechanism','Stem collar against guide lower face at return'),
        C.cyl(prefix+'_cam_press_collar',(0,0,.176),.009,.003,steel,(0,0,1),7900,True,True,ALL_TIERS,'mechanism','Stem collar against guide upper face at full release')])
    for g in [cam.geoms[0],cam.geoms[2],cam.geoms[3],hook.geoms[-1],hook.geoms[1]]:
        g.solref=(.001,1.);g.solimp=(.99,.999,.001);g.contact_priority=1
    guides=[]
    for axis in (0,1):
        for sign in (-1,1):
            p=[cam.pos[0],cam.pos[1],cam.pos[2]+.10];size=[.018,.018,.016]
            p[axis]+=sign*(.018+.00475)/2;size[axis]=(.018-.00475)/2
            size[1 if axis==0 else 0]=.018 if axis==0 else .00475
            n=f'{prefix}_cam_guide_{axis}_{sign}';guides.append(n)
            world.geoms.append(C.box(n,p,size,steel,7900,True,True,ALL_TIERS,'mechanism','Prepared cam stem guide'))
    # Mount enters the front guide wall, leaving its stem bore unobstructed.
    world.geoms.append(C.box(prefix+'_cam_mount',(cam.pos[0],(wall+cam.pos[1]-.018)/2,cam.pos[2]+.10),
        (.018,(cam.pos[1]-.018-wall)/2,.012),steel,7900,True,True,ALL_TIERS,'mechanism','Car-at-landing cam mounting support'))
    for b in (hook,cam):
        model.add_body(b);model.meta.setdefault('mechanism_mass_bodies',[]).append(b.name)
        model.meta.setdefault('physical_inertia_joints',[]).append(b.joint.name)
    return {'leaf':name,'joint':name+'_slide','hook_joint':hook.joint.name,'cam_joint':cam.joint.name,
        'bar_geom':bar,'hook_geom':prefix+'_hook_nose','roller_geom':prefix+'_roller','cam_geom':prefix+'_cam_shoe',
        'hook_stop_geoms':[prefix+'_roller_arm',stop],
        'cam_return_collar_geom':prefix+'_cam_return_collar','cam_press_collar_geom':prefix+'_cam_press_collar',
        'bearing_geoms':bearing,'cam_guide_geoms':guides,'max_cam_force_N':30.,'cam_travel_m':.057,'cam_safety_range_m':[-.010,.071],
        'released_angle_rad':.65,'closed_m':.006,'seated_m':.0005,'stroke_m':float(model.body(name).joint.range[1]-.010)}
