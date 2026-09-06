"""Original supported vault boltwork and crane journals.

The spur reduction is an explicitly ideal keyed-gear relation. Bolt loads
travel through actual cranks, pin-connected rods and rigid carriers. This is
not an OEM, security, blast-pressure or structural-strength certification.
"""
from __future__ import annotations

import math
import numpy as np

from ..ir import ALL_TIERS, Body, Joint, Equality, Site, QUAT_ID, quat_z_to, quat_from_axis_angle
from .. import hardware as H
from . import common as C
from .lock_stock import cut_stock
from .marine_dogs import bearing_y
from .marine_linkage import _gear


def slider_progress(angle, radius, length):
    """Inward motion on the branch with the connecting rod pointing outward."""
    radicand=length*length-(radius*math.sin(angle))**2
    if radicand<=0:raise ValueError('Vault connecting rod cannot reach its slider')
    return radius+length-radius*math.cos(angle)-math.sqrt(radicand)


def crank_radius(stroke, length, angle):
    lo,hi=0.,length*.80
    if stroke<=0 or slider_progress(angle,hi,length)<=stroke:
        raise ValueError('Invalid vault crank stroke')
    for _ in range(70):
        mid=(lo+hi)/2
        if slider_progress(angle,mid,length)<stroke:lo=mid
        else:hi=mid
    return (lo+hi)/2


def _bearing(body,name,center,material,axis=(0,1,0),inner=.010,outer=.020,half=.008):
    """Real ring sectors; inner chord clearance is checked in native QA."""
    before=len(body.geoms)
    bearing_y(body,name,(0,0,0),material,inner=inner,outer=outer,half_length=half,semantic='mechanism')
    # bearing_y's bore is Y. Rotate its authored vertices/pose to the axis.
    from ..ir import quat_to_mat,mat_to_quat
    a=np.asarray(axis,float);a/=np.linalg.norm(a)
    old=np.array([0.,1.,0.]);dot=float(np.dot(old,a))
    q=QUAT_ID if dot>1-1e-12 else quat_from_axis_angle(np.cross(old,a),math.acos(dot))
    R=quat_to_mat(q)
    for geom in body.geoms[before:]:
        geom.pos=tuple(np.asarray(center)+R@np.asarray(geom.pos))
        geom.quat=tuple(mat_to_quat(R@quat_to_mat(geom.quat)))
    return [g.name for g in body.geoms[before:]]


def _bar(body,name,a,b,r,material,semantic='mechanism'):
    a,b=np.asarray(a,float),np.asarray(b,float);d=b-a
    body.geoms.append(C.cyl(name,tuple((a+b)/2),r,float(np.linalg.norm(d))/2,material,
                            tuple(d),7850,True,True,ALL_TIERS,semantic,name.replace('_',' ')))
    return name


def _stock_cut(model,leaf,lo,hi,name):
    before=sum(g.mass() for g in leaf.geoms if g.semantic in ('leaf','glass'))
    cut=cut_stock(leaf,lo,hi,name)
    after=sum(g.mass() for g in leaf.geoms if g.semantic in ('leaf','glass'))
    cut['removed_mass_kg']=before-after
    model.meta.setdefault('vault_stock_cuts',[]).append(cut)
    return cut


def _shaft(model,leaf,body,mount,t,steel,*,both=False,front=.14,outside_radius=.04):
    x,_,z=body.pos
    low=-(t/2+front);high=t/2+.14 if both else -t/2+.040
    body.geoms.append(C.cyl(body.name+'_spindle',(0,(low+high)/2,0),.009,(high-low)/2,
                            steel,(0,1,0),7850,True,True,ALL_TIERS,'mechanism','Keyed retained operator spindle'))
    _stock_cut(model,leaf,(x-.0105,-t/2-.001,z-.0105),(x+.0105,(t/2+.001 if both else -t/2+.042),z+.0105),body.name+'_shaft_bore')
    planes=[-t/2-.016,-t/2-.074]
    if both:planes.append(t/2+.016)
    for k,y in enumerate(planes):
        _bearing(mount,f'{body.name}_bearing_{k}',(0,y,0),steel)
        if k==0 or y>0:
            # Ring back face sits on actual stock around the aperture.
            mount.geoms.append(C.box(f'{body.name}_bearing_foot_{k}',(.030,y,0),(.012,abs(y)-t/2,.023),
                                    steel,7850,True,True,ALL_TIERS,'mechanism','Bearing flange welded to prepared leaf'))
        else:
            support_z=-.040 if outside_radius<.10 else 0.
            mount.geoms.append(C.box(f'{body.name}_bearing_stay_{k}',(outside_radius,(-t/2+y)/2,support_z),(.012,(-t/2-y)/2,.023),
                                    steel,7850,True,True,ALL_TIERS,'mechanism','Outer bearing support to leaf'))
            mount.geoms.append(C.box(f'{body.name}_bearing_bridge_{k}',((outside_radius+.020)/2,y,0),((outside_radius-.020)/2+.003,.006,.010),
                                    steel,7850,True,True,ALL_TIERS,'mechanism','Bearing flange bridge outside gear sweep'))
            if support_z:
                mount.geoms.append(C.box(f'{body.name}_bearing_return_{k}',(outside_radius,y,support_z/2),(.012,.006,abs(support_z)/2+.010),
                                        steel,7850,True,True,ALL_TIERS,'mechanism','Rear-plane return connecting low bearing stay'))
        for sign in (-1,1):
            yy=y+sign*.012
            body.geoms.append(C.cyl(f'{body.name}_retainer_{k}_{sign}',(0,yy,0),.014,.003,steel,
                                    (0,1,0),7850,True,True,ALL_TIERS,'mechanism','Axial shaft collar'))


def _wheel(body,t,steel):
    y=-(t/2+.14);radius=.20;tube=.011
    for k in range(32):
        a,b=2*math.pi*k/32,2*math.pi*(k+1)/32
        _bar(body,f'wheel_rim_{k}',(radius*math.cos(a),y,radius*math.sin(a)),
             (radius*math.cos(b),y,radius*math.sin(b)),tube,steel,'operator')
    for k in range(5):
        a=2*math.pi*k/5
        _bar(body,f'wheel_spoke_{k}',(0,y,0),(radius*math.cos(a),y,radius*math.sin(a)),.007,steel,'operator')
    for k in range(8):
        a=2*math.pi*(4*k+.5)/32
        grip_radius=radius*math.cos(math.pi/32)  # actual straight rim segment midpoint
        body.sites.append(Site(f'wheel_grip_{k}',(grip_radius*math.cos(a),y-tube,grip_radius*math.sin(a)),
                               tuple(quat_z_to((0,-1,0))),.008,'grip'))


def _lever(body,t,u,steel):
    for face,tag in ((-1,'n'),(1,'p')):
        y=face*(t/2+.14)
        _bar(body,f'{body.name}_lever_{tag}',(0,y,0),(-u*.22,y,0),.0125,steel,'operator')
        body.sites.append(Site(f'{body.name}_grip_{tag}',(-u*.20,y+face*.0125,0),
                               tuple(quat_z_to((0,face,0))),.008,'grip'))


def _crank_rod(model,leaf,crank,mount,*,name,u,t,steel,stroke,angle,bolt_z,edge,bolt_radius=.016):
    length=.20;radius=crank_radius(stroke,length,angle);lane=-(t/2+.102)
    x,_,z=crank.pos;carrier_x=x+u*(radius+length)
    crank.geoms.append(C.box(name+'_crank_arm',(u*radius/2,lane+.009,0),
                            (radius/2,.004,.012),steel,7850,True,True,ALL_TIERS,'mechanism','Keyed crank to real rod pin'))
    _bar(crank,name+'_crank_pin',(u*radius,lane-.007,0),(u*radius,lane+.011,0),.004,steel)
    crank.geoms.append(C.cyl(name+'_pin_cap',(u*radius,lane-.008,0),.007,.002,steel,(0,1,0),7850,
                            True,True,ALL_TIERS,'mechanism','Retained crank pin'))
    carrier=model.add_body(Body(name+'_carrier',leaf.name,(carrier_x,0,z),
        joint=Joint(name+'_slide','slide',(-u,0,0),range=(-.003,stroke+.003),damping=20.*len(bolt_z),
                    frictionloss=25.*len(bolt_z),robot_interactive=False,role='lock'),
        semantic='lock',label='Rigid bolt carrier on bored guides'))
    low,high=min(bolt_z)-z,max(bolt_z)-z
    carrier_lane=lane+.015
    carrier.geoms.append(C.box(name+'_carrier_bar',(0,carrier_lane,(low+high+.032)/2),(.012,.008,(high-low+.068)/2),
                              steel,7850,True,True,ALL_TIERS,'lock','Rigid multi-bolt connecting bar'))
    _bar(carrier,name+'_carrier_pin',(0,lane-.007,0),(0,lane+.025,0),.004,steel)
    carrier.geoms.append(C.cyl(name+'_carrier_pin_cap',(0,lane-.008,0),.007,.002,steel,(0,1,0),7850,
                              True,True,ALL_TIERS,'mechanism','Retained carrier pin'))
    rod=model.add_body(Body(name+'_rod',crank.name,(u*radius,lane,0),
        joint=Joint(name+'_rod_hinge','hinge',(0,-u,0),range=(-math.pi,math.pi),damping=.005,
                    frictionloss=.005,role='mechanism',robot_interactive=False),
        semantic='mechanism',label='Rigid rod with two bored pin eyes'))
    rod.geoms.append(C.box(name+'_rod_bar',(u*length/2,0,0),(length/2-.007,.003,.008),
                           steel,7850,True,True,ALL_TIERS,'mechanism','Steel pin-connected rod'))
    for label,xx in (('a',0.),('b',u*length)):
        _bearing(rod,name+'_eye_'+label,(xx,0,0),steel,inner=.0047,outer=.009,half=.003)
    model.equalities.append(Equality('connect',name+'_rod_pin',rod.name,carrier.name,
        anchor=(u*length,0,0),tiers=ALL_TIERS,solref=(.002,1.),solimp=(.999,.9999,.0001),
        label='Actual rod eye on carrier pin; no remote follower'))
    bolts=[];guides=[]
    for k,zz in enumerate(bolt_z):
        localz=zz-z;start=carrier_x-u*.012;end=edge+u*stroke
        center=(start+end)/2;half=abs(end-start)/2
        bolt=f'{name}_bolt_{k}'
        carrier.geoms.append(C.cyl(bolt,(center-carrier_x,0,localz),bolt_radius,half,steel,(u,0,0),7850,
                                   True,True,ALL_TIERS,'lock','Solid steel locking bolt'))
        carrier.geoms.append(C.box(bolt+'_web',(0,carrier_lane/2,localz),(.010,abs(carrier_lane)/2,.010),steel,7850,
                                   True,True,ALL_TIERS,'lock','Continuous carrier-to-bolt steel web'))
        # Separate local cavities preserve the heavy panel between bolt rows.
        xa,xb=sorted((start-u*(stroke+.002),edge+u*.001))
        _stock_cut(model,leaf,(xa,-.018,zz-.018),(xb,.018,zz+.018),bolt+'_path')
        xa,xb=sorted((carrier_x-u*stroke-u*.012,carrier_x+u*.012))
        _stock_cut(model,leaf,(xa,-t/2-.001,zz-.012),(xb,.018,zz+.012),bolt+'_web_slot')
        for j,xx in enumerate((edge-u*.012,edge-u*.033)):
            _stock_cut(model,leaf,(xx-.008,-.026,zz-.026),(xx+.008,.026,zz+.026),bolt+f'_bush_cut_{j}')
            guides+=_bearing(mount,bolt+f'_guide_{j}',(xx-x,0,zz-z),steel,axis=(1,0,0),inner=bolt_radius+.0014,outer=.0265,half=.008)
        bolts.append(bolt)
    # Actual translational end stops: operator limits only guard numerical
    # overtravel. The upper bar tab loads fixed brackets tied to intact stock.
    tab_z=high+.060
    tab=name+'_carrier_stop_tab'
    carrier.geoms.append(C.box(tab,(0,carrier_lane,tab_z),(.012,.008,.010),steel,7850,
                              True,True,ALL_TIERS,'mechanism','Carrier tab against physical end stops'))
    stops=[]
    for label,xx in (('thrown',carrier_x+u*.019),('released',carrier_x-u*(stroke+.019))):
        stop=name+'_stop_'+label;stops.append(stop)
        mount.geoms.append(C.box(stop,(xx-x,carrier_lane+.011,tab_z),(.007,.019,.010),steel,7850,
                                True,True,ALL_TIERS,'mechanism','Carrier end stop'))
        mount.geoms.append(C.box(stop+'_mount',(xx-x,(carrier_lane+.030-t/2)/2,tab_z),
                                (.014,(-t/2-(carrier_lane+.030))/2,.014),steel,7850,True,True,ALL_TIERS,
                                'mechanism','End-stop bracket welded to intact leaf face'))
    # At the extended dead centre, the slider alone cannot prevent a crank
    # turning through zero. A separate physical crank stop supplies that path.
    stop=name+'_crank_return_stop';xx=u*radius*.7;yy=lane+.009
    mount.geoms.append(C.box(stop,(xx,yy,-.020),(.010,.008,.008),steel,7850,True,True,ALL_TIERS,
                            'mechanism','Physical zero-angle crank stop'))
    outside=u*.135
    mount.geoms.append(C.box(stop+'_front_bridge',((xx+outside)/2,yy,-.030),(abs(outside-xx)/2+.010,.008,.010),
                            steel,7850,True,True,ALL_TIERS,'mechanism','Crank stop support beyond gear plane'))
    mount.geoms.append(C.box(stop+'_stand',(outside,(yy-t/2)/2,-.030),(.010,(-t/2-yy)/2,.010),
                            steel,7850,True,True,ALL_TIERS,'mechanism','Crank stop stand outside gear perimeter'))
    for first,second in [(tab,other) for other in stops]+[(name+'_crank_arm',stop)]:
        model.meta.setdefault('native_contact_pairs',[]).append({'geom1':first,'geom2':second,
            'solref':[.001,1.],'solimp':[.99,.999,.0001],'friction':[.05,.05,.0001,.00001,.00001]})
    model.meta.setdefault('mechanism_mass_bodies',[]).extend([carrier.name,rod.name])
    return {'name':name,'input_joint':crank.joint.name,'carrier_joint':carrier.joint.name,'rod_joint':rod.joint.name,
            'crank_body':crank.name,'rod_body':rod.name,'carrier_body':carrier.name,'connect':name+'_rod_pin',
            'crank_radius_m':radius,'rod_length_m':length,'angle_rad':angle,'stroke_m':stroke,'u':u,
            'bolt_geoms':bolts,'guide_geoms':guides,'contact_pin_geoms':[name+'_crank_pin',name+'_carrier_pin'],
            'carrier_stop_tab':tab,'carrier_stop_geoms':stops,'crank_stop_pair':[name+'_crank_arm',stop],
            'scope':'Actual keyed crank, bored rod eyes and rigid bolt carrier; ideal hinge/slide bearings.'}


def _crane_journals(model,leaf,spec,steel):
    u=float(model.meta['u']);v=float(model.meta['v']);t=spec['leaf']['thickness'];height=spec['leaf']['height']
    pin=np.asarray(leaf.joint.pos,float);world=model.body('world_env');hx=leaf.pos[0]
    moving=model.add_body(Body('vault_crane_arms',leaf.name,semantic='mechanism',label='Bored crane sleeves and leaf arms'))
    rows=[]
    levels=[.35,.05+height-.30] if spec['hinge']['count']==2 else [.35,.05+height/2,.05+height-.30]
    for k,z in enumerate(levels):
        center=(pin[0],pin[1],z)
        sleeves=_bearing(moving,f'vault_crane_{k}_sleeve',center,steel,axis=(0,0,1),inner=.032,outer=.043,half=.080)
        moving.geoms.append(C.box(f'vault_crane_{k}_arm',(u*.1325,pin[1],z),(.0875,.020,.048),steel,7850,
                                  True,True,ALL_TIERS,'hinge','Steel crane arm keyed to bored sleeve'))
        moving.geoms.append(C.box(f'vault_crane_{k}_leaf_mount',(u*.200,v*(t/2+.0225),z),(.055,.0225,.060),steel,7850,
                                  True,True,ALL_TIERS,'hinge','Crane arm welded to leaf face'))
        shaft=f'vault_crane_{k}_journal'
        world.geoms.append(C.cyl(shaft,(hx+pin[0],pin[1],z),.030,.145,steel,(0,0,1),7850,
                                 True,True,ALL_TIERS,'hinge','Frame-mounted crane journal'))
        fixed=[]
        for side in (-1,1):
            zz=z+side*.120
            n=f'vault_crane_{k}_frame_{side}';fixed.append(n)
            world.geoms.append(C.box(n,(hx-u*.048,v*(t/2+.0475),zz),(.032,.0475,.020),steel,7850,
                                     True,True,ALL_TIERS,'hinge','Journal bearing block welded into vault jamb'))
            world.geoms.append(C.cyl(f'vault_crane_{k}_thrust_{side}',(hx+pin[0],pin[1],z+side*.083),.045,.0025,
                                     steel,(0,0,1),7850,True,True,ALL_TIERS,'hinge','Retained thrust washer below/above rotating sleeve'))
        rows.append({'journal':shaft,'sleeves':sleeves,'frame_blocks':fixed,'arm':f'vault_crane_{k}_arm',
                     'leaf_mount':f'vault_crane_{k}_leaf_mount','journal_radius_m':.030,'bore_inner_vertex_radius_m':.032})
    model.meta.setdefault('mechanism_mass_bodies',[]).append(moving.name)
    model.meta['vault_crane_journals']=rows


def _fixed_pulls(model,leaf,spec,steel,u):
    t=spec['leaf']['thickness'];x=u*(.006+spec['leaf']['width']-.055);z=spec['leaf']['height']/2
    body=model.add_body(Body('vault_fixed_pulls',leaf.name,semantic='mechanism',label='Independent fixed leaf-opening grips'))
    for face,tag in ((-1,'n'),(1,'p')):
        y=face*(t/2+.065)
        for k,zz in enumerate((z-.12,z+.12)):
            body.geoms.append(C.box(f'vault_pull_pad_{tag}_{k}',(x,face*(t/2+.003),zz),(.026,.003,.028),
                                   steel,7850,True,True,ALL_TIERS,'operator','Pull grip foot on leaf stock'))
            _bar(body,f'vault_pull_standoff_{tag}_{k}',(x,face*(t/2+.006),zz),(x,y,zz),.012,steel,'operator')
        _bar(body,f'vault_pull_bar_{tag}',(x,y,z-.12),(x,y,z+.12),.012,steel,'operator')
        leaf.sites.append(Site(f'vault_leaf_grip_{tag}',(x,y+face*.012,z),tuple(quat_z_to((0,face,0))),.008,'grip'))
    model.meta.setdefault('mechanism_mass_bodies',[]).append(body.name)


def _account_prepared_stock(model,phys,spec):
    """Deduct actual prepared stock before the common mass reconciler runs."""
    import copy
    mass=phys['mass'];rows=mass['per_body']
    if len(rows)!=1 or rows[0]['body']!='leaf' or model.meta.get('vault_material_accounting'):
        raise ValueError('Vault material deduction needs one fresh leaf budget')
    row=rows[0];removed=sum(c['removed_mass_kg'] for c in model.meta['vault_stock_cuts'])
    if not (0<=removed<row['slab_kg']):raise ValueError('Vault prepared stock exceeds leaf material')
    old=copy.deepcopy(row);replaced={key:row['hardware_parts'].get(key,0.) for key in ('operator','hinges_half')}
    hardware=sum(replaced.values());delta=removed+hardware
    for key in replaced:
        row['hardware_parts'][key]=0.;mass['hardware_parts'][key]-=replaced[key]
    row['slab_kg']-=removed;row['hardware_kg']-=hardware;row['total_kg']-=delta
    mass['slab_kg']-=removed;mass['hardware_kg']-=hardware;mass['total_kg']-=delta
    mass['dynamics_mass_kg']=row['total_kg'];mass['slab_area_density_kg_m2']=row['slab_kg']/(row['width']*row['height'])
    row.update(removed_stock_kg=removed,catalogue_mechanisms_replaced_kg=replaced)
    reference=mass['reference_unit'];mass['uncut_reference_unit']=copy.deepcopy(reference)
    reference.update(slab_kg=row['slab_kg'],hardware_kg=row['hardware_kg'],total_kg=row['total_kg'],
                     slab_area_density_kg_m2=mass['slab_area_density_kg_m2'],hardware_parts=copy.deepcopy(row['hardware_parts']))
    reference['formula']+='; subtract exact prepared stock and replaced operator/crane allowances'
    phys['per_body_dynamics']['leaf']['mass'].update(copy.deepcopy(row),dynamics_mass_kg=row['total_kg'])
    model.meta['vault_material_accounting']={'removed_stock_kg':removed,'removed_geometry_volume_m3':sum(c['removed_geometry_volume_m3'] for c in model.meta['vault_stock_cuts']),
        'replaced_catalogue_kg':replaced,'uncut_row':old,'material_row_before_mechanism_BOM':copy.deepcopy(row),
        'density_source':'Each exact cut piece retains its original authored geom mass density; successive subtractions do not count intersecting cuts twice.',
        'hinge_friction_scope':'Prototype retains the full authored pre-cut primary friction; native mechanism inertia is separately accounted. No friction reduction is used to obtain release.'}


def rebuild_vault_hardware(model,spec,phys):
    """Replace the old boltwork in a freshly built, not-yet-reconciled model."""
    if spec['family'] not in ('vault','blast'):raise ValueError('Vault/blast source required')
    leaf=model.body('leaf');u=float(model.meta['u']);v=float(model.meta['v']);t=spec['leaf']['thickness'];height=spec['leaf']['height'];width=spec['leaf']['width']
    old={b.name for b in model.bodies if b.name=='wheel' or b.name.startswith(('bolt_','dog_'))}
    model.bodies=[b for b in model.bodies if b.name not in old]
    model.equalities=[e for e in model.equalities if not e.name.startswith(('wheel_bolt_','lever_bolt_'))]
    leaf.geoms=[g for g in leaf.geoms if not g.name.startswith('hinge_')]
    steel=C.mat_from_material(model,'stainless','mat_vault_mechanism')
    edge=u*(.006+width);nominal_throw=.05
    strike_gap=max(0.,float(spec['opening']['width'])-width-.006)
    added_travel=max(0.,strike_gap-.004)
    stroke=nominal_throw+added_travel
    wheel=H.OPERATORS[spec['operator']['model']].kind=='wheel';groups=[];backed=[]
    if wheel:
        nominal_throw=float(H.LATCHES[spec['latch']['model']].throw)
        stroke=nominal_throw+added_travel
        x=u*(.006+width*.55);z=height*.5
        nominal=H.OPERATORS['wheel_vault'].travel
        driver=model.add_body(Body('wheel',leaf.name,(x,0,z),joint=Joint('wheel_hinge','hinge',(0,u,0),range=(-.05,nominal+.1),damping=.1,frictionloss=5.,role='operator'),semantic='operator'))
        mount=model.add_body(Body('vault_wheel_mount',leaf.name,(x,0,z),semantic='mechanism'))
        _shaft(model,leaf,driver,mount,t,steel);_wheel(driver,t,steel)
        output=model.add_body(Body('vault_reduction',leaf.name,(x,0,z+.12),joint=Joint('vault_reduction_hinge','hinge',(0,-u,0),range=(-.02,nominal/5+.02),damping=.05,frictionloss=.05,role='mechanism',robot_interactive=False),semantic='mechanism'))
        output_mount=model.add_body(Body('vault_reduction_mount',leaf.name,output.pos,semantic='mechanism'))
        _shaft(model,leaf,output,output_mount,t,steel,front=.11,outside_radius=.125)
        _gear(driver,'vault_input_gear',-(t/2+.043),.020,20,steel)
        _gear(output,'vault_output_gear',-(t/2+.043),.100,100,steel,phase=math.pi/100)
        model.equalities.append(Equality('joint','vault_spur_ratio',output.joint.name,driver.joint.name,(0,.2,0,0,0),tiers=ALL_TIERS,
                                solref=(.002,1.),solimp=(.999,.9999,.0001),label='Ideal keyed 20:100 spur reduction; no tooth compliance/strength model'))
        n=spec['kinematics']['bolts'];zs=[.05+height*(k+.5)/n for k in range(n)]
        group=_crank_rod(model,leaf,output,output_mount,name='vault_bolts',u=u,t=t,steel=steel,stroke=stroke,angle=nominal/5,bolt_z=zs,edge=edge,bolt_radius=.0125 if n==4 else .016)
        group.update(operator_joint=driver.joint.name,operator_body=driver.name,operator_nominal_range=[0.,nominal],ratio=.2,gear_equality='vault_spur_ratio')
        groups.append(group);backed.extend([driver.name,mount.name,output.name,output_mount.name])
        model.meta['operator_joint']=driver.joint.name
    else:
        for k,z in enumerate((.05+height*.25,.05+height*.75)):
            x=edge-u*(.20+crank_radius(stroke,.20,1.5708)+.10)
            driver=model.add_body(Body(f'dog_{k}',leaf.name,(x,0,z),joint=Joint(f'dog_{k}_hinge','hinge',(0,-u,0),range=(-.03,1.6108),damping=1.,frictionloss=6.,role='lock'),semantic='operator'))
            mount=model.add_body(Body(f'vault_lever_mount_{k}',leaf.name,driver.pos,semantic='mechanism'))
            _shaft(model,leaf,driver,mount,t,steel,both=True);_lever(driver,t,u,steel)
            group=_crank_rod(model,leaf,driver,mount,name=f'vault_bolt_{k}',u=u,t=t,steel=steel,stroke=stroke,angle=1.5708,bolt_z=[z],edge=edge)
            group.update(operator_joint=driver.joint.name,operator_body=driver.name,operator_nominal_range=[0.,1.5708],ratio=1.)
            groups.append(group);backed.extend([driver.name,mount.name])
        model.meta['operator_joint']='dog_0_hinge'
    model.meta.setdefault('mechanism_mass_bodies',[]).extend(backed)
    _crane_journals(model,leaf,spec,steel)
    _fixed_pulls(model,leaf,spec,steel,u)
    model.meta.setdefault('physical_inertia_joints',[]).extend(b.joint.name for b in model.bodies if b.joint and b.name!='leaf')
    model.meta['vault_boltwork']={'groups':groups,'manual_force_cap_N':66.7,'catalogue_throw_m':nominal_throw,'closed_leaf_to_jamb_gap_m':strike_gap,'gap_compensation_travel_m':added_travel,'actual_bolt_travel_m':stroke,'scope':'Prepared local stock; supported shafts; ideal spur pair only where declared; native rods, pins, carriers and frame bolt contacts. No blast/security/strength certification.'}
    model.meta['native_timestep_s']=min(.00025,model.meta.get('native_timestep_s',.002))
    model.meta['native_arena_memory_mib']=max(64,model.meta.get('native_arena_memory_mib',16))
    # A thick vault needs a real closing rebate across its enlarged running
    # gap. The generic 11 mm moulding otherwise misses the leaf entirely.
    from dataclasses import replace
    stops=[];frame_geoms=[]
    for g in model.body('world_env').geoms:
        if g.name.startswith('stop_strike'):
            jamb_x=u*spec['opening']['width']/2
            inner=leaf.pos[0]+edge-u*.020
            g.pos=((jamb_x+inner)/2,g.pos[1],g.pos[2])
            g.size=(abs(jamb_x-inner)/2,g.size[1],g.size[2])
            g.label='Frame-backed vault closing rebate with 20 mm leaf overlap'
            # The off-axis thick leaf's non-swing-side pull crosses this plane
            # during initial opening. Use two actual backed plates with a clear
            # handgrip window, not a solid rebate through the grip trajectory.
            low,high=g.pos[2]-g.size[2],g.pos[2]+g.size[2]
            gaplow,gaphigh=height/2-.18,height/2+.18
            for tag,a,b in (('lower',low,min(high,gaplow)),('upper',max(low,gaphigh),high)):
                if b>a:
                    part=replace(g,name=g.name+'_'+tag,pos=(g.pos[0],g.pos[1],(a+b)/2),size=(g.size[0],g.size[1],(b-a)/2))
                    frame_geoms.append(part);stops.append(part.name)
        else:frame_geoms.append(g)
    model.body('world_env').geoms=frame_geoms
    if not stops:raise ValueError('Vault requires an actual closing rebate')
    model.meta['vault_closing_stops']=stops
    model.meta['vault_primary_nominal_range']=list(leaf.joint.range)
    leaf.joint.range=(-.01,leaf.joint.range[1])
    _account_prepared_stock(model,phys,spec)
    return model.meta['vault_boltwork']


def resolve_vault_configuration(model,qpos,meta):
    """Inspection-only exact crank/rod branch; never called during physics."""
    def at(name):return int(model.jnt_qposadr[model.joint(name).id])
    for row in meta.get('vault_boltwork',{}).get('groups',[]):
        theta=float(qpos[at(row['operator_joint'])])*row['ratio']
        qpos[at(row['input_joint'])]=theta
        qpos[at(row['carrier_joint'])]=slider_progress(theta,row['crank_radius_m'],row['rod_length_m'])
        # Rod absolute inclination is -asin(r sin(theta)/L). Its local
        # hinge counter-rotates relative to the crank parent.
        qpos[at(row['rod_joint'])]=-math.asin(row['crank_radius_m']*math.sin(theta)/row['rod_length_m'])-theta
