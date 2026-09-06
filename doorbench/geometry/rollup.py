"""Original articulated rolling curtain, fixed barrel and flared guide mouths.

The material gauge, corrugation envelope and real reserve wraps are separate
quantities. All native motion is integrated through ordinary pin joints and
barrel/slat/guide contacts; a rigid translated sheet is never hidden in a hood.
"""
from __future__ import annotations
import math
import numpy as np
from ..ir import Body,Joint,Site,ALL_TIERS,QUAT_ID,quat_from_axis_angle
from .. import hardware as H
from . import common as C


def counterbalance_parameters(params,curtain_mass,bottom_mass,fraction):
    """Size a linear torsion spring from the two static endpoint loads.

    The winding radius grows as material winds onto the barrel. A continuous
    area-conservation spiral estimates that radius; actual motion/contact is
    still entirely native. The open bottom bar and remaining hanging curtain
    need nonzero spring torque, as in real pre-tensioned rolling-door springs.
    This is an original engineering approximation, not an OEM spring rating.
    """
    length=params['physical_curtain_length_m'];density=curtain_mass/length
    r0=params['joint_radius_m'];depth=params['profile_depth_m']
    closed=params['barrel_z_m']-params['closed_bottom_z_m']
    opened=params['barrel_z_m']-params['open_bottom_z_m']
    travel=closed-opened
    r1=math.sqrt(r0*r0+depth*travel/math.pi)
    angle=2*math.pi*(r1-r0)/depth
    closed_torque=fraction*(density*closed+bottom_mass)*9.81*r0
    open_torque=fraction*(density*opened+bottom_mass)*9.81*r1
    stiffness=(closed_torque-open_torque)/angle if fraction else 0.
    reference=closed_torque/stiffness if stiffness else 0.
    return {'fraction':fraction,'scope':'Authored assistance fraction at estimated static endpoints; original linear torsion spring with nonzero open pretension, not guaranteed balance throughout travel',
        'mass_basis_kg':curtain_mass+bottom_mass,'curtain_mass_kg':curtain_mass,'bottom_bar_mass_kg':bottom_mass,
        'closed_hanging_length_m':closed,'open_hanging_length_m':opened,'closed_effective_radius_m':r0,'open_effective_radius_m':r1,
        'estimated_open_angle_rad':angle,'closed_torque_Nm':closed_torque,'open_torque_Nm':open_torque,
        'stiffness_Nm_rad':stiffness,'springref_rad':reference,
        'formula':'r_open=sqrt(r_closed²+profile_depth*travel/pi); q_open=2*pi*(r_open-r_closed)/profile_depth; T=fraction*g*(linear_curtain_mass*hanging_length+bottom_mass)*r; k=(T_closed-T_open)/q_open',
        'source_scope':'Cookson installation manual p8 requires open-position pretension and endpoint balance; these dimensions and calculated rates are original, not manufacturer supplied.'}


def curtain_dimensions(spec):
    height=float(spec['leaf']['height']);width=float(spec['leaf']['width']);grille=spec['leaf']['slab']=='rollup_alu_grille'
    profile=.014 if not grille else .010
    radius=.14;barrel_z=height+.50;barrel_y=max(.40,spec['opening']['wall_thickness']/2+.30)
    straight=math.ceil((barrel_z-.02)/.075);pitch=(barrel_z-.02)/straight
    joint_radius=math.hypot(radius+profile/2+.001,pitch/2)
    angle=2*math.asin(pitch/(2*joint_radius));wrap_count=4
    theta0=math.pi-wrap_count*angle
    arc=[(barrel_y+joint_radius*math.cos(theta0+i*angle),barrel_z+joint_radius*math.sin(theta0+i*angle)) for i in range(wrap_count+1)]
    nodes=arc+[(barrel_y-joint_radius,barrel_z-i*pitch) for i in range(1,straight+1)]
    a=np.asarray(nodes);vectors=a[:-1]-a[1:];angles=np.arctan2(vectors[:,0],vectors[:,1])
    return {'width_m':width,'opening_height_m':float(spec['opening']['height']),'barrel_radius_m':radius,'barrel_y_m':barrel_y,'barrel_z_m':barrel_z,
        'profile_depth_m':profile,'material_gauge_m':float(spec['leaf']['thickness']),'grille':grille,
        'pitch_m':pitch,'slat_count':straight+wrap_count,'initial_wrap_slats':wrap_count,'joint_radius_m':joint_radius,
        'physical_curtain_length_m':pitch*(straight+wrap_count),'closed_bottom_z_m':.02,'open_bottom_z_m':height+.10,
        # Keep the complete bottom slat captured at the open stop. A flare
        # beginning below it permits the reinforced bar to pitch past the
        # stop angle, even though a lug briefly touches that angle.
        'guide_end_z_m':height+.17,'funnel_top_z_m':height+.26,'guide_y_m':barrel_y-joint_radius,
        'initial_nodes_yz':nodes,'initial_panel_angles_rad':angles.tolist(),'initial_attachment_angle_rad':theta0}


def _sheet_slat(model,body,spec,params,index,row):
    """Actual profile visual and labelled convex contact envelope.

    The closed convex envelope is appropriate to a corrugated steel strip's
    gross winding contact, but does not resolve its submillimetre folds. Grille
    holes remain physical holes and are never filled by this envelope.
    """
    width=params['width_m'];pitch=params['pitch_m'];depth=params['profile_depth_m'];material=C.mat_from_finish(model,spec['leaf']['finish'],'mat_rollup_curtain')
    if params['grille']:
        body.geoms.append(C.box(f'curtain_slat_{index}_rail',(0,0,-pitch/2),(width/2,depth/2,.0125),material,2700,
            True,True,ALL_TIERS,'leaf','Aluminium grille cross rail'))
        for k,x in enumerate(np.linspace(-width/2+.012,width/2-.012,max(2,math.ceil(width/.20)+1))):
            body.geoms.append(C.box(f'curtain_slat_{index}_link_{k}',(x,0,-pitch/2),(.0015,depth/2,pitch/2-.0005),material,2700,
                True,True,ALL_TIERS,'leaf','Grille articulation link'))
        body.geoms.append(C.cyl(f'curtain_slat_{index}_hinge_wire',(0,0,0),.002,width/2,material,(1,0,0),2700,
            True,True,ALL_TIERS,'hinge','Grille hinge wire'))
    else:
        points=np.array([[0,0],[-depth/2,-pitch*.2],[-depth/2,-pitch*.8],[0,-pitch]])
        for k,(a,b) in enumerate(zip(points[:-1],points[1:])):
            vector=np.r_[0.,b-a];mid=(a+b)/2
            body.geoms.append(C.obox(f'curtain_slat_{index}_sheet_{k}',(0,*mid),(1,0,0),(0,-vector[2],vector[1]),0,0,0,
                width/2,float(np.linalg.norm(vector))/2,params['material_gauge_m']/2,material,False,ALL_TIERS,'leaf','Corrugated steel sheet'))
        envelope=C.box(f'curtain_slat_{index}_envelope',(0,0,-pitch/2),(width/2,depth/2,pitch/2-.001),material,7850,
            True,False,ALL_TIERS,'leaf','Convex gross-contact envelope of corrugated slat',friction=(.05,.001,.0001),mass=0.)
        body.geoms.append(envelope)
    for geom in body.geoms:
        if geom.collision:
            geom.friction=(.08,.001,.0001);geom.solref=(.002,1.);geom.solimp=(.99,.999,.001)


def build_rollup(spec,phys,model):
    from .garage_tiltup import _bearing_eye
    p=curtain_dimensions(spec);width=p['width_m'];height=spec['leaf']['height'];radius=p['barrel_radius_m'];y=p['barrel_y_m'];z=p['barrel_z_m'];pitch=p['pitch_m'];count=p['slat_count']
    world=C.add_floor_and_wall(model,spec,wall_half_width=max(3.,width/2+1.),wall_height=z+.40)
    steel=C.mat_from_material(model,'steel_galvanized','mat_rollup_structure');aluminium=C.mat_from_material(model,'aluminum_dark','mat_rollup_barrel')
    guide=p['guide_y_m'];end=p['guide_end_z_m'];depth=p['profile_depth_m']
    for sign,tag in ((-1,'l'),(1,'r')):
        for face in (-1,1):
            world.geoms.append(C.box(f'curtain_guide_{tag}_{face}',(sign*(width/2-.008),guide+face*(depth/2+.009),end/2),(.025,.005,end/2),steel,7850,
                True,True,ALL_TIERS,'track','Curtain guide face',friction=(.08,.001,.0001),solref=(.002,1.)))
            a=np.array([sign*(width/2-.008),guide+face*(depth/2+.009),end]);b=np.array([a[0],guide+face*.09,p['funnel_top_z_m']]);delta=b-a
            from .sectional import _segment
            funnel=_segment(f'curtain_funnel_{tag}_{face}',a,b,.005,steel);funnel.solref=(.002,1.);funnel.solimp=(.99,.999,.001)
            world.geoms.append(funnel)
        world.geoms.append(C.box(f'curtain_guide_web_{tag}',(sign*(width/2+.021),guide,end/2),(.004,depth/2+.014,end/2),steel,7850,semantic='track',label='Outer guide web'))
        for j,zz in enumerate(np.linspace(.2,end-.1,5)):
            wall_back=spec['opening']['wall_thickness']/2
            world.geoms.append(C.box(f'curtain_guide_wall_bracket_{tag}_{j}',(sign*(width/2+.055),(wall_back+guide)/2,zz),(.026,(guide-wall_back)/2,.005),steel,7850,semantic='frame',label='Guide fixing to side jamb'))
        _bearing_eye(world,f'curtain_barrel_bearing_{tag}',(sign*(width/2+.11),y,z),steel,inner=.0125,outer=.045,half_length=.015)
        world.geoms.append(C.box(f'curtain_barrel_bracket_{tag}',(sign*(width/2+.20),(spec['opening']['wall_thickness']/2+y)/2,z),(.025,(y-spec['opening']['wall_thickness']/2)/2,.08),steel,7850,semantic='frame',label='Barrel bearing bracket anchored to header'))
        world.geoms.append(C.box(f'curtain_bearing_bridge_{tag}',(sign*(width/2+.17),y,z+.040),(.055,.025,.005),steel,7850,semantic='frame',label='Bearing bracket upper bridge'))
    mass=phys['mass']['total_kg'];cb=float(spec['kinematics'].get('counterbalance_fraction',0.))
    nominal_angle=(z-.25)/(radius+.02)
    # A coaxial spring is an actual torsion counterbalance. The closed force
    # fraction and original linear curve are stated explicitly; this is not
    # whole-travel force compensation or an actuator in disguise.
    preload=cb*mass*9.81*radius;stiffness=preload/nominal_angle if cb else 0.
    barrel=Body('curtain_barrel',None,(0,y,z),QUAT_ID,semantic='mechanism',label='Rotating winding barrel')
    barrel.joint=Joint('curtain_drum_hinge','hinge',(-1,0,0),range=(-.10,nominal_angle+2.),damping=1.,stiffness=stiffness,springref=nominal_angle,
        role='primary',robot_interactive=False,label='Real winding barrel; opening progress is measured at the bottom bar')
    shell_mass=2*math.pi*radius*width*.002*2700
    barrel.geoms.append(C.cyl('curtain_barrel_shell',(0,0,0),radius,width/2+.03,aluminium,(1,0,0),2700,True,True,ALL_TIERS,'mechanism','2 mm aluminium winding drum (convex contact shell)',mass=shell_mass))
    barrel.geoms[-1].friction=(.08,.001,.0001);barrel.geoms[-1].solref=(.002,1.);barrel.geoms[-1].solimp=(.99,.999,.001)
    barrel.geoms.append(C.cyl('curtain_barrel_axle',(0,0,0),.012,width/2+.16,steel,(1,0,0),7850,True,True,ALL_TIERS,'hinge','Supported steel barrel axle'))
    model.add_body(barrel);model.meta.setdefault('mechanism_mass_bodies',[]).append(barrel.name)
    # The spring sits on the barrel axis, between shaft and stationary anchor.
    world.geoms.append(C.cyl('curtain_torsion_anchor',(width/2+.19,y,z),.035,.025,steel,(1,0,0),7850,True,True,ALL_TIERS,'mechanism','Fixed torsion-spring anchor'))
    if cb:barrel.geoms.append(C.cyl('curtain_torsion_spring',(width/2-.12,0,0),.022,.12,steel,(1,0,0),7850,False,True,ALL_TIERS,'mechanism','Coaxial counterbalance spring',mass=.25))
    parent=barrel;panels=[];angles=p['initial_panel_angles_rad'];nodes=p['initial_nodes_yz'];rows={r['body']:r for r in phys['mass']['per_body']}
    for i in range(count):
        pos=(0,nodes[0][0]-y,nodes[0][1]-z) if i==0 else (0,0,-pitch)
        angle=angles[i] if i==0 else angles[i]-angles[i-1]
        panel=Body(f'curtain_slat_{i}',parent.name,pos,tuple(quat_from_axis_angle((-1,0,0),angle)),semantic='leaf',label=f'Interlocking curtain slat {i+1}')
        panel.joint=Joint(f'curtain_slat_{i}_hinge','hinge',(-1,0,0),range=(-1.2,1.2),damping=.03,role='mechanism',robot_interactive=False,
            label='Interlocking slat flexure')
        _sheet_slat(model,panel,spec,p,i,rows[panel.name]);panel.sites.append(Site(f'curtain_node_{i}',(0,0,0),role='mechanism'))
        model.add_body(panel);panels.append(panel);parent=panel
    bottom=panels[-1];bottom.sites.append(Site('curtain_bottom',(0,0,-pitch),role='mechanism'))
    bottom_bar=Body('curtain_bottom_reinforcement',bottom.name,semantic='mechanism',label='Formed bottom-bar reinforcement and seal')
    bottom_bar.geoms.append(C.box('curtain_bottom_bar',(0,0,-pitch+.035),(width/2-.050,.018,.040),steel,7850,True,True,ALL_TIERS,'leaf','Formed hollow steel bottom bar ending before side channels, 1.2 kg/m',mass=1.2*(width-.10)))
    rubber=C.mat_rgba(model,'mat_curtain_seal',(.07,.07,.08,1),.9)
    bottom_bar.geoms.append(C.box('curtain_astragal',(0,0,-pitch-.010),(width/2-.040,.015,.010),rubber,1100,True,True,ALL_TIERS,'seal','Bottom rubber seal clear of guide ends'))
    # Real up-stops prevent the bottom bar from winding into the bellmouth.
    # Their rear lugs bypass the running channels, then bear against angles
    # fixed to the guide posts. They do not limit a hidden primary coordinate.
    # The bellmouth flares 74 mm behind the running plane. Keep the lug
    # beyond that swept flange, and connect it inboard of the guide. A lug
    # merely behind the straight channel would catch the bellmouth first.
    # Slide-bolt connections also pass below the released keeper.
    lug_y=.105
    lug_z=.020 if spec['lock']['model']=='garage_slide_lock' else .035
    stop_z=p['open_bottom_z_m']+lug_z+.006
    for sign,tag in ((-1,'l'),(1,'r')):
        bottom_bar.geoms.append(C.box(f'curtain_stop_lug_standoff_{tag}',(sign*(width/2-.065),(.013+lug_y)/2,-pitch+lug_z),(.012,(lug_y-.013)/2,.006),steel,7850,semantic='mechanism',label='Bottom stop lug connection to reinforcement'))
        bottom_bar.geoms.append(C.box(f'curtain_stop_lug_{tag}',(sign*(width/2-.01),lug_y,-pitch+lug_z),(.045,.009,.006),steel,7850,semantic='mechanism',label='Bottom-bar up-stop lug behind guide channel, inboard of padlock shackle'))
        world.geoms.append(C.box(f'curtain_up_stop_{tag}',(sign*(width/2+.035),guide+.0825,stop_z+.006),(.025,.0675,.006),steel,7850,semantic='track',label='Full-open stop angle covering the captured lug sweep, fixed to guide post'))
        world.geoms.append(C.box(f'curtain_up_stop_mount_{tag}',(sign*(width/2+.06),guide+(lug_y-.009)/2,stop_z+.018),(.006,(lug_y+.021)/2,.022),steel,7850,semantic='frame',label='Up-stop angle anchor on guide post'))
        for geom in [*bottom_bar.geoms[-2:],*world.geoms[-2:]]:
            geom.solref=(.002,1.);geom.solimp=(.99,.999,.001)
        post_low=height-.10;post_high=stop_z+.040
        world.geoms.append(C.box(f'curtain_up_stop_post_{tag}',(sign*(width/2+.055),guide+.012,(post_low+post_high)/2),(.018,.022,(post_high-post_low)/2),steel,7850,semantic='frame',label='Stop extension post overlapping the upper jamb fixing bracket'))
    model.add_body(bottom_bar);model.meta['mechanism_mass_bodies'].append(bottom_bar.name)
    balance=counterbalance_parameters(p,mass,bottom_bar.inertial()[0],cb)
    barrel.joint.stiffness=balance['stiffness_Nm_rad'];barrel.joint.springref=balance['springref_rad']
    operator=_add_operator(model,spec,bottom,p,steel)
    if spec['operator']['model']=='none':
        balance=counterbalance_parameters(p,mass,bottom_bar.inertial()[0]+model.bodies[-1].inertial()[0],cb)
        barrel.joint.stiffness=balance['stiffness_Nm_rad'];barrel.joint.springref=balance['springref_rad']
    # Enclosed hood clears the complete growing coil, including its end faces.
    for side in (-1,1):_bearing_eye(world,f'curtain_hood_end_{side}',(side*(width/2+.065),y,z),steel,inner=.013,outer=.255,half_length=.012)
    world.geoms.append(C.box('curtain_hood_top',(0,y,z+.265),(width/2+.077,.267,.010),steel,7850,True,True,ALL_TIERS,'mechanism','Enclosed hood top'))
    world.geoms.append(C.box('curtain_hood_back',(0,y+.267,z),(width/2+.077,.010,.265),steel,7850,True,True,ALL_TIERS,'mechanism','Enclosed hood back'))
    world.geoms.append(C.box('curtain_hood_front',(0,y-.267,z+.07),(width/2+.077,.010,.195),steel,7850,True,True,ALL_TIERS,'mechanism','Hood front with lower curtain exit'))
    opener=spec['kinematics'].get('opener','none_manual')
    if opener=='motor_disengaged':
        world.geoms.append(C.box('curtain_disengaged_motor',(width/2+.31,y,z),(.10,.10,.12),aluminium,500,True,True,ALL_TIERS,'mechanism','Motor with mechanically disengaged clutch'))
    _add_locks(model,spec,bottom,world,p,steel)
    world.sites.extend([Site('approach_point',(0,-2.,0),role='approach'),Site('goal_point',(0,2.,0),role='goal'),Site('door_plane_center',(0,0,spec['opening']['height']/2),role='pass_plane')])
    model.meta.update({'primary_joint':'curtain_drum_hinge','operator_joint':None,'handle_height':.065,'counterbalance_fraction':cb,'native_timestep_s':.0005,
        'native_arena_memory_mib':64 if p['grille'] else 16,
        'rollup_curtain':{'schema_version':1,'kind':'native_barrel_wound_articulated_curtain','dimensions':p,'primary_joint':'curtain_drum_hinge',
            'slat_joints':[b.joint.name for b in panels],'slat_bodies':[b.name for b in panels],'manual_grip_site':'lift_handle_grip',
            'progress':{'site':'curtain_bottom','closed_z_m':.02,'open_z_m':height+.10},
            'up_stops':{'kind':'bottom_bar_lugs_against_fixed_guide_angles','lug_names':['curtain_stop_lug_l','curtain_stop_lug_r'],'stop_names':['curtain_up_stop_l','curtain_up_stop_r'],'nominal_bottom_z_m':p['open_bottom_z_m'],'rear_offset_m':lug_y},
            'drive':{'mode':'manual','opener':opener,'manual_max_force_N':120.,'chain_hoist_supported':False if opener=='chain_hoist' else None},
            'operator':operator,
            'counterbalance':balance,
            'collision_scope':'Corrugated steel uses labelled convex per-slat envelopes; grille apertures remain open. Original generic geometry, not manufacturer-certified CAD.',
            'reference':'https://www.cooksondoor.com/docs/default-source/o-m-manuals/service-door-installation-and-maintenance-manuals.pdf?sfvrsn=6'},
        'mechanical_export_support':{'mjcf':'Native articulated slat/barrel/guide contact mechanics','urdf':'Articulated bodies export; dynamic winding requires appropriate contact and self-collision configuration','usd':'Articulated bodies export; winding contact fidelity is not independently certified'}})
    if opener=='chain_hoist':
        from .rollup_hoist import add_chain_hoist
        model.meta['rollup_hoist']=add_chain_hoist(model,spec,p,barrel,world,steel)
        from .hoist_keeper import add_chain_keeper
        add_chain_keeper(model,spec)
        model.meta['native_arena_memory_mib']=64
        model.meta['rollup_curtain']['drive'].update(mode='manual_chain',chain_hoist_supported=True,
            force_application='Actual circulating material-link grip; opposite strands open and close')
        model.meta['mechanical_export_support'].update(
            urdf='Free material-chain root and gear coupling export; complete winding/chain contact dynamics require independent simulator configuration',
            usd='Free material-chain root is explicitly unsupported/fixed for inspection; no chain-hoist dynamics parity claim')
    return barrel


def _add_operator(model,spec,bottom,p,steel):
    """Preserve the authored pull type and its real offset force application.

    D-pulls are mounted horizontally on the bottom bar. Ring pulls have a
    native hinge and an open circular grip; lifting a ring is not a mandatory
    latch-release action. An unspecified operator receives a real lift handle.
    """
    selected=spec['operator']['model'];op=H.OPERATORS[selected];pitch=p['pitch_m']
    if op.kind=='ring_pull':
        # Stand the upward-folded ring clear of the next two slats as they
        # begin curving into the hood. A flush pivot lets the real ring snag
        # those slats during the native full-open/closing transition.
        pivot=(0.,-.065,-pitch+.13)
        bottom.geoms.append(C.box('curtain_ring_mount',(0.,-.057,-pitch+.09),(.020,.004,.055),steel,7850,semantic='operator',label='Ring hinge mounting strap attached to bottom bar'))
        bottom.geoms.append(C.box('curtain_ring_mount_spacer',(0.,-.0375,-pitch+.055),(.020,.0195,.020),steel,7850,semantic='operator',label='Ring mounting standoff welded to bottom reinforcement'))
        bottom.geoms.append(C.cyl('curtain_ring_pin',pivot,.004,.023,steel,(1,0,0),7850,True,True,ALL_TIERS,'operator','Horizontal ring pivot pin'))
        handle=Body('curtain_lift_handle',bottom.name,pivot,semantic='operator',label='Hinged ring lift pull')
        handle.joint=Joint('curtain_ring_hinge','hinge',(-1,0,0),range=(0.,math.pi),damping=.015,role='mechanism',robot_interactive=False,label='Free ring can lift upright; no latch release')
        material=C.mat_from_material(model,op.material,'mat_rollup_ring');radius=.06
        from .sectional import _segment
        from .garage_tiltup import _bearing_eye
        _bearing_eye(handle,'curtain_ring_bearing',(0.,0.,0.),steel,inner=.0045,outer=.0085,half_length=.012)
        points=[(radius*math.sin(a),-.012,-radius+radius*math.cos(a)) for a in np.linspace(0,2*math.pi,25)]
        for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
            handle.geoms.append(_segment(f'curtain_ring_segment_{i}',a,b,.007,material,'operator'))
        handle.sites.append(Site('lift_handle_grip',(0.,-.012,-2*radius),role='grip'));model.add_body(handle)
        return {'specified_model':selected,'realized_model':selected,'kind':'hinged_ring_pull','grip_site':'lift_handle_grip','turn_required':False,'free_articulation_joint':handle.joint.name}
    realized=selected if op.kind=='pull' else 'pull_lift_garage';op=H.OPERATORS[realized]
    # A 200 mm upright handle would extend below the floor when installed on
    # an 80 mm bottom bar. Its horizontal installation is explicit geometry.
    rotation=tuple(quat_from_axis_angle((0,1,0),math.pi/2)) if op.style_params.get('shape')=='d_pull' else QUAT_ID
    handle=Body('curtain_lift_handle',bottom.name,(0.,0.,-pitch+.045),rotation,semantic='operator',label='Horizontal bottom-bar lift handle')
    C.add_pull(model,handle,op,1.,0.,0.,.036,-1.,name='lift_handle')
    handle.sites.append(Site('lift_handle_grip',handle.sites[-1].pos,role='grip'));model.add_body(handle)
    if selected=='none':model.meta.setdefault('mechanism_mass_bodies',[]).append(handle.name)
    return {'specified_model':selected,'realized_model':realized,'kind':'fixed_pull','grip_site':'lift_handle_grip','turn_required':False}


def _add_locks(model,spec,bottom,world,p,steel):
    kind=spec['lock']['model'];width=p['width_m'];pitch=p['pitch_m'];guide=p['guide_y_m'];z=.065
    if kind=='garage_slide_lock':
        bolt,_=C.add_barrel_bolt(model,bottom,'rollup_slide_lock',(-width/2,.018,-pitch+.045),(-1,0,0),(0,1,0),.18,.012,.095,
            bool(spec['lock'].get('engaged')),steel,protrusion=.080,standoff=.035,tiers=ALL_TIERS,role='lock',rod_semantic='lock',joint_name='rollup_slide_lock_slide',grip_site='slide_lock_grip')
        C.add_keeper_loop(world.geoms,'rollup_lock_keeper',(-width/2-.055,guide+.018,z),(-width/2-.055,guide+.053,z),(-1,0,0),(0,1,0),.006,steel,ALL_TIERS,base=.03,bar=.005,bar_len=.014)
        world.geoms.append(C.box('rollup_lock_mount',(-width/2-.09,guide+.005,z),(.05,.013,.04),steel,7850,True,True,ALL_TIERS,'lock','Keeper mounting on guide post'))
        if spec['lock'].get('engaged') and not spec['lock'].get('robot_side_release'):bolt.joint.range=(0.,.001)
        model.meta['rollup_lock']={'kind':kind,'joint':bolt.joint.name,'grip_site':'slide_lock_grip','released_q':.095}
    elif kind=='padlock':
        from .garage_locks import add_tiltup_lock
        b0=len(model.bodies);g0=len(bottom.geoms);w0=len(world.geoms)
        add_tiltup_lock(model,bottom,world,spec,.02+pitch,mount_height=z)
        offset=.060+.018-spec['leaf']['thickness']/2
        for g in bottom.geoms[g0:]:g.pos=(g.pos[0],g.pos[1]+offset,g.pos[2])
        for b in model.bodies[b0:]:
            if b.parent==bottom.name:b.pos=(b.pos[0],b.pos[1]+offset,b.pos[2])
        for g in world.geoms[w0:]:g.pos=(g.pos[0],g.pos[1]+guide+offset,g.pos[2])
        bottom.geoms.append(C.box('rollup_hasp_standoff',(width/2-.14,.048,-pitch+.045),(.035,.030,.026),steel,7850,True,True,ALL_TIERS,'lock','Bottom-bar hasp spacer clear of guide'))
        model.meta['rollup_lock']=model.meta['garage_lock_hardware']
