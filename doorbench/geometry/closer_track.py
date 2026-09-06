"""Generic single-arm track closer with an electrically retained roller detent.

LCN's4040SE installation establishes arm/track/roller/solenoid topology and
power-loss/test-switch release. Internal detent dimensions here are authored
engineering geometry, not recovered OEM CAD or a fire-safety certification.
"""
from __future__ import annotations
import math
import hashlib
import numpy as np
from ..ir import Body,Joint,Site,Equality,ALL_TIERS,QUAT_ID,quat_from_axis_angle
from . import common as C
from .closer_mounts import frame_face,frame_backing,shoe_ring


def track_curve(pinion,track_y,hinge,axis_sign,side,length,maximum,samples=721):
    P0=np.asarray(pinion,float)[:2];H=np.asarray(hinge,float)[:2]
    theta=np.linspace(0.,maximum,samples);heads=[];carriage=[]
    for angle in theta:
        a=axis_sign*angle;c,s=math.cos(a),math.sin(a);P=H+np.array([[c,-s],[s,c]])@(P0-H)
        dy=track_y-P[1]
        if abs(dy)>=length-.001:raise ValueError('Track arm cannot reach its native slider')
        dx=side*math.sqrt(length*length-dy*dy)
        carriage.append(P[0]+dx);heads.append(math.atan2(dy,dx)-a)
    q=np.unwrap(heads);q-=q[0];sign=float(np.sign(q[-1]));ratio=np.gradient(sign*q,theta,edge_order=2)
    if ratio.min()<.05:raise ValueError('Track pinion reaches a singular or reversing branch')
    return (theta,q,ratio,sign),np.asarray(carriage)


def add_track_closer(model,world,leaf,spec,phys,u,v,hx,Hh,t,Wo,jamb_t):
    """Connected native hinge+slider+connect topology, retained in every tier."""
    from .. import hardware as H
    from ..closer_pinion import configure_pinion
    cl=H.CLOSERS[spec['closer']['model']];mat=C.mat_from_material(model,'aluminum_dark','mat_closer')
    pfx='' if leaf.name=='leaf' else leaf.name+'_';n=lambda name:pfx+name
    def box(body,name,pos,size,*,collision=True,material=mat,density=2700):
        g=C.box(n(name),pos,size,material,density,collision,True,ALL_TIERS,'closer',name.replace('_',' '));body.geoms.append(g);return g.name
    l,w,h=cl.body_size;zc=Hh-.08;z=float(spec['opening']['height'])+.040
    leafpos=np.asarray(C.body_world_pos(model,leaf));xp=u*.25
    surface,mount=frame_face(world,hx+u*.46,leafpos[2]+z,v)
    stand=max(0.,v*(surface-leafpos[1])+.018-(t/2+h/2));yp=v*(t/2+stand+h/2)
    if stand>1e-6:box(leaf,'closer_mount_spacer',(xp,v*(t/2+stand/2),zc),(l/2,stand/2,w/2))
    box(leaf,'closer_body_col',(xp,yp,zc),(l/2,h/2,w/2),density=2000)
    by=surface+v*.036;bylocal=by-leafpos[1];L=.43;maximum=math.radians(float(spec['kinematics']['max_open_deg']))
    curve,xs=track_curve((xp,yp),bylocal,leaf.joint.pos,leaf.joint.axis[2],u,L,maximum)
    bx=float(xs[0]);th=math.atan2(bylocal-yp,bx-xp)
    arm=Body(n('closer_arm_main'),leaf.name,(xp,yp,z),tuple(quat_from_axis_angle((0,0,1),th)),tiers=ALL_TIERS,semantic='closer')
    arm.joint=Joint(n('closer_pinion'),'hinge',(0,0,1),(0,0,0),None,damping=.01,role='mechanism',robot_interactive=False)
    arm.geoms.append(C.cyl(n('closer_pinion_shaft'),(0,0,(zc-z)/2),.008,(z-zc)/2,mat,(0,0,1),7850,False,True,ALL_TIERS,'closer','Pinion shaft seated in housing'))
    box(arm,'closer_arm_main_geom',((L-.022)/2,0,0),((L-.022)/2,.008,.005),collision=True,density=7850)
    box(arm,'closer_shoe_neck',(L-.011,0,0),(.011,.003,.003),collision=True,density=7850)
    arm.geoms.append(C.cyl(n('closer_shoe_pivot'),(L,0,.0155),.005,.0175,mat,(0,0,1),7850,True,True,ALL_TIERS,'closer','Arm tip enters open slider bearing'))
    model.add_body(arm)
    # The carriage runs along a retained C channel; it is not a body floating
    # beside a decorative track. Its central bore surrounds the arm-tip pin.
    travel=(xs-bx)*u;low=float(travel.min())-.003;high=float(travel.max())+.003
    slider=Body(n('closer_track_slide'),None,(hx+bx,by,leafpos[2]+z),QUAT_ID,tiers=ALL_TIERS,semantic='closer')
    slider.joint=Joint(n('closer_track_joint'),'slide',(u,0,0),(0,0,0),(low,high),damping=.05,role='mechanism',robot_interactive=False)
    shoes=shoe_ring(slider.geoms,(0,0,.023),mat,name=n('closer_shoe_block'))
    model.add_body(slider)
    xlo=float(xs.min()+hx)-.025;xhi=float(xs.max()+hx)+.025;center=(xlo+xhi)/2;span=(xhi-xlo)/2
    # Full backplate reaches the structural header through a real spacer.
    plate=box(world,'closer_bracket',(center,surface+v*.004,leafpos[2]+z+.023),(span,.004,.022))
    # Duplicate backing only for this track; helper's standard32 mm patch is
    # extended to the complete real plate footprint.
    prior=len(world.geoms);mount,spacer=frame_backing(world,center,leafpos[2]+z+.023,.022,v,surface,mat)
    for geom in world.geoms[prior:]:geom.name=n(geom.name);geom.size=(span,geom.size[1],geom.size[2])
    spacer=n(spacer) if spacer else None
    for geom in world.geoms:
        if geom.name==mount:geom.tiers=ALL_TIERS
    guides=[]
    guides.append(box(world,'closer_track_top',(center,(surface+v*.008+by+v*.014)/2,leafpos[2]+z+.036),(span,abs(by+v*.014-surface-v*.008)/2,.003)))
    for dz in (.012,.034):
        guides.append(box(world,f'closer_track_front_{dz}',(center,by+v*.014,leafpos[2]+z+dz),(span,.003,.003)))
    # Retaining lips stay out of the central pin's14 mm slot.
    for side in (-1,1):
        guides.append(box(world,f'closer_track_lip_{side}',(center,by+side*.011,leafpos[2]+z+.010),(span,.004,.002)))
    for side,x in ((-1,xlo),(1,xhi)):
        guides.append(box(world,f'closer_track_end_{side}',(x,by,leafpos[2]+z+.023),(.003,.017,.014)))
    connect=n('closer_arm_connect')
    model.equalities.append(Equality('connect',connect,arm.name,slider.name,anchor=(L,0,0),tiers=ALL_TIERS,label='Arm tip pinned to native track carriage',solref=(.004,1.),solimp=(.99,.999,.001)))
    model.contact_excludes.append((arm.name,leaf.name))
    row={'schema':'doorbench.closer-mount.v1','kind':'single_arm_track','leaf_joint':leaf.joint.name,'leaf_body':leaf.name,
         'leaf_support_geoms':[g.name for g in leaf.geoms if g.semantic in ('leaf','glass')],
         'housing_spacer_geom':n('closer_mount_spacer') if stand>1e-6 else None,
         'main_geom':n('closer_arm_main_geom'),'fore_geom':n('closer_arm_main_geom'),'neck_geom':n('closer_shoe_neck'),
         'body_geom':n('closer_body_col'),'shaft_geom':n('closer_pinion_shaft'),'frame_plate':plate,'frame_geom':mount,'frame_spacer':spacer,
         'shoe_geoms':shoes,'pivot_geom':n('closer_shoe_pivot'),'main_joint':arm.joint.name,'track_joint':slider.joint.name,
         'connect':connect,'track_geoms':guides,'spacer_m':stand,'frame_surface_y_m':surface,'track_side':u,
         'scope':'Native single-arm pinion and frame slider; physically retained channel; solenoid roller detent'}
    model.meta.setdefault('closer_mounts',[]).append(row)
    report=configure_pinion(model,leaf,arm,spec,phys,pinion=(xp,yp),shoe=(bx,bylocal),elbow=None,lengths=None,curve_data=curve)
    report['mechanism']='native_pinion_spring_single_arm_track'
    phys['closer']['mechanism']=report['mechanism']
    add_track_detent(model,world,slider,row,curve,xs,spec,mat,u,v,leafpos[2]+z,by,hx)
    report['unmodeled_features']=[f for f in report['unmodeled_features'] if f!='hold_open']
    report['hold_open_scope']='Native capture at the selected point, powered holding, test-button and power-loss release, finite-force manual breakout; recapture following overtravel is not guaranteed'
    report['overtravel_recapture_validated']=False
    model.meta.setdefault('notes',[]).append('Single-arm track uses generic dimensioned components, informed by4040SE topology; no OEM/fire-safety certification.')
    return row


def resolve_track_configuration(model,q,row):
    """Initial/inspection pose only; native stepping never calls this solve."""
    import mujoco
    d=mujoco.MjData(model);d.qpos[:]=q;mujoco.mj_kinematics(model,d)
    main=model.joint(row['main_joint']).id;slide=model.joint(row['track_joint']).id
    a=int(model.jnt_qposadr[main]);sa=int(model.jnt_qposadr[slide]);eq=model.equality(row['connect']).id
    bmain=int(model.jnt_bodyid[main]);target=int(model.eq_obj2id[eq]);P=d.xanchor[main]
    B=d.xpos[target]+d.xmat[target].reshape(3,3)@model.eq_data[eq,3:6]
    axis=d.xaxis[slide];length=float(np.linalg.norm(model.eq_data[eq,:3]));v=B-P
    parallel=float(v@axis);perp=v-parallel*axis;rem=length*length-float(perp@perp)
    if rem<=1e-8:raise ValueError('Closer track circle cannot reach guide line')
    q[sa]+=math.sqrt(rem)-parallel
    if not model.jnt_range[slide,0]-1e-6<=q[sa]<=model.jnt_range[slide,1]+1e-6:raise ValueError('Closer track carriage would leave its physical rail')
    d.qpos[:]=q;mujoco.mj_kinematics(model,d);B=d.xpos[target]+d.xmat[target].reshape(3,3)@model.eq_data[eq,3:6]
    heading=math.atan2(B[1]-P[1],B[0]-P[0]);base=math.atan2(d.xmat[bmain].reshape(3,3)[1,0],d.xmat[bmain].reshape(3,3)[0,0])-q[a]
    q[a]=(heading-base+math.pi)%(2*math.pi)-math.pi
    d.qpos[:]=q;mujoco.mj_kinematics(model,d)
    endpoint=d.xpos[bmain]+d.xmat[bmain].reshape(3,3)@model.eq_data[eq,:3]
    if np.linalg.norm(endpoint-B)>2e-6:raise ValueError('Track initial pose fails actual native connect anchor')


def add_track_detent(model,world,slider,row,curve,xs,spec,mat,u,v,z,by,hx):
    """Real cam/roller contact carries hold load; a separate solenoid retains cam.

    Both cam flanks are beveled, allowing manual breakout. Removing power lets
    the return spring lift the cam clear. There is no leaf torque detent/weld.
    """
    import trimesh
    pfx='' if row['leaf_body']=='leaf' else row['leaf_body']+'_';n=lambda name:pfx+name
    theta=curve[0];maximum=float(theta[-1]);hold=min(math.pi/2,maximum)
    # Mount within the manufacturer's85–110 degree region, leaving enough
    # native travel to move the roller over the actual cam and enter its seat.
    if hold<math.radians(84.9):raise ValueError('Track has insufficient opening travel for its hold-open detent')
    xhold=hx+float(np.interp(hold,theta,xs));yroller=by+v*.065;zr=z+.023
    roller=Body(n('closer_track_roller'),slider.name,(0,v*.065,.023),QUAT_ID,tiers=ALL_TIERS,semantic='closer')
    roller.joint=Joint(n('closer_track_roller_spin'),'hinge',(0,1,0),(0,0,0),None,damping=.00002,role='mechanism',robot_interactive=False)
    roller.geoms.append(C.cyl(n('closer_detent_roller'),(0,0,0),.007,.004,mat,(0,1,0),7850,True,True,ALL_TIERS,'closer','Free rotating steel hold-open roller'))
    roller.geoms[0].friction=(.02,.0001,.00001);model.add_body(roller)
    # Seat the axle in the outer carriage wall (9 mm from its center), not
    # through the separate vertical arm-tip pin in the carriage's bore.
    slider.geoms.append(C.cyl(n('closer_roller_axle'),(0,v*.038,.023),.003,.029,mat,(0,1,0),7850,False,True,ALL_TIERS,'closer','Axle seated in carriage wall and rolling follower'))
    # Roller is on the opening side of the nose at its nominal held point.
    # This contact geometry, not an angle threshold, defines the actual seat.
    cx=xhold+u*(.007*math.sqrt(1+3.5**2)/3.5);cz=zr+.007
    # Coil block is behind the stem; side support legs connect it to the rail
    # backplate while leaving the central moving cam/roller workspace open.
    fixed=[]
    def box(body,name,pos,size,density=7850):
        g=C.box(n(name),pos,size,mat,density,True,True,ALL_TIERS,'closer',name.replace('_',' '),friction=(.04,.0001,.00001));body.geoms.append(g);return g.name
    back_y=by-v*.023
    for side in (-1,1):
        fixed.append(box(world,f'closer_solenoid_support_{side}',(cx+side*.024,(back_y+yroller)/2,cz+.040),(.004,abs(yroller-back_y)/2+.006,.006)))
    for side in (-1,1):
        fixed.append(box(world,f'closer_solenoid_back_connection_{side}',(cx+side*.024,back_y,cz+.011),(.004,.006,.035)))
    # Hollow pole and coil surround a real 8 mm stem passage.
    for axis in (0,1):
        for side in (-1,1):
            p=[cx,yroller,cz+.046];p[axis]+=side*.012
            half=[.008,.020,.020] if axis==0 else [.004,.008,.020]
            fixed.append(box(world,f'closer_solenoid_pole_{axis}_{side}',tuple(p),tuple(half)))
            world.geoms[-1].margin=.0001;world.geoms[-1].solref=(.002,1.);world.geoms[-1].solimp=(.99,.999,.0001)
    cam=Body(n('closer_hold_plunger'),None,(cx,yroller,cz),QUAT_ID,tiers=ALL_TIERS,semantic='closer')
    cam.joint=Joint(n('closer_hold_release'),'slide',(0,0,1),(0,0,0),(0.,.012),stiffness=1500.,springref=.010,damping=20.,limit_solref=(.002,1.),role='mechanism',robot_interactive=False)
    # The cam has a steep3.5:1 holding flank and a gentle14:25 entry ramp.
    # Holding force is sized through that mechanical advantage, not applied
    # directly to the door. The second cam provides the opposite flank.
    points=[(u*x,y,zz) for x,zz in [(-.004,.007),(0.,-.007),(.025,.007)] for y in (-.006,.006)]
    mesh=trimesh.convex.convex_hull(np.asarray(points,float))
    geom=C.mesh_geom(n('closer_hold_cam'),n('closer_hold_cam_mesh')+'_'+hashlib.sha256(mesh.vertices.tobytes()).hexdigest()[:12],mesh,(0,0,0),QUAT_ID,mat,7850,True,ALL_TIERS,'closer','Two-sided native cam for reversible roller detent');geom.friction=(.02,.0001,.00001);cam.geoms.append(geom)
    # The second cam makes a real roller pocket. Entering from either side
    # first lifts the plunger; it drops around the roller at the hold point.
    leftmesh=trimesh.convex.convex_hull(np.asarray([(-x,y,zz) for x,y,zz in points],float))
    leftkey=n('closer_hold_cam_left_mesh')+'_'+hashlib.sha256(leftmesh.vertices.tobytes()).hexdigest()[:12]
    left=C.mesh_geom(n('closer_hold_cam_left'),leftkey,leftmesh,(-2*u*(.007*math.sqrt(1+3.5**2)/3.5),0,0),QUAT_ID,mat,7850,True,ALL_TIERS,'closer','Opposite roller pocket cam');left.friction=(.02,.0001,.00001);cam.geoms.append(left)
    box(cam,'closer_hold_cam_bridge',(-u*.00728,0,.010),(.024,.006,.003))
    cam.geoms.append(C.cyl(n('closer_solenoid_stem'),(0,0,.039),.0035,.032,mat,(0,0,1),7850,False,True,ALL_TIERS,'closer','Continuous cam stem through solenoid bore'))
    armature=box(cam,'closer_solenoid_armature',(0,0,.068),(.020,.020,.002))
    model.add_body(cam)
    # Accessible momentary test switch is a separate real button. Depressing
    # it interrupts coil current; it does not move the leaf or the cam itself.
    button=Body(n('closer_test_button'),None,(cx+u*.052,yroller+v*.005,cz+.040),QUAT_ID,tiers=ALL_TIERS,semantic='operator')
    button.joint=Joint(n('closer_test_release'),'slide',(0,-v,0),(0,0,0),(0.,.005),stiffness=800.,springref=0.,damping=.3,role='operator',label='Interrupt closer hold-open power')
    button.geoms.append(C.cyl(n('closer_test_cap'),(0,0,0),.007,.003,mat,(0,v,0),1400,True,True,ALL_TIERS,'operator','Momentary electrical release button'))
    from ..ir import quat_z_to
    button.sites.append(Site(n('closer_test_push'),(0,v*.003,0),quat_z_to((0,v,0)),.004,'push'))
    model.add_body(button)
    for side in (-1,1):
        box(world,f'closer_test_switch_wall_x_{side}',(cx+u*.052+side*.010,yroller-v*.001,cz+.040),(.002,.012,.014),1400)
        box(world,f'closer_test_switch_wall_z_{side}',(cx+u*.052,yroller-v*.001,cz+.040+side*.012),(.008,.012,.002),1400)
    box(world,'closer_test_switch_back',(cx+u*.052,yroller-v*.006,cz+.040),(.012,.001,.014),1400)
    fixed.append(box(world,'closer_test_switch_mount',(cx+u*.040,(back_y+yroller-v*.008)/2,cz+.040),(.028,abs(yroller-v*.008-back_y)/2,.004)))
    model.contact_excludes.append((roller.name,slider.name))
    torque=float(np.interp(hold,theta,model.meta['closer_pinion_calibration'][-1]['table']['achieved_door_torque_Nm']))
    dx=np.gradient(xs,theta,edge_order=2);slider_force=torque/abs(float(np.interp(hold,theta,dx)))
    force=max(40.,1.8*slider_force/3.5+15.)
    law={'schema':'doorbench.track-hold.v1','leaf_joint':row['leaf_joint'],'plunger_joint':cam.joint.name,'button_joint':button.joint.name,
         'roller_geom':n('closer_detent_roller'),'cam_geom':n('closer_hold_cam'),'cam_geoms':[n('closer_hold_cam'),n('closer_hold_cam_left')],'armature_geom':armature,
         'pole_geoms':[g for g in fixed if 'pole' in g],'support_geoms':fixed,'button_site':n('closer_test_push'),'button_release_threshold_m':.003,
         'nominal_hold_angle_rad':hold,'coil_force_at_seat_N':force,'magnetic_gap_scale_m':.004,
         'powered_by_default':True,'required_slider_holding_force_N':slider_force,
         'scope':'Original physical roller/cam/return-spring/solenoid model; electrical input and magnetic field idealized; no OEM force calibration'}
    model.meta.setdefault('closer_track_holds',[]).append(law)
    return law
