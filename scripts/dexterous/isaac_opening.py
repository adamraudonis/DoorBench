#!/usr/bin/env python3
"""Live PhysX robot motor-control experiment. Saved controls are never saved poses.

The initial seed is written once during reset. Every simulated transition is
computed by Isaac Sim, including the free robot base, contacts and passive door.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--robot-usd',required=True)
p.add_argument('--robot-source-prim',default='/H1')
p.add_argument('--door-usd',required=True)
p.add_argument('--motors',required=True)
p.add_argument('--reference',required=True)
p.add_argument('--output',required=True)
p.add_argument('--seconds',type=float,default=12.)
p.add_argument('--record',action='store_true')
p.add_argument('--time-scale',type=float,default=1.,help='Slower motor-reference clock; physics dt is unchanged')
p.add_argument('--view',choices=['wide','hand'],default='wide')
p.add_argument('--upright-gain',type=float,default=0.,help='Post-opening IMU ankle feedback; bounded robot motors only')
p.add_argument('--native-robot',help='Enable closed-loop kinematic teacher using this native robot XML for FK only')
p.add_argument('--grip-rotation-fraction',type=float,default=1.,help='Fraction of operator rotation tracked by palm orientation; physical contacts remain unconstrained')
p.add_argument('--arm-impedance',type=float,default=1.,help='Software arm position-gain multiplier at the 500 Hz motor loop; native force caps remain unchanged')
p.add_argument('--grip-impedance',type=float,default=1.,help='Finger position-gain multiplier; native force caps remain unchanged')
p.add_argument('--grip-reset-targets',action='store_true',help='Hold reset finger posture instead of a frozen native-policy action')
p.add_argument('--finger-curl',type=float,default=0.,help='Additional bounded tendon curl target in radians')
p.add_argument('--grip-force',type=float,default=0.,help='Privileged inward finger-force reference in N, applied only through bounded motors; gates pressing on measured opposing contacts')
p.add_argument('--torso-damping',type=float,default=0.,help='Additional bounded waist velocity feedback in Nm s/rad')
p.add_argument('--stance-qp',action='store_true',help='Privileged inverse-dynamics motor controller for standing')
p.add_argument('--press-feedforward',action='store_true',help='Task-space pressure via bounded robot motors')
p.add_argument('--mechanism-test',action='store_true',help='Non-robot calibration: apply known forces directly to door joints')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
for name in ('arm_impedance','time_scale','seconds'):
    value=getattr(a,name)
    if not math.isfinite(value) or value<=0:p.error(f'--{name.replace("_","-")} must be finite and positive')
if not math.isfinite(a.grip_force) or a.grip_force<0:p.error('--grip-force must be finite and nonnegative')
if not math.isfinite(a.grip_impedance) or a.grip_impedance<0:p.error('--grip-impedance must be finite and nonnegative')
if not math.isfinite(a.finger_curl):p.error('--finger-curl must be finite')
if not math.isfinite(a.torso_damping) or a.torso_damping<0:p.error('--torso-damping must be finite and nonnegative')
if a.record:a.enable_cameras=True
launcher=AppLauncher(a);app=launcher.app
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from doorbench.dexterous.contact_audit import opposition
from doorbench.dexterous.reset import check_joint_reset
from doorbench.dexterous.isaac_materials import bind_robot_contact_material,check_solver_materials,check_solver_offsets
from pxr import Usd,UsdPhysics,UsdGeom,UsdShade,Sdf,PhysxSchema,PhysicsSchemaTools,Gf
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from omni.physx import get_physx_simulation_interface


def add_latch_tendon(stage):
    """Passive one-sided length constraint; angular gearing uses metres/degree."""
    prefix='/World/Door/Articulation/Joints/'
    prims={n:stage.GetPrimAtPath(prefix+n) for n in ('leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide')}
    scale=float(prims['leaf_latch_bolt_slide'].GetAttribute('doorbench:latch_coupling_scale').Get())
    for name,c in [('leaf_hinge',0.),('leaf_handle_hinge',-scale),('leaf_latch_bolt_slide',1.)]:
        prim=prims[name]
        if name=='leaf_hinge':
            PhysxSchema.PhysxTendonAxisRootAPI.Apply(prim,'latch')
            axis=PhysxSchema.PhysxTendonAxisAPI(prim,'latch')
        else:axis=PhysxSchema.PhysxTendonAxisAPI.Apply(prim,'latch')
        angular=prim.IsA(UsdPhysics.RevoluteJoint)
        axis.CreateGearingAttr([c*np.pi/180 if angular else c])
        axis.CreateForceCoefficientAttr([c])
    root=PhysxSchema.PhysxTendonAxisRootAPI.Apply(prims['leaf_hinge'],'latch')
    root.CreateStiffnessAttr(0.)
    root.CreateDampingAttr(0.)
    root.CreateLimitStiffnessAttr(100000.)
    root.CreateLowerLimitAttr(0.)
    root.CreateUpperLimitAttr(10.)
    root.CreateRestLengthAttr(0.)
    return scale


def main():
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    ref=json.loads(Path(a.reference).read_text());motors=json.loads(Path(a.motors).read_text())
    dt=.002
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt,device=a.device,
        render_interval=20,physx=sim_utils.PhysxCfg(solver_type=1,min_position_iteration_count=32,
                                               min_velocity_iteration_count=8)))
    stage=sim.stage
    # Isaac Lab disables CPU contact reporting until a ContactSensor is created.
    # This adapter consumes raw solved contacts instead of that sensor wrapper.
    sim.carb_settings.set_bool('/physics/disableContactProcessing',False)
    if getattr(sim,'_app_control_on_stop_handle',None) is not None:
        sim._app_control_on_stop_handle.unsubscribe();sim._app_control_on_stop_handle=None
    # The importer saved a whole stage; reference its H1 subtree explicitly.
    root=UsdGeom.Xform.Define(stage,'/World/H1').GetPrim()
    root.GetReferences().AddReference(str(Path(a.robot_usd).resolve()),a.robot_source_prim)
    sim_utils.UsdFileCfg(usd_path=str(Path(a.door_usd).resolve())).func('/World/Door',sim_utils.UsdFileCfg(usd_path=str(Path(a.door_usd).resolve())))
    # The exported door file has a default prim; its authored world floor is retained.
    # door.usda already has a solid floor at z=0; a second floor doubles contacts.
    sim_utils.DomeLightCfg(intensity=1800.).func('/World/Light',sim_utils.DomeLightCfg(intensity=1800.))
    # Diagnostic gold makes black finger pads distinguishable from the lever.
    material=UsdShade.Material.Define(stage,'/World/DiagnosticHandle')
    shader=UsdShade.Shader.Define(stage,'/World/DiagnosticHandle/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.55,.27,.045))
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(.7)
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.35)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Door/Articulation/leaf_handle')):
        if prim.IsA(UsdGeom.Gprim):UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    contact_material_audit=bind_robot_contact_material(stage,'/World/H1',motors.get('contact_material'))
    (out/'contact-material-audit.json').write_text(json.dumps(contact_material_audit,indent=2)+'\n')
    roots=[]
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            if prim.GetName()=='worldBody' and not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(prim)):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            else:roots.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive=UsdPhysics.DriveAPI.Apply(prim,'angular')
            drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(0.)
            for schema in list(prim.GetAppliedSchemas()):
                if 'Tendon' in schema:prim.RemoveAPI(getattr(PhysxSchema,schema.split(':')[0]),schema.split(':')[1])
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateDisableGravityAttr(False);rb.CreateMaxDepenetrationVelocityAttr(.5)
            rb.CreateMaxAngularVelocityAttr(5729.58)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            c=PhysxSchema.PhysxCollisionAPI.Apply(prim);c.CreateContactOffsetAttr(.001);c.CreateRestOffsetAttr(0.)
    assert len(roots)==1,roots
    art_api=PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(roots[0]))
    art_api.CreateEnabledSelfCollisionsAttr(True)
    art_api.CreateSolverPositionIterationCountAttr(32);art_api.CreateSolverVelocityIterationCountAttr(8)
    scale=add_latch_tendon(stage)
    joint_effort_limits={name:0. for name in motors['joint_names']}
    for motor in motors['actuators']:
        for name,coefficient in motor['terms'].items():
            joint_effort_limits[name]+=abs(coefficient)*max(abs(x) for x in motor['force_range'])
    assert all(limit>0 for limit in joint_effort_limits.values())
    robot=Articulation(ArticulationCfg(prim_path='/World/H1',spawn=None,
        articulation_root_prim_path=roots[0][len('/World/H1'):],
        actuators={'motor':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,effort_limit_sim=joint_effort_limits)}))
    door=Articulation(ArticulationCfg(prim_path='/World/Door',spawn=None,
        articulation_root_prim_path='/Articulation',
        actuators={'passive':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=None,damping=None)}))
    camera=None;writer=None
    if a.record:
        from isaaclab.sensors import Camera,CameraCfg
        camera=Camera(CameraCfg(prim_path='/World/Camera',update_period=0.,height=720,width=960,
            data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=48. if a.view=='hand' else 24.,clipping_range=(.02,100.))))
    contacts=[]
    all_contacts=[]
    report_counts=[0,0]
    def contact_report(headers,data):
        report_counts[0]+=len(headers);report_counts[1]+=len(data)
        for h in headers:
            paths=[str(PhysicsSchemaTools.intToSdfPath(getattr(h,k))) for k in ('collider0','collider1')]
            if not any('Door' in x for x in paths) or not any('H1' in x for x in paths):continue
            for c in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]:
                contacts.append(dict(colliders=paths,position=list(c.position),normal=list(c.normal),
                                     force_N=float(np.linalg.norm(c.impulse)/dt),separation_m=float(c.separation)))
    subscription=get_physx_simulation_interface().subscribe_contact_report_events(contact_report)
    sim.reset()
    # Imported collider instances are invisible to ordinary PrimRange traversal.
    # Set every solver shape at setup and verify the values actually used by PhysX.
    contact_offsets=robot.root_physx_view.get_contact_offsets()
    rest_offsets=robot.root_physx_view.get_rest_offsets()
    contact_material_audit['imported_contact_offset_range_m']=[float(contact_offsets.min()),float(contact_offsets.max())]
    robot.root_physx_view.set_rest_offsets(torch.zeros_like(rest_offsets),torch.tensor([0],dtype=torch.int32))
    robot.root_physx_view.set_contact_offsets(torch.full_like(contact_offsets,.001),torch.tensor([0],dtype=torch.int32))
    contact_material_audit.update(check_solver_offsets(robot.root_physx_view.get_contact_offsets().cpu().numpy(),
                                                       robot.root_physx_view.get_rest_offsets().cpu().numpy()))
    contact_material_audit.update(check_solver_materials(robot.root_physx_view.get_material_properties()[0].cpu().numpy(),motors['contact_material']))
    (out/'contact-material-audit.json').write_text(json.dumps(contact_material_audit,indent=2)+'\n')
    sim.carb_settings.set_bool('/physics/disableContactProcessing',False)
    print('CONTACT_REPORTING_ENABLED '+str(sim.carb_settings.get('/physics/disableContactProcessing')),flush=True)
    hand_pattern='/World/H1/pelvis/rh_*'
    hand_bodies=sim.physics_sim_view.create_rigid_body_view(hand_pattern)
    hand_contacts=sim.physics_sim_view.create_rigid_contact_view(hand_pattern,
        filter_patterns=['/World/Door/Articulation/leaf_handle','/World/Door/Articulation/leaf'],max_contact_data_count=4096)
    hand_paths=list(hand_bodies.prim_paths)
    print('TACTILE_BODIES '+json.dumps(hand_paths),flush=True)
    grip_prim=stage.GetPrimAtPath('/World/Door/Articulation/leaf_handle/leaf_handle_lever_col_n')
    grip=UsdGeom.Capsule(grip_prim)
    assert grip,'This development contact audit requires the named lever capsule'
    grip_center=np.array(grip_prim.GetAttribute('xformOp:translate').Get())
    gq=grip_prim.GetAttribute('xformOp:orient').Get()
    grip_axis=np.array(gq.Transform(Gf.Vec3f(0,0,1)))
    grip_radius=float(grip.GetRadiusAttr().Get());grip_half=float(grip.GetHeightAttr().Get())/2
    if camera:
        eye,target=([.9,-3.5,1.65],[.05,0.,.95]) if a.view=='wide' else ([.02,-.38,1.09],[.26,-.08,.914])
        camera.set_world_poses_from_view(eyes=torch.tensor([eye],device=a.device),
                                         targets=torch.tensor([target],device=a.device))
        import imageio.v2 as imageio
        writer=imageio.get_writer(out/'live-isaac.mp4',fps=25,codec='libx264',quality=8)
    rnames=list(robot.joint_names);dnames=list(door.joint_names)
    assert set(rnames)==set(motors['joint_names']),(set(rnames)^set(motors['joint_names']))
    index={n:i for i,n in enumerate(rnames)}
    q=torch.tensor([[ref['initial_joints'][n] for n in rnames]],device=a.device)
    limits=robot.root_physx_view.get_dof_limits()[0].cpu().numpy()
    check_joint_reset(rnames,q[0].cpu().numpy(),limits)
    robot.write_joint_state_to_sim(q,torch.zeros_like(q))
    initial_root=list(ref['initial_root'])
    if a.mechanism_test:initial_root[1]-=2.
    robot.write_root_pose_to_sim(torch.tensor([initial_root],device=a.device))
    robot.write_root_velocity_to_sim(torch.zeros((1,6),device=a.device))
    arm=torch.tensor([[motors['passive'][n]['armature'] for n in rnames]],device=a.device)
    robot.write_joint_armature_to_sim(arm)
    robot.write_joint_friction_coefficient_to_sim(torch.zeros_like(q))
    matrix=np.zeros((len(motors['actuators']),len(rnames)))
    for i,motor in enumerate(motors['actuators']):
        for n,c in motor['terms'].items():matrix[i,index[n]]=c
    kp=np.array([m['kp'] for m in motors['actuators']]);bias=np.array([m['bias'] for m in motors['actuators']])
    right_fingers=np.array([motor['name'].startswith('rh_') and 'WRJ' not in motor['name'] for motor in motors['actuators']])
    right_arm=np.array([(motor['name'].startswith('right_') and not any(n in motor['name'] for n in ('hip','knee','ankle'))) or motor['name'].startswith('rh_A_WRJ') for motor in motors['actuators']])
    arm_gain=right_arm*(a.arm_impedance-1)+right_fingers*(a.grip_impedance-1)
    extra_damping=np.array([.03 if finger and gain!=0 else (.8 if 'WRJ' in motor['name'] else 10.) if gain>0 else 0. for motor,gain,finger in zip(motors['actuators'],arm_gain,right_fingers)])
    for i,motor in enumerate(motors['actuators']):
        if motor['name']=='torso':extra_damping[i]+=a.torso_damping
    reset_lengths=matrix@q[0].cpu().numpy()
    curl_motors=[i for i,motor in enumerate(motors['actuators']) if motor['name'] in ['rh_A_'+f+'J0' for f in ('FF','MF','RF','LF')]]
    force_ranges=np.array([m['force_range'] for m in motors['actuators']])
    damp=np.array([motors['passive'][n]['damping'] for n in rnames])
    friction=np.array([motors['passive'][n]['friction'] for n in rnames])
    target=torch.zeros((1,len(dnames)),device=a.device)
    for i,n in enumerate(dnames):
        prim=stage.GetPrimAtPath('/World/Door/Articulation/Joints/'+n)
        target[0,i]=prim.GetAttribute('doorbench:target_si').Get() or 0.
    controls=np.array(ref['controls']);rows=[]
    teacher=None;teacher_info={};teacher_control=None
    if a.native_robot:
        from physx_teacher import HandleTeacher
        teacher=HandleTeacher(a.native_robot,motors,ref,stance_qp=a.stance_qp,grip_rotation_fraction=a.grip_rotation_fraction,grip_force=a.grip_force)
    release_time=None
    ankle_motors=[i for i,motor in enumerate(motors['actuators']) if any(n in motor['terms'] for n in ('left_ankle','right_ankle'))]
    (out/'configuration.json').write_text(json.dumps(dict(args=vars(a),robot_joint_names=rnames,door_joint_names=dnames,
        dt=dt,robot_mass_kg=float(robot.root_physx_view.get_masses().sum()),latch_scale=scale,
        simulator_effort_limits=robot.root_physx_view.get_dof_max_forces()[0].cpu().tolist(),
        runtime_pose_writes=0,direct_door_commands=bool(a.mechanism_test),contact_material_audit=contact_material_audit,
        scope='Direct-force mechanism calibration; NOT robot opening' if a.mechanism_test else 'Privileged near-handle motor reference; live PhysX; no traversal'),indent=2)+'\n')
    sources=[Path(__file__),Path(__file__).with_name('physx_teacher.py')]+[Path(__file__).resolve().parents[2]/'doorbench/dexterous'/n for n in ('stance.py','reset.py','contact_audit.py','isaac_materials.py')]
    inputs=[Path(a.robot_usd),Path(a.door_usd),Path(a.motors),Path(a.reference)]
    if a.native_robot:inputs.append(Path(a.native_robot))
    (out/'provenance.json').write_text(json.dumps(dict(captured_before_steps_unix=time.time(),
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+inputs if p.exists()},
        camera_note='Diagnostic gold handle material; physical properties unchanged'),indent=2)+'\n')
    for source in sources:
        if source.exists():(out/('source-'+source.name)).write_bytes(source.read_bytes())
    stage.GetRootLayer().Export(str((out/'scene.usda').resolve()))
    time_origin=float(sim.current_time)
    for step in range(round(a.seconds/dt)):
        pos=robot.data.joint_pos[0].cpu().numpy();vel=robot.data.joint_vel[0].cpu().numpy()
        ctrl=controls[min(int(step*dt*50/a.time_scale),len(controls)-1)].copy()
        if teacher:
            if step%10==0:
                body=door.data.body_state_w[0,:,:7].cpu().numpy()
                teacher_control,teacher_info=teacher.command(step*dt,robot.data.root_state_w[0].cpu().numpy(),dict(zip(rnames,pos)),
                    body[door.body_names.index('leaf')],body[door.body_names.index('leaf_handle')],
                    float(door.data.joint_pos[0,dnames.index('leaf_handle_hinge')]),float(door.data.joint_pos[0,dnames.index('leaf_hinge')]),
                    velocities=dict(zip(rnames,vel)),hand_forces=dict(zip(hand_paths,hand_contacts.get_contact_force_matrix(dt=dt).cpu().numpy().sum(axis=1))),grasp=rows[-1]['grasp_opposition'] if rows else None)
            ctrl=teacher_control.copy()
        if teacher_info.get('phase') not in ('release','withdraw'):
            if a.grip_reset_targets:ctrl[right_fingers]=reset_lengths[right_fingers]
            ctrl[curl_motors]+=a.finger_curl
        if a.upright_gain:
            gravity=robot.data.projected_gravity_b[0].cpu().numpy()
            pitch=float(np.arctan2(gravity[0],-gravity[2]))
            rate=float(robot.data.root_ang_vel_b[0,1])
            blend=np.clip(step*dt/1.5,0.,1.)
            ctrl[ankle_motors]+=blend*np.clip(a.upright_gain*pitch+.1*rate,-.3,.3)
        ctrl=np.clip(ctrl,[m['control_range'][0] for m in motors['actuators']],[m['control_range'][1] for m in motors['actuators']])
        lengths=matrix@pos;speeds=matrix@vel
        feedforward=teacher.feedforward if teacher and a.press_feedforward else np.zeros(len(kp))
        impedance=kp*arm_gain*(ctrl-lengths)-extra_damping*speeds
        forces=np.clip(kp*ctrl+bias[:,0]+bias[:,1]*lengths+bias[:,2]*speeds+feedforward+impedance,force_ranges[:,0],force_ranges[:,1])
        torque=matrix.T@forces-damp*vel-friction*np.tanh(vel/.001)
        robot.set_joint_effort_target(torch.tensor(torque[None],device=a.device,dtype=torch.float32))
        door.set_joint_position_target(target);door.set_joint_velocity_target(torch.zeros_like(target))
        if a.mechanism_test:
            effort=torch.zeros_like(target)
            effort[0,dnames.index('leaf_handle_hinge')]=1.5 if step*dt>.3 else 0.
            effort[0,dnames.index('leaf_hinge')]=3. if step*dt>1.5 else 0.
            door.set_joint_effort_target(effort)
        robot.write_data_to_sim();door.write_data_to_sim()
        contacts.clear();sim.step(render=False)
        if camera and step%20==0:
            if a.view=='hand':
                # Move only the diagnostic camera; never the robot or door.
                pose=door.data.body_state_w[0,door.body_names.index('leaf_handle'),:7].cpu().numpy()
                hrot=Rotation.from_quat([*pose[4:7],pose[3]]).as_matrix()
                center=pose[:3]+hrot@grip_center
                camera.set_world_poses_from_view(eyes=torch.tensor([[.15,-.38,1.45]],device=a.device,dtype=torch.float32),
                                                targets=torch.tensor(np.array([center]),device=a.device,dtype=torch.float32))
            sim.render()
        if abs(float(sim.current_time)-time_origin-(step+1)*dt)>.0001:
            raise RuntimeError('Physics clock changed outside the explicit motor timestep')
        robot.update(dt);door.update(dt)
        all_contacts.extend(dict(time_s=(step+1)*dt,**c) for c in contacts)
        if step==0:
            (out/'initial-body-poses.json').write_text(json.dumps(dict(
                robot=dict(zip(robot.body_names,robot.data.body_state_w[0,:,:7].cpu().tolist())),
                door=dict(zip(door.body_names,door.data.body_state_w[0,:,:7].cpu().tolist()))),indent=2)+'\n')
        if step%10==0:
            state=robot.data.root_state_w[0].cpu().numpy()
            up=robot.data.projected_gravity_b[0].cpu().numpy()
            row=dict(time_s=(step+1)*dt,sim_time_s=float(sim.current_time)-time_origin,root=state.tolist(),torso_tilt_deg=float(np.degrees(np.arccos(np.clip(-up[2],-1,1)))),
                     joints=robot.data.joint_pos[0].cpu().tolist(),door=dict(zip(dnames,door.data.joint_pos[0].cpu().tolist())),
                     contacts=list(contacts),max_motor_force=float(abs(forces).max()))
            measured=hand_contacts.get_contact_force_matrix(dt=dt).cpu().numpy()
            row['hand_forces_N']=dict(zip(hand_paths,measured.sum(axis=1).tolist()))
            row['hand_forces_handle_N']=dict(zip(hand_paths,measured[:,0,:].tolist()))
            row['hand_forces_panel_N']=dict(zip(hand_paths,measured[:,1,:].tolist()))
            force,point,normal,distance,count,start=[x.cpu().numpy() for x in hand_contacts.get_contact_data(dt)]
            pose=door.data.body_state_w[0,door.body_names.index('leaf_handle'),:7].cpu().numpy()
            rot=Rotation.from_quat([*pose[4:7],pose[3]]).as_matrix();center=pose[:3]+rot@grip_center;axis=rot@grip_axis
            detailed=[]
            for i,path in enumerate(hand_paths):
                for k in range(int(start[i,0]),int(start[i,0]+count[i,0])):
                    delta=point[k]-center;axial=float(delta@axis);radial=float(np.linalg.norm(delta-axial*axis))
                    digit=path.rsplit('/',1)[-1][3:5]
                    on_grip=abs(axial)<=grip_half and abs(radial-grip_radius)<.004
                    detailed.append(dict(body=path,digit=digit,position=point[k].tolist(),normal=normal[k].tolist(),
                        normal_force_N=float(force[k,0]),separation_m=float(distance[k,0]),on_grip_surface=on_grip))
            row['hand_contacts']=detailed
            row['grasp_opposition']=opposition([c for c in detailed if c['on_grip_surface']],center,axis)
            row['teacher']=teacher_info
            row['motor_targets']=ctrl.tolist()
            row['motor_feedforward']=feedforward.tolist()
            row['motor_forces']=forces.tolist()
            row['joint_torque_command']=torque.tolist()
            row['joint_torque_sent']=robot._joint_effort_target_sim[0].cpu().tolist()
            rows.append(row)
            if step%250==0:
                progress={k:row[k] for k in ('time_s','door','torso_tilt_deg','teacher')}
                progress['hand_force_N']=float(np.linalg.norm(measured,axis=-1).sum())
                progress['grasp_opposition']=row['grasp_opposition']
                (out/'latest.json').write_text(json.dumps(row)+'\n')
                (out/'progress.json').write_text(json.dumps(progress)+'\n')
                print('PHYSX_PROGRESS '+json.dumps(progress),flush=True)
        if camera and step%20==0:
            camera.update(dt*20);frame=camera.data.output['rgb'][0].cpu().numpy()[...,:3]
            writer.append_data(frame)
            if step%500==0:imageio.imwrite(out/f'frame-{step:05d}.png',frame)
        if not torch.isfinite(robot.data.joint_pos).all():raise RuntimeError('Nonfinite robot state')
        if rows and (rows[-1]['root'][2]<.45 or rows[-1]['torso_tilt_deg']>45):
            (out/'early-stop.json').write_text(json.dumps(dict(reason='Robot fell',time_s=(step+1)*dt))+'\n')
            break
    (out/'trace.json').write_text(json.dumps(rows)+'\n')
    (out/'contacts.json').write_text(json.dumps(all_contacts)+'\n')
    (out/'contact-report-counts.json').write_text(json.dumps(report_counts)+'\n')
    if writer:writer.close()
    print('PHYSX_RUN_COMPLETE',flush=True)

failed=False
try:main()
except BaseException:
    failed=True
    import traceback
    traceback.print_exc()
    Path(a.output).mkdir(parents=True,exist_ok=True)
    (Path(a.output)/'error.txt').write_text(traceback.format_exc())
finally:app.close()
if failed:raise SystemExit(1)
