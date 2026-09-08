#!/usr/bin/env python3
"""Live PhysX robot motor-control experiment. Saved controls are never saved poses.

The initial seed is written once during reset. Every simulated transition is
computed by Isaac Sim, including the free robot base, contacts and passive door.
"""
import argparse
import hashlib
import gzip
import json
import math
import time
import os
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
p.add_argument('--sensor-layout',help='Record finite robot-mounted sensors; teacher remains privileged')
p.add_argument('--time-scale',type=float,default=1.,help='Slower motor-reference clock; physics dt is unchanged')
p.add_argument('--view',choices=['wide','hand'],default='wide')
p.add_argument('--upright-gain',type=float,default=0.,help='Post-opening IMU ankle feedback; bounded robot motors only')
p.add_argument('--native-robot',help='Enable closed-loop kinematic teacher using this native robot XML for FK only')
p.add_argument('--acquisition',action='store_true',help='Execute the shared contact-free acquisition teacher from reference.path_qpos; privileged development only')
p.add_argument('--acquisition-middle-finger-force',type=float,help='Explicit acquisition middle-finger preload in N; original motor caps unchanged')
p.add_argument('--operate-after-acquisition',action='store_true',help='After 0.5 s of actual qualified grasp, press the lever and hold a partial opening through robot motors')
p.add_argument('--open-on-latch-clear',action='store_true',help='Start the smooth opening ramp on measured release, without waiting for the press-reference timer')
p.add_argument('--operator-compliance-gain',type=float,default=0.,help='Bounded palm-reference integral compensation for actual operator-angle error; motor and mechanism limits unchanged')
p.add_argument('--sensor-policy-checkpoint',help='Execute the recurrent actor using only robot sensor packets; no teacher fallback')
p.add_argument('--sensor-objective',choices=['acquisition','partial-opening'],default='partial-opening',help='Declared curriculum qualification; neither establishes traversal')
p.add_argument('--sensor-reset-preflight',help='Required frozen native reset receipt for a sensor-only actor')
p.add_argument('--reset-from-acquisition-path',action='store_true',help='Use the frozen contact-free first configuration at reset only')
p.add_argument('--grasp-profile',choices=['distal-pad-v1','volar-phalange-v1'],default='distal-pad-v1')
p.add_argument('--grasp-profile-definition',help='Frozen declaration required for an opt-in grasp profile')
p.add_argument('--full-sequence-reset',help='Start from the frozen walking reset and run continuous walk/lower/prepare/acquire/operate')
p.add_argument('--preparation-reference',help='Contact-free readiness path; screened again at the actual stopped pose')
p.add_argument('--locomotion-checkpoint',help='Frozen original H1 locomotion checkpoint')
p.add_argument('--native-door',help='Matching unstepped native door geometry for readiness collision checks')
p.add_argument('--grip-rotation-fraction',type=float,default=1.,help='Fraction of operator rotation tracked by palm orientation; physical contacts remain unconstrained')
p.add_argument('--arm-impedance',type=float,default=1.,help='Software arm position-gain multiplier at the 500 Hz motor loop; native force caps remain unchanged')
p.add_argument('--grip-impedance',type=float,default=1.,help='Finger position-gain multiplier; native force caps remain unchanged')
p.add_argument('--grip-reset-targets',action='store_true',help='Hold reset finger posture instead of a frozen native-policy action')
p.add_argument('--finger-curl',type=float,default=0.,help='Additional bounded tendon curl target in radians')
p.add_argument('--grip-force',type=float,default=0.,help='Privileged inward finger-force reference in N, applied only through bounded motors; gates pressing on measured opposing contacts')
p.add_argument('--torso-damping',type=float,default=0.,help='Additional bounded waist velocity feedback in Nm s/rad')
p.add_argument('--stance-qp',action='store_true',help='Privileged inverse-dynamics motor controller for standing')
p.add_argument('--press-feedforward',action='store_true',help='Task-space pressure via bounded robot motors')
p.add_argument('--panel-push',action='store_true',help='Development: reacquire open-palm panel contact after handle release, through bounded motors')
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
if a.acquisition and (not a.native_robot or a.panel_push or a.mechanism_test):p.error('Acquisition requires --native-robot and a separate acquisition-only trial')
if a.operate_after_acquisition and not a.acquisition:p.error('--operate-after-acquisition requires --acquisition')
if a.open_on_latch_clear and not a.operate_after_acquisition:p.error('--open-on-latch-clear requires --operate-after-acquisition')
from doorbench.dexterous.control_mode import validate_sensor_actor_mode
try:validate_sensor_actor_mode(a)
except ValueError as error:p.error(str(error))
if a.full_sequence_reset and (not a.operate_after_acquisition or not all((a.preparation_reference,a.locomotion_checkpoint,a.native_door))):
    p.error('Full sequence requires acquisition, operation, preparation, locomotion checkpoint and native door geometry')
if a.full_sequence_reset and a.acquisition_middle_finger_force is not None:
    p.error('Full sequence currently uses the frozen original five-digit preload')
if a.record or a.sensor_layout:a.enable_cameras=True
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
    (out/'motor-contract.json').write_bytes(Path(a.motors).read_bytes())
    sequence_reset=json.loads(Path(a.full_sequence_reset).read_text()) if a.full_sequence_reset else None
    physics_audit_enabled=bool(a.acquisition or a.sensor_policy_checkpoint)
    if a.acquisition or a.reset_from_acquisition_path:
        ref['initial_joints']=dict(zip(ref['acquisition']['joint_names'],ref['acquisition']['path_qpos'][0]))
    if a.sensor_policy_checkpoint:
        from doorbench.dexterous.sensor_reset_preflight import validate_sensor_reset_preflight
        actor_reset=validate_sensor_reset_preflight(a.sensor_reset_preflight,reference=a.reference,motors=a.motors,
            robot_usd=a.robot_usd,door_usd=a.door_usd)
        ref['initial_root']=actor_reset['root_xyz_wxyz']
        ref['initial_joints']=dict(zip(actor_reset['joint_order'],actor_reset['joint_position']))
        (out/'sensor-reset-preflight.json').write_bytes(Path(a.sensor_reset_preflight).read_bytes())
    if a.grasp_profile!='distal-pad-v1':
        if not a.grasp_profile_definition:raise ValueError('Opt-in grasp profile requires its frozen declaration')
        declaration=json.loads(Path(a.grasp_profile_definition).read_text())
        if declaration.get('profile')!=a.grasp_profile or declaration.get('robot_xml_sha256')!=motors.get('source_xml_sha256'):
            raise ValueError('Grasp profile declaration differs from the actual calibrated embodiment')
        (out/'grasp-profile-definition.json').write_bytes(Path(a.grasp_profile_definition).read_bytes())
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
        if prim.IsA(UsdGeom.Gprim) and not a.sensor_layout:UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
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
    from doorbench.dexterous.isaac_tendons import author_passive_tendons
    tendon_audit=author_passive_tendons(stage,'/World/H1',motors.get('passive_tendons',[]))
    (out/'passive-tendon-audit.json').write_text(json.dumps(tendon_audit,indent=2)+'\n')
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
    camera=None;writer=None;hand_camera=None;hand_writer=None
    if a.record:
        from isaaclab.sensors import Camera,CameraCfg
        camera=Camera(CameraCfg(prim_path='/World/Camera',update_period=0.,height=720,width=960,
            data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=48. if a.view=='hand' else 24.,clipping_range=(.02,100.))))
        if sequence_reset:
            hand_camera=Camera(CameraCfg(prim_path='/World/HandReviewCamera',update_period=0.,height=720,width=720,
                data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=48.,clipping_range=(.02,100.))))
    sensor_recorder=None
    if a.sensor_layout:
        from doorbench.dexterous.isaac_sensor_recording import IsaacSensorRecorder
        sensor_recorder=IsaacSensorRecorder(stage,a.sensor_layout,out/'sensors',control_source='sensor_actor' if a.sensor_policy_checkpoint else 'privileged_teacher')
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
    if motors.get('passive_tendons'):
        from doorbench.dexterous.isaac_tendons import verify_passive_backend
        tendon_audit['backend_readback']=verify_passive_backend(robot.root_physx_view,motors['passive_tendons'])
        (out/'passive-tendon-audit.json').write_text(json.dumps(tendon_audit,indent=2)+'\n')
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
        if hand_camera:hand_writer=imageio.get_writer(out/'live-isaac-hand.mp4',fps=25,codec='libx264',quality=8)
    initial_leaf_pose=door.data.body_state_w[0,door.body_names.index('leaf'),:7].cpu().numpy()
    initial_leaf_rotation=Rotation.from_quat([*initial_leaf_pose[4:7],initial_leaf_pose[3]]).as_matrix()
    rnames=list(robot.joint_names);dnames=list(door.joint_names)
    assert set(rnames)==set(motors['joint_names']),(set(rnames)^set(motors['joint_names']))
    index={n:i for i,n in enumerate(rnames)}
    reset_joints=sequence_reset['joints'] if sequence_reset else ref['initial_joints']
    q=torch.tensor([[reset_joints[n] for n in rnames]],device=a.device)
    limits=robot.root_physx_view.get_dof_limits()[0].cpu().numpy()
    check_joint_reset(rnames,q[0].cpu().numpy(),limits)
    robot.write_joint_state_to_sim(q,torch.zeros_like(q))
    initial_root=list(sequence_reset['initial_root'] if sequence_reset else ref['initial_root'])
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
    controls=None if a.sensor_policy_checkpoint else np.array(ref['controls']);rows=[]
    teacher=None;teacher_info={};teacher_control=None;sequence=None;operation=None;sensor_actor=None
    if a.acquisition:
        from doorbench.dexterous.acquisition_teacher import AcquisitionTeacher
        teacher=AcquisitionTeacher(a.native_robot,motors,ref,middle_finger_force=a.acquisition_middle_finger_force)
        operation=None
        if a.operate_after_acquisition:
            from doorbench.dexterous.operation_teacher import DoorOperationTeacher
            joint_geometry={}
            for role,name,child in [('operator','leaf_handle_hinge','leaf_handle'),('leaf','leaf_hinge','leaf')]:
                joint=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/World/Door/Articulation/Joints/'+name))
                if not joint or [str(path) for path in joint.GetBody1Rel().GetTargets()]!=['/World/Door/Articulation/'+child]:
                    raise ValueError('Operation requires the declared measured child-body joint frame')
                basis=np.eye(3)['XYZ'.index(joint.GetAxisAttr().Get())]
                joint_geometry[role+'_origin']=np.array(joint.GetLocalPos1Attr().Get())
                joint_geometry[role+'_axis']=np.array(joint.GetLocalRot1Attr().Get().Transform(Gf.Vec3f(*map(float,basis))))
            operation=DoorOperationTeacher(teacher,joint_geometry,wait_for_press_completion=not a.open_on_latch_clear,operator_compliance_gain=a.operator_compliance_gain)
            if sequence_reset:
                from doorbench.dexterous.full_sequence_teacher import FullSequenceTeacher
                sequence=FullSequenceTeacher(a.native_robot,motors,ref,
                    json.loads(Path(a.preparation_reference).read_text()),sequence_reset,a.locomotion_checkpoint,
                    joint_geometry,door_xml=a.native_door,operation_options=dict(
                        wait_for_press_completion=not a.open_on_latch_clear,
                        operator_compliance_gain=a.operator_compliance_gain))
                teacher=sequence.acquisition;operation=sequence.operation
            (out/'operation-protocol.json').write_text(json.dumps(dict(
                joint_geometry={k:v.tolist() for k,v in joint_geometry.items()},
                qualification='Path fraction >= 0.999 and uninterrupted 0.5 s of measured valid five-pad grasp',
                operator_target_rad=.87,leaf_target_rad=.08,press_duration_s=5.,opening_duration_s=3.,
                wait_for_press_completion=not a.open_on_latch_clear,
                operator_compliance_gain=a.operator_compliance_gain,operator_compliance_limit_rad=.15,
                freeze_compliance_on_release=True,
                scope='Continuous walking, lowering, readiness, acquisition and partial opening; no traversal' if sequence else 'Contact-free acquisition to lever, latch and partial opening; no approach or traversal',
                final_hold='The final 0.5 s must pass the unchanged strict five-pad check and hold leaf angle in [0.075, 0.10] rad; report intermediate digit unloads separately'),indent=2)+'\n')
    elif a.native_robot:
        from physx_teacher import HandleTeacher
        if a.panel_push:
            from panel_push_teacher import PanelPushTeacher
            HandleTeacher=PanelPushTeacher
        teacher=HandleTeacher(a.native_robot,motors,ref,stance_qp=a.stance_qp,grip_rotation_fraction=a.grip_rotation_fraction,grip_force=a.grip_force)
    if a.sensor_policy_checkpoint:
        from doorbench.dexterous.sensor_policy_controller import SensorPolicyController
        sensor_actor=SensorPolicyController(a.sensor_policy_checkpoint,motor_contract=motors,
            sensor_layout=sensor_recorder.layout,physics_dt_s=dt,device=a.device)
        sensor_actor.reset_episode()
    release_time=None
    ankle_motors=[i for i,motor in enumerate(motors['actuators']) if any(n in motor['terms'] for n in ('left_ankle','right_ankle'))]
    (out/'configuration.json').write_text(json.dumps(dict(args=vars(a),robot_joint_names=rnames,door_joint_names=dnames,
        dt=dt,robot_mass_kg=float(robot.root_physx_view.get_masses().sum()),latch_scale=scale,
        simulator_effort_limits=robot.root_physx_view.get_dof_max_forces()[0].cpu().tolist(),
        runtime_pose_writes=0,direct_door_commands=bool(a.mechanism_test),contact_material_audit=contact_material_audit,
        scope='Sensor-only recurrent force actor; declared curriculum objective; no teacher or traversal claim' if sensor_actor else 'Continuous walk/lower/prepare/acquire/partial opening; privileged live PhysX; no traversal' if sequence else 'Contact-free acquisition and partial opening; privileged live PhysX; no traversal' if a.operate_after_acquisition else 'Contact-free acquisition teacher; privileged live PhysX; no opening or traversal' if a.acquisition else 'Direct-force mechanism calibration; NOT robot opening' if a.mechanism_test else 'Privileged near-handle motor reference; live PhysX; no traversal'),indent=2)+'\n')
    sources=[Path(__file__),Path(__file__).with_name('physx_teacher.py')]+[Path(__file__).resolve().parents[2]/'doorbench/dexterous'/n for n in ('stance.py','reset.py','contact_audit.py','isaac_materials.py')]
    if a.panel_push:sources.append(Path(__file__).with_name('panel_push_teacher.py'))
    if a.acquisition:sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in ('acquisition_teacher.py','isaac_tendons.py','grasp_verification.py','isaac_pad_audit.py')]
    if a.operate_after_acquisition:sources.append(Path(__file__).resolve().parents[2]/'doorbench/dexterous/operation_teacher.py')
    if sequence:sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in
        ('full_sequence_teacher.py','approach_teacher.py','approach_lowering.py','locomotion.py','locomotion_approach.py','locomotion_manipulation.py','locomotion_posture.py','isaac_sensors.py')]
    inputs=[Path(a.robot_usd),Path(a.door_usd),Path(a.motors),Path(a.reference)]
    if a.native_robot:inputs.append(Path(a.native_robot))
    if a.grasp_profile_definition:inputs.append(Path(a.grasp_profile_definition))
    if sequence:
        inputs += [Path(v) for v in (a.full_sequence_reset,a.preparation_reference,a.locomotion_checkpoint,a.native_door)]
        for name,value in [('body-reset',a.full_sequence_reset),('preparation-reference',a.preparation_reference)]:
            (out/(name+'.json')).write_bytes(Path(value).read_bytes())
    if sensor_actor:
        inputs += [Path(a.sensor_policy_checkpoint),Path(a.sensor_reset_preflight)]
        sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in
            ('sensor_actor.py','sensor_policy_controller.py','control_mode.py','isaac_pad_audit.py','isaac_tendons.py','grasp_verification.py','motor_contract_identity.py','sensor_reset_preflight.py')]
    if a.sensor_layout:
        inputs.append(Path(a.sensor_layout))
        sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name
            for name in ('sensor_contract.py','isaac_sensors.py','isaac_sensor_recording.py')]
    (out/'provenance.json').write_text(json.dumps(dict(captured_before_steps_unix=time.time(),
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+inputs if p.exists()},
        camera_note='Native materials and fixed robot sensor cameras' if a.sensor_layout else 'Diagnostic gold handle material; physical properties unchanged'),indent=2)+'\n')
    for source in sources:
        if source.exists():(out/('source-'+source.name)).write_bytes(source.read_bytes())
    stage.GetRootLayer().Export(str((out/'scene.usda').resolve()))
    if sensor_recorder:sensor_recorder.initialize(sim.physics_sim_view,rnames)
    mechanical_audit=None
    if a.panel_push or physics_audit_enabled:
        from doorbench.dexterous.isaac_sensors import all_scene_contact_paths
        scene_paths=all_scene_contact_paths(stage)
        robot_paths=[p for p in scene_paths if p.startswith('/World/H1/')]
        audit_contacts=sim.physics_sim_view.create_rigid_contact_view(robot_paths,
            filter_patterns=[list(scene_paths) for _ in robot_paths],max_contact_data_count=16384)
        audit_paths=list(audit_contacts.sensor_paths)
        audit_filters=np.array(audit_contacts.filter_paths).reshape(len(audit_paths),-1)
        invariant_getters={'mass':robot.root_physx_view.get_masses,'limits':robot.root_physx_view.get_dof_limits,
            'effort_caps':robot.root_physx_view.get_dof_max_forces,'materials':robot.root_physx_view.get_material_properties,
            'contact_offsets':robot.root_physx_view.get_contact_offsets,'rest_offsets':robot.root_physx_view.get_rest_offsets}
        invariants={n:f().cpu().numpy().copy() for n,f in invariant_getters.items()}
        if motors.get('passive_tendons'):
            invariant_getters.update({name:getattr(robot.root_physx_view,name) for name in tendon_audit['backend_readback']})
            invariants.update({n:f().cpu().numpy().copy() for n,f in invariant_getters.items()})
        mechanical_audit=dict(scope='Joint limits and loopbacks at 500 Hz; scene contacts at '+('500' if physics_audit_enabled else '50')+' Hz. Privileged evaluator only.',
            max_joint_stop_penetration_rad=0.,max_self_penetration_m=0.,max_nonfoot_environment_penetration_m=0.,
            max_hand_door_penetration_m=0.,max_loopback_violation_rad=0.,contact_samples=0,capacity=16384)
    loop_pairs=[(index[f'{side}_{digit}J1'],index[f'{side}_{digit}J2']) for side in ('rh','lh') for digit in ('FF','MF','RF','LF')]
    acquisition_states={k:[] for k in ('time_s','root','joints','motor_forces','door','door_velocity','torso_tilt_deg')}
    pad_evaluator=None;pad_steps=[]
    if physics_audit_enabled:
        from doorbench.dexterous.isaac_pad_audit import PhysXShadowPadAudit
        pad_evaluator=PhysXShadowPadAudit(hand_bodies,hand_contacts,handle_filter_index=0,profile=a.grasp_profile)
        pose=door.data.body_state_w[0,door.body_names.index('leaf_handle'),:7].cpu().numpy()
        rotation=Rotation.from_quat([*pose[4:7],pose[3]]).as_matrix()
        pad_steps.append(pad_evaluator.read(physics_dt=dt,time_s=0.,center=pose[:3]+rotation@grip_center,axis=rotation@grip_axis,half_length=grip_half,radius=grip_radius))
        acquisition_reset=dict(root=robot.data.root_state_w[0].cpu().tolist(),joints=robot.data.joint_pos[0].cpu().tolist(),door=dict(zip(dnames,door.data.joint_pos[0].cpu().tolist())),
            contact_evidence_note='t=0 contact buffers before the first explicit step; full static native path/initial clearances are recorded separately with the reference')
        (out/'acquisition-reset.json').write_text(json.dumps(acquisition_reset,indent=2)+'\n')
    foot_loads=np.zeros(2);right_hand_contact_count=0;right_hand_buffered_contact_count=0
    sequence_steps=[];max_motor_delivery_error=0.
    if sequence:
        feet=['left_ankle_link','right_ankle_link']
        foot_rows=[next(i for i,path in enumerate(audit_paths) if path.rsplit('/',1)[-1]==name) for name in feet]
        foot_bodies=[robot.body_names.index(name) for name in feet]
        foot_initial=None
        hand_audit_rows=[i for i,path in enumerate(audit_paths) if path.rsplit('/',1)[-1].startswith(('rh_','lh_'))]
    def checkpoint_prefix():
        # Periodic atomic checkpoints survive a native shutdown that bypasses
        # Python exceptions. They are explicitly incomplete, never scored passes.
        (out/'trace.partial.json.tmp').write_text(json.dumps(rows)+'\n')
        os.replace(out/'trace.partial.json.tmp',out/'trace.partial.json')
        if physics_audit_enabled:
            np.savez_compressed(out/'acquisition-physics.partial.tmp.npz',**acquisition_states)
            os.replace(out/'acquisition-physics.partial.tmp.npz',out/'acquisition-physics.partial.npz')
            with gzip.open(out/'acquisition-pad-steps.partial.tmp.gz','wt') as stream:json.dump(pad_steps,stream)
            os.replace(out/'acquisition-pad-steps.partial.tmp.gz',out/'acquisition-pad-steps.partial.json.gz')
        if sequence:
            with gzip.open(out/'full-sequence-steps.partial.tmp.gz','wt') as stream:json.dump(sequence_steps,stream)
            os.replace(out/'full-sequence-steps.partial.tmp.gz',out/'full-sequence-steps.partial.json.gz')
        (out/'partial-evidence.json').write_text(json.dumps(dict(status='incomplete',passed=False,
            evidence_only=True,time_s=rows[-1]['time_s'] if rows else 0.,
            max_motor_delivery_error_Nm=max_motor_delivery_error,mechanical_audit=mechanical_audit))+'\n')
    time_origin=float(sim.current_time)
    try:
        for step in range(round(a.seconds/dt)):
            pos=robot.data.joint_pos[0].cpu().numpy();vel=robot.data.joint_vel[0].cpu().numpy()
            ctrl=np.zeros(len(kp)) if sensor_actor else controls[min(int(step*dt*50/a.time_scale),len(controls)-1)].copy()
            if teacher and not a.acquisition:
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
            feedforward=teacher.feedforward if teacher and not a.acquisition and a.press_feedforward else np.zeros(len(kp))
            impedance=kp*arm_gain*(ctrl-lengths)-extra_damping*speeds
            forces=np.clip(kp*ctrl+bias[:,0]+bias[:,1]*lengths+bias[:,2]*speeds+feedforward+impedance,force_ranges[:,0],force_ranges[:,1])
            if a.acquisition:
                body=door.data.body_state_w[0,:,:7].cpu().numpy()
                measured_args=(step*dt,robot.data.root_state_w[0].cpu().numpy(),dict(zip(rnames,pos)),dict(zip(rnames,vel)),body[door.body_names.index('leaf_handle')])
                loads=dict(zip(hand_paths,hand_contacts.get_contact_force_matrix(dt=dt).cpu().numpy().sum(axis=1)))
                if operation:
                    angles={role:float(door.data.joint_pos[0,dnames.index(name)]) for role,name in [('operator','leaf_handle_hinge'),('leaf','leaf_hinge'),('latch','leaf_latch_bolt_slide')]}
                    if sequence:
                        all_loads=audit_contacts.get_contact_force_matrix(dt=dt).cpu().numpy().copy().sum(axis=1)
                        # PhysX exposes tangential patch forces separately from its
                        # normal-force matrix. Native hand-load compensation uses both.
                        from doorbench.dexterous.isaac_sensors import _collapse_pairs
                        patch_friction,patch_points,patch_counts,patch_starts=[v.cpu().numpy().copy() for v in audit_contacts.get_friction_data(dt)]
                        patches=_collapse_pairs(patch_friction.reshape(16384,3),patch_points.reshape(16384,3),patch_counts,patch_starts,capacity=16384)
                        for i,(_,vectors) in enumerate(patches):all_loads[i]+=vectors.sum(axis=0)
                        loads={audit_paths[i]:all_loads[i] for i in hand_audit_rows}
                        forces,teacher_info=sequence.force(*measured_args[:4],foot_loads,measured_args[4],
                            body[door.body_names.index('leaf')],angles,loads,
                            grasp_qualified=pad_steps[-1]['valid_pad_grasp'],hand_contact_count=right_hand_contact_count)
                        if sequence.readiness_screen is not None and not (out/'actual-preparation-screen.json').exists():
                            (out/'actual-preparation-screen.json').write_text(json.dumps(sequence.readiness_screen,indent=2)+'\n')
                            (out/'actual-preparation-reference.json').write_text(json.dumps(sequence.actual_preparation)+'\n')
                    else:
                        forces,teacher_info=operation.force(*measured_args,body[door.body_names.index('leaf')],angles,loads,grasp_qualified=pad_steps[-1]['valid_pad_grasp'])
                else:forces,teacher_info=teacher.force(*measured_args,loads)
                ctrl=teacher.target.copy();feedforward=np.zeros_like(forces)
            if sensor_actor:
                packet=sensor_recorder.builder.observe(now_s=step*dt,previous_action=sensor_actor.previous_action)
                forces=sensor_actor.force(packet,now_s=step*dt)
                if step==0:sensor_recorder.record_initial_decision(packet,forces)
                teacher_info=dict(phase='sensor_policy',runtime_inputs='numeric robot sensor packet and local acquisition clock',teacher_fallback=False)
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
            if (camera or sensor_recorder) and step%20==0:
                review_camera=hand_camera if hand_camera else camera if a.view=='hand' else None
                if review_camera:
                    # Diagnostic camera only: stay on the approach side as the door
                    # opens. A fixed world offset became occluded by the leaf.
                    pose=door.data.body_state_w[0,door.body_names.index('leaf_handle'),:7].cpu().numpy()
                    leaf_pose=door.data.body_state_w[0,door.body_names.index('leaf'),:7].cpu().numpy()
                    hrot=Rotation.from_quat([*pose[4:7],pose[3]]).as_matrix()
                    lrot=Rotation.from_quat([*leaf_pose[4:7],leaf_pose[3]]).as_matrix()
                    center=pose[:3]+hrot@grip_center
                    eye=center+lrot@initial_leaf_rotation.T@np.array([.25,-.40,.20])
                    review_camera.set_world_poses_from_view(eyes=torch.tensor(np.array([eye]),device=a.device,dtype=torch.float32),
                        targets=torch.tensor(np.array([center]),device=a.device,dtype=torch.float32))
                sim.render()
            if abs(float(sim.current_time)-time_origin-(step+1)*dt)>.0001:
                raise RuntimeError('Physics clock changed outside the explicit motor timestep')
            robot.update(dt);door.update(dt)
            if physics_audit_enabled:
                if sequence and foot_initial is None:foot_initial=robot.data.body_state_w[0,foot_bodies,2].cpu().numpy().copy()
                delivered=robot.root_physx_view.get_dof_actuation_forces()[0].cpu().numpy()
                max_motor_delivery_error=max(max_motor_delivery_error,float(np.max(np.abs(delivered-torque))))
            if mechanical_audit is not None:
                qnew=robot.data.joint_pos[0].cpu().numpy()
                mechanical_audit['max_joint_stop_penetration_rad']=max(mechanical_audit['max_joint_stop_penetration_rad'],
                    float(np.maximum(limits[:,0]-qnew,qnew-limits[:,1]).max()))
                mechanical_audit['max_loopback_violation_rad']=max(mechanical_audit['max_loopback_violation_rad'],max(float(qnew[j1]-qnew[j2]) for j1,j2 in loop_pairs))
                if physics_audit_enabled or step%10==0:
                    af,ap,an,ad,ac,ast=[v.cpu().numpy().copy() for v in audit_contacts.get_contact_data(dt)]
                    if ac.sum()>=16384:raise RuntimeError('Mechanical contact audit buffer exhausted')
                    mechanical_audit['contact_samples']+=1
                    if sequence:
                        foot_loads[:]=0.
                        right_hand_buffered_contact_count=sum(int(ac[i].sum()) for i,path in enumerate(audit_paths) if path.rsplit('/',1)[-1].startswith('rh_'))
                        right_hand_contact_count=0
                        for i,path in enumerate(audit_paths):
                            if not path.rsplit('/',1)[-1].startswith('rh_'):continue
                            for j in range(ac.shape[1]):
                                for k in range(int(ast[i,j]),int(ast[i,j]+ac[i,j])):
                                    # PhysX emits offset contacts before surfaces
                                    # touch. Retain them in debug/audits, but zero-load
                                    # positive-gap points cannot block a clear hand.
                                    right_hand_contact_count+=int(float(ad[k,0])<=0. or float(af[k,0])>=.05)
                    for i,path in enumerate(audit_paths):
                        for j in range(ac.shape[1]):
                            for k in range(int(ast[i,j]),int(ast[i,j]+ac[i,j])):
                                depth=max(0.,-float(ad[k,0]));other=audit_filters[i,j]
                                if sequence and i in foot_rows and other.rsplit('/',1)[-1]=='floor':
                                    foot_loads[foot_rows.index(i)]+=float(af[k,0]*an[k,2])
                                if other.startswith('/World/H1/'):
                                    key='max_self_penetration_m'
                                elif path.rsplit('/',1)[-1].startswith('rh_') and other.startswith('/World/Door/Articulation/'):
                                    key='max_hand_door_penetration_m'
                                elif path.rsplit('/',1)[-1] in ('left_ankle_link','right_ankle_link') and other.rsplit('/',1)[-1]=='floor':
                                    continue
                                else:key='max_nonfoot_environment_penetration_m'
                                mechanical_audit[key]=max(mechanical_audit[key],depth)
            if sequence:
                if step%250==0:
                    debug_contacts=[]
                    for i,path in enumerate(audit_paths):
                        if not path.rsplit('/',1)[-1].startswith('rh_'):continue
                        for j in range(ac.shape[1]):
                            for k in range(int(ast[i,j]),int(ast[i,j]+ac[i,j])):
                                debug_contacts.append(dict(body=path,other=str(audit_filters[i,j]),
                                    normal_force_N=float(af[k,0]),distance_m=float(ad[k,0])))
                    (out/'readiness-debug.json').write_text(json.dumps(dict(time_s=(step+1)*dt,
                        foot_loads_N=foot_loads.tolist(),buffered_hand_contact_count=right_hand_buffered_contact_count,
                        physical_hand_contact_count=right_hand_contact_count,teacher=teacher_info,hand_contacts=debug_contacts),indent=2)+'\n')
                sequence_steps.append(dict(time_s=(step+1)*dt,phase=teacher_info['phase'],
                    foot_loads_N=foot_loads.tolist(),foot_height_m=robot.data.body_state_w[0,foot_bodies,2].cpu().tolist(),
                    right_hand_contact_count=right_hand_contact_count,buffered_hand_contact_count=right_hand_buffered_contact_count))
            if physics_audit_enabled:
                acquisition_states['time_s'].append((step+1)*dt)
                acquisition_states['root'].append(robot.data.root_state_w[0].cpu().numpy().copy())
                acquisition_states['joints'].append(robot.data.joint_pos[0].cpu().numpy().copy())
                acquisition_states['motor_forces'].append(forces.copy())
                acquisition_states['door'].append(door.data.joint_pos[0].cpu().numpy().copy())
                acquisition_states['door_velocity'].append(door.data.joint_vel[0].cpu().numpy().copy())
                acquisition_states['torso_tilt_deg'].append(float(np.degrees(np.arccos(np.clip(-robot.data.projected_gravity_b[0,2].item(),-1,1)))))
                pose=door.data.body_state_w[0,door.body_names.index('leaf_handle'),:7].cpu().numpy()
                rotation=Rotation.from_quat([*pose[4:7],pose[3]]).as_matrix()
                pad_steps.append(pad_evaluator.read(physics_dt=dt,time_s=(step+1)*dt,center=pose[:3]+rotation@grip_center,axis=rotation@grip_axis,half_length=grip_half,radius=grip_radius))
                if step%500==0:
                    (out/'latest-pad-audit.json').write_text(json.dumps(pad_steps[-1],indent=2)+'\n')
            if sensor_recorder:
                previous_action=2*(forces-force_ranges[:,0])/(force_ranges[:,1]-force_ranges[:,0])-1
                sensor_recorder.update(robot_data=robot.data,dt=dt,time_s=(step+1)*dt,
                    previous_action=previous_action,rendered=step%20==0)
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
            if hand_camera and step%20==0:
                hand_camera.update(dt*20);hand_frame=hand_camera.data.output['rgb'][0].cpu().numpy()[...,:3]
                hand_writer.append_data(hand_frame)
                if step%500==0:imageio.imwrite(out/f'hand-frame-{step:05d}.png',hand_frame)
            if (step+1)%1000==0:checkpoint_prefix()
            if sensor_recorder and (step+1)%2500==0:sensor_recorder.finish(complete=False)
            if (step+1)%50==0 and (out/'stop.request').exists():
                (out/'early-stop.json').write_text(json.dumps(dict(reason='Requested graceful diagnostic stop',time_s=(step+1)*dt))+'\n')
                break
            if not torch.isfinite(robot.data.joint_pos).all():raise RuntimeError('Nonfinite robot state')
            if rows and (rows[-1]['root'][2]<.45 or rows[-1]['torso_tilt_deg']>45):
                (out/'early-stop.json').write_text(json.dumps(dict(reason='Robot fell',time_s=(step+1)*dt))+'\n')
                break
    except BaseException:
        # Preserve the actual executed prefix even when a controller or backend
        # error prevents normal qualification. These files never imply a pass.
        (out/'trace.json').write_text(json.dumps(rows)+'\n')
        if physics_audit_enabled:
            np.savez_compressed(out/'acquisition-physics.npz',**acquisition_states)
            with gzip.open(out/'acquisition-pad-steps.json.gz','wt') as stream:json.dump(pad_steps,stream)
        if sequence:
            with gzip.open(out/'full-sequence-steps.json.gz','wt') as stream:json.dump(sequence_steps,stream)
        if sensor_recorder and sensor_recorder.times:sensor_recorder.finish(complete=False)
        if writer:writer.close()
        if hand_writer:hand_writer.close()
        raise
    (out/'trace.json').write_text(json.dumps(rows)+'\n')
    if physics_audit_enabled:np.savez_compressed(out/'acquisition-physics.npz',**acquisition_states)
    if physics_audit_enabled:
        with gzip.open(out/'acquisition-pad-steps.json.gz','wt') as stream:json.dump(pad_steps,stream)
    (out/'contacts.json').write_text(json.dumps(all_contacts)+'\n')
    (out/'contact-report-counts.json').write_text(json.dumps(report_counts)+'\n')
    if mechanical_audit is not None:
        mechanical_audit['checks']={
            'joint_stops':mechanical_audit['max_joint_stop_penetration_rad']<.02,
            'documented_loopbacks':mechanical_audit['max_loopback_violation_rad']<.02,
            'self_collision':mechanical_audit['max_self_penetration_m']<.003,
            'environment_collision':mechanical_audit['max_nonfoot_environment_penetration_m']<.003,
            'working_hand_collision':mechanical_audit['max_hand_door_penetration_m']<.003,
            'plant_parameters_unchanged':all(np.array_equal(invariants[n],f().cpu().numpy()) for n,f in invariant_getters.items())}
        mechanical_audit['passed']=all(mechanical_audit['checks'].values())
        (out/'mechanical-audit.json').write_text(json.dumps(mechanical_audit,indent=2)+'\n')
    if physics_audit_enabled:
        tail=[r for r in pad_steps if r['sim_time_s']>=a.seconds-.5-1e-8]
        root_states=np.asarray(acquisition_states['root']);motor_forces=np.asarray(acquisition_states['motor_forces'])
        checks=dict(mechanical_audit['checks'])
        checks.update(complete_physics_steps=len(pad_steps)==round(a.seconds/dt)+1,
            closed_leaf_start=abs(acquisition_reset['door']['leaf_hinge'])<=.001,
            resting_operator_start=abs(acquisition_reset['door']['leaf_handle_hinge'])<=.001,
            initial_hand_door_contact_buffer_empty=pad_steps[0]['active_contact_count']==0,
            finite=bool(all(np.isfinite(np.asarray(v)).all() for v in acquisition_states.values())),
            upright=bool(len(rows) and max(acquisition_states['torso_tilt_deg'])<12 and root_states[:,2].min()>.7),
            motor_delivery_matches_command=max_motor_delivery_error<1e-4,
            native_motor_caps=bool(np.all(motor_forces>=force_ranges[:,0]-1e-5) and np.all(motor_forces<=force_ranges[:,1]+1e-5)),
            sustained_pad_grasp=bool(len(tail)>=round(.5/dt)+1 and all(r['valid_pad_grasp'] for r in tail)))
        report=dict(scope='Sensor-only actor physical acquisition/hold audit within declared curriculum task' if sensor_actor else 'Acquisition/hold audit within a continuous operation trial; see operation-report.json for mechanism outcome' if operation else 'Live PhysX privileged acquisition only; no approach/opening/traversal or sensor-only claim',passed=all(checks.values()),checks=checks,
            grasp_profile=a.grasp_profile,
            original_distal_pad_hold=bool(len(tail)>=round(.5/dt)+1 and all(r.get('distal_pad_grasp',r)['valid_pad_grasp'] for r in tail)),
            max_motor_delivery_error_Nm=max_motor_delivery_error,
            physics_dt_s=dt,duration_s=acquisition_states['time_s'][-1],runtime_robot_pose_writes=0,direct_door_commands=False,
            initial_contact_evidence_note=acquisition_reset['contact_evidence_note'],final_pad_grasp=pad_steps[-1])
        if sensor_actor:
            actor_checks=dict(checks)
            actor_checks.update(motor_delivery=max_motor_delivery_error<1e-4,
                no_wrong_pad_patch=all(c['pad_qualified'] for row in pad_steps for c in row['contacts']))
            positions=np.asarray(acquisition_states['door']);times=np.asarray(acquisition_states['time_s'])
            leaf=positions[:,dnames.index('leaf_hinge')];handle=positions[:,dnames.index('leaf_handle_hinge')];bolt=positions[:,dnames.index('leaf_latch_bolt_slide')]
            if a.sensor_objective=='partial-opening':
                actor_checks.update(operator_and_latch_released=bool(handle.max()>=.8 and bolt.max()>=.011),
                    partial_leaf_opening_held=bool(np.all((leaf[times>=a.seconds-.5]>=.075)&(leaf[times>=a.seconds-.5]<=.10))),
                    opening_bounded_for_transfer=bool(leaf.max()<=.12))
            actor_report=dict(report,scope='Closed-loop sensor-only actor curriculum trial; no traversal claim',
                objective=a.sensor_objective,checks=actor_checks,passed=all(actor_checks.values()),
                closed_loop_evaluated=True,checkpoint_sha256=sensor_actor.checkpoint_sha256,
                final_leaf_rad=float(leaf[-1]),maximum_handle_rad=float(handle.max()),
                maximum_bolt_retraction_m=float(bolt.max()),max_motor_delivery_error_Nm=max_motor_delivery_error,
                teacher_fallback=False,runtime_actor_inputs='Stereo RGB, tactile bins, encoders, IMU, previous action, relative sensor ages and validity',
                evaluator_privilege='Simulator state/geometry used only for reset, recording, failure termination and scoring; never provided to the actor')
            (out/'actor-report.json').write_text(json.dumps(actor_report,indent=2)+'\n')
            (out/'report.json').write_text(json.dumps(actor_report,indent=2)+'\n')
            print('ACTOR_RESULT '+json.dumps({k:v for k,v in actor_report.items() if k!='final_pad_grasp'}),flush=True)
        (out/'acquisition-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('ACQUISITION_RESULT '+json.dumps({k:v for k,v in report.items() if k!='final_pad_grasp'}),flush=True)
        if operation:
            positions=np.asarray(acquisition_states['door']);times=np.asarray(acquisition_states['time_s'])
            leaf=positions[:,dnames.index('leaf_hinge')]
            handle=positions[:,dnames.index('leaf_handle_hinge')]
            bolt=positions[:,dnames.index('leaf_latch_bolt_slide')]
            operation_checks=dict(checks)
            operation_checks.update(acquisition_precedes_operation=operation.started is not None,
                operator_driven_to_release=bool(handle.max()>=.80 and bolt.max()>=.011),
                partial_leaf_opening_held=bool(np.all((leaf[times>=a.seconds-.5]>=.075)&(leaf[times>=a.seconds-.5]<=.10))),
                opening_bounded_for_transfer=bool(leaf.max()<=.12))
            offset=sequence.acquisition_started if sequence and sequence.acquisition_started is not None else 0.
            absolute_operation_start=operation.started+offset if operation.started is not None else None
            operation_rows=[r for r in pad_steps if absolute_operation_start is not None and r['sim_time_s']>=absolute_operation_start]
            operation_report=dict(report,scope='Continuous live PhysX walk/lower/prepare/acquire/partial opening; no traversal or sensor-only claim' if sequence else 'Live PhysX contact-free acquisition to lever, latch and partial opening; no approach/traversal or sensor-only claim',
                checks=operation_checks,passed=all(operation_checks.values()),
                maximum_handle_rad=float(handle.max()),maximum_leaf_rad=float(leaf.max()),
                final_leaf_rad=float(leaf[-1]),maximum_bolt_retraction_m=float(bolt.max()),
                operation_invalid_grasp_samples=sum(not r['valid_pad_grasp'] for r in operation_rows),
                operation_digit_unload_samples=sum(any(v<.2 for v in r['digit_forces_N'].values()) for r in operation_rows),
                operation_opposition_failure_samples=sum(r.get('minimum_pairwise_finger_alignment',1.)<=.5 or r.get('maximum_thumb_finger_dot',-1.)>=-.5 for r in operation_rows),
                diagnostic_counter_note='Digit unload counts measured force below 0.2 N; invalid grasp includes geometry and surface failures. Older reports conflated these counters.',
                operation_invalid_pad_patch_samples=sum(any(not c['pad_qualified'] for c in r['contacts']) for r in operation_rows),
                operation_reference=dict(operation_start_s=operation.started,opening_start_s=operation.open_started,final_goals=operation.info))
            if operation.started is not None:
                operation_report['operation_reference'].update(palm_position_in_handle_m=operation.p_relative.tolist(),palm_rotation_in_handle=operation.r_relative.tolist())
            if sequence:
                operation_report['operation_reference']['clock_offset_s']=offset
                operation_report['operation_reference']['absolute_operation_start_s']=absolute_operation_start
                operation_checks.update(
                    separated_start=bool(np.linalg.norm(np.array(initial_root[:2])-sequence_reset['goal_xy'])>=.5-1e-9),
                    walked_from_separate_start=bool(np.linalg.norm(root_states[-1,:2]-np.array(initial_root[:2]))>.5),
                    both_feet_swung=bool(all(any(r['foot_loads_N'][i]<10 and r['foot_height_m'][i]>foot_initial[i]+.015 for r in sequence_steps) for i in (0,1))),
                    readiness_screen_passed=bool(sequence.readiness_screen and sequence.readiness_screen['passed']),
                    actual_contact_free_preparation=bool(sequence.prep_started is not None and sequence.acquisition_started is not None and
                        all(r['right_hand_contact_count']==0 for r in sequence_steps if sequence.prep_started<=r['time_s']<=sequence.acquisition_started)),
                    body_solver_succeeded=sequence.body.controller.solver_failures==0,
                    operating_pad_patches_valid=operation_report['operation_invalid_pad_patch_samples']==0,
                    all_pad_patches_valid=all(c['pad_qualified'] for row in pad_steps for c in row['contacts']),
                    motor_delivery_matches_command=max_motor_delivery_error<1e-4)
                operation_report.update(passed=all(operation_checks.values()),checks=operation_checks,
                    all_episode_invalid_pad_patch_samples=sum(any(not c['pad_qualified'] for c in row['contacts']) for row in pad_steps),
                    initial_goal_distance_m=float(np.linalg.norm(np.array(initial_root[:2])-sequence_reset['goal_xy'])),
                    max_motor_delivery_error_Nm=max_motor_delivery_error,
                    handoffs=sequence.handoffs,readiness_screen=sequence.readiness_screen,
                    blocked_reason=sequence.blocked_reason)
                with gzip.open(out/'full-sequence-steps.json.gz','wt') as stream:json.dump(sequence_steps,stream)
                (out/'full-sequence-report.json').write_text(json.dumps(operation_report,indent=2)+'\n')
            (out/'operation-report.json').write_text(json.dumps(operation_report,indent=2)+'\n')
            (out/'report.json').write_text(json.dumps(operation_report,indent=2)+'\n')
            print('OPERATION_RESULT '+json.dumps({k:v for k,v in operation_report.items() if k!='final_pad_grasp'}),flush=True)
    if sensor_recorder:sensor_recorder.finish(complete=len(sensor_recorder.times)==round(a.seconds/dt) and not (out/'early-stop.json').exists())
    if writer:writer.close()
    if hand_writer:hand_writer.close()
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
