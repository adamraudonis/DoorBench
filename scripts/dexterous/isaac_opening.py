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
from doorbench.dexterous.json_record_stream import write_json_record_array


def validate_traversal_mode(args):
    """Fail unsupported combinations before starting the Isaac application."""
    if not args.traverse:
        if (getattr(args,'traversal_stow_phase_seconds',5.)!=5. or
                getattr(args,'opening_handoff_policy','first-crossing-v1')!='first-crossing-v1'):
            raise ValueError('Explicit traversal settings require --traverse')
        return
    if not args.full_sequence_reset or not args.full_opening:
        raise ValueError('--traverse requires --full-sequence-reset and --full-opening')
    if args.sensor_policy_checkpoint or getattr(args,'sensor_balance_calibration',None) or getattr(args,'sensor_locomotion_calibration',None) or args.mechanism_test or args.panel_push:
        raise ValueError('Traversal is an explicit privileged motor-teacher mode')
    if args.target_aperture < 1.2 or args.time_scale != 1.:
        raise ValueError('Traversal requires >=1.2 rad aperture and the unchanged controller clock')


def actual_motor_delivery(delivered_joint_forces, pre_step_velocity, matrix,
                          inverse, damping, friction):
    """Reconstruct submitted motor inputs from backend actuation-input readback.

    get_dof_actuation_forces returns the submitted generalized input, not
    independently measured solver joint reaction/torque. That input includes our
    explicit native damping/friction subtraction. Use the velocity from the
    command's start, not the integrated endpoint. Residuals cannot be hidden in
    the eight unactuated differential finger coordinates.
    """
    delivered = np.asarray(delivered_joint_forces, float)
    velocity = np.asarray(pre_step_velocity, float)
    if delivered.shape != (69,) or velocity.shape != (69,):
        raise ValueError('Actual 69-joint delivery and matching pre-step velocity required')
    active = delivered + damping*velocity + friction*np.tanh(velocity/.001)
    motor = inverse @ active
    residual = float(np.max(np.abs(matrix.T @ motor-active)))
    if not np.isfinite(motor).all() or residual > 1e-5:
        raise ValueError('Actual joint delivery does not lie in the original motor transmission')
    return motor, residual


def controller_root_state(robot_data, *, traverse):
    """New traversal uses actor-origin world velocity required by native qvel.

    IsaacLab root_state_w combines actor-frame pose with COM linear velocity.
    root_link_state_w includes the world angular-velocity cross-offset term.
    Angular velocity is world-frame in both. Preserve prior modes explicitly.
    """
    return robot_data.root_link_state_w if traverse else robot_data.root_state_w


def independent_traversal_checks(base_checks, opening_report, steps, *, dt,
                                 end_s, maximum_seconds, controller_done, failure):
    """Reduce actual endpoint/interval records independently of wrapper .done."""
    if (not isinstance(base_checks,dict) or
            any(not isinstance(value,(bool,np.bool_)) for value in base_checks.values()) or
            not isinstance(controller_done,(bool,np.bool_))):
        raise ValueError('Independent physical/completion checks must be explicit booleans')
    if not np.isfinite([dt,end_s,maximum_seconds]).all() or min(dt,end_s,maximum_seconds)<=0:
        raise ValueError('Finite positive physical episode clocks are required')
    if opening_report is not None and not isinstance(opening_report.get('passed'),(bool,np.bool_)):
        raise ValueError('Opening-prefix outcome must be an explicit boolean')
    for row in steps:
        root=np.asarray(row['root'],float);feet=np.asarray(row['foot_loads_N'],float)
        evidence=row['continuation_evidence']
        if (root.shape!=(13,) or feet.shape!=(2,) or not np.isfinite([*root,*feet,row['time_s'],row['pose_time_s']]).all() or
                any(not isinstance(value,(bool,np.bool_)) for value in
                    (row['post_started'],row['passage_completed'],row['right_pad_patches_valid'],evidence['physics_qualified']))):
            raise ValueError('Malformed measured traversal state or qualification')
        if row['post_started'] and not np.isfinite(row['minimum_body_y_m']):
            raise ValueError('Actual post-opening body clearance is required')
        if not np.isfinite(evidence['left_hand_load_N']) or evidence['left_hand_load_N']<0:
            raise ValueError('Actual nonnegative left-hand load is required')
        for key in ('left_hand_contacts','right_environment_contacts'):
            if (not isinstance(evidence[key],(int,np.integer)) or
                    isinstance(evidence[key],(bool,np.bool_)) or evidence[key]<0):
                raise ValueError('Actual nonnegative integer contact counts required')
    checks = {k:bool(v) for k,v in base_checks.items() if k!='sustained_pad_grasp'}
    times = np.asarray([row['time_s'] for row in steps], float)
    expected = round(end_s/dt)
    coverage = (expected > 0 and len(steps)==expected+1 and
        np.allclose(times,np.arange(expected+1)*dt,atol=1e-8,rtol=0))
    tail = [row for row in steps if end_s-1.-1e-8 <= row['time_s'] <= end_s+1e-8]
    enough = (len(tail)==round(1./dt)+1 and
        abs(tail[-1]['time_s']-end_s)<1e-8 and abs(tail[0]['time_s']-(end_s-1.))<1e-8)
    def quiet(row):
        root=np.asarray(row['root'],float);ev=row['continuation_evidence']
        return bool(row['post_started'] and row['minimum_body_y_m']>.2 and
            row['passage_completed'] and np.linalg.norm(root[7:9])<.03 and
            min(row['foot_loads_N'])>=10. and ev['left_hand_contacts']==0 and
            ev['left_hand_load_N']<.1 and ev['right_environment_contacts']==0 and
            ev['physics_qualified'] and row['right_pad_patches_valid'])
    held = bool(enough and all(quiet(row) for row in tail))
    excursion = (max(float(np.linalg.norm(np.asarray(row['root'])[:2]-np.asarray(tail[0]['root'])[:2]))
                     for row in tail) if tail else float('inf'))
    checks.update(qualified_opening_prefix=bool(opening_report and opening_report['passed']),
        complete_controller_measurements=bool(coverage and end_s<=maximum_seconds+1e-8),
        contact_epoch_aligned=bool(steps and all(abs(row['pose_time_s']-row['time_s'])<1e-8 and
            np.allclose(row['contact_interval_s'],[max(0.,row['time_s']-dt),row['time_s']],atol=1e-8,rtol=0)
            for row in steps)),
        all_interval_physics_qualified=bool(steps and all(row['continuation_evidence']['physics_qualified'] for row in steps)),
        all_interval_right_pad_patches_valid=bool(steps and all(row['right_pad_patches_valid'] for row in steps)),
        full_body_passage_and_quiet_unloaded_finish=held,
        final_second_xy_excursion_below_3cm=bool(enough and excursion<.03),
        continuous_controller_completed=bool(controller_done), no_controller_failure=failure is None)
    return checks


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
p.add_argument('--sensor-gyro-profile',choices=['backend-angular-velocity-v1','pose-delta-angle-v1'],default='backend-angular-velocity-v1',help='Explicit ideal own-body delta-angle sensor alternative; accelerometer and estimator remain unchanged')
p.add_argument('--time-scale',type=float,default=1.,help='Slower motor-reference clock; physics dt is unchanged')
p.add_argument('--view',choices=['wide','hand'],default='wide')
p.add_argument('--upright-gain',type=float,default=0.,help='Post-opening IMU ankle feedback; bounded robot motors only')
p.add_argument('--native-robot',help='Enable closed-loop kinematic teacher using this native robot XML for FK only')
p.add_argument('--acquisition',action='store_true',help='Execute the shared contact-free acquisition teacher from reference.path_qpos; privileged development only')
p.add_argument('--acquisition-stance-profile',choices=['landed-foot-v1'],help='Explicit body-origin feedback and actual-foot-frame acquisition stance')
p.add_argument('--acquisition-middle-finger-force',type=float,help='Explicit acquisition middle-finger preload in N; original motor caps unchanged')
p.add_argument('--acquisition-index-finger-force',type=float,help='Explicit acquisition index-finger preload in N; original motor caps unchanged')
p.add_argument('--acquisition-pressure-segment',choices=['distal'],help='Apply the grasp-force reference through distal geometry only')
p.add_argument('--operation-grasp-offset-in-handle-m',nargs=3,type=float,help='Optional handle-frame reference recenter, at most 10 mm; one-second smooth ramp')
p.add_argument('--operation-index-proximal-offset-rad',type=float,default=0.,help='Bounded index proximal reference offset, smoothly applied during operation')
p.add_argument('--operation-index-tendon-offset-rad',type=float,default=0.,help='Bounded summed index tendon reference offset, split over FFJ1/2')
p.add_argument('--operation-min-acquisition-seconds',type=float,default=0.,help='Earliest qualified grasp-to-operation handoff')
p.add_argument('--hold-attained-grasp',action='store_true',help='Capture original coupled finger targets after a qualified partial opening')
p.add_argument('--operation-actual-pad-control',action='store_true')
p.add_argument('--operation-material-profile',choices=('actual-material-v1','actual-material-v2','actual-tangent-v1','measured-pressure-v1'),default='actual-material-v1')
p.add_argument('--attained-hold-stage',choices=('acquisition','operator','aperture','opening'),default='opening')
p.add_argument('--operate-after-acquisition',action='store_true',help='After 0.5 s of actual qualified grasp, press the lever and hold a partial opening through robot motors')
p.add_argument('--open-on-latch-clear',action='store_true',help='Start the smooth opening ramp on measured release, without waiting for the press-reference timer')
p.add_argument('--operator-compliance-gain',type=float,default=0.,help='Bounded palm-reference integral compensation for actual operator-angle error; motor and mechanism limits unchanged')
p.add_argument('--sensor-policy-checkpoint',help='Execute the recurrent actor using only robot sensor packets; no teacher fallback')
p.add_argument('--sensor-balance-calibration',help='Opt-in stationary sensor-only balance calibration; no learned policy or acquisition claim')
p.add_argument('--sensor-locomotion-calibration',help='Separate five-second pinned H1 sensor-locomotion diagnostic; no door task claim')
p.add_argument('--sensor-locomotion-stop-after-seconds',type=float,help='Explicit 10s walking/stopping component with a 3s stop request')
p.add_argument('--sensor-locomotion-robot',help='Bound robot-only model for IMU mounting and motor calibration')
p.add_argument('--sensor-locomotion-checkpoint',help='Pinned original Unitree H1 locomotion network')
p.add_argument('--sensor-balance-robot',help='Static robot-only XML calibration for the sensor balance estimator')
p.add_argument('--sensor-arm-schedule',help='Opt-in frozen six-second scripted arm schedule over sensor-only balance; not a learned door policy')
p.add_argument('--sensor-reach-protocol',help='Frozen eleven-second contact-free coordinated reach protocol over sensor-only balance')
p.add_argument('--sensor-reach-route',help='Joint-only route bound to the reach protocol; no world or door state')
p.add_argument('--sensor-acquisition-protocol',help='Frozen nineteen-second sensor-feedback scripted grasp protocol; not opening or a learned policy')
p.add_argument('--sensor-acquisition-route',help='Joint-only full acquisition route bound to its separate protocol')
p.add_argument('--sensor-objective',choices=['acquisition','partial-opening'],default='partial-opening',help='Declared curriculum qualification; neither establishes traversal')
p.add_argument('--sensor-reset-preflight',help='Required frozen native reset receipt for a sensor-only actor')
p.add_argument('--reset-from-acquisition-path',action='store_true',help='Use the frozen contact-free first configuration at reset only')
p.add_argument('--grasp-profile',choices=['distal-pad-v1','volar-phalange-v1'],default='distal-pad-v1')
p.add_argument('--grasp-profile-definition',help='Frozen declaration required for an opt-in grasp profile')
p.add_argument('--joint-passive-profile',choices=['legacy-tanh-v1','backend-dry-v2'],default='legacy-tanh-v1',help='Versioned robot passive-joint adapter; backend dry friction requires separate qualification')
p.add_argument('--full-sequence-reset',help='Start from the frozen walking reset and run continuous walk/lower/prepare/acquire/operate')
p.add_argument('--preparation-reference',help='Contact-free readiness path; screened again at the actual stopped pose')
p.add_argument('--locomotion-checkpoint',help='Frozen original H1 locomotion checkpoint')
p.add_argument('--native-door',help='Matching unstepped native door geometry for readiness collision checks')
p.add_argument('--full-opening',action='store_true',help='Privileged acquisition, lever, bimanual transfer and loaded aperture development; no approach/traversal')
p.add_argument('--traverse',action='store_true',help='Continue a qualified full walking/opening episode through measured release, stow, rise, passage and quiet stop without resetting')
p.add_argument('--traversal-stow-phase-seconds',type=float,choices=(4.,5.),default=5.,help='Explicit sequential stow timing; the verified native sequence uses four seconds')
p.add_argument('--opening-handoff-policy',choices=('first-crossing-v1','loaded-hold-v2'),default='first-crossing-v1',help='Explicit measured opening handoff; the verified native sequence requires loaded-hold-v2')
p.add_argument('--left-palm-targets',help='Source-bound screened left-palm workspace targets')
p.add_argument('--right-release-screen',help='Frozen axial right-hand release path')
p.add_argument('--bimanual-runtime-screen',help='Source/design-bound runtime geometry re-screen for another platform')
p.add_argument('--target-aperture',type=float,default=1.2,help='Declared full-opening aperture in radians')
p.add_argument('--follow-leaf-during-transfer',action='store_true',help='Let the right grip follow actual panel motion during left-hand support; full-opening development only')
p.add_argument('--panel-profile',choices=('plain-v1','hybrid-surface-v2'),help='Explicit development panel controller; original physical limits remain unchanged')
p.add_argument('--palm-load-target',type=float,help='Explicit development palm-pressure target in N; requires full opening')
p.add_argument('--transfer-load-target',type=float,default=4.,help='Declared left-panel support target in N before right-hand release; original motor caps unchanged')
p.add_argument('--left-planning-profile',choices=('strict-v1','intermediate-clearance-3mm-v1'),help='Independently replan the left approach from the current episode state')
p.add_argument('--whole-body-return-path',help='Opt-in exact-attained-state screened lever return; requires continuous walking stance')
p.add_argument('--whole-body-ungrip-path',help='Opt-in screened whole-body withdrawal after the lever-return path')
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
p.add_argument('--operation-opening-trigger-rad',type=float,default=.80,help='Controller transition only; final operator and latch acceptance thresholds remain unchanged')
p.add_argument('--operation-leaf-target-rad',type=float,default=.08,help='Commanded partial opening; physical acceptance bounds stay unchanged')
p.add_argument('--operation-operator-lead-limit-rad',type=float,help='Bound palm reference around measured handle angle; original release thresholds remain unchanged')
p.add_argument('--operation-leaf-lead-limit-rad',type=float,help='Explicit measured-door lead bound for a new standalone opening trial')
p.add_argument('--operation-hub-geometry',type=str,help='Original hub descriptor from the independently gated native prerequisite')
p.add_argument('--operation-hub-clearance-m',type=float,default=.004,help='Prospective 4–8 mm hub-avoidance activation; force cap remains 3 N')
p.add_argument('--operation-operator-follow-after-leaf-rad',type=float,help='Blend toward the measured handle angle after the leaf clears the latch')
p.add_argument('--validate-arguments-only',action='store_true',help='Validate CLI combinations without starting SimulationApp')
a=p.parse_args()
if a.operation_operator_follow_after_leaf_rad is not None and (not .015<=a.operation_operator_follow_after_leaf_rad<=.05 or not a.operate_after_acquisition or a.full_opening or a.full_sequence_reset):p.error('Operator follow requires standalone operation and .015..0.05 rad')
if a.operation_operator_lead_limit_rad is not None and (not .01<=a.operation_operator_lead_limit_rad<=.15 or not a.operate_after_acquisition or a.full_opening or a.full_sequence_reset):p.error('Operator lead requires standalone operation and .01..0.15 rad')
if a.operation_leaf_lead_limit_rad is not None and (not .002<=a.operation_leaf_lead_limit_rad<=.03 or not a.operate_after_acquisition or a.full_opening or a.full_sequence_reset):p.error('Leaf lead bound requires standalone operation and .002..0.03 rad')
if not .075<=a.operation_leaf_target_rad<=.10:p.error('Partial opening command must be .075..0.10 rad')
if a.operation_leaf_target_rad!=.08 and (not a.operate_after_acquisition or a.full_opening or a.full_sequence_reset):p.error('Leaf command override requires standalone operation')
if a.sensor_gyro_profile!='backend-angular-velocity-v1' and not a.sensor_layout:
    p.error('Alternate gyroscope profile requires --sensor-layout')
for name in ('arm_impedance','time_scale','seconds'):
    value=getattr(a,name)
    if not math.isfinite(value) or value<=0:p.error(f'--{name.replace("_","-")} must be finite and positive')
if not math.isfinite(a.grip_force) or a.grip_force<0:p.error('--grip-force must be finite and nonnegative')
if not math.isfinite(a.grip_impedance) or a.grip_impedance<0:p.error('--grip-impedance must be finite and nonnegative')
if not math.isfinite(a.finger_curl):p.error('--finger-curl must be finite')
if not math.isfinite(a.torso_damping) or a.torso_damping<0:p.error('--torso-damping must be finite and nonnegative')
if a.acquisition and (not a.native_robot or a.panel_push or a.mechanism_test):p.error('Acquisition requires --native-robot and a separate acquisition-only trial')
if a.acquisition_stance_profile and (not a.acquisition or a.full_sequence_reset):p.error('Landed-foot acquisition is a separate explicit acquisition trial')
if a.operate_after_acquisition and not a.acquisition:p.error('--operate-after-acquisition requires --acquisition')
if a.acquisition_pressure_segment and (not a.acquisition or a.full_sequence_reset or a.full_opening):p.error('Pressure segment requires a separate acquisition trial')
if not math.isfinite(a.operation_min_acquisition_seconds) or a.operation_min_acquisition_seconds<0:p.error('Finite nonnegative operation handoff time required')
if a.operation_actual_pad_control and (not a.operate_after_acquisition or a.full_sequence_reset or a.full_opening or a.hold_attained_grasp):p.error('Actual material pads require a separate standalone operation trial')
if a.hold_attained_grasp and (not a.operate_after_acquisition or a.full_sequence_reset or a.full_opening):p.error('Attained grasp hold requires a standalone operation trial')
for value,limit in [(a.operation_index_proximal_offset_rad,.1),(a.operation_index_tendon_offset_rad,.12)]:
    if not math.isfinite(value) or abs(value)>limit:p.error('Finite bounded index posture offsets required')
if (a.operation_index_proximal_offset_rad or a.operation_index_tendon_offset_rad) and (not a.operate_after_acquisition or a.full_sequence_reset or a.full_opening):p.error('Index offsets require standalone operation')
if a.operation_grasp_offset_in_handle_m is not None:
    if not all(math.isfinite(v) for v in a.operation_grasp_offset_in_handle_m) or sum(v*v for v in a.operation_grasp_offset_in_handle_m)>.01**2:p.error('Finite grasp offset within 10 mm required')
if (a.operation_grasp_offset_in_handle_m is not None or a.operation_min_acquisition_seconds) and (not a.operate_after_acquisition or a.full_sequence_reset or a.full_opening):p.error('Operation offsets/timing require a separate operation trial')
if a.open_on_latch_clear and not a.operate_after_acquisition:p.error('--open-on-latch-clear requires --operate-after-acquisition')
from doorbench.dexterous.control_mode import validate_sensor_actor_mode,validate_sensor_balance_protocol
try:validate_sensor_actor_mode(a)
except ValueError as error:p.error(str(error))
if a.full_sequence_reset and (not a.operate_after_acquisition or not all((a.preparation_reference,a.locomotion_checkpoint,a.native_door))):
    p.error('Full sequence requires acquisition, operation, preparation, locomotion checkpoint and native door geometry')
if a.full_sequence_reset and a.acquisition_middle_finger_force is not None:
    p.error('Full sequence currently uses the frozen original five-digit preload')
if a.acquisition_index_finger_force is not None and (not a.acquisition or not math.isfinite(a.acquisition_index_finger_force) or a.acquisition_index_finger_force<0):
    p.error('Index-finger preload requires acquisition and a finite nonnegative value')
if a.full_opening and (not a.operate_after_acquisition or not all((a.native_door,a.left_palm_targets,a.right_release_screen))):
    p.error('Full opening requires acquisition/operation and screened native geometry')
if not math.isfinite(a.target_aperture) or a.target_aperture<=0:p.error('Aperture must be finite and positive')
if a.full_opening and any(v is not None for v in (a.acquisition_index_finger_force,a.acquisition_middle_finger_force)):
    p.error('Full opening uses its explicitly frozen default acquisition forces')
if (a.follow_leaf_during_transfer or a.panel_profile is not None or a.palm_load_target is not None) and not a.full_opening:
    p.error('Panel controller options require --full-opening')
if a.palm_load_target is not None and (not math.isfinite(a.palm_load_target) or not 2<a.palm_load_target<=10):
    p.error('Palm target must be finite, above 2 N and at most 10 N')
if not math.isfinite(a.transfer_load_target) or not 2<a.transfer_load_target<=10:
    p.error('Transfer target must be finite, above 2 N and at most 10 N')
if a.transfer_load_target!=4. and not a.full_opening:
    p.error('A nondefault transfer target requires --full-opening')
if a.left_planning_profile is not None and not a.full_opening:
    p.error('Attained left planning requires --full-opening')
if a.whole_body_return_path and not (a.full_opening and a.full_sequence_reset):
    p.error('Whole-body return requires full opening and the continuous landed stance')
if a.whole_body_ungrip_path and not a.whole_body_return_path:
    p.error('Whole-body ungrip requires its screened whole-body return path')
try:validate_sensor_balance_protocol(a)
except ValueError as error:p.error(str(error))
balance_scope=('Sensor-only analytical balance with scripted torso, arm and finger joint goals; acquisition only; RGB unused; no learned policy, opening or traversal claim'
               if a.sensor_acquisition_protocol else 'Sensor-only analytical balance with scripted torso, arm and finger joint goals; contact-free reach only; RGB unused; no learned policy, acquisition or door task claim'
               if a.sensor_reach_protocol else 'Sensor-only analytical balance with scripted arm/wrist joint goals; RGB unused; no learned policy, acquisition or door task claim'
               if a.sensor_arm_schedule else 'Sensor-only analytical stationary balance; RGB unused; no learned policy, acquisition or door task claim')
try:validate_traversal_mode(a)
except ValueError as error:p.error(str(error))
if not .70<=a.operation_opening_trigger_rad<=.80:raise ValueError('Opening trigger must be .70.. .80 rad')
if not math.isfinite(a.operation_hub_clearance_m) or not .004<=a.operation_hub_clearance_m<=.008:p.error('Hub clearance activation must be 4–8 mm')
if a.validate_arguments_only:
    print(json.dumps(dict(arguments_valid=True,physics_started=False)));raise SystemExit(0)
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
    import importlib.metadata
    import platform
    versions={}
    for package in ('numpy','scipy','mujoco','osqp','torch','isaacsim','isaaclab'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    (out/'runtime-versions.json').write_text(json.dumps(dict(python=platform.python_version(),
        platform=platform.platform(),packages=versions,
        note='Exact command replay requires the original calculator and optimizer dependencies, not only source hashes.'),indent=2)+'\n')
    ref=json.loads(Path(a.reference).read_text());motors=json.loads(Path(a.motors).read_text())
    (out/'motor-contract.json').write_bytes(Path(a.motors).read_bytes())
    sequence_reset=json.loads(Path(a.full_sequence_reset).read_text()) if a.full_sequence_reset else None
    sensor_control=bool(a.sensor_policy_checkpoint or a.sensor_balance_calibration or a.sensor_locomotion_calibration)
    physics_audit_enabled=bool(a.acquisition or sensor_control)
    if a.acquisition or a.reset_from_acquisition_path:
        ref['initial_joints']=dict(zip(ref['acquisition']['joint_names'],ref['acquisition']['path_qpos'][0]))
    if sensor_control:
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
        if sequence_reset or a.full_opening or sensor_control or a.operate_after_acquisition:
            hand_camera=Camera(CameraCfg(prim_path='/World/HandReviewCamera',update_period=0.,height=720,width=720,
                data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=48.,clipping_range=(.02,100.))))
    sensor_recorder=None
    if a.sensor_layout:
        from doorbench.dexterous.isaac_sensor_recording import IsaacSensorRecorder
        sensor_recorder=IsaacSensorRecorder(stage,a.sensor_layout,out/'sensors',control_source='sensor_actor' if sensor_control else 'privileged_teacher',gyro_profile=a.sensor_gyro_profile)
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
    from doorbench.dexterous.isaac_joint_passive import passive_profile,configure_backend,PassivePropertyInvariant
    passive_declaration=passive_profile(motors,rnames,a.joint_passive_profile)
    passive_guard=None
    if a.joint_passive_profile=='backend-dry-v2':
        passive_receipt=configure_backend(robot.root_physx_view,passive_declaration)
        damp=passive_declaration['explicit_damping'];friction=passive_declaration['explicit_friction']
        passive_guard=PassivePropertyInvariant(passive_declaration,physics_dt_s=dt)
    else:
        passive_receipt=dict(profile=a.joint_passive_profile,scope='Historical explicit damping and tanh(v/.001) friction; retained for replay')
    (out/'joint-passive-profile.json').write_text(json.dumps(passive_receipt,indent=2)+'\n')
    target=torch.zeros((1,len(dnames)),device=a.device)
    for i,n in enumerate(dnames):
        prim=stage.GetPrimAtPath('/World/Door/Articulation/Joints/'+n)
        target[0,i]=prim.GetAttribute('doorbench:target_si').Get() or 0.
    controls=None if sensor_control else np.array(ref['controls']);rows=[]
    teacher=None;teacher_info={};teacher_control=None;sequence=None;operation=None;sensor_actor=None;full_opening=None;opening_geometry=None;continuous=None
    if a.acquisition:
        from doorbench.dexterous.acquisition_teacher import AcquisitionTeacher
        hub_geometry=None
        if a.operation_hub_geometry:
            if not a.operate_after_acquisition or a.full_opening or sequence_reset:raise ValueError('Hub avoidance currently requires standalone operation')
            hub_geometry=json.loads(Path(a.operation_hub_geometry).read_text())
            (out/'hub-geometry.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(a.operation_hub_geometry).read_bytes()).hexdigest(),geometry=hub_geometry),indent=2)+'\n')
        teacher=AcquisitionTeacher(a.native_robot,motors,ref,handle_hub_geometry=hub_geometry,middle_finger_force=a.acquisition_middle_finger_force,index_finger_force=a.acquisition_index_finger_force,stance_profile=a.acquisition_stance_profile,pressure_segment=a.acquisition_pressure_segment or 'nearest')
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
            operation=DoorOperationTeacher(teacher,joint_geometry,index_proximal_offset_rad=a.operation_index_proximal_offset_rad,index_tendon_offset_rad=a.operation_index_tendon_offset_rad,release_operator_threshold=a.operation_opening_trigger_rad,handle_hub_avoidance=hub_geometry is not None,hub_clearance_m=a.operation_hub_clearance_m,leaf_target=a.operation_leaf_target_rad,leaf_lead_limit_rad=a.operation_leaf_lead_limit_rad,operator_lead_limit_rad=a.operation_operator_lead_limit_rad,operator_follow_after_leaf_rad=a.operation_operator_follow_after_leaf_rad,wait_for_press_completion=not a.open_on_latch_clear,operator_compliance_gain=a.operator_compliance_gain,min_acquisition_seconds=a.operation_min_acquisition_seconds,grasp_offset_in_handle_m=a.operation_grasp_offset_in_handle_m or (0.,0.,0.),fixed_pad_control=a.operation_actual_pad_control,pad_control_profile=a.operation_material_profile if a.operation_actual_pad_control else "commanded-material-v1",hold_attained_grasp=a.hold_attained_grasp,attained_hold_stage=a.attained_hold_stage)
            if sequence_reset and not a.full_opening:
                from doorbench.dexterous.full_sequence_teacher import FullSequenceTeacher
                sequence=FullSequenceTeacher(a.native_robot,motors,ref,
                    json.loads(Path(a.preparation_reference).read_text()),sequence_reset,a.locomotion_checkpoint,
                    joint_geometry,door_xml=a.native_door,operation_options=dict(
                        wait_for_press_completion=not a.open_on_latch_clear,
                        operator_compliance_gain=a.operator_compliance_gain),
                    acquisition_options=dict(index_finger_force=a.acquisition_index_finger_force))
                teacher=sequence.acquisition;operation=sequence.operation
            if a.full_opening:
                from doorbench.dexterous.full_opening_teacher import FullOpeningTeacher
                from doorbench.dexterous.isaac_opening_measurements import OpeningGeometryMeasurements
                opening_options=dict(open_on_latch_clear=a.open_on_latch_clear,
                    operator_compliance_gain=a.operator_compliance_gain,target_aperture=a.target_aperture,
                    follow_leaf_during_transfer=a.follow_leaf_during_transfer,
                    panel_profile=a.panel_profile or 'hybrid-surface-v2',palm_load_target=a.palm_load_target,
                    transfer_load_target=a.transfer_load_target,left_planning_profile=a.left_planning_profile,
                    whole_body_return_path=a.whole_body_return_path,whole_body_ungrip_path=a.whole_body_ungrip_path)
                if sequence_reset:
                    from doorbench.dexterous.walking_opening_teacher import WalkingOpeningTeacher
                    sequence_type=WalkingOpeningTeacher
                    if a.traverse:
                        from doorbench.dexterous.continuous_door_teacher import ContinuousDoorTeacher
                        sequence_type=ContinuousDoorTeacher
                    sequence=sequence_type(a.native_robot,motors,ref,
                        json.loads(Path(a.preparation_reference).read_text()),sequence_reset,a.locomotion_checkpoint,
                        joint_geometry,door_xml=a.native_door,left_targets=a.left_palm_targets,
                        release_screen=a.right_release_screen,runtime_screen=a.bimanual_runtime_screen,
                        opening_options=opening_options,**({'maximum_seconds':a.seconds,
                            'post_phase_seconds':a.traversal_stow_phase_seconds,'handoff_policy':a.opening_handoff_policy} if a.traverse else {}))
                    if a.traverse:
                        continuous=sequence;sequence=continuous.walking
                    full_opening=sequence.opening
                else:
                    full_opening=FullOpeningTeacher(a.native_robot,motors,ref,joint_geometry,
                        door_xml=a.native_door,left_targets=a.left_palm_targets,
                        release_screen=a.right_release_screen,runtime_screen=a.bimanual_runtime_screen,
                        **opening_options)
                teacher=full_opening.acquisition;operation=None
                opening_geometry=OpeningGeometryMeasurements(a.native_door,a.native_robot,rnames)
            (out/'operation-protocol.json').write_text(json.dumps(dict(
                role='Partial-opening stage diagnostic only; full-opening-protocol.json governs this run' if full_opening else 'Declared operation trial protocol',
                joint_geometry={k:v.tolist() for k,v in joint_geometry.items()},
                qualification='Path fraction >= 0.999 and uninterrupted 0.5 s of measured valid five-pad grasp',
                operator_target_rad=.87,leaf_target_rad=.08,press_duration_s=5.,opening_duration_s=3.,
                wait_for_press_completion=not a.open_on_latch_clear,
                operator_compliance_gain=a.operator_compliance_gain,operator_compliance_limit_rad=.15,
                freeze_compliance_on_release=True,fixed_pad_control=a.operation_actual_pad_control,pad_control_profile=a.operation_material_profile if a.operation_actual_pad_control else "commanded-material-v1",hold_attained_grasp=a.hold_attained_grasp,
                scope='Continuous walking, lowering, readiness, acquisition and partial opening; no traversal' if sequence else 'Contact-free acquisition to lever, latch and partial opening; no approach or traversal',
                final_hold='The final 0.5 s must pass the unchanged strict five-pad check and hold leaf angle in [0.075, 0.10] rad; report intermediate digit unloads separately'),indent=2)+'\n')
            if full_opening:
                (out/'full-opening-protocol.json').write_text(json.dumps(dict(
                    maximum_seconds=a.seconds,target_aperture_rad=a.target_aperture,
                    terminal_event='Freeze opening prefix at first measured target crossing; continue physical traversal' if continuous else 'First measured target-aperture crossing or declared timeout',
                    final_hold='Final uninterrupted 0.5 s of actual left-palm projected load >= 2 N',
                    right_release='Requires prior 0.5 s of opposed right-hand grip and actual left-panel support >= 2 N',
                    scope='Opening prefix of uninterrupted approach/open/traverse trial' if continuous else 'Continuous approach through bimanual aperture; no traversal' if sequence else 'Initialized contact-free acquisition through bimanual loaded aperture; no approach/traversal',
                    original_caps_and_physics=True,open_on_latch_clear=a.open_on_latch_clear,
                    operator_compliance_gain=a.operator_compliance_gain,
                    follow_leaf_during_transfer=a.follow_leaf_during_transfer,
                    panel_profile=full_opening.panel_profile,palm_load_target=full_opening.push.target_palm_load,
                    transfer_load_target=a.transfer_load_target,
                    geometry_source_hashes=opening_geometry.sources),indent=2)+'\n')
                if continuous:
                    (out/'traversal-protocol.json').write_text(json.dumps(dict(
                        maximum_seconds=a.seconds,physics_dt_s=dt,stow_profile='sequential-v2',phase_seconds=a.traversal_stow_phase_seconds,
                        handoff_policy=a.opening_handoff_policy,
                        opening='Requires independently qualified opening prefix; any invalid pad or physical interval fails the whole episode',
                        handoff='Actual root/q/dq/body poses/door velocities and preceding delivered motor forces at unchanged global clock; no plant reset',
                        final_hold='Whole body y>0.2 m; horizontal root speed<0.03 m/s; both feet>=10 N; hands clear and LH normal load<0.1 N for1 s; XY excursion<0.03 m',
                        original_caps_and_physics=True,runtime_pose_writes=0,
                        scope='Privileged continuous controller development; no sensor-only policy claim'),indent=2)+'\n')
    elif a.native_robot:
        from physx_teacher import HandleTeacher
        if a.panel_push:
            from panel_push_teacher import PanelPushTeacher
            HandleTeacher=PanelPushTeacher
        teacher=HandleTeacher(a.native_robot,motors,ref,stance_qp=a.stance_qp,grip_rotation_fraction=a.grip_rotation_fraction,grip_force=a.grip_force)
    if a.sensor_locomotion_calibration:
        from doorbench.dexterous.sensor_locomotion import SensorLocomotionController
        sensor_actor=SensorLocomotionController.from_calibration(a.sensor_locomotion_robot,motors,sensor_recorder.layout,
            a.sensor_locomotion_calibration,a.sensor_locomotion_checkpoint)
        if a.sensor_locomotion_stop_after_seconds is not None:
            from doorbench.dexterous.sensor_walk_stop import SensorWalkStopController
            sensor_actor=SensorWalkStopController.from_calibration(a.sensor_locomotion_robot,motors,sensor_recorder.layout,a.sensor_locomotion_calibration,a.sensor_locomotion_checkpoint,a.sensor_locomotion_stop_after_seconds)
        sensor_actor.reset_episode()
    elif a.sensor_policy_checkpoint:
        from doorbench.dexterous.sensor_policy_controller import SensorPolicyController
        sensor_actor=SensorPolicyController(a.sensor_policy_checkpoint,motor_contract=motors,
            sensor_layout=sensor_recorder.layout,physics_dt_s=dt,device=a.device)
        sensor_actor.reset_episode()
    elif a.sensor_balance_calibration:
        if a.sensor_acquisition_protocol:
            from doorbench.dexterous.sensor_acquisition_runtime import SensorAcquisitionBalanceRuntime
            sensor_actor=SensorAcquisitionBalanceRuntime(a.sensor_balance_robot,motors,sensor_recorder.layout,
                a.sensor_balance_calibration,a.sensor_acquisition_protocol,a.sensor_acquisition_route)
        elif a.sensor_reach_protocol:
            from doorbench.dexterous.sensor_reach_runtime import SensorReachBalanceRuntime
            sensor_actor=SensorReachBalanceRuntime(a.sensor_balance_robot,motors,sensor_recorder.layout,
                a.sensor_balance_calibration,a.sensor_reach_protocol,a.sensor_reach_route)
        elif a.sensor_arm_schedule:
            from doorbench.dexterous.sensor_arm_balance_runtime import SensorArmBalanceRuntime
            sensor_actor=SensorArmBalanceRuntime(a.sensor_balance_robot,motors,sensor_recorder.layout,
                a.sensor_balance_calibration,a.sensor_arm_schedule)
        else:
            from doorbench.dexterous.sensor_balance_runtime import SensorBalanceRuntime
            sensor_actor=SensorBalanceRuntime(a.sensor_balance_robot,motors,sensor_recorder.layout,
                a.sensor_balance_calibration)
        sensor_actor.reset_episode()
    release_time=None
    ankle_motors=[i for i,motor in enumerate(motors['actuators']) if any(n in motor['terms'] for n in ('left_ankle','right_ankle'))]
    (out/'configuration.json').write_text(json.dumps(dict(args=vars(a),robot_joint_names=rnames,door_joint_names=dnames,
        dt=dt,robot_mass_kg=float(robot.root_physx_view.get_masses().sum()),latch_scale=scale,
        root_state_convention='actor-origin pose and world actor-origin linear/angular velocity' if (continuous or a.sensor_locomotion_calibration or a.acquisition_stance_profile) else 'legacy IsaacLab actor pose plus world COM linear/angular velocity',
        balance_root_state_convention='balance-steps uses root_link_state_w: actor-origin pose and world actor-origin linear/angular velocity; evaluator only' if a.sensor_balance_calibration else None,
        simulator_effort_limits=robot.root_physx_view.get_dof_max_forces()[0].cpu().tolist(),
        runtime_pose_writes=0,direct_door_commands=bool(a.mechanism_test),contact_material_audit=contact_material_audit,
        scope='Uninterrupted approach, opening, release and traversal; privileged live PhysX development' if continuous else 'Continuous approach through bimanual loaded aperture; privileged live PhysX; no traversal' if full_opening and sequence else 'Contact-free acquisition through bimanual loaded aperture; privileged live PhysX; no approach/traversal' if full_opening else balance_scope if a.sensor_balance_calibration else 'Sensor-only recurrent force actor; declared curriculum objective; no teacher or traversal claim' if sensor_actor else 'Continuous walk/lower/prepare/acquire/partial opening; privileged live PhysX; no traversal' if sequence else 'Contact-free acquisition and partial opening; privileged live PhysX; no traversal' if a.operate_after_acquisition else 'Contact-free acquisition teacher; privileged live PhysX; no opening or traversal' if a.acquisition else 'Direct-force mechanism calibration; NOT robot opening' if a.mechanism_test else 'Privileged near-handle motor reference; live PhysX; no traversal'),indent=2)+'\n')
    sources=[Path(__file__),Path(__file__).with_name('physx_teacher.py')]+[Path(__file__).resolve().parents[2]/'doorbench/dexterous'/n for n in ('stance.py','reset.py','contact_audit.py','isaac_materials.py','isaac_joint_passive.py')]
    if a.panel_push:sources.append(Path(__file__).with_name('panel_push_teacher.py'))
    if a.acquisition:sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in ('acquisition_teacher.py','isaac_tendons.py','grasp_verification.py','isaac_pad_audit.py')]
    if a.operation_hub_geometry:sources.append(Path(__file__).resolve().parents[2]/'doorbench/dexterous/handle_hub_avoidance.py')
    if a.operate_after_acquisition:sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in ('operation_teacher.py','isaac_opening_measurements.py')]
    if sequence:sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in
        ('full_sequence_teacher.py','approach_teacher.py','approach_lowering.py','locomotion.py','locomotion_approach.py','locomotion_manipulation.py','locomotion_posture.py','isaac_sensors.py')]
    inputs=[Path(a.robot_usd),Path(a.door_usd),Path(a.motors),Path(a.reference)]
    if a.native_robot:inputs.append(Path(a.native_robot))
    if a.grasp_profile_definition:inputs.append(Path(a.grasp_profile_definition))
    if sequence:
        inputs += [Path(v) for v in (a.full_sequence_reset,a.preparation_reference,a.locomotion_checkpoint,a.native_door)]
        for name,value in [('body-reset',a.full_sequence_reset),('preparation-reference',a.preparation_reference)]:
            (out/(name+'.json')).write_bytes(Path(value).read_bytes())
    if full_opening:
        inputs += [Path(v) for v in (a.left_palm_targets,a.right_release_screen,a.native_door,a.bimanual_runtime_screen) if v]
        for name in ('whole_body_return_path','whole_body_ungrip_path'):
            if getattr(a,name):
                path=Path(getattr(a,name));inputs.append(path)
                (out/(name.replace('_','-')+'.json')).write_bytes(path.read_bytes())
        sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in
            ('full_opening_teacher.py','walking_opening_teacher.py','full_opening_audit.py','isaac_opening_measurements.py','bimanual_transfer.py','bimanual_runtime.py',
             'panel_continuation.py','right_hand_release.py','robot_design_identity.py')]
        if a.left_planning_profile:
            sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in
                ('runtime_left_planner.py','landed_left_planner.py','landed_left_audit.py','left_approach_clearance.py')]
        if a.whole_body_return_path:
            sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in
                ('whole_body_return.py','right_release_return.py')]
        if a.whole_body_ungrip_path:
            sources.append(Path(__file__).resolve().parents[2]/'doorbench/dexterous/whole_body_ungrip.py')
    if continuous:
        sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in
            ('continuous_door_teacher.py','post_opening_teacher.py','post_opening.py','post_opening_route.py',
             'passage.py','isaac_post_opening_measurements.py')]
    if sensor_actor:
        inputs.append(Path(a.sensor_reset_preflight))
        if a.sensor_locomotion_calibration:
            inputs += [Path(a.sensor_locomotion_calibration),Path(a.sensor_locomotion_robot),Path(a.sensor_locomotion_checkpoint)]
            (out/'sensor-locomotion-calibration.json').write_bytes(Path(a.sensor_locomotion_calibration).read_bytes())
            sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in
                ('sensor_locomotion.py','sensor_walk_stop.py','sensor_balance.py','locomotion_manipulation.py','stance.py','motor_target_control.py','locomotion.py')]
        if a.sensor_policy_checkpoint:inputs.append(Path(a.sensor_policy_checkpoint))
        if a.sensor_balance_calibration:
            inputs += [Path(a.sensor_balance_calibration),Path(a.sensor_balance_robot)]
            (out/'sensor-balance-calibration.json').write_bytes(Path(a.sensor_balance_calibration).read_bytes())
            sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in
                ('sensor_balance_runtime.py','sensor_balance.py','locomotion_manipulation.py','stance.py','isaac_post_opening_measurements.py')]
            if a.sensor_arm_schedule:
                inputs.append(Path(a.sensor_arm_schedule))
                (out/'balance-arm-schedule.json').write_bytes(Path(a.sensor_arm_schedule).read_bytes())
                sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in
                    ('sensor_arm_balance_runtime.py','sensor_arm_balance.py','arm_balance_schedule.py')]
            if a.sensor_reach_protocol:
                inputs += [Path(a.sensor_reach_protocol),Path(a.sensor_reach_route)]
                (out/'balance-reach-protocol.json').write_bytes(Path(a.sensor_reach_protocol).read_bytes())
                (out/'balance-reach-route.json').write_bytes(Path(a.sensor_reach_route).read_bytes())
                sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in
                    ('sensor_reach_runtime.py','sensor_reach_evaluation.py','sensor_reach_balance.py','reach_balance_schedule.py')]
            if a.sensor_acquisition_protocol:
                inputs += [Path(a.sensor_acquisition_protocol),Path(a.sensor_acquisition_route)]
                (out/'balance-acquisition-protocol.json').write_bytes(Path(a.sensor_acquisition_protocol).read_bytes())
                (out/'balance-acquisition-route.json').write_bytes(Path(a.sensor_acquisition_route).read_bytes())
                sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in
                    ('sensor_acquisition_runtime.py','sensor_acquisition_evaluation.py','sensor_acquisition_evidence.py',
                     'sensor_acquisition_schedule.py','sensor_reach_runtime.py','sensor_reach_evaluation.py',
                     'sensor_reach_balance.py','reach_balance_schedule.py','isaac_acquisition_contacts.py')]
        sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name for name in
            ('sensor_actor.py','sensor_policy_controller.py','control_mode.py','isaac_pad_audit.py','isaac_tendons.py','grasp_verification.py','motor_contract_identity.py','sensor_reset_preflight.py','teacher_query_recording.py')]
    if a.sensor_layout:
        inputs.append(Path(a.sensor_layout))
        sources += [Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name
            for name in ('sensor_contract.py','isaac_sensors.py','isaac_sensor_recording.py','pose_gyro.py')]
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
        if passive_guard:
            invariant_getters.update(joint_friction=robot.root_physx_view.get_dof_friction_properties,
                                    joint_armature=robot.root_physx_view.get_dof_armatures)
        invariants={n:f().cpu().numpy().copy() for n,f in invariant_getters.items()}
        if motors.get('passive_tendons'):
            invariant_getters.update({name:getattr(robot.root_physx_view,name) for name in tendon_audit['backend_readback']})
            invariants.update({n:f().cpu().numpy().copy() for n,f in invariant_getters.items()})
        mechanical_audit=dict(scope='Joint limits and loopbacks at 500 Hz; scene contacts at '+('500' if physics_audit_enabled else '50')+' Hz. Privileged evaluator only.',
            max_joint_stop_penetration_rad=0.,max_self_penetration_m=0.,max_nonfoot_environment_penetration_m=0.,
            max_hand_door_penetration_m=0.,max_loopback_violation_rad=0.,contact_samples=0,capacity=16384)
    loop_pairs=[(index[f'{side}_{digit}J1'],index[f'{side}_{digit}J2']) for side in ('rh','lh') for digit in ('FF','MF','RF','LF')]
    # Keep measured joint velocities for every physical state archive, including
    # standalone acquisition. Later source-state planning must not assume rest
    # or finite-difference positions to invent an unrecorded initial velocity.
    acquisition_states={k:[] for k in ('time_s','root','joints','joint_velocity','motor_forces','door','door_velocity','torso_tilt_deg')}
    if continuous:
        acquisition_states.update({k:[] for k in ('actual_motor_forces','actual_joint_effort',
            'continuation_body_poses','actual_foot_loads','legacy_root_state_w')})
        motor_inverse=np.linalg.pinv(matrix.T)
        if np.linalg.matrix_rank(matrix.T)!=61 or not np.allclose(motor_inverse@matrix.T,np.eye(61),atol=1e-12,rtol=0):
            raise ValueError('Exact full-rank 61-motor transmission required for delivered-force readback')
        last_actual_motor_forces,_=actual_motor_delivery(robot.root_physx_view.get_dof_actuation_forces()[0].cpu().numpy(),
            robot.data.joint_vel[0].cpu().numpy(),matrix,motor_inverse,damp,friction)
        (out/'motor-readback-contract.json').write_text(json.dumps(dict(
            api='ArticulationView.get_dof_actuation_forces',
            semantics='Submitted generalized actuation-input readback from the backend; not independently measured joint torque',
            independent_reaction_api='get_dof_projected_joint_forces is a solver joint-reaction diagnostic and is not inverted as a motor command',
            archive_fields=dict(actual_joint_effort='Legacy field name: backend submitted generalized input',
                actual_motor_forces='Legacy field name: motor-equivalent reconstruction of submitted input'),
            initial_interval_s=[0.,0.],initial_motor_input=last_actual_motor_forces.tolist(),
            t0_note='Unstepped reset input and empty contact interval; not force evidence from an executed physics step',
            root_controller_field='root_link_state_w',legacy_diagnostic_field='legacy_root_state_w',
            root_body_name=robot.body_names[0],root_com_offset_in_actor_m=robot.data.body_com_pos_b[0,0].cpu().tolist(),
            angular_velocity_frame='world; converted to body-local only inside native free-joint calculators'),indent=2)+'\n')
    pad_evaluator=None;pad_steps=[]
    if physics_audit_enabled:
        from doorbench.dexterous.isaac_pad_audit import PhysXShadowPadAudit
        pad_evaluator=PhysXShadowPadAudit(hand_bodies,hand_contacts,handle_filter_index=0,profile=a.grasp_profile)
        pose=door.data.body_state_w[0,door.body_names.index('leaf_handle'),:7].cpu().numpy()
        rotation=Rotation.from_quat([*pose[4:7],pose[3]]).as_matrix()
        pad_steps.append(pad_evaluator.read(physics_dt=dt,time_s=0.,center=pose[:3]+rotation@grip_center,axis=rotation@grip_axis,half_length=grip_half,radius=grip_radius))
        acquisition_reset=dict(root=controller_root_state(robot.data,traverse=bool(continuous or a.sensor_locomotion_calibration or a.acquisition_stance_profile))[0].cpu().tolist(),joints=robot.data.joint_pos[0].cpu().tolist(),door=dict(zip(dnames,door.data.joint_pos[0].cpu().tolist())),
            contact_evidence_note='t=0 contact buffers before the first explicit step; full static native path/initial clearances are recorded separately with the reference')
        (out/'acquisition-reset.json').write_text(json.dumps(acquisition_reset,indent=2)+'\n')
    foot_loads=np.zeros(2);right_hand_contact_count=0;right_hand_buffered_contact_count=0
    sequence_steps=[];full_opening_steps=[];full_aperture_crossed=False;max_motor_delivery_error=0.
    balance_steps=[];balance_contact_stream=None;balance_arm_initial=None;balance_reach_initial=None
    if a.sensor_reach_protocol or a.sensor_acquisition_protocol:
        balance_reach_initial=dict(root13_actororigin=robot.data.root_link_state_w[0].cpu().tolist(),
            joint_position={name:float(robot.data.joint_pos[0,i].item()) for i,name in enumerate(rnames)})
        if a.sensor_acquisition_protocol:
            from doorbench.dexterous.isaac_acquisition_contacts import acquisition_hand_contact_counts
            initial_buffers=[v.cpu().numpy().copy() for v in audit_contacts.get_contact_data(dt)]
            af0,ap0,an0,ad0,ac0,ast0=initial_buffers
            from doorbench.dexterous.isaac_post_opening_measurements import continuation_contact_summary
            continuation_contact_summary(audit_paths,audit_filters,af0,an0,ad0,ac0,ast0,capacity=16384,physics_qualified=True)
            initial_patches=[]
            for i in range(len(audit_paths)):
                for j in range(ac0.shape[1]):
                    for k in range(int(ast0[i,j]),int(ast0[i,j]+ac0[i,j])):
                        initial_patches.append(dict(sensor=i,filter=j,slot=k,position=ap0[k].tolist(),normal=an0[k].tolist(),
                            force_N=float(af0[k,0]),distance_m=float(ad0[k,0])))
            balance_reach_initial.update(acquisition_hand_contact_counts(audit_paths,audit_filters,initial_patches),
                initial_door_position={name:float(door.data.joint_pos[0,dnames.index(name)].item())
                    for name in ('leaf_hinge','leaf_handle_hinge')},initial_contact_patches=initial_patches,
                contact_note='Reset-time contact buffers; bound static reset screening remains separately required')
        (out/('balance-acquisition-reset.json' if a.sensor_acquisition_protocol else 'balance-reach-reset.json')).write_text(json.dumps(balance_reach_initial,indent=2)+'\n')
    if a.sensor_arm_schedule:
        balance_arm_initial={name:float(robot.data.joint_pos[0,rnames.index(name)].item())
            for name in sensor_actor.goal_names}
        (out/'balance-arm-reset.json').write_text(json.dumps(balance_arm_initial,indent=2)+'\n')
    if a.sensor_balance_calibration:
        balance_contact_stream=gzip.open(out/'balance-contacts.jsonl.gz','wt',compresslevel=1)
        (out/'balance-contact-layout.json').write_text(json.dumps(dict(sensor_paths=audit_paths,
            filter_paths=audit_filters.tolist(),capacity=16384,
            scope='Occupied actual post-step normal contact slots; evaluator only'),indent=2)+'\n')
    traversal_steps=[];frozen_opening_report=None;max_transmission_residual=0.
    traversal_contact_stream=None
    if continuous:
        traversal_contact_stream=gzip.open(out/'traversal-contacts.jsonl.gz','wt',compresslevel=1)
        (out/'traversal-contact-layout.json').write_text(json.dumps(dict(sensor_paths=audit_paths,
            filter_paths=audit_filters.tolist(),capacity=16384,body_pose_order=list(continuous.post.pose_names),
            motor_names=list(continuous.motor_names),robot_joint_names=rnames,
            normal_and_friction_buffers_are_independent=True,
            scope='All occupied actual PhysX normal-contact and friction-patch slots; unused buffer capacity is omitted'),indent=2)+'\n')
    if sequence:
        feet=['left_ankle_link','right_ankle_link']
        foot_rows=[next(i for i,path in enumerate(audit_paths) if path.rsplit('/',1)[-1]==name) for name in feet]
        foot_bodies=[robot.body_names.index(name) for name in feet]
        foot_initial=None
    def read_full_opening_measurement(t):
        from doorbench.dexterous.isaac_opening_measurements import contact_force_pairs,panel_surface_loads,hand_contact_loads
        normal=audit_contacts.get_contact_force_matrix(dt=dt).cpu().numpy().copy()
        vectors,points,counts,starts=[v.cpu().numpy().copy() for v in audit_contacts.get_friction_data(dt)]
        pairs=contact_force_pairs(normal,vectors.reshape(16384,3),counts,starts,capacity=16384)
        measured_body=door.data.body_state_w[0,:,:7].cpu().numpy().copy()
        hp=measured_body[door.body_names.index('leaf_handle')];lp=measured_body[door.body_names.index('leaf')]
        angles={role:float(door.data.joint_pos[0,dnames.index(name)]) for role,name in
                [('operator','leaf_handle_hinge'),('leaf','leaf_hinge'),('latch','leaf_latch_bolt_slide')]}
        geometry=opening_geometry.read(time_s=t,pose_time_s=t,root=controller_root_state(robot.data,traverse=bool(continuous or a.sensor_locomotion_calibration or a.acquisition_stance_profile))[0].cpu().numpy(),
            joints=dict(zip(rnames,robot.data.joint_pos[0].cpu().numpy())),angles=angles,
            body_poses=dict(zip(robot.body_names,robot.data.body_state_w[0,:,:7].cpu().numpy())),
            handle_pose=hp,leaf_pose=lp)
        surface=panel_surface_loads(audit_paths,audit_filters,pairs,lp)
        loads=hand_contact_loads(audit_paths,pairs)
        result=dict(geometry=geometry,surface=surface,angles=angles,hand_forces=loads,
            root_height_m=float(robot.data.root_state_w[0,2]),
            torso_tilt_deg=float(np.degrees(np.arccos(np.clip(-robot.data.projected_gravity_b[0,2].item(),-1,1)))))
        if continuous:
            from doorbench.dexterous.isaac_post_opening_measurements import continuation_contact_summary
            af,ap,an,ad,ac,ast=[v.cpu().numpy().copy() for v in audit_contacts.get_contact_data(dt)]
            result['continuation']=continuation_contact_summary(audit_paths,audit_filters,af,an,ad,ac,ast,
                capacity=16384,physics_qualified=True)
            result['body_poses']={name:robot.data.body_state_w[0,robot.body_names.index(name),:7].cpu().numpy().copy()
                for name in continuous.post.pose_names if not name.startswith('leaf')}
            result['body_poses'].update(leaf=lp.copy(),leaf_handle=hp.copy())
            result['door_velocities']=dict(zip(dnames,door.data.joint_vel[0].cpu().numpy().copy()))
            def sparse_buffer(count, start, arrays):
                pairs=[];slots=[]
                for i,j in zip(*np.nonzero(count)):
                    n,first=int(count[i,j]),int(start[i,j])
                    if n:
                        pairs.append([int(i),int(j),first,n]);slots.extend(range(first,first+n))
                return dict(pairs=pairs,slots=slots,**{key:np.asarray(value)[slots].tolist() for key,value in arrays.items()})
            traversal_contact_stream.write(json.dumps(dict(time_s=t,pose_time_s=t,
                contact_interval_s=[max(0.,t-dt),t],
                normal=sparse_buffer(ac,ast,dict(force_N=af,point_world=ap,normal_world=an,distance_m=ad)),
                friction=sparse_buffer(counts,starts,dict(force_N=vectors.reshape(16384,3),point_world=points.reshape(16384,3)))),
                separators=(',',':'))+'\n')
        return result
    full_measurement=read_full_opening_measurement(0.) if full_opening else None

    def current_physics_checks():
        """Independent active-plant reduction at this exact executed prefix."""
        roots=np.asarray(acquisition_states['root']);commands=np.asarray(acquisition_states['motor_forces'])
        n=len(acquisition_states['time_s'])
        return dict(complete_physics_steps=bool(n and len(pad_steps)==n+1 and
                mechanical_audit['contact_samples']==n and all(len(v)==n for v in acquisition_states.values()) and
                np.allclose(acquisition_states['time_s'],np.arange(1,n+1)*dt,atol=1e-8,rtol=0)),
            joint_stops=mechanical_audit['max_joint_stop_penetration_rad']<.02,
            documented_loopbacks=mechanical_audit['max_loopback_violation_rad']<.02,
            self_collision=mechanical_audit['max_self_penetration_m']<.003,
            environment_collision=mechanical_audit['max_nonfoot_environment_penetration_m']<.003,
            working_hand_collision=mechanical_audit['max_hand_door_penetration_m']<.003,
            plant_parameters_unchanged=all(np.array_equal(invariants[n],f().cpu().numpy()) for n,f in invariant_getters.items()),
            closed_leaf_start=abs(acquisition_reset['door']['leaf_hinge'])<=.001,
            resting_operator_start=abs(acquisition_reset['door']['leaf_handle_hinge'])<=.001,
            initial_hand_door_contact_buffer_empty=pad_steps[0]['active_contact_count']==0,
            finite=bool(all(np.isfinite(np.asarray(v)).all() for v in acquisition_states.values())),
            upright=bool(len(roots) and max(acquisition_states['torso_tilt_deg'])<12 and roots[:,2].min()>.7),
            motor_delivery_matches_command=max_motor_delivery_error<1e-4,
            native_motor_caps=bool(len(commands) and np.all(commands>=force_ranges[:,0]-1e-5) and np.all(commands<=force_ranges[:,1]+1e-5) and
                (not continuous or (np.all(np.asarray(acquisition_states['actual_motor_forces'])>=force_ranges[:,0]-1e-5) and
                                    np.all(np.asarray(acquisition_states['actual_motor_forces'])<=force_ranges[:,1]+1e-5)))))

    def freeze_opening_prefix():
        nonlocal frozen_opening_report
        if frozen_opening_report is not None or continuous.opening_audit is None:
            return
        from doorbench.dexterous.full_opening_audit import full_opening_checks
        offset=sequence.acquisition_started
        full_checks=full_opening_checks(current_physics_checks(),steps=full_opening_steps,pad_steps=pad_steps,
            physics_dt=dt,maximum_seconds=a.seconds,target_aperture=a.target_aperture,
            operation_started=full_opening.operation_started+offset if full_opening.operation_started is not None else None,
            release_started=full_opening.release.started+offset if full_opening.release.started is not None else None)
        full_checks.update(separated_start=bool(np.linalg.norm(np.array(initial_root[:2])-sequence_reset['goal_xy'])>=.5-1e-9),
            walked_from_separate_start=bool(np.linalg.norm(np.asarray(acquisition_states['root'])[-1,:2]-np.array(initial_root[:2]))>.5),
            both_feet_swung=bool(all(any(r['foot_loads_N'][i]<10 and r['foot_height_m'][i]>foot_initial[i]+.015 for r in sequence_steps) for i in (0,1))),
            readiness_screen_passed=bool(sequence.readiness_screen and sequence.readiness_screen['passed']),
            actual_contact_free_preparation=bool(sequence.prep_started is not None and sequence.acquisition_started is not None and
                all(r['right_hand_contact_count']==0 for r in sequence_steps if sequence.prep_started<=r['time_s']<=sequence.acquisition_started)),
            body_solver_succeeded=sequence.body.controller.solver_failures==0,
            continuous_walk_to_opening=sequence.acquisition_started is not None,
            composition_crossing_qualified=continuous.opening_audit['passed'])
        frozen_opening_report=dict(scope='Frozen opening prefix of a continuous live PhysX traversal trial',
            passed=all(full_checks.values()),checks=full_checks,physics_dt_s=dt,
            duration_s=continuous.opening_audit['time_s'],maximum_seconds=a.seconds,
            target_aperture_rad=a.target_aperture,termination='frozen_actual_aperture_crossing',
            final_leaf_rad=full_opening_steps[-1]['angles']['leaf'],
            final_palm_load_N=full_opening_steps[-1]['surface']['palm_normal_load_N'],
            grasp_profile=a.grasp_profile,handoffs=dict(full_opening.handoffs),opening_clock_offset_s=offset,
            runtime_robot_pose_writes=0,direct_door_commands=False,native_mirror_steps=0,
            max_motor_delivery_error_Nm=max_motor_delivery_error,
            handoff_state_sha256=continuous.opening_audit['handoff_state_sha256'])
        (out/'full-opening-report.json').write_text(json.dumps(frozen_opening_report,indent=2)+'\n')
        (out/'opening-qualification.json').write_text(json.dumps(continuous.opening_audit,indent=2)+'\n')
        if continuous.handoff is not None:
            (out/'continuation-handoff.json').write_text(json.dumps(continuous.handoff,indent=2)+'\n')
        with gzip.open(out/'full-opening-steps.json.gz','wt') as stream:write_json_record_array(stream,full_opening_steps)
        if not frozen_opening_report['passed']:
            raise ValueError('Independent opening prefix audit failed; traversal cannot continue')

    def save_traversal_evidence():
        if not continuous:return
        with gzip.open(out/'traversal-steps.json.gz','wt') as stream:write_json_record_array(stream,traversal_steps)
        (out/'continuous-controller.json').write_text(json.dumps(dict(info=continuous.info,
            failure=continuous.failure,handoffs=continuous.handoffs,opening_audit=continuous.opening_audit,
            handoff=continuous.handoff),indent=2)+'\n')
    def save_attained_left_plan():
        if not full_opening:return
        for name,value in (('actual-left-planning',full_opening.left_planning_receipt),
                           ('actual-left-targets',full_opening.left_planning_targets)):
            if value is not None:(out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    def checkpoint_prefix():
        save_attained_left_plan()
        if a.sensor_balance_calibration:
            with gzip.open(out/'balance-steps.partial.json.gz','wt') as stream:write_json_record_array(stream,balance_steps)
            balance_contact_stream.flush()
        # Periodic atomic checkpoints survive a native shutdown that bypasses
        # Python exceptions. They are explicitly incomplete, never scored passes.
        (out/'trace.partial.json.tmp').write_text(json.dumps(rows)+'\n')
        os.replace(out/'trace.partial.json.tmp',out/'trace.partial.json')
        if physics_audit_enabled:
            np.savez_compressed(out/'acquisition-physics.partial.tmp.npz',**acquisition_states)
            os.replace(out/'acquisition-physics.partial.tmp.npz',out/'acquisition-physics.partial.npz')
            with gzip.open(out/'acquisition-pad-steps.partial.tmp.gz','wt') as stream:write_json_record_array(stream,pad_steps)
            os.replace(out/'acquisition-pad-steps.partial.tmp.gz',out/'acquisition-pad-steps.partial.json.gz')
        if full_opening:
            with gzip.open(out/'full-opening-steps.partial.tmp.gz','wt') as stream:write_json_record_array(stream,full_opening_steps)
            os.replace(out/'full-opening-steps.partial.tmp.gz',out/'full-opening-steps.partial.json.gz')
        if sequence:
            with gzip.open(out/'full-sequence-steps.partial.tmp.gz','wt') as stream:write_json_record_array(stream,sequence_steps)
            os.replace(out/'full-sequence-steps.partial.tmp.gz',out/'full-sequence-steps.partial.json.gz')
        if continuous:
            with gzip.open(out/'traversal-steps.partial.tmp.gz','wt') as stream:write_json_record_array(stream,traversal_steps)
            os.replace(out/'traversal-steps.partial.tmp.gz',out/'traversal-steps.partial.json.gz')
            traversal_contact_stream.flush()
        (out/'partial-evidence.json').write_text(json.dumps(dict(status='incomplete',passed=False,
            evidence_only=True,time_s=rows[-1]['time_s'] if rows else 0.,
            max_motor_delivery_error_Nm=max_motor_delivery_error,mechanical_audit=mechanical_audit))+'\n')
    teacher_queries=None
    if sensor_actor:
        from doorbench.dexterous.teacher_query_recording import TeacherQueryRecorder
        teacher_queries=TeacherQueryRecorder(out/'teacher-query-evidence',joint_names=rnames,hand_body_names=hand_paths,dt=dt)
    time_origin=float(sim.current_time)
    from doorbench.dexterous.wall_phase_timer import WallPhaseTimer
    wall_timing=WallPhaseTimer()
    try:
        for step in range(round(a.seconds/dt)):
            wall_timing.start()
            delivery_failure=None
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
                measured_args=(step*dt,controller_root_state(robot.data,traverse=bool(continuous or a.sensor_locomotion_calibration or a.acquisition_stance_profile))[0].cpu().numpy(),dict(zip(rnames,pos)),dict(zip(rnames,vel)),body[door.body_names.index('leaf_handle')])
                loads=dict(zip(hand_paths,hand_contacts.get_contact_force_matrix(dt=dt).cpu().numpy().sum(axis=1)))
                if full_opening:
                    state=full_measurement
                    geometry=state['geometry'];surface=state['surface']
                    physical=bool(state['root_height_m']>.7 and state['torso_tilt_deg']<12 and
                        mechanical_audit['max_joint_stop_penetration_rad']<.02 and mechanical_audit['max_loopback_violation_rad']<.02 and
                        all(mechanical_audit[k]<.003 for k in ('max_self_penetration_m','max_nonfoot_environment_penetration_m','max_hand_door_penetration_m')) and
                        max_motor_delivery_error<1e-4)
                    evidence=dict(grasp_qualified=pad_steps[-1]['valid_pad_grasp'],physics_qualified=physical,
                        right_pad_patches_valid=all(c['pad_qualified'] for c in pad_steps[-1]['contacts']),
                        hand_contact_count=right_hand_contact_count,left_panel_load_N=surface['total_normal_load_N'],
                        left_palm_load_N=surface['palm_normal_load_N'],right_lever_clearance_m=geometry['right_lever_clearance_m'])
                    opening_measurements=dict(evidence=evidence,right_palm_pose=geometry['right_palm_pose'],
                        pose_time_s=geometry['time_s'],contact_interval_s=(max(0.,step*dt-dt),step*dt))
                    if continuous:
                        continuation=state['continuation']
                        continuation_evidence=dict(continuation['evidence'],physics_qualified=physical)
                        evidence['hand_contact_count']+=continuation_evidence['left_hand_contacts']
                        forces,continuous_info=continuous.force(*measured_args[:4],continuation['foot_loads'],measured_args[4],
                            body[door.body_names.index('leaf')],state['angles'],state['hand_forces'],**opening_measurements,
                            body_poses=state['body_poses'],door_velocities=state['door_velocities'],
                            continuation_evidence=continuation_evidence,applied_motor_forces=last_actual_motor_forces,
                            release_normal_world=continuation['release_normal_world'])
                        teacher_info=dict(continuous_info['component'],continuous_phase=continuous_info['phase'],
                            continuous_completed=continuous_info['completed'])
                        traversal_steps.append(dict(time_s=step*dt,pose_time_s=geometry['time_s'],
                            contact_interval_s=list(opening_measurements['contact_interval_s']),
                            phase=teacher_info['phase'],root=measured_args[1].tolist(),
                            foot_loads_N=continuation['foot_loads'].tolist(),
                            continuation_evidence=continuation_evidence,
                            right_pad_patches_valid=evidence['right_pad_patches_valid'],
                            post_started=continuous.handoff is not None,
                            minimum_body_y_m=teacher_info.get('minimum_body_y_m'),
                            passage_completed=teacher_info.get('passage_completed',False),
                            preceding_actual_motor_forces=last_actual_motor_forces.tolist()))
                        freeze_opening_prefix()
                        if continuous.done:
                            # Current measurements end the last actual step. Do
                            # not apply another command after the qualified stop.
                            save_traversal_evidence()
                            break
                    elif sequence:
                        forces,teacher_info=sequence.force(*measured_args[:4],foot_loads,measured_args[4],
                            body[door.body_names.index('leaf')],state['angles'],state['hand_forces'],**opening_measurements)
                    else:
                        forces,teacher_info=full_opening.force(*measured_args,body[door.body_names.index('leaf')],
                            state['angles'],state['hand_forces'],**opening_measurements)
                elif operation:
                    angles={role:float(door.data.joint_pos[0,dnames.index(name)]) for role,name in [('operator','leaf_handle_hinge'),('leaf','leaf_hinge'),('latch','leaf_latch_bolt_slide')]}
                    # Both standalone and walking-to-operation teachers need the
                    # measured tangential load as well as the normal matrix.
                    from doorbench.dexterous.isaac_opening_measurements import contact_force_pairs,hand_contact_loads
                    normal=audit_contacts.get_contact_force_matrix(dt=dt).cpu().numpy().copy()
                    patch_friction,patch_points,patch_counts,patch_starts=[v.cpu().numpy().copy() for v in audit_contacts.get_friction_data(dt)]
                    pairs=contact_force_pairs(normal,patch_friction.reshape(16384,3),patch_counts,patch_starts,capacity=16384)
                    loads=hand_contact_loads(audit_paths,pairs)
                    if sequence:
                        forces,teacher_info=sequence.force(*measured_args[:4],foot_loads,measured_args[4],
                            body[door.body_names.index('leaf')],angles,loads,
                            grasp_qualified=pad_steps[-1]['valid_pad_grasp'],hand_contact_count=right_hand_contact_count)
                        if sequence.readiness_screen is not None and not (out/'actual-preparation-screen.json').exists():
                            (out/'actual-preparation-screen.json').write_text(json.dumps(sequence.readiness_screen,indent=2)+'\n')
                            (out/'actual-preparation-reference.json').write_text(json.dumps(sequence.actual_preparation)+'\n')
                    else:
                        forces,teacher_info=operation.force(*measured_args,body[door.body_names.index('leaf')],angles,loads,grasp_qualified=pad_steps[-1]['valid_pad_grasp'])
                elif not full_opening:forces,teacher_info=teacher.force(*measured_args,loads)
                if sequence and sequence.readiness_screen is not None and not (out/'actual-preparation-screen.json').exists():
                    (out/'actual-preparation-screen.json').write_text(json.dumps(sequence.readiness_screen,indent=2)+'\n')
                    (out/'actual-preparation-reference.json').write_text(json.dumps(sequence.actual_preparation)+'\n')
                ctrl=teacher.target.copy();feedforward=np.zeros_like(forces)
            if sensor_actor:
                if step==0 and (a.sensor_locomotion_calibration or getattr(sensor_actor,'action_semantics',None)=='original_motor_target_v1'):
                    # Encoders are available at reset without inventing a past
                    # IMU/tactile interval or advancing physics.
                    ji=sensor_recorder.joint_indices
                    sensor_recorder.builder.push('joint_position',pos[ji],capture_s=0.)
                    sensor_recorder.builder.push('joint_velocity',vel[ji],capture_s=0.)
                measured_body=door.data.body_state_w[0,:,:7].cpu().numpy()
                teacher_queries.record(time_s=step*dt,root_state=robot.data.root_state_w[0].cpu().numpy(),
                    joint_position=pos,joint_velocity=vel,handle_pose=measured_body[door.body_names.index('leaf_handle')],
                    leaf_pose=measured_body[door.body_names.index('leaf')],
                    door_position=door.data.joint_pos[0,[dnames.index(n) for n in ('leaf_handle_hinge','leaf_hinge','leaf_latch_bolt_slide')]].cpu().numpy(),
                    right_hand_forces_world=hand_contacts.get_contact_force_matrix(dt=dt).cpu().numpy().sum(axis=1))
                packet=sensor_recorder.builder.observe(now_s=step*dt,previous_action=sensor_actor.previous_action)
                if a.sensor_acquisition_protocol and getattr(sensor_actor._controller,'reflex',None) is not None:
                    if step==0:pressure_inputs={k:[] for k in ('tactile','sensor_time_s','sensor_valid')}
                    for k in pressure_inputs:pressure_inputs[k].append(packet[k].copy())
                forces=sensor_actor.force(packet,now_s=step*dt)
                if step==0:sensor_recorder.record_initial_decision(packet,forces)
                teacher_info=dict(phase='sensor_policy',runtime_inputs='numeric robot sensor packet and local acquisition clock',teacher_fallback=False) if not a.sensor_balance_calibration else dict(
                    **sensor_actor.last_info,phase='sensor_acquisition_balance' if a.sensor_acquisition_protocol else 'sensor_reach_balance' if a.sensor_reach_protocol else 'sensor_arm_balance' if a.sensor_arm_schedule else 'sensor_balance',teacher_fallback=False)
            if a.sensor_locomotion_calibration:
                teacher_info=dict(**sensor_actor.last_info,phase='sensor_locomotion',teacher_fallback=False)
            torque=matrix.T@forces-damp*vel-friction*np.tanh(vel/.001)
            robot.set_joint_effort_target(torch.tensor(torque[None],device=a.device,dtype=torch.float32))
            door.set_joint_position_target(target);door.set_joint_velocity_target(torch.zeros_like(target))
            if a.mechanism_test:
                effort=torch.zeros_like(target)
                effort[0,dnames.index('leaf_handle_hinge')]=1.5 if step*dt>.3 else 0.
                effort[0,dnames.index('leaf_hinge')]=3. if step*dt>1.5 else 0.
                door.set_joint_effort_target(effort)
            robot.write_data_to_sim();door.write_data_to_sim()
            wall_timing.mark('controller_and_submission')
            contacts.clear();sim.step(render=False)
            wall_timing.mark('physics_step')
            if passive_guard:
                try:passive_guard.check(robot.root_physx_view,time_s=(step+1)*dt)
                except ValueError:
                    (out/'joint-passive-invariants.json').write_text(json.dumps(passive_guard.receipt(),indent=2)+'\n')
                    raise
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
                    if a.sensor_balance_calibration:
                        # This camera is diagnostic only. Balance starts with
                        # the hand above the lever; a lever-only crop misses it.
                        palm=robot.data.body_state_w[0,robot.body_names.index('rh_palm'),:3].cpu().numpy()
                        center=(center+palm)/2.
                        eye=center+lrot@initial_leaf_rotation.T@np.array([.40,-.64,.32])
                    review_camera.set_world_poses_from_view(eyes=torch.tensor(np.array([eye]),device=a.device,dtype=torch.float32),
                        targets=torch.tensor(np.array([center]),device=a.device,dtype=torch.float32))
                sim.render()
            if abs(float(sim.current_time)-time_origin-(step+1)*dt)>.0001:
                raise RuntimeError('Physics clock changed outside the explicit motor timestep')
            robot.update(dt);door.update(dt)
            wall_timing.mark('render_and_state_refresh')
            if physics_audit_enabled:
                if sequence and foot_initial is None:foot_initial=robot.data.body_state_w[0,foot_bodies,2].cpu().numpy().copy()
                delivered=robot.root_physx_view.get_dof_actuation_forces()[0].cpu().numpy()
                max_motor_delivery_error=max(max_motor_delivery_error,float(np.max(np.abs(delivered-torque))))
                if continuous:
                    try:
                        last_actual_motor_forces,residual=actual_motor_delivery(delivered,vel,matrix,motor_inverse,damp,friction)
                        max_transmission_residual=max(max_transmission_residual,residual)
                        if np.any(last_actual_motor_forces<force_ranges[:,0]-1e-5) or np.any(last_actual_motor_forces>force_ranges[:,1]+1e-5):
                            raise ValueError('Actual reconstructed motor delivery exceeded original force caps')
                    except ValueError as error:
                        delivery_failure=str(error)
                        # Retain the actual failed interval before stopping. This
                        # algebraic reconstruction remains explicitly unqualified.
                        last_actual_motor_forces=motor_inverse@(delivered+damp*vel+friction*np.tanh(vel/.001))
                        (out/'motor-delivery-failure.json').write_text(json.dumps(dict(time_s=(step+1)*dt,
                            error=delivery_failure,actual_joint_effort=delivered.tolist(),
                            pre_step_velocity=vel.tolist(),requested_motor_forces=forces.tolist()))+'\n')
            if mechanical_audit is not None:
                qnew=robot.data.joint_pos[0].cpu().numpy()
                mechanical_audit['max_joint_stop_penetration_rad']=max(mechanical_audit['max_joint_stop_penetration_rad'],
                    float(np.maximum(limits[:,0]-qnew,qnew-limits[:,1]).max()))
                mechanical_audit['max_loopback_violation_rad']=max(mechanical_audit['max_loopback_violation_rad'],max(float(qnew[j1]-qnew[j2]) for j1,j2 in loop_pairs))
                if physics_audit_enabled or step%10==0:
                    af,ap,an,ad,ac,ast=[v.cpu().numpy().copy() for v in audit_contacts.get_contact_data(dt)]
                    if ac.sum()>=16384:raise RuntimeError('Mechanical contact audit buffer exhausted')
                    mechanical_audit['contact_samples']+=1
                    if sequence or full_opening:
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
                    if a.sensor_balance_calibration:
                        from doorbench.dexterous.isaac_post_opening_measurements import continuation_contact_summary
                        summary=continuation_contact_summary(audit_paths,audit_filters,af,an,ad,ac,ast,
                            capacity=16384,physics_qualified=True)
                        touched=0;occupied=[]
                        for i,path in enumerate(audit_paths):
                            for j in range(ac.shape[1]):
                                for k in range(int(ast[i,j]),int(ast[i,j]+ac[i,j])):
                                    occupied.append(dict(sensor=i,filter=j,slot=k,position=ap[k].tolist(),
                                        normal=an[k].tolist(),force_N=float(af[k,0]),distance_m=float(ad[k,0])))
                                    if path.rsplit('/',1)[-1].startswith(('rh_','lh_')):
                                        touched+=int(float(ad[k,0])<=0. or float(af[k,0])>1e-8)
                        balance_contact_stream.write(json.dumps(dict(interval_start_s=step*dt,
                            interval_end_s=(step+1)*dt,contacts=occupied),separators=(',',':'))+'\n')
                        balance_steps.append(dict(time_s=(step+1)*dt,
                            root13_actororigin=robot.data.root_link_state_w[0].cpu().tolist(),
                            torso_tilt_deg=float(np.degrees(np.arccos(np.clip(-robot.data.projected_gravity_b[0,2].item(),-1,1)))),
                            foot_floor_loads=summary['foot_loads'].tolist(),hand_contact_count=touched,
                            controller_info=sensor_actor.last_info))
                        if a.sensor_arm_schedule:
                            balance_steps[-1]['actual_arm_joint_position']={name:float(robot.data.joint_pos[0,rnames.index(name)].item())
                                for name in sensor_actor.goal_names}
                        if a.sensor_reach_protocol or a.sensor_acquisition_protocol:
                            balance_steps[-1]['actual_joint_position']={name:float(robot.data.joint_pos[0,i].item())
                                for i,name in enumerate(rnames)}
                        if a.sensor_acquisition_protocol:
                            balance_steps[-1].update(acquisition_hand_contact_counts(audit_paths,audit_filters,occupied))
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
                acquisition_states['root'].append(controller_root_state(robot.data,traverse=bool(continuous or a.sensor_locomotion_calibration or a.acquisition_stance_profile))[0].cpu().numpy().copy())
                acquisition_states['joints'].append(robot.data.joint_pos[0].cpu().numpy().copy())
                acquisition_states['joint_velocity'].append(robot.data.joint_vel[0].cpu().numpy().copy())
                acquisition_states['motor_forces'].append(forces.copy())
                acquisition_states['door'].append(door.data.joint_pos[0].cpu().numpy().copy())
                acquisition_states['door_velocity'].append(door.data.joint_vel[0].cpu().numpy().copy())
                acquisition_states['torso_tilt_deg'].append(float(np.degrees(np.arccos(np.clip(-robot.data.projected_gravity_b[0,2].item(),-1,1)))))
                if continuous:
                    acquisition_states['legacy_root_state_w'].append(robot.data.root_state_w[0].cpu().numpy().copy())
                    acquisition_states['actual_motor_forces'].append(last_actual_motor_forces.copy())
                    acquisition_states['actual_joint_effort'].append(delivered.copy())
                pose=door.data.body_state_w[0,door.body_names.index('leaf_handle'),:7].cpu().numpy()
                rotation=Rotation.from_quat([*pose[4:7],pose[3]]).as_matrix()
                actual_pad=pad_evaluator.read(physics_dt=dt,time_s=(step+1)*dt,center=pose[:3]+rotation@grip_center,axis=rotation@grip_axis,
                    half_length=grip_half,radius=grip_radius,include_evidence=bool(a.sensor_acquisition_protocol or a.acquisition))
                if a.sensor_acquisition_protocol:
                    balance_steps[-1]['pad_evidence']=actual_pad.pop('raw_evidence')
                pad_steps.append(actual_pad)
                if step%500==0:
                    (out/'latest-pad-audit.json').write_text(json.dumps(pad_steps[-1],indent=2)+'\n')
            if full_opening:
                full_measurement=read_full_opening_measurement((step+1)*dt)
                if frozen_opening_report is None:
                    full_opening_steps.append(dict(time_s=(step+1)*dt,geometry=full_measurement['geometry'],
                        surface=full_measurement['surface'],angles=full_measurement['angles'],teacher=teacher_info))
                if continuous:
                    acquisition_states['continuation_body_poses'].append(np.array([full_measurement['body_poses'][name] for name in continuous.post.pose_names]))
                    acquisition_states['actual_foot_loads'].append(full_measurement['continuation']['foot_loads'].copy())
            if delivery_failure is not None:
                raise RuntimeError(delivery_failure)
            if sensor_recorder:
                previous_action=2*(forces-force_ranges[:,0])/(force_ranges[:,1]-force_ranges[:,0])-1
                sensor_recorder.update(robot_data=robot.data,dt=dt,time_s=(step+1)*dt,
                    previous_action=previous_action,rendered=step%20==0)
            all_contacts.extend(dict(time_s=(step+1)*dt,**c) for c in contacts)
            if step==0:
                (out/'initial-body-poses.json').write_text(json.dumps(dict(
                    time_s=(step+1)*dt, pose_epoch='post-first-physics-step',
                    pose_convention='world body-origin xyz and wxyz quaternion',
                    robot=dict(zip(robot.body_names,robot.data.body_state_w[0,:,:7].cpu().tolist())),
                    door=dict(zip(door.body_names,door.data.body_state_w[0,:,:7].cpu().tolist()))),indent=2)+'\n')
            if step%10==0:
                state=controller_root_state(robot.data,traverse=bool(continuous or a.sensor_locomotion_calibration or a.acquisition_stance_profile))[0].cpu().numpy()
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
            wall_timing.mark('audit_and_state_recording')
            if camera and step%20==0:
                camera.update(dt*20);frame=camera.data.output['rgb'][0].cpu().numpy()[...,:3]
                writer.append_data(frame)
                if step%500==0:imageio.imwrite(out/f'frame-{step:05d}.png',frame)
            if hand_camera and step%20==0:
                hand_camera.update(dt*20);hand_frame=hand_camera.data.output['rgb'][0].cpu().numpy()[...,:3]
                hand_writer.append_data(hand_frame)
                if step%500==0:imageio.imwrite(out/f'hand-frame-{step:05d}.png',hand_frame)
            wall_timing.mark('camera_and_video_output')
            if (step+1)%1000==0:
                checkpoint_prefix()
                if teacher_queries:teacher_queries.finish(complete=False,executed_steps=len(acquisition_states['time_s']))
            if sensor_recorder and (step+1)%2500==0:sensor_recorder.finish(complete=False)
            wall_timing.finish('periodic_checkpoints')
            if (step+1)%250==0:
                (out/'wall-timing.json').write_text(json.dumps(wall_timing.receipt(),indent=2)+'\n')
            if (step+1)%50==0 and (out/'stop.request').exists():
                (out/'early-stop.json').write_text(json.dumps(dict(reason='Requested graceful diagnostic stop',time_s=(step+1)*dt))+'\n')
                break
            if full_opening and full_measurement['angles']['leaf']>=a.target_aperture:
                full_aperture_crossed=True
                if not continuous:break
            if not torch.isfinite(robot.data.joint_pos).all():raise RuntimeError('Nonfinite robot state')
            if rows and (rows[-1]['root'][2]<.45 or rows[-1]['torso_tilt_deg']>45):
                (out/'early-stop.json').write_text(json.dumps(dict(reason='Robot fell',time_s=(step+1)*dt))+'\n')
                break
    except BaseException as run_error:
        (out/'wall-timing.json').write_text(json.dumps(wall_timing.receipt(),indent=2)+'\n')
        if passive_guard:
            (out/'joint-passive-invariants.json').write_text(json.dumps(passive_guard.receipt(),indent=2)+'\n')
        save_attained_left_plan()
        if balance_contact_stream:
            balance_contact_stream.close()
            with gzip.open(out/'balance-steps.json.gz','wt') as stream:write_json_record_array(stream,balance_steps)
            failed_balance=dict(passed=False,scope=balance_scope,
                error=str(run_error),duration_s=balance_steps[-1]['time_s'] if balance_steps else 0.,
                physical_evidence_complete=False,teacher_fallback=False)
            for name in ('balance-report.json','report.json'):
                (out/name).write_text(json.dumps(failed_balance,indent=2)+'\n')
        # Preserve the actual executed prefix even when a controller or backend
        # error prevents normal qualification. These files never imply a pass.
        (out/'trace.json').write_text(json.dumps(rows)+'\n')
        if physics_audit_enabled:
            np.savez_compressed(out/'acquisition-physics.npz',**acquisition_states)
            with gzip.open(out/'acquisition-pad-steps.json.gz','wt') as stream:write_json_record_array(stream,pad_steps)
        if full_opening:
            with gzip.open(out/'full-opening-steps.json.gz','wt') as stream:write_json_record_array(stream,full_opening_steps)
        if sequence:
            with gzip.open(out/'full-sequence-steps.json.gz','wt') as stream:write_json_record_array(stream,sequence_steps)
        if continuous:
            traversal_contact_stream.close()
            try:freeze_opening_prefix()
            except Exception as audit_error:
                (out/'opening-freeze-error.txt').write_text(str(audit_error)+'\n')
            save_traversal_evidence()
            (out/'contacts.json').write_text(json.dumps(all_contacts)+'\n')
            (out/'mechanical-audit.partial.json').write_text(json.dumps(mechanical_audit,indent=2)+'\n')
            failure_report=dict(passed=False,status='controller_or_backend_failure',scope='Uninterrupted privileged traversal development',
                error=str(run_error),duration_s=acquisition_states['time_s'][-1] if acquisition_states['time_s'] else 0.,
                opening_prefix=frozen_opening_report,controller_failure=continuous.failure,
                runtime_robot_pose_writes=0,direct_door_commands=False)
            for name in ('traversal-report.json','report.json'):
                (out/name).write_text(json.dumps(failure_report,indent=2)+'\n')
        if sensor_recorder:sensor_recorder.finish(complete=False)
        if teacher_queries:teacher_queries.finish(complete=False,executed_steps=len(acquisition_states['time_s']))
        if writer:writer.close()
        if hand_writer:hand_writer.close()
        raise
    (out/'wall-timing.json').write_text(json.dumps(wall_timing.receipt(),indent=2)+'\n')
    save_attained_left_plan()
    if balance_contact_stream:
        balance_contact_stream.close()
        with gzip.open(out/'balance-steps.json.gz','wt') as stream:write_json_record_array(stream,balance_steps)
    (out/'trace.json').write_text(json.dumps(rows)+'\n')
    if physics_audit_enabled:np.savez_compressed(out/'acquisition-physics.npz',**acquisition_states)
    if physics_audit_enabled:
        with gzip.open(out/'acquisition-pad-steps.json.gz','wt') as stream:write_json_record_array(stream,pad_steps)
    if full_opening:
        with gzip.open(out/'full-opening-steps.json.gz','wt') as stream:write_json_record_array(stream,full_opening_steps)
    save_traversal_evidence()
    if traversal_contact_stream:traversal_contact_stream.close()
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
        report=dict(scope='Right-hand acquisition diagnostic only; intentional later release means this is not the full-opening run result. See full-opening-report.json' if full_opening else 'Acquisition diagnostic remains incomplete during the separate balance experiment; balance-report.json is the scoped result' if a.sensor_balance_calibration else 'Sensor-only actor physical acquisition/hold audit within declared curriculum task' if sensor_actor else 'Acquisition/hold audit within a continuous operation trial; see operation-report.json for mechanism outcome' if operation else 'Live PhysX privileged acquisition only; no approach/opening/traversal or sensor-only claim',passed=all(checks.values()),checks=checks,
            grasp_profile=a.grasp_profile,
            original_distal_pad_hold=bool(len(tail)>=round(.5/dt)+1 and all(r.get('distal_pad_grasp',r)['valid_pad_grasp'] for r in tail)),
            max_motor_delivery_error_Nm=max_motor_delivery_error,
            physics_dt_s=dt,duration_s=acquisition_states['time_s'][-1],runtime_robot_pose_writes=0,direct_door_commands=False,
            initial_contact_evidence_note=acquisition_reset['contact_evidence_note'],final_pad_grasp=pad_steps[-1])
        if sensor_actor and not (a.sensor_balance_calibration or a.sensor_locomotion_calibration):
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
        if a.sensor_balance_calibration:
            balance_checks={k:v for k,v in checks.items() if k!='sustained_pad_grasp'}
            if a.sensor_acquisition_protocol:
                from doorbench.dexterous.sensor_acquisition_evaluation import evaluate_sensor_acquisition_balance
                balance_report=evaluate_sensor_acquisition_balance(balance_steps,balance_checks,
                    robot_xml=a.sensor_balance_robot,motors=motors,
                    initial_root13_actororigin=balance_reach_initial['root13_actororigin'],
                    initial_joint_position=balance_reach_initial['joint_position'],
                    initial_hand_contact_count=balance_reach_initial['hand_contact_count'],
                    initial_door_position=balance_reach_initial['initial_door_position'],
                    calibration=a.sensor_balance_calibration,protocol=a.sensor_acquisition_protocol,
                    joint_route=a.sensor_acquisition_route,physics_dt_s=dt,expected_duration_s=a.seconds,
                    reflex_inputs=(sensor_recorder.layout,pressure_inputs) if getattr(sensor_actor._controller,'reflex',None) is not None else None)
            elif a.sensor_reach_protocol:
                from doorbench.dexterous.sensor_reach_evaluation import evaluate_sensor_reach_balance
                balance_report=evaluate_sensor_reach_balance(balance_steps,balance_checks,
                    robot_xml=a.sensor_balance_robot,motors=motors,
                    initial_root13_actororigin=balance_reach_initial['root13_actororigin'],
                    initial_joint_position=balance_reach_initial['joint_position'],
                    calibration=a.sensor_balance_calibration,protocol=a.sensor_reach_protocol,
                    joint_route=a.sensor_reach_route,physics_dt_s=dt,expected_duration_s=a.seconds)
            elif a.sensor_arm_schedule:
                from doorbench.dexterous.sensor_arm_balance_runtime import evaluate_sensor_arm_balance
                balance_report=evaluate_sensor_arm_balance(balance_steps,balance_checks,
                    initial_arm_joint_position=balance_arm_initial,schedule=json.loads(Path(a.sensor_arm_schedule).read_text()),
                    physics_dt_s=dt,expected_duration_s=a.seconds)
                balance_report['schedule_sha256']=sensor_actor.schedule_sha256
            else:
                from doorbench.dexterous.sensor_balance_runtime import evaluate_sensor_balance
                balance_report=evaluate_sensor_balance(balance_steps,balance_checks,
                    physics_dt_s=dt,expected_duration_s=a.seconds)
            balance_report.update(calibration_sha256=hashlib.sha256(Path(a.sensor_balance_calibration).read_bytes()).hexdigest(),
                joint_passive_profile=a.joint_passive_profile,
                runtime_robot_pose_writes=0,direct_door_commands=False,teacher_fallback=False,
                runtime_controller_inputs='Encoders, local IMU, foot tactile and previous command; fixed robot/posture calibration; RGB captured but unused'+('; separate frozen joint-only torso, arm and finger route' if a.sensor_reach_protocol or a.sensor_acquisition_protocol else '; separate frozen scripted arm/wrist joint schedule' if a.sensor_arm_schedule else ''),
                force_readback='Submitted backend actuation input, not an independent joint torque sensor')
            for name in ('balance-report.json','report.json'):
                (out/name).write_text(json.dumps(balance_report,indent=2)+'\n')
            print('BALANCE_RESULT '+json.dumps(balance_report),flush=True)
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
                teacher_hand_loads='Both hands, world-space normal plus distinct friction patches from actual PhysX contact buffers',
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
                with gzip.open(out/'full-sequence-steps.json.gz','wt') as stream:write_json_record_array(stream,sequence_steps)
                (out/'full-sequence-report.json').write_text(json.dumps(operation_report,indent=2)+'\n')
            (out/'operation-report.json').write_text(json.dumps(operation_report,indent=2)+'\n')
            (out/'report.json').write_text(json.dumps(operation_report,indent=2)+'\n')
            print('OPERATION_RESULT '+json.dumps({k:v for k,v in operation_report.items() if k!='final_pad_grasp'}),flush=True)
        if full_opening and (not continuous or frozen_opening_report is None):
            from doorbench.dexterous.full_opening_audit import full_opening_checks
            full_checks=full_opening_checks(checks,steps=full_opening_steps,pad_steps=pad_steps,
                physics_dt=dt,maximum_seconds=a.seconds,target_aperture=a.target_aperture,
                operation_started=None if full_opening.operation_started is None else full_opening.operation_started+(sequence.acquisition_started if sequence else 0.),
                release_started=None if full_opening.release.started is None else full_opening.release.started+(sequence.acquisition_started if sequence else 0.))
            if sequence:
                full_checks.update(separated_start=bool(np.linalg.norm(np.array(initial_root[:2])-sequence_reset['goal_xy'])>=.5-1e-9),
                    walked_from_separate_start=bool(np.linalg.norm(root_states[-1,:2]-np.array(initial_root[:2]))>.5),
                    both_feet_swung=bool(all(any(r['foot_loads_N'][i]<10 and r['foot_height_m'][i]>foot_initial[i]+.015 for r in sequence_steps) for i in (0,1))),
                    readiness_screen_passed=bool(sequence.readiness_screen and sequence.readiness_screen['passed']),
                    actual_contact_free_preparation=bool(sequence.prep_started is not None and sequence.acquisition_started is not None and
                        all(r['right_hand_contact_count']==0 for r in sequence_steps if sequence.prep_started<=r['time_s']<=sequence.acquisition_started)),
                    body_solver_succeeded=sequence.body.controller.solver_failures==0,
                    continuous_walk_to_opening=sequence.acquisition_started is not None)
                with gzip.open(out/'full-sequence-steps.json.gz','wt') as stream:write_json_record_array(stream,sequence_steps)
            end=acquisition_states['time_s'][-1]
            full_report=dict(scope='Continuous live PhysX approach, acquisition and bimanual loaded aperture; no traversal or sensor-only claim' if sequence else 'Live PhysX contact-free acquisition, lever/latch operation and bimanual loaded aperture; no approach/traversal or sensor-only claim',
                passed=all(full_checks.values()),checks=full_checks,physics_dt_s=dt,duration_s=end,
                maximum_seconds=a.seconds,target_aperture_rad=a.target_aperture,
                termination='declared_aperture_crossing' if full_aperture_crossed else 'declared_timeout_or_failure',
                final_leaf_rad=full_opening_steps[-1]['angles']['leaf'],
                final_palm_load_N=full_opening_steps[-1]['surface']['palm_normal_load_N'],
                grasp_profile=a.grasp_profile,handoffs=full_opening.handoffs,
                opening_clock_offset_s=sequence.acquisition_started if sequence else 0.,
                sequence_handoffs=sequence.handoffs if sequence else None,
                runtime_robot_pose_writes=0,direct_door_commands=False,native_mirror_steps=0,
                max_motor_delivery_error_Nm=max_motor_delivery_error,
                clearance_scope='Authored native geometry at synchronized actual PhysX state; independent PhysX contact/penetration gates retained',
                initial_contact_evidence_note=acquisition_reset['contact_evidence_note'])
            (out/'full-opening-report.json').write_text(json.dumps(full_report,indent=2)+'\n')
            (out/'report.json').write_text(json.dumps(full_report,indent=2)+'\n')
            print('FULL_OPENING_RESULT '+json.dumps(full_report),flush=True)
        if continuous:
            end=acquisition_states['time_s'][-1]
            final_checks=independent_traversal_checks(current_physics_checks(),frozen_opening_report,
                traversal_steps,dt=dt,end_s=end,maximum_seconds=a.seconds,
                controller_done=continuous.done,failure=continuous.failure)
            final_checks['no_requested_early_stop']=not (out/'early-stop.json').exists()
            final_checks['actual_motor_transmission_residual']=max_transmission_residual<=1e-5
            final_checks['delivered_motor_command_continuity']=continuous.maximum_motor_delivery_error<=1e-5
            traversal_report=dict(scope='Uninterrupted live PhysX approach, opening, release, stow, rise and passage; privileged teacher',
                passed=all(final_checks.values()),checks=final_checks,duration_s=end,
                physics_dt_s=dt,maximum_seconds=a.seconds,
                termination='qualified_whole_body_quiet_finish' if continuous.done else 'declared_timeout_or_failure',
                opening_prefix=frozen_opening_report,controller_failure=continuous.failure,
                handoffs=continuous.handoffs,final_root=acquisition_states['root'][-1].tolist(),
                final_leaf_rad=float(acquisition_states['door'][-1][dnames.index('leaf_hinge')]),
                final_traversal_measurement=traversal_steps[-1] if traversal_steps else None,
                maximum_actual_motor_delivery_error_Nm=continuous.maximum_motor_delivery_error,
                maximum_transmission_residual_Nm=max_transmission_residual,
                stow_profile='sequential-v2',phase_seconds=a.traversal_stow_phase_seconds,handoff_policy=a.opening_handoff_policy,
                runtime_robot_pose_writes=0,direct_door_commands=False,native_mirror_steps=0,
                force_readback='Submitted backend actuation input plus matching pre-step native passive terms, inverted through full-rank original61motor transmission; not an independent torque measurement',
                body_pose_order=list(continuous.post.pose_names),grasp_profile=a.grasp_profile,
                clearance_scope='Actual root/joints and validated body origins in unchanged native-shape calculator; live PhysX penetration gates remain active')
            for name in ('traversal-report.json','report.json'):
                (out/name).write_text(json.dumps(traversal_report,indent=2)+'\n')
            print('TRAVERSAL_RESULT '+json.dumps({k:v for k,v in traversal_report.items() if k!='final_traversal_measurement'}),flush=True)
    if a.sensor_locomotion_calibration:
        loco_checks=current_physics_checks()
        loco_checks['declared_duration']=len(acquisition_states['time_s'])==round(a.seconds/dt)
        roots=np.asarray(acquisition_states['root'])
        if a.sensor_locomotion_stop_after_seconds is not None:
            tail=roots[np.asarray(acquisition_states['time_s'])>=a.seconds-1.-1e-8]
            loco_checks['supported_stance_handoff']=sensor_actor.balance is not None
            loco_checks['stance_solver_clean']=sensor_actor.balance is not None and sensor_actor.balance.qp_failures==0
            loco_checks['quiet_final_second']=bool(len(tail)>=500 and np.max(np.linalg.norm(tail[:,7:9],axis=1))<.03 and np.max(np.ptp(tail[:,:2],axis=0))<.03)
        loco_report=dict(passed=all(loco_checks.values()),checks=loco_checks,
            scope='Pinned H1 sensor-only locomotion component; constant body command, no learned vision task policy or door opening',
            duration_s=float(acquisition_states['time_s'][-1]),runtime_robot_pose_writes=0,teacher_actions=0,
            action_semantics=sensor_actor.action_semantics,checkpoint_sha256=sensor_actor.checkpoint_sha256,
            root_displacement_world_m=(roots[-1,:3]-np.asarray(acquisition_reset['root'])[:3]).tolist(),
            controller_inputs='Joint encoders, own mounted IMU delta angle, internal gait clock and static calibration',
            initial_orientation_calibration='upright; yaw/XY arbitrary',joint_passive_profile=a.joint_passive_profile,
            maximum_torso_tilt_deg=max(acquisition_states['torso_tilt_deg']),
            max_motor_delivery_error_Nm=max_motor_delivery_error,vision_task_policy=False,full_task_qualified=False)
        if a.sensor_locomotion_stop_after_seconds is not None:
            loco_report.update(scope='Sensor-driven walking, gait braking and supported stance transition; no vision navigation or door task',controller_final_info=sensor_actor.last_info,controller_inputs='Joint encoders, own mounted IMU, foot tactile grids, internal gait clock, static calibration and a fixed stop request')
        for name in ('locomotion-report.json','report.json'):
            (out/name).write_text(json.dumps(loco_report,indent=2)+'\n')
        print('LOCOMOTION_RESULT '+json.dumps(loco_report),flush=True)
    if passive_guard:
        (out/'joint-passive-invariants.json').write_text(json.dumps(passive_guard.receipt(),indent=2)+'\n')
    completed_recording=(len(acquisition_states['time_s'])==round(a.seconds/dt) or
        bool(continuous.done if continuous else full_opening and full_aperture_crossed)) and not (out/'early-stop.json').exists()
    if sensor_recorder:sensor_recorder.finish(complete=completed_recording and len(sensor_recorder.times)==len(acquisition_states['time_s']))
    if teacher_queries:teacher_queries.finish(complete=completed_recording,executed_steps=len(acquisition_states['time_s']))
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
    if a.sensor_balance_calibration and not (Path(a.output)/'balance-report.json').exists():
        for name in ('balance-report.json','report.json'):
            (Path(a.output)/name).write_text(json.dumps(dict(passed=False,
                scope=balance_scope+'; setup or finalization failure; no qualified physical result',
                error=traceback.format_exc()),indent=2)+'\n')
finally:app.close()
if failed:raise SystemExit(1)
