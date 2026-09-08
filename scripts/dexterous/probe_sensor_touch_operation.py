#!/usr/bin/env python3
"""Uninterrupted acquisition and local-distal-touch regulated motor press."""
import argparse
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import time
import subprocess
import re
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.motor_contract_identity import motor_contract_fingerprint
from doorbench.dexterous.native_transition_audit import NativeTransitionRecorder
from doorbench.dexterous.native_transition_archive import NativeTransitionArchive, STATE_FIELDS, BODY_FIELDS
from doorbench.dexterous.sensor_distal_touch_control import SensorDistalTouchController
from doorbench.dexterous.sensor_reach_balance import SensorReachBalanceController
from doorbench.dexterous.sensor_acquisition_schedule import ScriptedAcquisitionSchedule
from doorbench.dexterous.sensor_handle_operation_schedule import ScriptedHandleOperationSchedule
from doorbench.dexterous.grasp_verification import native_grasp_sample, audit_grasp_steps, shadow_surface_qualified
from doorbench.dexterous.sensor_contract import ActorObservationBuilder
from scripts.dexterous.export_sensor_layout import export_layout


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def intended_contact_geometry(m,d,c,lever):
    """Evaluator-only original distal-pad geometry; no motor/controller input."""
    if lever not in c.geom:return False
    other=int(c.geom[1] if c.geom[0]==lever else c.geom[0]);body=int(m.geom_bodyid[other])
    match=re.fullmatch(r'robot/rh_(ff|mf|rf|lf|th)(.+)',m.body(body).name)
    if not match:return False
    R=d.xmat[body].reshape(3,3);local=R.T@(c.pos-d.xpos[body])
    outward=R.T@((1 if c.geom[0]==other else -1)*c.frame[:3])
    center=d.geom_xpos[lever];axis=d.geom_xmat[lever].reshape(3,3)[:,2]
    relative=c.pos-center;axial=float(relative@axis);radial=relative-axial*axis
    if np.linalg.norm(radial)<1e-8:return False
    alignment=float((R@outward)@(-radial/np.linalg.norm(radial)))
    return bool(shadow_surface_qualified(match.group(1),match.group(2),local,outward) and
                m.geom_size[lever,1]-abs(axial)>=.001 and alignment>.8)


def screen_motion(sim,arm,route):
    """Screen full nominal joint route on detached scene; intended lever only."""
    m,d=sim.m,sim.d;w=mujoco.MjData(m);w.qpos[:]=d.qpos;start=d.qpos.copy();clock=float(d.time)
    qa=np.array([m.jnt_qposadr[m.joint('robot/'+name).id] for name in arm.goal_names])
    lever=m.geom('leaf_handle_lever_col_n').id
    bad=[];max_penetration=0.;first_contact=None;intended_count=0
    for i,u in enumerate(np.linspace(0.,1.,1001)):
        w.qpos[qa]=route.values_at_fraction(float(u));mujoco.mj_kinematics(m,w);mujoco.mj_collision(m,w)
        failures=[]
        for name,value in zip(arm.goal_names,w.qpos[qa]):
            limits=m.jnt_range[m.joint('robot/'+name).id]
            if not limits[0]<=value<=limits[1]:failures.append(dict(reason='authored joint bound',joint=name,value=float(value)))
        for digit in ('FF','MF','RF','LF'):
            distal=float(w.qpos[m.joint('robot/rh_'+digit+'J1').qposadr[0]])
            middle=float(w.qpos[m.joint('robot/rh_'+digit+'J2').qposadr[0]])
            if distal>middle+1e-7:failures.append(dict(reason='passive loopback goal',digit=digit,difference_rad=distal-middle))
        for c in w.contact[:w.ncon]:
            names=[m.body(m.geom_bodyid[g]).name for g in c.geom];geoms=[m.geom(g).name for g in c.geom]
            hand=any(n.startswith(('robot/rh_','robot/lh_')) for n in names)
            if hand and c.dist<=0:
                if first_contact is None:first_contact=float(u)
                intended=intended_contact_geometry(m,w,c,lever)
                intended_count+=int(intended)
                if not intended or u<=.45:failures.append(dict(reason='unintended or premature hand contact',bodies=names,distance_m=float(c.dist)))
            if any(n.startswith('robot/') for n in names) and not ('floor' in geoms and any(n.endswith('_ankle_link') for n in names)):
                max_penetration=max(max_penetration,-float(c.dist))
                if c.dist<-.003:failures.append(dict(reason='nonfoot penetration',bodies=names,distance_m=float(c.dist)))
        if failures:bad.append(dict(index=i,fraction=float(u),failures=failures))
    unchanged=np.array_equal(d.qpos,start) and d.time==clock
    return dict(passed=not bad and unchanged,samples=1001,first_hand_contact_fraction=first_contact,
        intended_distal_lever_contact_samples=intended_count,
        source='unstepped detached full-scene native FK/collision calculator',active_plant_unchanged=unchanged,
        moving_joints=list(arm.goal_names),fixed='initial pelvis/legs,left arm/hand,door; coordinated torso yaw explicitly enabled',
        end_palm_position_world_m=w.site_xpos[m.site('robot/rh_palm_touch').id].tolist(),
        source_path_stop_fraction=1.,max_nonfoot_penetration_m=max_penetration,bad_samples=bad,
        scope='Nominal static full route with original distal-pad geometry admission; actual grasp qualification remains independent')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('robot','door','motors','reference','output','press-plan','preload-screen'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--calibration',type=Path,default=Path(__file__).resolve().parents[2]/'configs/dexterous/sensor-balance-v1.json')
    p.add_argument('--schedule',type=Path,default=Path(__file__).resolve().parents[2]/'configs/dexterous/sensor-acquisition-balance-v1.json')
    p.add_argument('--seconds',type=float,default=36.)
    p.add_argument('--gravity-correction',type=float,default=.2)
    p.add_argument('--initial-velocity',type=float,default=0.,help='Declared reset-only lateral velocity perturbation')
    p.add_argument('--impedance-protocol',type=Path,help='Opt-in separately frozen post-acquisition stiffness comparison')
    a=p.parse_args()
    if not 0<a.seconds<=40 or not np.isfinite([a.seconds,a.initial_velocity]).all() or abs(a.initial_velocity)>.1:p.error('Bounded finite protocol required')
    if a.output.exists():p.error('Fresh evidence directory required')
    a.output.mkdir(parents=True)
    robot=Path(a.robot);door=a.door if a.door.is_dir() else a.door.parent
    ref=json.loads(a.reference.read_text());motors=json.loads(a.motors.read_text());layout=export_layout(robot)
    calibration=json.loads(a.calibration.read_text());desired=calibration['desired_posture']
    if calibration['robot_xml_sha256']!=sha(robot) or calibration['physics_dt_s']!=.002 or calibration['motor_contract_sha256']!=motor_contract_fingerprint(motors):
        raise ValueError('Frozen constant calibration differs from the authored robot')
    reset_angles=dict(zip(ref['acquisition']['joint_names'],ref['acquisition']['path_qpos'][0]))
    if desired!=reset_angles:raise ValueError('This comparison requires the same fixed posture as the frozen actor reset')
    schedule=json.loads(a.schedule.read_text())
    if schedule['schema']!='doorbench.scripted-sensor-acquisition.v1' or schedule['source_reference_sha256']!=sha(a.reference) or schedule['grasp_profile']!='distal-pad-v1' or schedule['required_grasp_hold_s']!=.5 or schedule['stop_fraction']!=1.:raise ValueError('Exact frozen contact-enabled route and original distal grasp protocol required')
    arm=SensorReachBalanceController(robot,motors,layout,desired,gravity_correction=a.gravity_correction,allow_torso_yaw=schedule['allow_torso_yaw'],maximum_goal_speed_radps=schedule['maximum_goal_speed_radps'],finger_impedance_multiplier=schedule.get('finger_impedance_multiplier',1.),finger_velocity_damping=schedule.get('finger_velocity_damping',0.),finger_target_velocity_damping=schedule.get('finger_target_velocity_damping',False))
    controller=arm.balance
    base_route=ScriptedAcquisitionSchedule(arm.goal_names,ref['acquisition']['joint_names'],ref['acquisition']['path_qpos'],start_s=schedule['start_s'],reach_seconds=schedule['reach_seconds'],settle_seconds=schedule['settle_seconds'])
    plan=json.loads(a.press_plan.read_text())
    if not plan['passed'] or plan['schema']!='doorbench.offline-sensor-press-plan.v1' or plan['robot_xml_sha256']!=sha(robot) or plan['door_xml_sha256']!=sha(door/'door.xml') or plan['source_reference_sha256']!=sha(a.reference) or plan['acquisition_seconds']!=base_route.duration_s:raise ValueError('Qualified matching offline press plan required')
    route=ScriptedHandleOperationSchedule(base_route,plan['joint_names'],plan['path_qpos'],press_seconds=plan['press_seconds'],settle_seconds=plan['settle_seconds'])
    if base_route.duration_s!=schedule['duration_s'] or a.seconds!=36. or route.duration_s!=plan['duration_s'] or route.sampled_maximum_goal_speed()>arm.maximum_goal_speed_radps:raise ValueError('Declared duration or joint-goal slew disagrees with route')
    preload_screen=json.loads(a.preload_screen.read_text())
    if not preload_screen['passed'] or preload_screen['press_plan_sha256']!=sha(a.press_plan) or preload_screen['source_trajectory_sha256']!=plan['source_trajectory_sha256']:raise ValueError('Bounded preload must have a matching offline screen')
    touch=SensorDistalTouchController(arm,route,layout)
    if a.impedance_protocol:
        from doorbench.dexterous.sensor_distal_touch_impedance import SensorDistalTouchImpedanceController
        touch=SensorDistalTouchImpedanceController(arm,route,layout,json.loads(a.impedance_protocol.read_text()),motors)
    builder=ActorObservationBuilder(joint_count=69,action_count=61,tactile_dimension=layout['tactile_dimension'])
    provenance=dict(scope=__doc__,controller_input='Numeric sensor.v2 packet+clock for balance, plus separately declared coordinated torso/right-arm/right-hand joint goals',
        calibration=dict(desired_joint_angles=desired,derived_local_root_height_m=float(controller.calibration_root[2]),initial_orientation='upright, arbitrary yaw zero'),
        robot_xml_sha256=sha(robot),door_xml_sha256=sha(door/'door.xml'),motor_contract_sha256=sha(a.motors),reference_sha256=sha(a.reference),calibration_sha256=sha(a.calibration),arm_schedule_sha256=sha(a.schedule),
        actual_reset_reference_used_only_by_evaluator=True,imu_timing='Native actual-step local IMU/tactile captured at interval start; available at interval end; endpoint encoders captured at end',
        rgb='Disabled and invalid throughout; controller validates shape but uses no RGB',
        command_adapter='Original motor transmissions/caps/passive physics; force-only command representation set once at reset',
        parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},sources={})
    sources=['doorbench/dexterous/sensor_reach_balance.py','doorbench/dexterous/sensor_balance.py','doorbench/dexterous/sensor_contract.py','doorbench/dexterous/locomotion_manipulation.py',
        'doorbench/dexterous/stance.py','doorbench/dexterous/grasp_verification.py','doorbench/dexterous/native_transition_audit.py',
        'doorbench/dexterous/native_warning_audit.py','doorbench/dexterous/environment.py','scripts/dexterous/probe_sensor_touch_operation.py','scripts/dexterous/export_sensor_layout.py']
    root=Path(__file__).resolve().parents[2]
    provenance['source_git_revision']=subprocess.run(['git','rev-parse','HEAD'],cwd=root,check=True,text=True,capture_output=True).stdout.strip()
    provenance['runtime_versions']=dict(python=sys.version,mujoco=mujoco.__version__,numpy=np.__version__)
    # Freeze the imported project dependency closure, including DoorEnv and the
    # actual force/contact auditor, rather than only the outer controller file.
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            path=Path(file).resolve()
            if path.is_relative_to(root) and path.suffix=='.py':sources.append(str(path.relative_to(root)))
    for src in sorted(set(sources)):
        target=a.output/'frozen-source'/src;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/src,target);provenance['sources'][src]=sha(target)
    for name,path in [('preload-screen.json',a.preload_screen),('press-plan.json',a.press_plan),('reference.json',a.reference),('motors.json',a.motors),('calibration.json',a.calibration),('schedule.json',a.schedule)]:shutil.copy2(path,a.output/name)
    write(a.output/'sensor-layout.json',layout);write(a.output/'provenance.json',provenance)
    if a.impedance_protocol:
        shutil.copy2(a.impedance_protocol,a.output/'impedance-protocol.json')
        provenance['impedance_protocol_sha256']=sha(a.impedance_protocol)
        write(a.output/'provenance.json',provenance)
    audit=json.loads(robot.with_suffix('.audit.json').read_text())
    sim=DexterousDoorEnv(door,robot,audit,frame_skip=1);sim.reset(images=False,randomize=False)
    m,d=sim.m,sim.d;names=controller.names
    ids=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids]
    aids=np.array([m.actuator('robot/'+n).id for n in controller.actions])
    d.qpos[sim.root_qadr:sim.root_qadr+7]=ref['initial_root'];d.qpos[qa]=controller.desired;d.qvel[:]=0
    d.qvel[sim.root_vadr]=a.initial_velocity
    m.actuator_gainprm[aids,0]=1.;m.actuator_biasprm[aids,:3]=0.;m.actuator_ctrlrange[aids]=controller.caps
    d.ctrl[:]=0.;mujoco.mj_forward(m,d)
    write(a.output/'reset.json',dict(root=d.qpos[sim.root_qadr:sim.root_qadr+7].tolist(),joints=dict(zip(names,d.qpos[qa])),qpos=d.qpos.tolist(),qvel=d.qvel.tolist()))
    screen=screen_motion(sim,arm,base_route)
    screen['schema']='doorbench.sensor-operation-static-screen.v1'
    screen['preload_screen_sha256']=sha(a.preload_screen)
    screen['press_plan_passed']=plan['passed'];screen['press_plan_sha256']=sha(a.press_plan)
    screen['binding']=dict(schedule_sha256=sha(a.schedule),robot_xml_sha256=sha(robot),door_xml_sha256=sha(door/'door.xml'),
        reset_sha256=sha(a.output/'reset.json'),reference_sha256=sha(a.reference),motor_contract_sha256=sha(a.motors),
        calibration_sha256=sha(a.calibration),sensor_layout_sha256=sha(a.output/'sensor-layout.json'),
        screen_source_sha256=sha(__file__),schedule_source_sha256=sha(root/'doorbench/dexterous/sensor_acquisition_schedule.py'),
        controller_source_sha256=sha(root/'doorbench/dexterous/sensor_reach_balance.py'))
    write(a.output/'static-screen.json',screen)
    if not screen['passed']:
        write(a.output/'report.json',dict(passed=False,scope='Static path rejected before physics',screen=screen))
        print(json.dumps({k:v for k,v in screen.items() if k!='bad_samples'},indent=2));return
    immutable=('body_mass','body_inertia','body_gravcomp','jnt_range','tendon_range','tendon_solref_lim','tendon_solimp_lim',
        'geom_friction','geom_contype','geom_conaffinity','dof_damping','dof_armature','dof_frictionloss','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange')
    originals={k:getattr(m,k).copy() for k in immutable}
    recorder=NativeTransitionRecorder(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge',profile='distal-pad-v1')
    initial_physics=native_grasp_sample(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge',profile='distal-pad-v1')
    caps=controller.caps;scale=np.maximum(abs(caps[:,0]),abs(caps[:,1]));previous=np.zeros(61)
    qposes=[d.qpos.copy()];qvels=[d.qvel.copy()];forces=[];rows=[];infos=[];packets=[];error=None;started=time.monotonic();handoff=None
    rawfile=NativeTransitionArchive(a.output/'actual-transitions',chunk_size=250)
    try:
        for step in range(round(a.seconds/m.opt.timestep)):
            t=float(d.time);packet=builder.observe(now_s=t,previous_action=previous)
            if handoff is None and t>=base_route.duration_s-1e-9:
                handoff=audit_grasp_steps([initial_physics,*rows],physics_dt=.002,expected_duration=base_route.duration_s,required_hold=.5)
                handoff['source_prefix_steps']=len(rows)
                write(a.output/'acquisition-handoff-audit.json',handoff)
                if not handoff['passed'] or any(r['invalid_loaded_distal_patches'] or r['unintended_hand_contacts'] for r in rows):
                    error='Unqualified acquisition; press safety evaluation stopped execution';break
            goals=route.goals(t)
            force,info=touch.force(packet,now_s=t)
            goals=dict(zip(info['goal_joint_names'],info['goal_joint_position_rad']))
            info['high_level_source']='frozen joint route with local distal touch preload/progression'
            packets.append(packet);infos.append(info);d.ctrl[aids]=force
            recorder.before_step();sim.plant.step()
            # Copy actual native sensor output before any other dynamics call.
            samples={'imu_gyro':d.sensor('robot/imu_gyro').data.copy(),'imu_accelerometer':d.sensor('robot/imu_accelerometer').data.copy(),
                     'tactile':d.sensordata[sim.tactile_indices].copy()}
            row,raw=recorder.after_step()
            row['left_and_right_hand_contacts']=0;row['unintended_hand_contacts']=0;floor_support=np.zeros(2)
            for c in raw['contacts']:
                bodies=[m.body(b).name for b in c['body']];geoms=[m.geom(g).name for g in c['geom']]
                if any(n.startswith(('robot/rh_','robot/lh_')) for n in bodies) and (c['distance_m']<=0 or c['wrench_contact_frame'][0]>1e-6):
                    row['left_and_right_hand_contacts']+=1
                    intended_pair='leaf_handle_lever_col_n' in geoms and any(re.fullmatch(r'robot/rh_(ff|mf|rf|lf|th).+',n) for n in bodies)
                    if not intended_pair:row['unintended_hand_contacts']+=1
                if 'floor' in geoms:
                    for i,side in enumerate(('left','right')):
                        if 'robot/'+side+'_ankle_link' in bodies:floor_support[i]+=max(0.,c['wrench_contact_frame'][0])
            row['bolt_slide_m']=float(d.qpos[m.jnt_qposadr[m.joint('leaf_latch_bolt_slide').id]])
            row['script_phase']='press' if t>=base_route.duration_s-1e-9 else 'acquisition'
            row['invalid_loaded_distal_patches']=sum(c['normal_force_N']>1e-6 and not c['pad_qualified'] for c in row['pad_grasp']['contacts'])
            row['actual_floor_support_normal_N']=floor_support.tolist();row['input_force_error_Nm']=float(np.max(np.abs(d.actuator_force[aids]-force)))
            row['actual_root_horizontal_speed_mps']=float(np.linalg.norm(d.qvel[sim.root_vadr:sim.root_vadr+2]))
            requested_q=controller.desired.copy()
            for name,goal in goals.items():requested_q[controller.names.index(name)]=goal
            actual_coordinates=controller.matrix@d.qpos[qa];requested_coordinates=controller.matrix@requested_q
            row['command_coordinate_tracking_error_max_rad']=float(np.max(abs(actual_coordinates[arm.arm_motors]-requested_coordinates[arm.arm_motors])))
            row['arm_tracking_error_max_rad']=float(max(abs(d.qpos[m.jnt_qposadr[m.joint('robot/'+name).id]]-goal) for name,goal in goals.items()))
            row['actual_root_angular_speed_radps']=float(np.linalg.norm(d.qvel[sim.root_vadr+3:sim.root_vadr+6]))
            if set(raw)!=set(STATE_FIELDS)|set(BODY_FIELDS)|{'contacts','mujoco_warning_interval'}:raise ValueError('Lossless archive does not cover every raw evidence field')
            rawfile.write(raw);rows.append(row)
            qposes.append(d.qpos.copy());qvels.append(d.qvel.copy());forces.append(force)
            end=float(d.time)
            for key,value in samples.items():builder.push(key,value,capture_s=t,available_s=end)
            builder.push('joint_position',d.qpos[qa].copy(),capture_s=end)
            builder.push('joint_velocity',d.qvel[va].copy(),capture_s=end)
            previous=force/scale
            if row['unintended_hand_contacts']>0 or row['max_nonfoot_penetration_m']>.003 or row['max_joint_limit_violation_rad']>.02 or row['max_shadow_loopback_violation_rad']>.02:
                error='Safety stop: unintended contact or original physical bound breached';break
            if row['torso_tilt_deg']>15 or row['root_height_m']<.65:
                error='Safety stop: tilt exceeded15degrees or pelvis below.65m';break
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
    finally:rawfile.close(complete=error is None and len(rows)==round(a.seconds/.002))
    duration=float(d.time);n=len(rows);final=[r for r in rows if r['sim_time_s']>=max(0,a.seconds-1.)]
    checks=dict(complete_requested_duration=abs(duration-a.seconds)<1e-9 and n==round(a.seconds/.002),
        execution_no_error=error is None,finite=all(r['finite'] for r in rows) and n>0,
        original_model_unchanged=all(np.array_equal(v,getattr(m,k)) for k,v in originals.items()),
        original_motor_caps=all(r['native_motor_limits'] for r in rows),
        delivered_motor_forces=all(r['input_force_error_Nm']<1e-5 for r in rows),
        joint_limits=all(r['max_joint_limit_violation_rad']<=.02 for r in rows),
        passive_loopbacks=all(r['max_shadow_loopback_violation_rad']<=.02 for r in rows),
        nonfoot_collision=all(r['max_nonfoot_penetration_m']<=.003 for r in rows),
        only_intended_lever_contacts=all(r['unintended_hand_contacts']==0 for r in rows),
        original_distal_pad_patches=all(r['invalid_loaded_distal_patches']==0 for r in rows),
        no_external_assistance=all(r['external_wrench_max']==0 and r['applied_generalized_force_max']==0 for r in rows),
        no_native_warnings=all(r['mujoco_warning_interval']['passed'] for r in rows),
        torso_upright=all(r['torso_tilt_deg']<=12 for r in rows),
        lowered_height=all(.82<=r['root_height_m']<=.92 for r in rows),
        final_quiet=bool(final) and all(r['actual_root_horizontal_speed_mps']<.03 and r['actual_root_angular_speed_radps']<.05 for r in final),
        final_both_feet=bool(final) and all(min(r['actual_floor_support_normal_N'])>30 for r in final),
        estimator_never_stepped=controller.d.time==0.,all_qp_solved=controller.qp_failures==0,
        cold_start_no_sensor_values=bool(infos) and infos[0]['cold_start'],
        geometric_path_screen=screen['passed'],
        arm_tracking=all(r['arm_tracking_error_max_rad']<.04 for r in rows))
    motor_protocol=schedule.get('tracking_profile')=='motor-transmission-v2'
    if motor_protocol:
        del checks['arm_tracking']
        checks['motor_coordinate_tracking']=all(r['command_coordinate_tracking_error_max_rad']<.04 for r in rows)
    arm_indices=np.array([m.jnt_qposadr[m.joint('robot/'+name).id] for name in arm.goal_names]);arm_actual=np.asarray(qposes)[:,arm_indices]
    excursions=np.max(abs(arm_actual-arm_actual[0]),axis=0)
    requested=np.array([route.goals(route.duration_s)[n]-route.goals(0.)[n] for n in arm.goal_names])
    if motor_protocol:
        full_actual=np.asarray(qposes)[:,qa];coordinates=full_actual@controller.matrix.T
        coordinate_excursions=np.max(abs(coordinates-coordinates[0]),axis=0)
        delta_q=np.zeros(69);delta_q[arm.indices]=requested;coordinate_delta=controller.matrix@delta_q
        checks['physical_reach_motion_delivered']=all(coordinate_excursions[i]>.7*abs(coordinate_delta[i]) for i in arm.arm_motors if abs(coordinate_delta[i])>.01)
    else:
        checks['physical_reach_motion_delivered']=all(excursions[i]>.7*abs(delta) for i,delta in enumerate(requested) if abs(delta)>.01)
    grasp_audit=audit_grasp_steps([initial_physics,*rows],physics_dt=.002,expected_duration=a.seconds,required_hold=.5)
    write(a.output/'grasp-audit.json',grasp_audit)
    checks['original_opposed_distal_grasp_hold']=grasp_audit['passed']
    checks['qualified_acquisition_before_press']=handoff is not None and handoff['passed']
    checks['local_press_route_completed']=touch.finished_at is not None and duration-touch.finished_at>=3.
    checks['sustained_handle_depression']=len(final)>=500 and all(r['handle_angle_rad']>=.8 and r['bolt_slide_m']>=.011 for r in final[-251:])
    actual_palm=d.site_xpos[m.site('robot/rh_palm_touch').id].copy()
    palm_error=float(np.linalg.norm(actual_palm-np.asarray(screen['end_palm_position_world_m'])))
    # The acquisition endpoint is no longer the target after the press.
    # Operation is independently scored by actual lever/latch and pad physics.
    with gzip.open(a.output/'physics.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    with gzip.open(a.output/'controller.jsonl.gz','wt') as f:
        for r in infos:f.write(json.dumps(r,separators=(',',':'))+'\n')
    np.savez_compressed(a.output/'trajectory.npz',qpos=qposes,qvel=qvels,force=forces,time=np.arange(len(qposes))*.002,joint_names=names,action_names=controller.actions)
    if packets:np.savez_compressed(a.output/'actor-inputs.npz',**{key:np.stack([p[key] for p in packets]) for key in packets[0] if not key.startswith('rgb')})
    report=dict(scope='Uninterrupted scripted acquisition plus local-distal-touch preload/progression; sensor-only state; no learned/vision/opening claim',passed=all(checks.values()),checks=checks,
        duration_s=duration,requested_duration_s=a.seconds,physics_steps=n,wall_seconds=time.monotonic()-started,error=error,
        max_torso_tilt_deg=max([r['torso_tilt_deg'] for r in rows],default=0),
        height_range_m=[min([r['root_height_m'] for r in rows],default=0),max([r['root_height_m'] for r in rows],default=0)],
        max_joint_violation_rad=max([r['max_joint_limit_violation_rad'] for r in rows],default=0),
        max_loopback_violation_rad=max([r['max_shadow_loopback_violation_rad'] for r in rows],default=0),
        max_nonfoot_penetration_m=max([r['max_nonfoot_penetration_m'] for r in rows],default=0),
        palm_endpoint_error_m=palm_error,source_path_stop_fraction=route.stop_fraction,
        tracking_profile=schedule.get('tracking_profile','individual-joint-v1'),
        maximum_motor_coordinate_tracking_error_rad=max([r['command_coordinate_tracking_error_max_rad'] for r in rows],default=0),
        maximum_arm_tracking_error_rad=max([r['arm_tracking_error_max_rad'] for r in rows],default=0),
        measured_arm_excursions_rad=dict(zip(arm.goal_names,excursions.tolist())),
        local_virtual_press_clock_s=touch.virtual_s,local_preload_offsets_rad=touch.delta.tolist(),
        max_handle_angle_rad=max([r['handle_angle_rad'] for r in rows],default=0),
        final_handle_angle_rad=rows[-1]['handle_angle_rad'] if rows else None,
        max_latch_retraction_m=max([r['bolt_slide_m'] for r in rows],default=0),
        grasp_audit=grasp_audit,acquisition_handoff_audit=handoff,final_grasp=rows[-1]['pad_grasp'] if rows else None,
        qp_failures=controller.qp_failures,provenance_sha256=sha(a.output/'provenance.json'))
    write(a.output/'report.json',report);print(json.dumps(report,indent=2));sim.close()

if __name__=='__main__':main()
