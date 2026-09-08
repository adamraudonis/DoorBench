#!/usr/bin/env python3
"""Bounded native sensor-only balance test; never a complete door task claim."""
import argparse
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.motor_contract_identity import motor_contract_fingerprint
from doorbench.dexterous.native_transition_audit import NativeTransitionRecorder
from doorbench.dexterous.sensor_balance import SensorBalanceController
from doorbench.dexterous.sensor_contract import ActorObservationBuilder
from scripts.dexterous.export_sensor_layout import export_layout


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('robot','door','motors','reference','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--calibration',type=Path,default=Path(__file__).resolve().parents[2]/'configs/dexterous/sensor-balance-v1.json')
    p.add_argument('--seconds',type=float,default=5.)
    p.add_argument('--gravity-correction',type=float,default=.2)
    p.add_argument('--initial-velocity',type=float,default=0.,help='Declared reset-only lateral velocity perturbation')
    a=p.parse_args()
    if not 0<a.seconds<=10 or not np.isfinite([a.seconds,a.initial_velocity]).all() or abs(a.initial_velocity)>.1:p.error('Bounded finite protocol required')
    if a.output.exists():p.error('Fresh evidence directory required')
    a.output.mkdir(parents=True)
    robot=Path(a.robot);door=a.door if a.door.is_dir() else a.door.parent
    ref=json.loads(a.reference.read_text());motors=json.loads(a.motors.read_text());layout=export_layout(robot)
    calibration=json.loads(a.calibration.read_text());desired=calibration['desired_posture']
    if calibration['robot_xml_sha256']!=sha(robot) or calibration['physics_dt_s']!=.002 or calibration['motor_contract_sha256']!=motor_contract_fingerprint(motors):
        raise ValueError('Frozen constant calibration differs from the authored robot')
    reset_angles=dict(zip(ref['acquisition']['joint_names'],ref['acquisition']['path_qpos'][0]))
    if desired!=reset_angles:raise ValueError('This comparison requires the same fixed posture as the frozen actor reset')
    controller=SensorBalanceController(robot,motors,layout,desired,gravity_correction=a.gravity_correction)
    builder=ActorObservationBuilder(joint_count=69,action_count=61,tactile_dimension=layout['tactile_dimension'])
    provenance=dict(scope=__doc__,controller_input='Exact numeric doorbench.sensors.v2 packet plus local clock only',
        calibration=dict(desired_joint_angles=desired,derived_local_root_height_m=float(controller.calibration_root[2]),initial_orientation='upright, arbitrary yaw zero'),
        robot_xml_sha256=sha(robot),door_xml_sha256=sha(door/'door.xml'),motor_contract_sha256=sha(a.motors),reference_sha256=sha(a.reference),calibration_sha256=sha(a.calibration),
        actual_reset_reference_used_only_by_evaluator=True,imu_timing='Native actual-step local IMU/tactile captured at interval start; available at interval end; endpoint encoders captured at end',
        rgb='Disabled and invalid throughout; controller validates shape but uses no RGB',
        command_adapter='Original motor transmissions/caps/passive physics; force-only command representation set once at reset',
        parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},sources={})
    sources=['doorbench/dexterous/sensor_balance.py','doorbench/dexterous/sensor_contract.py','doorbench/dexterous/locomotion_manipulation.py',
        'doorbench/dexterous/stance.py','doorbench/dexterous/grasp_verification.py','doorbench/dexterous/native_transition_audit.py',
        'doorbench/dexterous/native_warning_audit.py','doorbench/dexterous/environment.py','scripts/dexterous/probe_sensor_balance.py','scripts/dexterous/export_sensor_layout.py']
    root=Path(__file__).resolve().parents[2]
    for src in sources:
        target=a.output/'frozen-source'/src;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/src,target);provenance['sources'][src]=sha(target)
    for name,path in [('reference.json',a.reference),('motors.json',a.motors),('calibration.json',a.calibration)]:shutil.copy2(path,a.output/name)
    write(a.output/'sensor-layout.json',layout);write(a.output/'provenance.json',provenance)
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
    immutable=('body_mass','body_inertia','body_gravcomp','jnt_range','tendon_range','tendon_solref_lim','tendon_solimp_lim',
        'geom_friction','geom_contype','geom_conaffinity','dof_damping','dof_armature','dof_frictionloss','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange')
    originals={k:getattr(m,k).copy() for k in immutable}
    recorder=NativeTransitionRecorder(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge')
    caps=controller.caps;scale=np.maximum(abs(caps[:,0]),abs(caps[:,1]));previous=np.zeros(61)
    qposes=[d.qpos.copy()];qvels=[d.qvel.copy()];forces=[];rows=[];infos=[];packets=[];error=None;started=time.monotonic()
    rawfile=gzip.open(a.output/'actual-transitions.jsonl.gz','wt')
    try:
        for step in range(round(a.seconds/m.opt.timestep)):
            t=float(d.time);packet=builder.observe(now_s=t,previous_action=previous)
            force,info=controller.force(packet,now_s=t)
            packets.append(packet);infos.append(info);d.ctrl[aids]=force
            recorder.before_step();sim.plant.step()
            # Copy actual native sensor output before any other dynamics call.
            samples={'imu_gyro':d.sensor('robot/imu_gyro').data.copy(),'imu_accelerometer':d.sensor('robot/imu_accelerometer').data.copy(),
                     'tactile':d.sensordata[sim.tactile_indices].copy()}
            row,raw=recorder.after_step()
            row['left_and_right_hand_contacts']=0;floor_support=np.zeros(2)
            for c in raw['contacts']:
                bodies=[m.body(b).name for b in c['body']];geoms=[m.geom(g).name for g in c['geom']]
                if any(n.startswith(('robot/rh_','robot/lh_')) for n in bodies) and (c['distance_m']<=0 or c['wrench_contact_frame'][0]>1e-6):row['left_and_right_hand_contacts']+=1
                if 'floor' in geoms:
                    for i,side in enumerate(('left','right')):
                        if 'robot/'+side+'_ankle_link' in bodies:floor_support[i]+=max(0.,c['wrench_contact_frame'][0])
            row['actual_floor_support_normal_N']=floor_support.tolist();row['input_force_error_Nm']=float(np.max(np.abs(d.actuator_force[aids]-force)))
            row['actual_root_horizontal_speed_mps']=float(np.linalg.norm(d.qvel[sim.root_vadr:sim.root_vadr+2]))
            row['actual_root_angular_speed_radps']=float(np.linalg.norm(d.qvel[sim.root_vadr+3:sim.root_vadr+6]))
            rawfile.write(json.dumps(raw,separators=(',',':'))+'\n');rows.append(row)
            qposes.append(d.qpos.copy());qvels.append(d.qvel.copy());forces.append(force)
            end=float(d.time)
            for key,value in samples.items():builder.push(key,value,capture_s=t,available_s=end)
            builder.push('joint_position',d.qpos[qa].copy(),capture_s=end)
            builder.push('joint_velocity',d.qvel[va].copy(),capture_s=end)
            previous=force/scale
            if row['torso_tilt_deg']>15 or row['root_height_m']<.65:
                error='Safety stop: tilt exceeded15degrees or pelvis below.65m';break
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
    finally:rawfile.close()
    duration=float(d.time);n=len(rows);final=[r for r in rows if r['sim_time_s']>=max(0,a.seconds-1.)]
    checks=dict(complete_requested_duration=abs(duration-a.seconds)<1e-9 and n==round(a.seconds/.002),
        execution_no_error=error is None,finite=all(r['finite'] for r in rows) and n>0,
        original_model_unchanged=all(np.array_equal(v,getattr(m,k)) for k,v in originals.items()),
        original_motor_caps=all(r['native_motor_limits'] for r in rows),
        delivered_motor_forces=all(r['input_force_error_Nm']<1e-5 for r in rows),
        joint_limits=all(r['max_joint_limit_violation_rad']<=.02 for r in rows),
        passive_loopbacks=all(r['max_shadow_loopback_violation_rad']<=.02 for r in rows),
        nonfoot_collision=all(r['max_nonfoot_penetration_m']<=.003 for r in rows),
        hands_away=all(r['left_and_right_hand_contacts']==0 for r in rows),
        no_external_assistance=all(r['external_wrench_max']==0 and r['applied_generalized_force_max']==0 for r in rows),
        no_native_warnings=all(r['mujoco_warning_interval']['passed'] for r in rows),
        torso_upright=all(r['torso_tilt_deg']<=12 for r in rows),
        lowered_height=all(.82<=r['root_height_m']<=.92 for r in rows),
        final_quiet=bool(final) and all(r['actual_root_horizontal_speed_mps']<.03 and r['actual_root_angular_speed_radps']<.05 for r in final),
        final_both_feet=bool(final) and all(min(r['actual_floor_support_normal_N'])>30 for r in final),
        estimator_never_stepped=controller.d.time==0.,all_qp_solved=controller.qp_failures==0,
        cold_start_no_sensor_values=bool(infos) and infos[0]['cold_start'])
    with gzip.open(a.output/'physics.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    with gzip.open(a.output/'controller.jsonl.gz','wt') as f:
        for r in infos:f.write(json.dumps(r,separators=(',',':'))+'\n')
    np.savez_compressed(a.output/'trajectory.npz',qpos=qposes,qvel=qvels,force=forces,time=np.arange(len(qposes))*.002,joint_names=names,action_names=controller.actions)
    if packets:np.savez_compressed(a.output/'actor-inputs.npz',**{key:np.stack([p[key] for p in packets]) for key in packets[0] if not key.startswith('rgb')})
    report=dict(scope='Sensor-only stationary lowered balance component; no reach/open/traverse success',passed=all(checks.values()),checks=checks,
        duration_s=duration,requested_duration_s=a.seconds,physics_steps=n,wall_seconds=time.monotonic()-started,error=error,
        max_torso_tilt_deg=max([r['torso_tilt_deg'] for r in rows],default=0),
        height_range_m=[min([r['root_height_m'] for r in rows],default=0),max([r['root_height_m'] for r in rows],default=0)],
        max_joint_violation_rad=max([r['max_joint_limit_violation_rad'] for r in rows],default=0),
        max_loopback_violation_rad=max([r['max_shadow_loopback_violation_rad'] for r in rows],default=0),
        max_nonfoot_penetration_m=max([r['max_nonfoot_penetration_m'] for r in rows],default=0),
        qp_failures=controller.qp_failures,provenance_sha256=sha(a.output/'provenance.json'))
    write(a.output/'report.json',report);print(json.dumps(report,indent=2));sim.close()

if __name__=='__main__':main()
