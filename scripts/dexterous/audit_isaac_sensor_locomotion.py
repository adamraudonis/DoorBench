#!/usr/bin/env python3
"""Independent sensor, FK and motor replay of completed Isaac locomotion.

Evaluator poses are used only for comparison, never passed to the controller.
This does not independently reconstruct scene contacts or qualify a door task.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from doorbench.dexterous.sensor_contract import SENSOR_KEYS
from doorbench.dexterous.sensor_locomotion import SensorLocomotionController
from scripts.dexterous import audit_pose_gyro_run
from scripts.dexterous.audit_pose_gyro_run import reconstruct_rates, sha


def archive(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','robot','checkpoint','calibration','original-layout','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous receipts')
    run=a.run;s=run/'sensors'
    read=lambda n:json.loads((run/n).read_text())
    task=read('locomotion-report.json');layout=read('sensors/layout.json')
    capture=read('sensors/report.json');producer=read('sensors/gyro-producer.json')
    contract=read('motor-contract.json');reset=read('acquisition-reset.json')
    config=read('configuration.json');scope=read('sensors/scope.json')
    physical=archive(run/'acquisition-physics.npz');numeric=archive(s/'actor-sensors.npz')
    initial=archive(s/'actor-initial-decision.npz');rgb=archive(s/'actor-rgb.npz')
    evidence=archive(s/'gyro-producer-evidence.npz')
    count=len(physical['time_s']);expected=np.arange(1,count+1)*.002
    # This audit cannot accidentally run a shadow plant.
    def forbidden(*args,**kwargs):raise RuntimeError('Audit cannot step physics')
    for n in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,n,forbidden)
    if task['action_semantics'] not in ('sensor_walk_supported_stop_v1','pinned_h1_sensor_locomotion_constant_command_v1'):raise ValueError('Require a declared locomotion component')
    stopping=task['action_semantics']=='sensor_walk_supported_stop_v1'
    if stopping:
        from doorbench.dexterous.sensor_walk_stop import SensorWalkStopController
        c=SensorWalkStopController.from_calibration(a.robot,contract,layout,a.calibration,a.checkpoint,config['args']['sensor_locomotion_stop_after_seconds'])
    else:c=SensorLocomotionController.from_calibration(a.robot,contract,layout,a.calibration,a.checkpoint)
    c.reset_episode()
    packet={k:v for k,v in initial.items() if k not in ('time_s','motor_forces')}
    names=layout['joint_order'];order=[config['robot_joint_names'].index(n) for n in names]
    roots=np.vstack([reset['root'],physical['root']])
    joints=np.vstack([reset['joints'],physical['joints']])[:,order]
    m=mujoco.MjModel.from_xml_path(str(a.robot));d=mujoco.MjData(m)
    qa=np.array([m.joint(n).qposadr[0] for n in names]);frames=[]
    for root,q in zip(roots,joints):
        d.qpos[:7]=root[:7];d.qpos[qa]=q;mujoco.mj_kinematics(m,d)
        frames.append(d.site('imu').xmat.reshape(3,3).copy())
    reconstructed,producer_frames=reconstruct_rates(evidence['time_s'],evidence['body_quaternion_xyzw_world'],layout['imu']['quaternion_wxyz_body'])
    rate_error=float(np.max(abs(reconstructed.astype(float)-numeric['imu_gyro'])))
    frame_error=float(np.max(np.linalg.norm(Rotation.from_matrix(producer_frames@np.swapaxes(frames,1,2)).as_rotvec(),axis=1)))
    force_error=gravity_error=0.
    for i in range(count):
        force=c.force(packet,i*.002)
        force_error=max(force_error,float(np.max(abs(force-physical['motor_forces'][i]))))
        R=Rotation.from_quat(roots[i,3:7][[1,2,3,0]]).as_matrix()
        gravity=c.last_info.get('estimated_projected_gravity')
        if stopping and c.balance is not None:gravity=Rotation.from_quat(c.balance.d.qpos[[4,5,6,3]]).as_matrix().T@np.array([0.,0.,-1.])
        gravity_error=max(gravity_error,float(np.linalg.norm(np.asarray(gravity)-R.T@np.array([0.,0.,-1.]))))
        packet={k:v[i].copy() for k,v in numeric.items() if k!='time_s'}
        for key in ('rgb_left','rgb_right'):
            slot=SENSOR_KEYS.index(key);stamp=packet['sensor_time_s'][slot]
            if not packet['sensor_valid'][slot]:packet[key]=np.zeros_like(initial[key]);continue
            index=np.searchsorted(rgb['time_s'],stamp+1e-9,side='right')-1
            if index<0 or abs(rgb['time_s'][index]-stamp)>1e-8:raise ValueError('Missing actual RGB frame')
            packet[key]=rgb[key][index].copy()
    old=json.loads(a.original_layout.read_text());old['imu']['gyro_profile']='pose-delta-angle-v1'
    valid=np.zeros(len(SENSOR_KEYS),bool)
    valid[[SENSOR_KEYS.index(k) for k in ('joint_position','joint_velocity')]]=True
    joint_error=float(np.max(abs(joints[1:]-numeric['joint_position'])))
    motor=c.walk.motor if stopping else c.motor
    expected_count=5000 if stopping else 2500
    checks=dict(complete_declared_intervals=count==expected_count and len(numeric['time_s'])==expected_count and len(evidence['time_s'])==expected_count+1,
        runtime_physical_checks_pass=task['passed'] and all(task['checks'].values()),
        actual_sensor_capture_complete=capture['capture_complete'] and capture['samples']==count,
        captured_numeric_hashes=all(sha(s/n)==h for n,h in capture['numeric_file_sha256'].items()),
        gyro_evidence_hash=sha(s/'gyro-producer-evidence.npz')==producer['evaluator_evidence_sha256'],
        layout_identity=layout==old and capture['input_layout_sha256']==sha(a.original_layout) and capture['effective_layout_sha256']==sha(s/'layout.json'),
        gyro_producer_scope=producer['profile']=='pose-delta-angle-v1' and producer['samples']==count and producer['recorded_actor_packets']==count and producer['interval_count_matches_packets'] and producer['failed_reason'] is None and producer['actor_receives_orientation'] is False and producer['physics_writes']==0 and producer['source_body_path']==scope['body_paths_by_name'][layout['imu']['body_name']],
        exact_epochs=all(np.allclose(x,expected,atol=1e-8,rtol=0) for x in (physical['time_s'],numeric['time_s'],evidence['time_s'][1:],numeric['sensor_time_s'][:,2])) and numeric['sensor_valid'][:,2].all(),
        initial_encoders_only=float(initial['time_s'])==0 and np.array_equal(initial['sensor_valid'],valid) and np.max(abs(initial['joint_position']-joints[0]))<1e-7,
        actual_joint_encoder_join=joint_error<1e-7,
        every_gyro_interval_reconstructed=rate_error<1e-7,
        producer_matches_actual_robot_FK=frame_error<1e-5,
        every_motor_command_replayed=force_error<1e-4,
        gravity_estimate_matches_evaluator=gravity_error<.01,
        original_caps=np.all(physical['motor_forces']>=motor.caps[:,0]-1e-5) and np.all(physical['motor_forces']<=motor.caps[:,1]+1e-5),
        no_runtime_pose_writes=task['runtime_robot_pose_writes']==0,
        no_teacher_actions=task['teacher_actions']==0,
        original_dry_profile=task['joint_passive_profile']=='backend-dry-v2',
        audit_never_steps=d.time==0 and (c.walk.d.time==0 and (c.balance is None or c.balance.d.time==0) if stopping else c.d.time==0))
    if stopping:
        tail=physical['root'][physical['time_s']>=9.-1e-8]
        checks.update(supported_stance_handoff=c.balance is not None,stance_solver_clean=c.balance is not None and c.balance.qp_failures==0,actual_final_second_quiet=len(tail)>=500 and np.max(np.linalg.norm(tail[:,7:9],axis=1))<.03 and np.max(np.ptp(tail[:,:2],axis=0))<.03)
    checks={k:bool(v) for k,v in checks.items()}
    paths=[a.robot,a.checkpoint,a.calibration,a.original_layout,Path(__file__),Path(audit_pose_gyro_run.__file__)]
    paths += [run/n for n in ('locomotion-report.json','motor-contract.json','configuration.json','acquisition-reset.json','acquisition-physics.npz','sensors/layout.json','sensors/report.json','sensors/scope.json','sensors/gyro-producer.json','sensors/gyro-producer-evidence.npz','sensors/actor-sensors.npz','sensors/actor-initial-decision.npz','sensors/actor-rgb.npz')]
    import importlib
    for module in ('sensor_locomotion','sensor_walk_stop','sensor_balance','locomotion_manipulation','stance','motor_target_control','locomotion'):
        paths.append(Path(importlib.import_module('doorbench.dexterous.'+module).__file__))
    result=dict(schema='doorbench.isaac-sensor-locomotion-audit.v1',passed=all(checks.values()),checks=checks,physics_steps=0,steps=count,scope=__doc__,maximum_motor_replay_error_Nm=force_error,maximum_gravity_vector_error=gravity_error,maximum_gyro_component_error_radps=rate_error,maximum_body_FK_rotation_error_rad=frame_error,maximum_joint_packet_error_rad=joint_error,input_sha256={str(n):sha(n) for n in paths},limitations=['Contact and motor delivery checks are bound runtime evidence, not independently reconstructed here.','Accelerometer and tactile values are not independently reconstructed.','RGB is replayed but not used for navigation. Walking/stance uses own encoders, IMU and, in stopping mode, foot touch.',f'This is a {count*.002:g}-second component result, not a door opening success.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
