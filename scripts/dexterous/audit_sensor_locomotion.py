#!/usr/bin/env python3
"""Replay recorded sensor control; compare estimates with evaluator-only truth.

No active-plant state is passed to the controller and no physics is advanced.
This is a locomotion component audit, not a door-task or vision-policy score.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
from doorbench.dexterous.native_sensor_demonstrations import digest
from doorbench.dexterous.sensor_locomotion import SensorLocomotionController
from scripts.dexterous.evaluate_native_sensor_policy import prepare_trial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    run=a.run;manifest=json.loads((run/'manifest.json').read_text());cfg=manifest['configuration']
    sensor=run/'own-sensors';report=json.loads((run/'report.json').read_text());capture=json.loads((sensor/'report.json').read_text())
    if cfg['action_semantics']!='pinned_h1_sensor_locomotion_constant_command_v1':raise ValueError('Require explicit locomotion baseline')
    sim,motors,layout,_=prepare_trial(Path(cfg['teacher_run']),Path(cfg['camera_profile']))
    c=SensorLocomotionController(cfg['robot'],motors,layout,cfg['upper_motor_posture'],cfg['locomotion_checkpoint'],cfg['body_command']);c.reset_episode()
    with np.load(sensor/'actor-initial-decision.npz',allow_pickle=False) as z:packet={k:z[k].copy() for k in z.files if k!='time_s'}
    packets=[]
    for chunk in capture['chunks']:
        if digest(sensor/chunk['file'])!=chunk['sha256']:raise ValueError('Sensor bytes changed')
        with np.load(sensor/chunk['file'],allow_pickle=False) as z:
            packets.extend({k:z[k][i].copy() for k in z.files if k!='time_s'} for i in range(chunk['rows']))
    with np.load(sensor/'actor-rgb.npz',allow_pickle=False) as z:rgb={k:z[k].copy() for k in z.files}
    from doorbench.dexterous.sensor_contract import SENSOR_KEYS
    force_error=gravity_error=gyro_error=0.;count=0;first=last=None;previous=None;bad_contacts=0;warnings=0
    initial_direction=None;min_height=float('inf');max_root_tilt=0.
    for raw in NativeTransitionArchive.read(run/'raw-transitions'):
        t=raw['interval_start_s'];force=c.force(packet,t)
        q=np.asarray(raw['qpos_before']);v=np.asarray(raw['qvel_before']);rq,rv=sim.root_qadr,sim.root_vadr
        rotation=Rotation.from_quat(q[rq+np.array([4,5,6,3])]).as_matrix()
        if first is None:first=q[rq:rq+3].copy();initial_direction=rotation[:,0]
        final_q=np.asarray(raw['qpos_after']);last=final_q[rq:rq+3].copy();min_height=min(min_height,last[2])
        final_R=Rotation.from_quat(final_q[rq+np.array([4,5,6,3])]).as_matrix()
        max_root_tilt=max(max_root_tilt,float(np.rad2deg(np.arccos(np.clip(final_R[2,2],-1,1)))))
        force_error=max(force_error,float(np.max(abs(force-np.asarray(raw['actuator_force'])[sim.actuators]))))
        gravity_error=max(gravity_error,float(np.linalg.norm(c.last_info['estimated_projected_gravity']-rotation.T@np.array([0.,0.,-1.]))))
        if previous is not None:gyro_error=max(gyro_error,float(np.linalg.norm(c.gyro-np.asarray(previous['qvel_before'])[rv+3:rv+6])))
        warn=raw.get('mujoco_warning_interval');warnings+=int(not warn or np.any(warn['before']) or np.any(warn['after']))
        for contact in raw['contacts']:
            if np.linalg.norm(contact['wrench_contact_frame'][:3])<1e-6:continue
            geoms=[sim.m.geom(g).name for g in contact['geom']];bodies=[sim.m.body(b).name for b in contact['body']]
            robot=[b.startswith('robot/') for b in bodies]
            if any(robot) and not all(robot):
                if 'floor' not in geoms or bodies[robot.index(True)] not in ('robot/left_ankle_link','robot/right_ankle_link'):bad_contacts+=1
        packet=packets[count]
        for key in ('rgb_left','rgb_right'):
            stamp=packet['sensor_time_s'][SENSOR_KEYS.index(key)];index=np.searchsorted(rgb['time_s'],stamp+1e-9,side='right')-1
            if index<0 or abs(rgb['time_s'][index]-stamp)>1e-8:raise ValueError('Missing actual camera frame')
            packet[key]=rgb[key][index].copy()
        previous=raw;count+=1
    displacement=last-first
    checks=dict(completed_duration=report['stop_reason']=='duration',complete_sensor_join=count==capture['samples'],
        exact_motor_replay=force_error<1e-5,estimated_gravity_close=gravity_error<.01,
        independently_reconstructed_gyro=gyro_error<1e-5,no_loaded_unintended_scene_contact=bad_contacts==0,
        no_warning_intervals=warnings==0,root_stable=max_root_tilt<10 and min_height>.9,
        no_runtime_pose_writes=report['runtime_pose_writes']==0,no_teacher_actions=report['teacher_actions']==0,
        original_force_delivery=report['maximum_motor_delivery_error_Nm']==0,
        no_external_assistance=report['final']['external_wrench_max']==0 and report['final']['applied_generalized_force_max']==0)
    checks={key:bool(value) for key,value in checks.items()}
    result=dict(passed=all(checks.values()),checks=checks,steps=count,duration_s=report['time_s'],
        displacement_world_m=displacement.tolist(),forward_displacement_m=float(displacement@initial_direction),
        maximum_motor_replay_error_Nm=force_error,maximum_gravity_vector_error=gravity_error,
        maximum_gyro_reconstruction_error_rad_s=gyro_error,maximum_root_tilt_deg=max_root_tilt,minimum_root_height_m=min_height,
        loaded_unintended_scene_contacts=bad_contacts,physics_steps=0,scope=__doc__,
        inputs={name:digest(run/name) for name in ('manifest.json','report.json','raw-transitions/manifest.json','own-sensors/report.json')},
        auditor_sha256=digest(__file__))
    (run/'independent-locomotion-audit.json').write_text(json.dumps(result,indent=2)+'\n');sim.close();print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
