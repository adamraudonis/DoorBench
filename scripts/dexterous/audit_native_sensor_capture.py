#!/usr/bin/env python3
"""Join finite own-sensor packets to actual native transitions, without stepping.

Checks every encoder/action/time row and samples the independent preintegration
IMU angular-rate and tactile-force calculations. Accelerometer reconstruction
and initial-decision coverage remain separate requirements; this does not
qualify a learned policy or dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
from doorbench.dexterous.sensor_contract import SENSOR_KEYS, AngularTaxelGrid


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path)
    a=parser.parse_args();sensor=a.run/'own-sensors';report=json.loads((sensor/'report.json').read_text())
    if report.get('capture_complete') is not True:raise ValueError('Sensor writer must finish first')
    layout=json.loads((sensor/'layout.json').read_text());config=json.loads((a.run/'manifest.json').read_text())['configuration']
    robot=Path(config['robot']);sim=DexterousDoorEnv(config['door'],robot,json.loads(robot.with_suffix('.audit.json').read_text()))
    m=sim.m;d=mujoco.MjData(m)
    with np.load(sensor/'actor-rgb.npz',allow_pickle=False) as z:rgb={k:z[k].copy() for k in z.files}
    checks=dict(complete_capture=True,strict_rgb=set(rgb)=={'time_s','rgb_left','rgb_right'},
        ordered_rgb=bool(np.all(np.diff(rgb['time_s'])>0)),finite=True,strict_numeric=True,
        clocks=True,encoder_values=True,actual_previous_action=True,valid_sensors=True,rgb_capture_times=True)
    for key in ('rgb_left','rgb_right'):
        checks[key+'_shape']=rgb[key].dtype==np.uint8 and rgb[key].shape==(len(rgb['time_s']),128,128,3)
    raw=iter(NativeTransitionArchive.read(a.run/'raw-transitions'))
    count=0;gyro_error=0.;gyro_samples=0;tactile_error=0.
    mounts=[]
    for cfg in layout['sensors']:
        sid=m.sensor('robot/'+cfg['name']).id;site=int(m.sensor_objid[sid])
        mounts.append((site,int(m.site_bodyid[site]),AngularTaxelGrid(cfg['width'],cfg['height'],tuple(cfg['fov_degrees']))))
    caps=m.actuator_forcerange[sim.actuators]
    expected=set(SENSOR_KEYS)-{'rgb_left','rgb_right'}|{'time_s','previous_action','sensor_time_s','sensor_valid'}
    for chunk in report['chunks']:
        path=sensor/chunk['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=chunk['sha256']:raise ValueError('Sensor chunk changed')
        with np.load(path,allow_pickle=False) as z:
            checks['strict_numeric'] &= set(z.files)==expected
            checks['finite'] &= all(np.isfinite(z[k]).all() for k in z.files)
            if len(z['time_s'])!=chunk['rows']:raise ValueError('Sensor row count differs')
            for i,t in enumerate(z['time_s']):
                row=next(raw);start=row['interval_start_s'];end=row['interval_end_s']
                q=np.asarray(row['qpos_after']);v=np.asarray(row['qvel_after'])
                checks['clocks'] &= abs(t-end)<1e-8
                checks['encoder_values'] &= np.array_equal(z['joint_position'][i],np.clip(q[sim.qadr].astype(np.float32),-20,20)) and np.array_equal(z['joint_velocity'][i],np.clip(v[sim.vadr].astype(np.float32),-100,100))
                force=np.asarray(row['actuator_force'])[sim.actuators]
                action=np.clip(2*(force-caps[:,0])/(caps[:,1]-caps[:,0])-1,-1,1).astype(np.float32)
                checks['actual_previous_action'] &= np.array_equal(z['previous_action'][i],action)
                checks['valid_sensors'] &= bool(z['sensor_valid'][i].all())
                for key,stamp in zip(SENSOR_KEYS,z['sensor_time_s'][i]):
                    if key.startswith('rgb_'):
                        checks['rgb_capture_times'] &= bool(np.any(rgb['time_s']==stamp) and 0<=t-stamp<.041)
                    else:checks['clocks'] &= abs(stamp-(end if key.startswith('joint_') else start))<1e-8
                if count%100==0:
                    d.qpos[:]=row['qpos_before'];d.qvel[:]=row['qvel_before']
                    mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_comVel(m,d);mujoco.mj_sensorVel(m,d)
                    gyro=np.clip(d.sensor('robot/imu_gyro').data.astype(np.float32),-100,100)
                    gyro_error=max(gyro_error,float(np.max(abs(gyro-z['imu_gyro'][i]))));gyro_samples+=1
                    body_contacts={}
                    for contact in row['contacts']:
                        world=np.asarray(contact['frame_world']).T@np.asarray(contact['wrench_contact_frame'][:3])
                        for side,body in enumerate(contact['body']):
                            points,forces=body_contacts.setdefault(body,([],[]))
                            points.append(contact['position_world_m']);forces.append((1 if side==1 else -1)*world)
                    tactile=[]
                    for site,body,grid in mounts:
                        points,forces=body_contacts.get(body,([],[]))
                        tactile.append(grid.bin_forces(np.asarray(points).reshape(-1,3),np.asarray(forces).reshape(-1,3),d.site_xpos[site],d.site_xmat[site].reshape(3,3)))
                    tactile_error=max(tactile_error,float(np.max(abs(np.clip(np.concatenate(tactile),-100,100)-z['tactile'][i]))))
                count+=1
    checks['complete_raw_join']=next(raw,None) is None and count==report['samples']
    checks['independent_sampled_gyro']=gyro_samples>0 and gyro_error<1e-6
    checks['independent_sampled_tactile']=gyro_samples>0 and tactile_error<1e-5
    out=dict(passed=all(checks.values()),checks=checks,samples=count,gyro_samples=gyro_samples,
        maximum_gyro_error_rad_s=gyro_error,maximum_tactile_error_N=tactile_error,scope=__doc__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sensor_report_sha256=hashlib.sha256((sensor/'report.json').read_bytes()).hexdigest(),
        raw_manifest_sha256=hashlib.sha256((a.run/'raw-transitions/manifest.json').read_bytes()).hexdigest())
    (sensor/'independent-join-audit.json').write_text(json.dumps(out,indent=2)+'\n');sim.close();print(json.dumps(out))
    return 0 if out['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
