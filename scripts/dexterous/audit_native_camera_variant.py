#!/usr/bin/env python3
"""Independently render sampled fixed-eye frames from raw actual transitions."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.native_transition_archive import unpacked
from doorbench.dexterous.sensor_contract import SENSOR_KEYS, ACTOR_KEYS


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--variant',type=Path,required=True)
    a=p.parse_args();receipt=json.loads((a.variant/'receipt.json').read_text());layout=json.loads((a.variant/'layout.json').read_text())
    if receipt.get('complete') is not True:raise ValueError('Require a finished camera variant')
    for name,sha in receipt['source_files'].items():
        if digest(name)!=sha:raise ValueError('Camera source changed: '+name)
    for name,sha in receipt['output_sha256'].items():
        if digest(a.variant/name)!=sha:raise ValueError('Camera output changed: '+name)
    c=json.loads((a.run/'manifest.json').read_text())['configuration'];robot=Path(c['robot'])
    if digest(robot)!=digest(a.run/'robot-input.xml') or digest(Path(c['door'])/'door.xml')!=digest(a.run/'door-input.xml'):
        raise ValueError('Physical replay model differs')
    s=DexterousDoorEnv(c['door'],robot,json.loads(robot.with_suffix('.audit.json').read_text()));m,d=s.m,s.d
    for cam in layout['cameras']:
        i=m.camera('robot/'+cam['name']).id
        if m.body(m.cam_bodyid[i]).name!='robot/'+cam['body_name']:raise ValueError('Changed physical camera body')
        m.cam_pos[i]=cam['position_body_m'];m.cam_quat[i]=cam['quaternion_wxyz_body'];m.cam_fovy[i]=cam['fovy_degrees']
    manifest=json.loads((a.run/'raw-transitions/manifest.json').read_text())
    def raw_at(t):
        chunk=next(c for c in manifest['chunks'] if c['interval_start_s']-1e-8<=t<c['interval_end_s']-1e-8)
        f=a.run/'raw-transitions'/chunk['file']
        if digest(f)!=chunk['sha256']:raise ValueError('Raw transition changed')
        with np.load(f,allow_pickle=False) as z:rows=list(unpacked({k:z[k] for k in z.files}))
        row=min(rows,key=lambda r:abs(r['interval_start_s']-t))
        if abs(row['interval_start_s']-t)>1e-8:raise ValueError('No exact actual frame epoch')
        return row
    def images(q):
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_camlight(m,d)
        return s.observe(images=True)
    checked=[];maximum=0
    with np.load(a.variant/'actor-rgb.npz',allow_pickle=False) as z:
        for i in sorted({0,len(z['time_s'])//8,len(z['time_s'])//4,len(z['time_s'])//2,3*len(z['time_s'])//4,len(z['time_s'])-1}):
            t=float(z['time_s'][i]);raw=raw_at(t-m.opt.timestep);rendered=images(raw['qpos_after'])
            for key in ('rgb_left','rgb_right'):
                error=int(np.max(abs(rendered[key].astype(int)-z[key][i].astype(int))));maximum=max(maximum,error)
            checked.append(t)
    raw=raw_at(0.);rendered=images(raw['qpos_before'])
    with np.load(a.variant/'actor-initial-decision.npz',allow_pickle=False) as z:
        checks=dict(initial_keys=set(z.files)==set(ACTOR_KEYS)|{'time_s','motor_forces'},
            initial_clock=float(z['time_s'])==0.,
            actual_initial_encoders=np.array_equal(z['joint_position'],np.clip(np.asarray(raw['qpos_before'])[s.qadr].astype(np.float32),-20,20)) and np.array_equal(z['joint_velocity'],np.clip(np.asarray(raw['qvel_before'])[s.vadr].astype(np.float32),-100,100)),
            initial_unobserved_imu_touch_invalid=all(not z['sensor_valid'][SENSOR_KEYS.index(k)] and not z[k].any() for k in ('imu_gyro','imu_accelerometer','tactile')),
            initial_observed_sensors_valid=all(z['sensor_valid'][SENSOR_KEYS.index(k)] for k in ('joint_position','joint_velocity','rgb_left','rgb_right')),
            initial_previous_action_zero=not z['previous_action'].any(),
            first_actual_motor_label=np.array_equal(z['motor_forces'],np.asarray(raw['actuator_force'])[s.actuators]),
            initial_camera_exact=all(np.array_equal(z[k],rendered[k]) for k in ('rgb_left','rgb_right')),
            sampled_cameras_pixel_exact=maximum==0)
    checks={k:bool(v) for k,v in checks.items()}
    report=dict(passed=all(checks.values()),checks=checks,checked_frame_times_s=checked,maximum_pixel_error=maximum,
        scope=__doc__+' No new physics or policy rollout; only recorded-state camera/initial-observation validation.',
        source_sha256=digest(__file__),variant_receipt_sha256=digest(a.variant/'receipt.json'))
    (a.variant/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n');s.close();print(json.dumps(report))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
