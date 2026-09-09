#!/usr/bin/env python3
"""Render a fixed camera variant at exact recorded physical states.

Never steps physics. Keeps the original capture unchanged. Produces a separate
camera archive and a reset observation with unavailable initial IMU/touch, bound
to the original trajectory and first actual command. Not a policy evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.camera_profile import apply_camera_profile
from doorbench.dexterous.sensor_contract import ActorObservationBuilder
from doorbench.dexterous.native_transition_archive import unpacked


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run=a.run;source=run/'own-sensors'
    config=json.loads((run/'manifest.json').read_text())['configuration'];robot=Path(config['robot'])
    if digest(robot)!=digest(run/'robot-input.xml'):raise ValueError('Robot differs from recorded run')
    layout=apply_camera_profile(json.loads((source/'layout.json').read_text()),json.loads(a.profile.read_text()))
    sim=DexterousDoorEnv(config['door'],robot,json.loads(robot.with_suffix('.audit.json').read_text()));m,d=sim.m,sim.d
    if digest(Path(config['door'])/'door.xml')!=digest(run/'door-input.xml'):raise ValueError('Door differs from recorded run')
    for camera in layout['cameras']:
        i=m.camera('robot/'+camera['name']).id
        if m.body(m.cam_bodyid[i]).name!='robot/'+camera['body_name']:raise ValueError('Camera body mismatch')
        m.cam_pos[i]=camera['position_body_m'];m.cam_quat[i]=camera['quaternion_wxyz_body'];m.cam_fovy[i]=camera['fovy_degrees']
    trace=json.loads((run/'trace.json').read_text());times=np.asarray([row['sim_time_s'] for row in trace])
    with np.load(run/'trajectory.npz',allow_pickle=False) as z:q=z['qpos'].copy()
    with np.load(source/'actor-rgb.npz',allow_pickle=False) as z:frame_times=z['time_s'].copy()
    indices=np.searchsorted(times,frame_times)
    if len(times)!=len(q) or np.any(indices>=len(times)) or not np.allclose(times[indices],frame_times,rtol=0,atol=1e-9):
        raise ValueError('Camera epochs must match exact actual trajectory samples')
    raw_manifest=json.loads((run/'raw-transitions/manifest.json').read_text());chunk=run/'raw-transitions'/raw_manifest['chunks'][0]['file']
    if digest(chunk)!=raw_manifest['chunks'][0]['sha256']:raise ValueError('Initial transition changed')
    with np.load(chunk,allow_pickle=False) as z:first=next(unpacked({k:z[k] for k in z.files}))
    if first['interval_start_s']!=0:raise ValueError('Require actual episode reset')
    a.output.mkdir(parents=True,exist_ok=False)
    frames={k:[] for k in ('rgb_left','rgb_right')}
    def render(pose):
        d.qpos[:]=pose;mujoco.mj_kinematics(m,d);mujoco.mj_camlight(m,d)
        return sim.observe(images=True)
    initial=render(first['qpos_before']);builder=ActorObservationBuilder(joint_count=69,action_count=61,tactile_dimension=layout['tactile_dimension'])
    d.qvel[:]=first['qvel_before']
    for key in ('joint_position','joint_velocity','rgb_left','rgb_right'):
        value=np.asarray(first['qvel_before'])[sim.vadr].astype(np.float32) if key=='joint_velocity' else initial[key]
        builder.push(key,value,capture_s=0.)
    packet=builder.observe(now_s=0.,previous_action=np.zeros(61))
    np.savez_compressed(a.output/'actor-initial-decision.npz',**packet,time_s=np.asarray(0.),motor_forces=np.asarray(first['actuator_force'])[sim.actuators])
    for j,index in enumerate(indices):
        observation=render(q[index])
        for key in frames:
            frames[key].append(observation[key])
            if j in (0,len(indices)//4,len(indices)//2,len(indices)-1):Image.fromarray(observation[key]).save(a.output/f'{key}-{j:05d}.png')
        if j%100==0:(a.output/'progress.json').write_text(json.dumps(dict(frames=j+1,total=len(indices),time_s=float(frame_times[j]))))
    np.savez_compressed(a.output/'actor-rgb.npz',time_s=frame_times,**{k:np.asarray(v) for k,v in frames.items()})
    (a.output/'layout.json').write_text(json.dumps(layout,indent=2)+'\n')
    receipt=dict(scope=__doc__,complete=True,frames=len(indices),physics_steps=0,
        camera_pose_refresh='mj_camlight after current-state kinematics, before render',
        initial_observation='Actual reset encoders and newly rendered fixed cameras; initial IMU/tactile invalid; label is first actual motor force',
        source_files={str(f):digest(f) for f in (run/'manifest.json',run/'trace.json',run/'trajectory.npz',source/'layout.json',chunk,a.profile,Path(__file__))},
        output_sha256={n:digest(a.output/n) for n in ('actor-rgb.npz','actor-initial-decision.npz','layout.json')},
        limitation='Camera variant only; requires original sensor/physics qualification plus independent frame and initial-packet audits before training')
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');sim.close();print(json.dumps(dict(frames=len(indices),output=str(a.output))))


if __name__=='__main__':main()
