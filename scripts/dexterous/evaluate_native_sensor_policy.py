#!/usr/bin/env python3
"""Execute an unassisted native sensor actor from the teacher's actual reset.

No recorded observations or teacher actions are replayed. This bounded probe
reports survival, mechanism and passage diagnostics, not full task qualification.
"""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from doorbench.dexterous.camera_profile import apply_camera_profile
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.native_sensor_capture import NativeSensorCapture
from doorbench.dexterous.native_transition_audit import NativeTransitionRecorder
from doorbench.dexterous.native_transition_archive import NativeTransitionArchive, unpacked
from doorbench.dexterous.native_sensor_demonstrations import digest
from doorbench.dexterous.sensor_policy_controller import SensorPolicyController
from doorbench.dexterous.provenance import capture
from scripts.dexterous.export_sensor_layout import export_layout
from scripts.dexterous.train_sensor_imitation import atomic_json


def prepare_trial(run, camera_profile):
    """Reconstruct the verified reset and original capped native force interface."""
    config = json.loads((run/'manifest.json').read_text())['configuration']
    robot = Path(config['robot']); door = Path(config['door'])
    if digest(robot) != digest(run/'robot-input.xml') or digest(door/'door.xml') != digest(run/'door-input.xml'):
        raise ValueError('Actual native source models differ from the admitted teacher')
    motors = json.loads((run/'motors-input.json').read_text())
    layout = apply_camera_profile(export_layout(robot), json.loads(camera_profile.read_text()))
    layout['capture_clock_profile'] = 'native-preintegration-inertial-tactile-v1'
    sim = DexterousDoorEnv(door, robot, json.loads(robot.with_suffix('.audit.json').read_text()))
    sim.reset(randomize=False, images=False)
    m,d = sim.m,sim.d
    raw_manifest = json.loads((run/'raw-transitions/manifest.json').read_text())
    chunk = raw_manifest['chunks'][0]; path = run/'raw-transitions'/chunk['file']
    if digest(path) != chunk['sha256']:
        raise ValueError('Actual reset evidence changed')
    with np.load(path, allow_pickle=False) as z:
        initial = next(unpacked({k:z[k] for k in z.files}))
    if initial['interval_start_s'] != 0 or len(initial['qpos_before']) != m.nq:
        raise ValueError('Require the original complete physical reset')
    d.qpos[:] = initial['qpos_before']; d.qvel[:] = initial['qvel_before']
    mujoco.mj_forward(m,d)
    caps = np.array([v['force_range'] for v in motors['actuators']])
    if not np.array_equal(caps, m.actuator_forcerange[sim.actuators]):
        raise ValueError('Native motor caps differ')
    # Same capped force-control interface as the qualified teacher.
    m.actuator_gainprm[sim.actuators,0]=1.;m.actuator_biasprm[sim.actuators,:3]=0.
    m.actuator_ctrlrange[sim.actuators]=caps;d.ctrl[sim.actuators]=0.
    for cam in layout['cameras']:
        i=m.camera('robot/'+cam['name']).id
        if m.body(m.cam_bodyid[i]).name!='robot/'+cam['body_name']:
            raise ValueError('Camera body differs')
        m.cam_pos[i]=cam['position_body_m'];m.cam_quat[i]=cam['quaternion_wxyz_body'];m.cam_fovy[i]=cam['fovy_degrees']
    return sim, motors, layout, chunk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-run', 'checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--camera-profile', type=Path, default=Path('configs/dexterous/h1-manipulation-cameras.json'))
    p.add_argument('--seconds', type=float, default=130.)
    p.add_argument('--max-wall-seconds', type=float, default=900.)
    a = p.parse_args()
    if not np.isfinite([a.seconds, a.max_wall_seconds]).all() or min(a.seconds, a.max_wall_seconds) <= 0:
        p.error('Finite positive rollout bounds required')
    if a.output.exists():
        raise FileExistsError('Preserve earlier physical rollouts')
    sim, motors, layout, chunk = prepare_trial(a.teacher_run, a.camera_profile)
    m,d = sim.m,sim.d
    config = json.loads((a.teacher_run/'manifest.json').read_text())['configuration']
    robot, door = Path(config['robot']), Path(config['door'])
    actor = SensorPolicyController(a.checkpoint, motor_contract=motors, sensor_layout=layout, physics_dt_s=.002)
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    configuration.update(robot=str(robot),door=str(door),control_source='sensor_actor',reset_chunk_sha256=chunk['sha256'],runtime_pose_writes=0)
    capture(Path(__file__).resolve().parents[2],a.output,configuration)
    sensors=NativeSensorCapture(sim,motors,layout,a.output/'own-sensors',control_source='sensor_actor')
    packet=sensors.initial_packet();actor.reset_episode()
    recorder=NativeTransitionRecorder(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge')
    archive=NativeTransitionArchive(a.output/'raw-transitions')
    started=time.monotonic();rows=[];reason='duration';maximum_delivery_error=0.
    for step in range(int(round(a.seconds/.002))):
        force=actor.force(packet,float(d.time));d.ctrl[sim.actuators]=force
        recorder.before_step();sim.plant.step();row,raw=recorder.after_step();archive.write(raw)
        maximum_delivery_error=max(maximum_delivery_error,float(np.max(abs(d.actuator_force[sim.actuators]-force))))
        packet=sensors.capture(start_s=raw['interval_start_s'],end_s=raw['interval_end_s'],actual_forces=d.actuator_force[sim.actuators])
        rows.append(row)
        if step%100==0:atomic_json(a.output/'progress.json',dict(time_s=float(d.time),diagnostics=sim.diagnostics(),control_source='sensor_actor'))
        if not row['finite'] or row['torso_tilt_deg']>35 or row['root_height_m']<.55:
            reason='fall_or_nonfinite';break
        if row['numerical_warnings']:
            reason='numerical_warning';break
        if time.monotonic()-started>a.max_wall_seconds:
            reason='wall_budget';break
    archive.close(complete=True);sensors.finish(complete=True)
    atomic_json(a.output/'physics-steps.json',rows)
    report=dict(control_source='sensor_actor',physical_rollout_evaluated=True,full_task_qualified=False,
        stop_reason=reason,time_s=float(d.time),runtime_pose_writes=0,teacher_actions=0,
        maximum_motor_delivery_error_Nm=maximum_delivery_error,final=sim.diagnostics(),scope=__doc__)
    atomic_json(a.output/'report.json',report);sim.close();print(json.dumps(report))


if __name__=='__main__':main()
