"""Read recorded motor commands and reconstruct the withdrawal handoff algebra.

No plant steps or API calls. The proposed first-command continuity is an
algebraic check, not evidence of improved physical grip or subsequent motion.
"""
import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

import mujoco
import numpy as np

from doorbench.dexterous.grasp_verification import scalar_transmission_matrix
from doorbench.dexterous.json_record_stream import iter_json_object_array
from doorbench.dexterous.landed_left_planner import LandedLeftScene
from doorbench.dexterous.motor_handoff import MotorHandoff
from doorbench.dexterous.withdrawal_motor_capture import WithdrawalMotorCapture


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--release-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise ValueError('Fresh diagnosis paths required')
    source, release = args.source_run.resolve(), args.release_run.resolve()
    manifest = json.loads((release/'manifest.json').read_text())
    cfg = manifest['configuration']
    config_path = Path(cfg['standing_withdrawal_path'])
    config = json.loads(config_path.read_text())
    if Path(config['source_run']).resolve() != source:
        raise ValueError('Release must bind its actual source')
    source_raw = json.loads((source/'raw-transitions/manifest.json').read_text())
    release_raw = json.loads((release/'raw-transitions/manifest.json').read_text())
    if not source_raw['complete'] or not release_raw['complete']:
        raise ValueError('Complete actual archives required')
    previous_chunk = source_raw['chunks'][-1]
    next_chunk = release_raw['chunks'][len(source_raw['chunks'])]
    paths = [source/'raw-transitions'/previous_chunk['file'],
             release/'raw-transitions'/previous_chunk['file'],
             release/'raw-transitions'/next_chunk['file']]
    for path, digest in zip(paths, [previous_chunk['sha256'], previous_chunk['sha256'], next_chunk['sha256']]):
        if sha(path) != digest:
            raise ValueError('Actual transition bytes changed')
    robot, door, motor_path = Path(cfg['robot']), Path(cfg['door'])/'door.xml', Path(cfg['motors'])
    if sha(robot) != manifest['inputs']['robot']['sha256'] or sha(door) != manifest['inputs']['door']['door.xml']:
        raise ValueError('Recorded model bytes required')
    motors = json.loads(motor_path.read_text())
    scene = LandedLeftScene(robot, door)
    m, d = scene.m, scene.d
    joint_ids = np.array([m.joint('robot/'+n).id for n in motors['joint_names']])
    qa, va = m.jnt_qposadr[joint_ids], m.jnt_dofadr[joint_ids]
    actuators = np.array([m.actuator('robot/'+a['name']).id for a in motors['actuators']])
    matrix = scalar_transmission_matrix(m, actuators, joint_ids)
    names = [a['name'] for a in motors['actuators']]
    caps = np.array([a['force_range'] for a in motors['actuators']])
    finger = np.array([i for i,n in enumerate(names) if n.startswith('rh_') and 'WRJ' not in n])
    arm = np.array([(n.startswith('right_') and not any(x in n for x in ('hip','knee','ankle')))
                    or n.startswith('rh_A_WRJ') for n in names])
    damping = np.array([(.8 if 'WRJ' in n else 10.) if is_arm else 20. if n == 'torso' else 0.
                        for n,is_arm in zip(names,arm)])
    damping -= np.array([a['bias'][2] for a in motors['actuators']])
    kd = damping.copy()
    kd[finger] *= np.sqrt(5.)
    with np.load(paths[0]) as z:
        preceding = z['controls'][-1, actuators].copy()
        previous_time = float(z['interval_start_s'][-1])
        initial_time = float(z['interval_end_s'][-1])
        initial_q, initial_v = z['qpos_after'][-1].copy(), z['qvel_after'][-1].copy()
        capture_q, capture_v = z['qpos_before'][-1].copy(), z['qvel_before'][-1].copy()
        prefix_commands = z['controls'][z['interval_start_s'] >= initial_time-.3-1e-8][:,actuators].copy()
    with np.load(paths[-1]) as z:
        mask = z['interval_start_s'] < initial_time+.3-1e-8
        times = z['interval_start_s'][mask].copy()
        commands = z['controls'][mask][:,actuators].copy()
        qpos, qvel = z['qpos_before'][mask].copy(), z['qvel_before'][mask].copy()
        if not np.array_equal(qpos[0],initial_q) or not np.array_equal(qvel[0],initial_v):
            raise ValueError('Handoff did not preserve exact attained state')
    if cfg['standing_transfer_handoff_seconds'] != 0:
        raise ValueError('This diagnosis tests the observed zero-handoff source path')
    actual_velocity = matrix @ initial_v[va]
    initial_damping = kd*actual_velocity
    capture = WithdrawalMotorCapture(caps)
    capture.observe(previous_time, preceding)
    corrected_first = MotorHandoff(capture.capture(initial_time), commands[0], caps, 1.).force(commands[0],0.)
    # Reconstruct every finger command from original preload, gravity, posture
    # and final outer MotorHandoff, using recorded states only.
    inverse = np.linalg.pinv(matrix[finger].T)
    d.qpos[:], d.qvel[:] = capture_q, capture_v
    mujoco.mj_forward(m,d)
    captured_gravity = inverse @ d.qfrc_bias[va]
    preload = preceding[finger]-captured_gravity
    target = (matrix @ initial_q[qa])[finger]
    kp = np.array([a['kp'] for a in motors['actuators']])[finger]*5.
    residuals, snapshots = [], []
    first_unblended = None
    for i,t in enumerate(times):
        d.qpos[:], d.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_forward(m,d)
        gravity = inverse @ d.qfrc_bias[va]
        proportional = kp*(target-(matrix@qpos[i,qa])[finger])
        velocity_term = -kd[finger]*(matrix@qvel[i,va])[finger]
        unblended = np.clip(preload+gravity+proportional+velocity_term,caps[finger,0],caps[finger,1])
        if first_unblended is None:
            first_unblended = unblended.copy()
        u = min(1.,max(0.,float(t-initial_time)))
        blend = u**3*(10+u*(-15+6*u))
        predicted = np.clip(unblended+(1-blend)*(preceding[finger]-first_unblended),caps[finger,0],caps[finger,1])
        residuals.append(float(np.max(abs(predicted-commands[i,finger]))))
        if i in (0,50,96,149):
            snapshots.append(dict(command_time_s=float(t),
                FFJ0_gravity_Nm=float(gravity[list(finger).index(names.index('rh_A_FFJ0'))]),
                motor_commands_Nm=dict(zip(names,commands[i].tolist()))))
    per_motor = [dict(name=n,source_command_Nm=float(preceding[i]),
        first_withdrawal_command_Nm=float(commands[0,i]),jump_Nm=float(commands[0,i]-preceding[i]),
        actual_source_motor_velocity_rad_s=float(actual_velocity[i]),
        effective_kd=float(kd[i]),new_damping_term_Nm=float(-initial_damping[i]),
        prefix_0p3s_command_range_Nm=[float(prefix_commands[:,i].min()),float(prefix_commands[:,i].max())],
        withdrawal_0p3s_command_range_Nm=[float(commands[:,i].min()),float(commands[:,i].max())])
        for i,n in enumerate(names)]
    captured_sources = {}
    source_text = {}
    with tarfile.open(release/'source.tar.gz') as tar:
        for name in ('standing_transfer.py','standing_withdrawal.py','attained_hand_tracking.py','attained_arm_tracking.py','motor_handoff.py'):
            path = 'doorbench/dexterous/'+name
            data = tar.extractfile(path).read()
            captured_sources[path] = hashlib.sha256(data).hexdigest()
            source_text[name] = data.decode('utf-8')
    branch = source_text['standing_transfer.py'].split('if self.handoff_seconds:',1)[1]
    if 'teacher.last_force=forces.copy()' not in branch:
        raise ValueError('Captured source no longer matches diagnosed cache-write branch')
    controller_rows = []
    with gzip.open(release/'controller-steps.json.gz','rt') as stream:
        for row in iter_json_object_array(stream):
            t = row['time_s']
            if any(abs(t-v)<1e-8 for v in (previous_time,initial_time,initial_time+.192)):
                controller_rows.append({k:row[k] for k in ('time_s','phase','arm_clipped_motors',
                    'maximum_arm_target_error_rad','maximum_finger_transmission_error_rad',
                    'return_palm_error_m','return_palm_rotation_error_rad') if k in row})
            if t>initial_time+.2:break
    inputs = paths+[source/'manifest.json',release/'manifest.json',release/'raw-transitions/manifest.json',
        source/'raw-transitions/manifest.json',release/'source.tar.gz',release/'controller-steps.json.gz',
        robot,door,motor_path,config_path,Path(__file__)]
    result = dict(schema='doorbench.withdrawal-command-handoff-diagnosis.v1',scope=__doc__,
        physics_steps=0,API_calls=0,first_command_time_s=float(times[0]),first_interval_end_s=float(times[0]+.002),
        preceding_command_time_s=previous_time,exact_attained_state_continuity=True,
        source_transfer_handoff_seconds=0.,finger_damping_hypothesis_supported=False,
        maximum_finger_kd=float(np.max(abs(kd[finger]))),
        maximum_first_finger_command_jump_Nm=float(np.max(abs(commands[0,finger]-preceding[finger]))),
        maximum_source_damping_term_Nm=float(np.max(abs(initial_damping))),
        maximum_actual_first_command_jump_Nm=float(np.max(abs(commands[0]-preceding))),
        maximum_corrected_algebraic_first_command_jump_Nm=float(np.max(abs(corrected_first-preceding))),
        maximum_reconstructed_finger_command_error_Nm=max(residuals),
        cause='StandingTransferTeacher zero-handoff path returns final arm/left overrides without updating nested acquisition.last_force; StandingWithdrawalTeacher captures that stale operation-only cache.',
        proposed_option={'capture_returned_motor_command':True},
        limitation='Only the first command continuity is established algebraically. Subsequent drift, forces, grasp, support and release require a new physical trial.',
        per_motor=per_motor,selected_snapshots=snapshots,controller_telemetry=controller_rows,
        captured_source_sha256=captured_sources,input_sha256={str(p.resolve()):sha(p) for p in inputs})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    largest=sorted(per_motor,key=lambda r:abs(r['jump_Nm']),reverse=True)[:8]
    lines=['The first withdrawal command captures a stale nested motor cache.','',
        f'The preceding command is at {previous_time:.3f} s; the first withdrawal command is at {times[0]:.3f} s (interval ends {times[0]+.002:.3f} s). State continuity is exact.', '',
        'Finger damping is zero for all RH fingers. Their first commands are unchanged. The largest source-velocity damping term across the robot is '+f'{max(abs(initial_damping)):.6f} Nm.', '',
        '| Motor | Previous Nm | First withdrawal Nm | Jump Nm |','|---|---:|---:|---:|']
    lines += [f"| {r['name']} | {r['source_command_Nm']:.6f} | {r['first_withdrawal_command_Nm']:.6f} | {r['jump_Nm']:+.6f} |" for r in largest]
    lines += ['',result['cause'],'',
        'The opt-in withdrawal-only capture records the final delegated command without changing it. The existing blend then reproduces that command on entry to floating-point precision. All original motor limits remain unchanged.', '',
        result['limitation'],'',f"Full per-motor terms, captured source hashes and evidence hashes: `{args.output.name}`."]
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('maximum_finger_kd','maximum_source_damping_term_Nm',
        'maximum_actual_first_command_jump_Nm','maximum_corrected_algebraic_first_command_jump_Nm',
        'maximum_reconstructed_finger_command_error_Nm')},indent=2))


if __name__ == '__main__':
    main()
