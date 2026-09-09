#!/usr/bin/env python3
"""Bounded teacher recovery from an actual sensor-policy approach prefix.

The privileged teacher observes each attained state, including during the actor
prefix. Its counterfactual labels are separate from the actually applied forces.
This is a recovery/data-collection diagnostic, never an unassisted policy score.
No runtime pose writes, root wrenches or replayed observations are used.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from doorbench.dexterous.approach_teacher import ApproachBodyTeacher
from doorbench.dexterous.native_sensor_capture import NativeSensorCapture
from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
from doorbench.dexterous.native_transition_audit import NativeTransitionRecorder
from doorbench.dexterous.provenance import capture
from doorbench.dexterous.sensor_policy_controller import SensorPolicyController
from scripts.dexterous.evaluate_native_sensor_policy import prepare_trial
from scripts.dexterous.train_sensor_imitation import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-run', 'checkpoint', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--camera-profile', type=Path, default=Path('configs/dexterous/h1-manipulation-cameras.json'))
    p.add_argument('--actor-seconds', type=float, required=True)
    p.add_argument('--seconds', type=float, default=2.)
    p.add_argument('--max-wall-seconds', type=float, default=180.)
    a = p.parse_args()
    if (not np.isfinite([a.actor_seconds, a.seconds, a.max_wall_seconds]).all()
            or not 0 <= a.actor_seconds < a.seconds <= 3. or a.max_wall_seconds <= 0
            or abs(a.actor_seconds/.002-round(a.actor_seconds/.002)) > 1e-8):
        p.error('Require a 2 ms-aligned actor prefix and a positive trial of at most 3 seconds')
    if a.output.exists():
        raise FileExistsError('Preserve previous recovery trials')
    sim, motors, layout, chunk = prepare_trial(a.teacher_run, a.camera_profile)
    m,d = sim.m,sim.d
    original = json.loads((a.teacher_run/'manifest.json').read_text())['configuration']
    ref = json.loads((a.teacher_run/'reference.json').read_text())
    reset = json.loads((a.teacher_run/'body_reset.json').read_text())
    teacher = ApproachBodyTeacher(original['robot'], motors, reset, original['checkpoint'],
        height=ref['initial_root'][2], handoff_delay=3., yaw_weight=2.)
    actor = SensorPolicyController(a.checkpoint, motor_contract=motors, sensor_layout=layout, physics_dt_s=.002)
    actor.reset_episode()
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(robot=original['robot'], door=original['door'],
        control_source='privileged_recovery', reset_chunk_sha256=chunk['sha256'])
    capture(Path(__file__).resolve().parents[2], a.output, config)
    sensors = NativeSensorCapture(sim,motors,layout,a.output/'own-sensors',control_source='privileged_recovery')
    packet = sensors.initial_packet()
    recorder = NativeTransitionRecorder(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge')
    archive = NativeTransitionArchive(a.output/'raw-transitions')
    feet = [m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    floor = m.geom('floor').id
    loads = np.zeros(2)
    qa = [m.jnt_qposadr[m.joint('robot/'+n).id] for n in teacher.names]
    va = [m.jnt_dofadr[m.joint('robot/'+n).id] for n in teacher.names]
    labels, rows = [], []
    reason = 'duration'; started = time.monotonic(); error = 0.
    try:
        for step in range(round(a.seconds/.002)):
            t = float(d.time); rq,rv = sim.root_qadr,sim.root_vadr
            rotation = Rotation.from_quat(d.qpos[rq+np.array([4,5,6,3])])
            root = np.r_[d.qpos[rq:rq+7],d.qvel[rv:rv+3],rotation.apply(d.qvel[rv+3:rv+6])]
            target, info = teacher.force(t,root,dict(zip(teacher.names,d.qpos[qa])),
                dict(zip(teacher.names,d.qvel[va])),loads)
            from_actor = step < round(a.actor_seconds/.002)
            force = actor.force(packet,t) if from_actor else target
            labels.append(dict(time_s=t,teacher_motor_forces=target.copy(),actor_applied=from_actor))
            d.ctrl[sim.actuators] = force
            recorder.before_step();sim.plant.step();row,raw = recorder.after_step();archive.write(raw)
            error = max(error,float(np.max(abs(d.actuator_force[sim.actuators]-force))))
            packet = sensors.capture(start_s=raw['interval_start_s'],end_s=raw['interval_end_s'],
                actual_forces=d.actuator_force[sim.actuators])
            row.update(actor_applied=from_actor,teacher_stage=info['stage'])
            rows.append(row); loads[:] = 0.
            for contact in raw['contacts']:
                if floor in contact['geom']:
                    other = contact['geom'][1] if contact['geom'][0] == floor else contact['geom'][0]
                    body = int(m.geom_bodyid[other])
                    if body in feet:
                        loads[feet.index(body)] += max(0.,contact['wrench_contact_frame'][0])
            if step % 100 == 0:
                atomic_json(a.output/'progress.json',dict(time_s=float(d.time),actor_applied=from_actor,diagnostics=sim.diagnostics()))
            if not row['finite'] or row['torso_tilt_deg'] > 35 or row['root_height_m'] < .55:
                reason = 'fall_or_nonfinite';break
            if row['numerical_warnings']:
                reason = 'numerical_warning';break
            if time.monotonic()-started > a.max_wall_seconds:
                reason = 'wall_budget';break
        archive.close(complete=True);sensors.finish(complete=True)
        np.savez_compressed(a.output/'counterfactual-labels.npz',
            **{k:np.asarray([r[k] for r in labels]) for k in labels[0]})
        atomic_json(a.output/'physics-steps.json',rows)
        report = dict(control_source='privileged_recovery',unassisted_policy_success=False,
            training_admitted=False,full_task_qualified=False,stop_reason=reason,time_s=float(d.time),
            actor_steps=sum(r['actor_applied'] for r in labels),teacher_steps=sum(not r['actor_applied'] for r in labels),
            runtime_pose_writes=0,native_teacher_mirror_steps=0,maximum_motor_delivery_error_Nm=error,
            maximum_torso_tilt_deg=max(r['torso_tilt_deg'] for r in rows),final=sim.diagnostics(),scope=__doc__)
        atomic_json(a.output/'report.json',report);print(json.dumps(report))
    finally:
        sim.close()


if __name__ == '__main__':
    main()
