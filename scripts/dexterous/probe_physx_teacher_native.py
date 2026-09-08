#!/usr/bin/env python3
"""Run the Isaac motor teacher on the unchanged native free-base plant.

CPU development only; this does not establish cross-engine parity or task skill.
The same bounded affine motor law is issued in force units, without helper forces.
"""
import argparse
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.contact_audit import lever_contacts
from doorbench.dexterous.provenance import capture
from physx_teacher import HandleTeacher
from panel_push_teacher import PanelPushTeacher


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('robot', 'door', 'motors', 'reference', 'output'):
        p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--seconds', type=float, default=16.)
    p.add_argument('--panel-push', action='store_true')
    p.add_argument('--native-stance', action='store_true', help='Use native measured contacts for a 500 Hz development stance loop')
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Use a new output directory, preserving every attempt')
    capture(Path(__file__).resolve().parents[2], a.output,
        {k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}, timing='before_native_teacher')
    (a.output/'run.pid').write_text(str(os.getpid()))
    pipeline = dict(stage='Native initialized opening development', started_at_unix=time.time(), completion_marker='NATIVE_TEACHER_COMPLETE')
    (a.output/'pipeline.json').write_text(json.dumps(pipeline))
    sim = DexterousDoorEnv(a.door, a.robot, json.loads(a.robot.with_suffix('.audit.json').read_text()))
    m, d = sim.m, sim.d
    ref = json.loads(a.reference.read_text()); motors = json.loads(a.motors.read_text())
    sim.reset(randomize=False, images=False)
    d.qpos[sim.root_qadr:sim.root_qadr+7] = ref['initial_root']
    names = motors['joint_names']; ids = [m.joint('robot/'+n).id for n in names]
    qa = m.jnt_qposadr[ids]; va = m.jnt_dofadr[ids]
    d.qpos[qa] = [ref['initial_joints'][n] for n in names]
    d.qvel[:] = 0.; mujoco.mj_forward(m, d)
    aids = [m.actuator('robot/'+v['name']).id for v in motors['actuators']]
    matrix = np.zeros((len(aids), len(ids))); index = {n:i for i,n in enumerate(names)}
    for i, motor in enumerate(motors['actuators']):
        for n, c in motor['terms'].items(): matrix[i,index[n]] = c
    kp = np.array([v['kp'] for v in motors['actuators']]); bias = np.array([v['bias'] for v in motors['actuators']])
    limits = np.array([v['force_range'] for v in motors['actuators']]); controls = np.array([v['control_range'] for v in motors['actuators']])
    arm = np.array([(v['name'].startswith('right_') and not any(n in v['name'] for n in ('hip','knee','ankle'))) or v['name'].startswith('rh_A_WRJ') for v in motors['actuators']])
    fingers = np.array([v['name'].startswith('rh_') and 'WRJ' not in v['name'] for v in motors['actuators']])
    damping = np.array([(.8 if 'WRJ' in v['name'] else 10.) if arm[i] else 20. if v['name']=='torso' else 0. for i,v in enumerate(motors['actuators'])])
    reset_lengths = matrix@d.qpos[qa]
    # Equivalent command-unit conversion only. Native transmission, force caps,
    # passive damping/armature, contacts, geometry, and joint stops remain intact.
    m.actuator_gainprm[aids,0] = 1.; m.actuator_biasprm[aids,:3] = 0.
    m.actuator_ctrlrange[aids] = limits
    teacher = (PanelPushTeacher if a.panel_push else HandleTeacher)(str(a.robot), motors, ref, stance_qp=True, grip_force=6.)
    if a.native_stance:
        from doorbench.dexterous.stance import StanceController
        native_stance = StanceController(sim)
    leaf = m.body('leaf').id; handle = m.body('leaf_handle').id
    hj = m.jnt_qposadr[m.joint('leaf_handle_hinge').id]; dj = m.jnt_qposadr[m.joint('leaf_hinge').id]
    bj = m.jnt_qposadr[m.joint('leaf_latch_bolt_slide').id]
    hand_bodies = [b for b in range(m.nbody) if m.body(b).name.startswith('robot/rh_')]
    hand_names = {b:m.body(b).name.removeprefix('robot/') for b in hand_bodies}
    rows = []; states = {k:[] for k in ('qpos','qvel','ctrl')}; failed = None
    def emit(v):
        text = json.dumps(v); print(text, flush=True)
        with (a.output/'run.log').open('a') as f: f.write(text+'\n')
    try:
        for step in range(round(a.seconds/m.opt.timestep)):
            if step%10 == 0:
                root = d.qpos[sim.root_qadr:sim.root_qadr+7].copy()
                rotation = d.xmat[sim.pelvis].reshape(3,3)
                root = np.r_[root, d.qvel[sim.root_vadr:sim.root_vadr+3], rotation@d.qvel[sim.root_vadr+3:sim.root_vadr+6]]
                hand_forces = {n:np.zeros(3) for n in hand_names.values()}
                for i,c in enumerate(d.contact[:d.ncon]):
                    w = np.zeros(6); mujoco.mj_contactForce(m, d, i, w)
                    force = c.frame.reshape(3,3).T@w[:3]
                    for sign,g in zip((-1,1),c.geom):
                        b = int(m.geom_bodyid[g])
                        if b in hand_names: hand_forces[hand_names[b]] += sign*force
                contact = lever_contacts(m,d,'leaf_handle_lever_col_n')
                target, info = teacher.command(float(d.time), root, dict(zip(names,d.qpos[qa])),
                    np.r_[d.xpos[leaf],d.xquat[leaf]],np.r_[d.xpos[handle],d.xquat[handle]],float(d.qpos[hj]),float(d.qpos[dj]),
                    dict(zip(names,d.qvel[va])),hand_forces,contact)
                if info.get('phase') not in ('release','withdraw'): target[fingers] = reset_lengths[fingers]
                target = np.clip(target, controls[:,0], controls[:,1])
            length = matrix@d.qpos[qa]; speed = matrix@d.qvel[va]
            force = kp*target+bias[:,0]+bias[:,1]*length+bias[:,2]*speed+teacher.feedforward+kp*(9*arm)*(target-length)-damping*speed
            d.ctrl[aids] = np.clip(force, limits[:,0], limits[:,1])
            if a.native_stance:
                stance_command, stance_status = native_stance.command()
                if stance_command is not None:
                    d.ctrl[native_stance.act] = stance_command
                info['native_stance'] = stance_status
            sim.plant.step()
            if step%10 == 0:
                contact = lever_contacts(m,d,'leaf_handle_lever_col_n')
                diag = sim.diagnostics()
                rows.append(dict(**diag, handle_rad=float(d.qpos[hj]), bolt_m=float(d.qpos[bj]), leaf_rad=float(d.qpos[dj]), teacher=info,
                    opposition=contact, hand_load_N=float(sum(np.linalg.norm(f) for f in hand_forces.values()))))
                for k in states: states[k].append(getattr(d,k).copy())
                if step%500 == 0: emit({k:rows[-1][k] for k in ('sim_time_s','leaf_rad','handle_rad','torso_tilt_deg','teacher')})
                if not diag['finite'] or diag['torso_tilt_deg'] > 12 or diag['root_height_m'] < .7:
                    failed = 'fell or nonfinite'; break
        report = dict(scope=__doc__, runtime_pose_writes=0, helper_forces=0, failed=failed, final_leaf_rad=rows[-1]['leaf_rad'],
            max_leaf_rad=max(r['leaf_rad'] for r in rows), max_handle_rad=max(r['handle_rad'] for r in rows),
            max_torso_tilt_deg=max(r['torso_tilt_deg'] for r in rows), duration_s=float(d.time))
        report['passed'] = not failed and report['final_leaf_rad'] >= .7 and report['max_handle_rad'] > .8
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        (a.output/'trace.json').write_text(json.dumps(rows)+'\n'); np.savez_compressed(a.output/'trajectory.npz', **states)
        pipeline.update(stage='Native teacher complete', result_passed=report['passed']); (a.output/'pipeline.json').write_text(json.dumps(pipeline))
        emit(report); emit('NATIVE_TEACHER_COMPLETE')
    finally:
        sim.close()


if __name__ == '__main__': main()
