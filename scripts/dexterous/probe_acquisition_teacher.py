#!/usr/bin/env python3
"""Test the shared Isaac acquisition controller on the native physical plant."""
import argparse
import gzip
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np

from doorbench.dexterous.acquisition_teacher import AcquisitionTeacher
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.grasp_verification import native_grasp_sample,audited_native_step,audit_grasp_steps
from doorbench.dexterous.provenance import capture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','reference','motors','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seconds',type=float,default=10.6)
    p.add_argument('--record-transitions',action='store_true',help='Actual force/pose-epoch archive and current-pose controller feedback')
    p.add_argument('--stance-profile',choices=['landed-foot-v1'],help='Explicit actual-foot-frame stance experiment; prior default unchanged')
    p.add_argument('--middle-finger-force',type=float)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Use a new output directory')
    if not json.loads((a.reference.parent/'geometry-audit.json').read_text())['passed']:raise ValueError('Unscreened acquisition candidate')
    ref=json.loads(a.reference.read_text());motors=json.loads(a.motors.read_text())
    capture(Path(__file__).resolve().parents[2],a.output,{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    (a.output/'reference.json').write_bytes(a.reference.read_bytes())
    (a.output/'run.pid').write_text(str(os.getpid()))
    pipeline=dict(stage='Shared acquisition controller native execution',started_at_unix=time.time(),completion_marker='ACQUISITION_TEACHER_COMPLETE')
    (a.output/'pipeline.json').write_text(json.dumps(pipeline))
    sim=DexterousDoorEnv(a.door,a.robot,json.loads(a.robot.with_suffix('.audit.json').read_text()));m,d=sim.m,sim.d
    teacher=AcquisitionTeacher(a.robot,motors,ref,middle_finger_force=a.middle_finger_force,stance_profile=a.stance_profile)
    sim.reset(randomize=False,images=False)
    d.qpos[sim.root_qadr:sim.root_qadr+7]=teacher.initial_root
    ids=np.array([m.joint('robot/'+n).id for n in teacher.names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids]
    d.qpos[qa]=teacher.path[0];d.qvel[:]=0;mujoco.mj_forward(m,d)
    aids=np.array([m.actuator('robot/'+v['name']).id for v in motors['actuators']])
    m.actuator_gainprm[aids,0]=1.;m.actuator_biasprm[aids,:3]=0.;m.actuator_ctrlrange[aids]=teacher.caps;d.ctrl[aids]=0.
    physics=[native_grasp_sample(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge')]
    recorder=archive=None;controller_steps=[]
    if a.record_transitions:
        from doorbench.dexterous.native_transition_audit import NativeTransitionRecorder
        from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
        recorder=NativeTransitionRecorder(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge');archive=NativeTransitionArchive(a.output/'raw-transitions')
    traces=[];states={key:[] for key in ('qpos','qvel','ctrl')}
    hand_names={b:m.body(b).name.removeprefix('robot/') for b in range(m.nbody) if m.body(b).name.startswith(('robot/rh_','robot/lh_'))}
    hb=m.body('leaf_handle').id;palm=m.site('robot/rh_palm_touch').id
    def emit(row):
        line=json.dumps(row);print(line,flush=True)
        with (a.output/'run.log').open('a') as f:f.write(line+'\n')
    try:
        for step in range(round(a.seconds/m.opt.timestep)):
            loads={name:np.zeros(3) for name in hand_names.values()}
            for k,c in enumerate(d.contact[:d.ncon]):
                wrench=np.zeros(6);mujoco.mj_contactForce(m,d,k,wrench);force=c.frame.reshape(3,3).T@wrench[:3]
                for sign,g in zip((-1,1),c.geom):
                    b=int(m.geom_bodyid[g])
                    if b in hand_names:loads[hand_names[b]]+=sign*force
            R=d.xmat[sim.pelvis].reshape(3,3)
            root=np.r_[d.qpos[sim.root_qadr:sim.root_qadr+7],d.qvel[sim.root_vadr:sim.root_vadr+3],R@d.qvel[sim.root_vadr+3:sim.root_vadr+6]]
            force,info=teacher.force(float(d.time),root,dict(zip(teacher.names,d.qpos[qa])),dict(zip(teacher.names,d.qvel[va])),np.r_[d.xpos[hb],d.xquat[hb]],loads)
            d.ctrl[aids]=force
            if recorder:
                recorder.before_step();sim.plant.step();sample,raw=recorder.after_step();archive.write(raw);physics.append(sample)
                controller_steps.append(dict(time_s=float(d.time)-m.opt.timestep,**info))
            else:physics.append(audited_native_step(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge'))
            if step%10==0:
                row=dict(**sim.diagnostics(),teacher=info,palm_error_m=float(np.linalg.norm(d.site_xpos[palm]-teacher.positions[-1])))
                traces.append(row)
                for key in states:states[key].append(getattr(d,key).copy())
                (a.output/'latest.json').write_text(json.dumps(row))
                if step%500==0:emit(row)
            if not physics[-1]['finite'] or physics[-1]['torso_tilt_deg']>35:break
        report=audit_grasp_steps(physics,physics_dt=m.opt.timestep,expected_duration=a.seconds)
        if archive:
            archive.close(complete=True)
            with gzip.open(a.output/'controller-steps.json.gz','wt') as stream:json.dump(controller_steps,stream)
            report['checks']['stance_solves_every_interval']=all(row['stance_status'] in ('solved','solved inaccurate') for row in controller_steps)
            report['checks']['no_warning_intervals']=all(row.get('mujoco_warning_interval',{}).get('passed',False) for row in physics[1:])
            report['passed']=all(report['checks'].values())
        report.update(scope='Shared privileged FK/motor controller; native acquisition only, no opening or traversal',runtime_robot_pose_writes=0,direct_door_commands=False,
            max_torso_tilt_deg=max(r['torso_tilt_deg'] for r in physics),final_palm_error_m=traces[-1]['palm_error_m'],final_contacts=physics[-1]['pad_grasp'])
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        (a.output/'trace.json').write_text(json.dumps(traces)+'\n')
        with gzip.open(a.output/'physics-steps.json.gz','wt') as f:json.dump(physics,f)
        np.savez_compressed(a.output/'trajectory.npz',**states)
        pipeline.update(stage='Shared acquisition controller completed',result_passed=report['passed']);(a.output/'pipeline.json').write_text(json.dumps(pipeline))
        emit({k:v for k,v in report.items() if k!='final_contacts'});emit('ACQUISITION_TEACHER_COMPLETE')
    finally:sim.close()
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
