#!/usr/bin/env python3
"""Replay a recorded native continuation through the numeric portable teacher.

No physics steps are performed. Actual pre-decision state, preceding interval
contacts and frozen source inputs are replayed in their original order. This
tests force equivalence and history, not another successful physical rollout.
"""
import argparse,gzip,hashlib,inspect,json,shutil,time
from pathlib import Path

import mujoco
import numpy as np

from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.native_transition_audit import _contact_solution
from doorbench.dexterous.post_opening_teacher import PostOpeningTeacher


from doorbench.dexterous.native_post_opening_measurements import measured_contacts,measured_state,outward_release_normal


def query_state(s,teacher,q,v):
    s.d.qpos[:]=q;s.d.qvel[:]=v;mujoco.mj_kinematics(s.m,s.d)
    return measured_state(s,teacher.names,teacher.door_names,teacher.pose_names)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=65.);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh replay output required')
    a.output.mkdir(parents=True);source=json.loads((a.trial/'report.json').read_text());args=source['arguments'];files=json.loads((a.trial/'inputs.json').read_text())['files']
    source_physics=source['passed'] is True and source['checks'].get('actual_step_contact_epochs') is True and source['checks'].get('actual_motor_delivery') is True
    frozen=[]
    for path in (Path(__file__),Path(inspect.getfile(PostOpeningTeacher)),Path(inspect.getfile(measured_contacts))):
        target=a.output/path.name;shutil.copy2(path,target);frozen.append(target)
    for key in ('robot','motors','initial_trajectory','body_reset','checkpoint','door.xml'):
        if hashlib.sha256(Path(files[key]['path']).read_bytes()).hexdigest()!=files[key]['sha256']:raise ValueError('Changed recorded input: '+key)
    robot=Path(args['robot']);motors=json.loads(Path(args['motors']).read_text());reset=json.loads(Path(args['body_reset']).read_text())
    s=DexterousDoorEnv(Path(args['door']),robot,json.loads(robot.with_suffix('.audit.json').read_text()));s.reset(images=False,randomize=False);m,d=s.m,s.d
    initial=np.load(args['initial_trajectory']);d.qpos[:]=initial['terminal_qpos'];d.qvel[:]=initial['terminal_qvel'];d.ctrl[:]=initial['terminal_ctrl'];d.time=0.
    ids=np.array([m.actuator('robot/'+x['name']).id for x in motors['actuators']]);caps=np.array([x['force_range'] for x in motors['actuators']])
    m.actuator_gainprm[ids,0]=1.;m.actuator_biasprm[ids,:3]=0.;m.actuator_ctrlrange[ids]=caps;mujoco.mj_forward(m,d)
    _,previous=_contact_solution(s);previous.update(interval_start_s=0.,interval_end_s=0.)
    normal=outward_release_normal(m,previous)
    teacher=PostOpeningTeacher(robot,motors,reset,args['checkpoint'],door_xml=Path(args['door'])/'door.xml',passage=args['passage'],inward_roll=args['inward_roll'])
    started=time.time();maximum=0.;worst=None;count=0;phases=[];rows=[];error=None;old_steps={n:getattr(mujoco,n) for n in ('mj_step','mj_step1','mj_step2')}
    def forbidden(*_args,**_kwargs):raise AssertionError('Replay must never perform a physics step')
    for name in old_steps:setattr(mujoco,name,forbidden)
    try:
        with gzip.open(a.trial/'actual-transitions.jsonl.gz','rt') as f:
            for line in f:
                raw=json.loads(line);t=float(raw['interval_start_s'])
                if t>=a.seconds-1e-8:break
                root,joints,velocities,kwargs=query_state(s,teacher,np.array(raw['qpos_before']),np.array(raw['qvel_before']))
                feet,loads,evidence=measured_contacts(m,previous,physics_qualified=source_physics)
                if count==0:kwargs.update(previous_motor_forces=initial['terminal_ctrl'][ids],release_normal_world=normal)
                force,info=teacher.force(t,root,joints,velocities,feet,loads,**kwargs,evidence=evidence,pose_time_s=t,contact_interval_s=[previous['interval_start_s'],previous['interval_end_s']])
                expected=np.array(raw['controls'])[ids];delta=float(np.max(abs(force-expected)))
                if delta>maximum:maximum=delta;worst=dict(time_s=t,motor=teacher.motor_names[int(np.argmax(abs(force-expected)))],expected=expected.tolist(),actual=force.tolist())
                phase=info['passage_stage'] if info['passage_stage'] not in (None,'post-opening component') else info['phase']
                if not phases or phases[-1]['phase']!=phase:phases.append(dict(time_s=t,phase=phase))
                if count%500==0:
                    row=dict(time_s=t,maximum_force_difference_Nm=maximum,current_difference_Nm=delta,phase=phase);rows.append(row);print(json.dumps(row),flush=True)
                count+=1;previous=raw
    except Exception as exc:
        error=repr(exc)
    finally:
        for name,value in old_steps.items():setattr(mujoco,name,value)
        s.close()
    checks=dict(complete=count==round(a.seconds/.002),force_equivalence=maximum<1e-5,no_runtime_steps=True,completed_without_error=error is None)
    report=dict(passed=all(checks.values()),checks=checks,calls=count,seconds=a.seconds,maximum_motor_force_difference_Nm=maximum,worst=worst,error=error,phases=phases,wall_seconds=time.time()-started,scope='Replay of recorded measurements only; no new physical rollout',initialization=teacher.initialization,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.trial/'actual-transitions.jsonl.gz',*frozen]})
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'trace.json').write_text(json.dumps(rows)+'\n')
    if teacher.plan is not None:(a.output/'geometry-screen.json').write_text(json.dumps(teacher.plan)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('initialization','worst','sources')}),flush=True)
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
