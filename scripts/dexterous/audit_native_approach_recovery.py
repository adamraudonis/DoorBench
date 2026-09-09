#!/usr/bin/env python3
"""Audit recovery labels, actual state/action joins and fixed-eye images.

This admits bounded privileged correction examples, not successful door tasks.
The existing sensor join audit samples gyro/touch; accelerometer reconstruction
is not included. Teacher queries and pixel reconstruction never step physics.
"""
import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from doorbench.dexterous.approach_teacher import ApproachBodyTeacher
from doorbench.dexterous.native_sensor_demonstrations import digest
from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
from doorbench.dexterous.sensor_contract import ACTOR_KEYS, SENSOR_KEYS
from scripts.dexterous.evaluate_native_sensor_policy import prepare_trial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    run=a.run;cfg=json.loads((run/'manifest.json').read_text())['configuration']
    source=Path(cfg['teacher_run']);source_cfg=json.loads((source/'manifest.json').read_text())['configuration']
    report=json.loads((run/'report.json').read_text());join=json.loads((run/'own-sensors/independent-join-audit.json').read_text())
    if report['control_source']!='privileged_recovery' or cfg['control_source']!='privileged_recovery':
        raise ValueError('Explicit recovery trial required')
    sim,motors,layout,_=prepare_trial(source,Path(cfg['camera_profile']));m,d=sim.m,sim.d
    reset=json.loads((source/'body_reset.json').read_text());ref=json.loads((source/'reference.json').read_text())
    teacher=ApproachBodyTeacher(source_cfg['robot'],motors,reset,source_cfg['checkpoint'],
        height=ref['initial_root'][2],handoff_delay=3.,yaw_weight=2.)
    with np.load(run/'counterfactual-labels.npz',allow_pickle=False) as z:
        if set(z.files)!={'time_s','teacher_motor_forces','actor_applied'}:raise ValueError('Invalid label fields')
        labels={k:z[k].copy() for k in z.files}
    sensor=run/'own-sensors';capture=json.loads((sensor/'report.json').read_text())
    with np.load(sensor/'actor-initial-decision.npz',allow_pickle=False) as z:initial={k:z[k].copy() for k in z.files}
    with np.load(sensor/'actor-rgb.npz',allow_pickle=False) as z:rgb={k:z[k].copy() for k in z.files}
    checks=dict(sensor_joins=join['passed'] is True and all(join['checks'].values()),
        bound_sensor_join=join['sensor_report_sha256']==digest(sensor/'report.json') and join['raw_manifest_sha256']==digest(run/'raw-transitions/manifest.json'),
        completed_recovery=report['stop_reason']=='duration' and report['time_s']>=cfg['seconds']-1e-8,
        explicit_privilege=capture['control_source']=='privileged_recovery' and not report['unassisted_policy_success'],
        original_force_delivery=report['maximum_motor_delivery_error_Nm']==0.,
        no_state_assistance=report['runtime_pose_writes']==0 and report['native_teacher_mirror_steps']==0,
        initial_fields=set(initial)==set(ACTOR_KEYS)|{'time_s'},
        initial_clock=float(initial['time_s'])==0 and not initial['previous_action'].any(),
        initial_validity=np.array_equal(initial['sensor_valid'],[k.startswith(('joint_','rgb_')) for k in SENSOR_KEYS]),
        initial_encoders=np.array_equal(initial['joint_position'],np.clip(d.qpos[sim.qadr],-20,20).astype(np.float32)) and np.array_equal(initial['joint_velocity'],np.clip(d.qvel[sim.vadr],-100,100).astype(np.float32)),
        exact_initial_images=True, exact_recorded_images=True, uninterrupted_states=True,
        exact_counterfactual_labels=True, exact_applied_teacher_forces=True, exact_switch_clock=True,
        original_motor_caps=True, every_interval_upright=True, finite_states=True,
        every_interval_no_warnings=True)
    mujoco.mj_camlight(m,d);pixels=sim.observe(images=True)
    for k in ('rgb_left','rgb_right'):checks['exact_initial_images'] &= np.array_equal(initial[k],pixels[k])
    feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')];floor=m.geom('floor').id
    qa=np.array([m.jnt_qposadr[m.joint('robot/'+n).id] for n in teacher.names]);va=np.array([m.jnt_dofadr[m.joint('robot/'+n).id] for n in teacher.names])
    loads=np.zeros(2);before_q=d.qpos.copy();before_v=d.qvel.copy();label_error=0.;frame_count=0;count=0
    caps=np.array([v['force_range'] for v in motors['actuators']])
    for i,raw in enumerate(NativeTransitionArchive.read(run/'raw-transitions')):
        q=np.asarray(raw['qpos_before']);v=np.asarray(raw['qvel_before']);t=raw['interval_start_s']
        checks['uninterrupted_states'] &= np.array_equal(q,before_q) and np.array_equal(v,before_v)
        rq,rv=sim.root_qadr,sim.root_vadr
        rotation=Rotation.from_quat(q[rq+np.array([4,5,6,3])]);root=np.r_[q[rq:rq+7],v[rv:rv+3],rotation.apply(v[rv+3:rv+6])]
        force,_=teacher.force(t,root,dict(zip(teacher.names,q[qa])),dict(zip(teacher.names,v[va])),loads)
        label_error=max(label_error,float(np.max(abs(force-labels['teacher_motor_forces'][i]))))
        checks['exact_counterfactual_labels'] &= np.array_equal(force,labels['teacher_motor_forces'][i]) and abs(labels['time_s'][i]-t)<1e-8
        from_actor=i<round(cfg['actor_seconds']/.002)
        checks['exact_switch_clock'] &= bool(labels['actor_applied'][i])==from_actor
        actual=np.asarray(raw['actuator_force'])[sim.actuators]
        if not from_actor:checks['exact_applied_teacher_forces'] &= np.array_equal(actual,force)
        checks['original_motor_caps'] &= bool(np.all((actual>=caps[:,0])&(actual<=caps[:,1])))
        before_q=np.asarray(raw['qpos_after']);before_v=np.asarray(raw['qvel_after'])
        checks['finite_states'] &= bool(np.isfinite(np.r_[q,v,before_q,before_v]).all())
        end_rotation=Rotation.from_quat(before_q[rq+np.array([4,5,6,3])]).as_matrix()
        tilt=np.rad2deg(np.arccos(np.clip(end_rotation[2,2],-1,1)))
        checks['every_interval_upright'] &= bool(tilt<=35 and before_q[rq+2]>=.55)
        warning=raw.get('mujoco_warning_interval')
        checks['every_interval_no_warnings'] &= bool(warning and not np.any(warning['before']) and not np.any(warning['after']))
        loads[:]=0.
        for c in raw['contacts']:
            if floor in c['geom']:
                other=c['geom'][1] if c['geom'][0]==floor else c['geom'][0];body=int(m.geom_bodyid[other])
                if body in feet:loads[feet.index(body)]+=max(0.,c['wrench_contact_frame'][0])
        if frame_count<len(rgb['time_s']) and abs(rgb['time_s'][frame_count]-raw['interval_end_s'])<1e-8:
            d.qpos[:]=before_q;d.qvel[:]=before_v;mujoco.mj_forward(m,d);mujoco.mj_camlight(m,d)
            pixels=sim.observe(images=True)
            for k in ('rgb_left','rgb_right'):checks['exact_recorded_images'] &= np.array_equal(pixels[k],rgb[k][frame_count])
            frame_count+=1
        count+=1
    checks['complete_label_and_camera_coverage']=count==len(labels['time_s'])==capture['samples'] and frame_count==len(rgb['time_s'])
    for relative in ('motors-input.json','body_reset.json','reference.json'):
        target=run/relative
        if target.exists():
            if digest(target)!=digest(source/relative):raise ValueError('Changed teacher snapshot')
        else:shutil.copyfile(source/relative,target)
    files=['manifest.json','report.json','counterfactual-labels.npz','motors-input.json','body_reset.json','reference.json',
        'raw-transitions/manifest.json','own-sensors/report.json','own-sensors/layout.json','own-sensors/independent-join-audit.json',
        'own-sensors/actor-initial-decision.npz','own-sensors/actor-rgb.npz']
    files += ['own-sensors/'+v['file'] for v in capture['chunks']]
    files += ['raw-transitions/'+v['file'] for v in json.loads((run/'raw-transitions/manifest.json').read_text())['chunks']]
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(schema='doorbench.native-approach-correction-audit.v1',passed=all(checks.values()),checks=checks,
        samples=count,rendered_frames=frame_count+1,physics_steps=0,maximum_label_error_Nm=label_error,
        files={name:digest(run/name) for name in files},source_sha256=digest(__file__),scope=__doc__)
    (run/'independent-correction-audit.json').write_text(json.dumps(result,indent=2)+'\n');sim.close()
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
