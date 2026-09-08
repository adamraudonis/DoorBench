#!/usr/bin/env python3
"""Verify consumed panel forces and summarize actual 100 Hz target demand."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve()
    rows=[json.loads(line) for line in gzip.open(run/'panel-targets.jsonl.gz','rt')]
    report=json.loads((run/'report.json').read_text())
    offset=report['opening_clock_offset_s']
    times=np.array([r['time_s'] for r in rows])
    clock=np.array([r['motor_target_update_time_s'] for r in rows])
    select=np.r_[True,np.diff(clock)>1e-7]
    updates=[r for r,s in zip(rows,select) if s]
    update_time=times[select]
    target=np.array([[r['consumed_motor_targets'][r['chain_motor_indices'][0]],*r['left_arm_targets']] for r in updates])
    dt=np.diff(update_time)
    velocity=np.diff(target,axis=0)/dt[:,None]
    acceleration=np.diff(velocity,axis=0)/np.diff(update_time[1:])[:,None]
    singular=np.array([r['weighted_task_jacobian_singular_values'] for r in updates])
    condition=singular[:,0]/singular[:,-1]
    manifest=json.loads((run/'raw-transitions/manifest.json').read_text())
    assert manifest['complete']
    matched=0;force_error=0.;geometry_clock_error=0.
    for chunk in manifest['chunks']:
        if chunk['interval_end_s'] < times[0]+offset-1e-8:continue
        file=run/'raw-transitions'/chunk['file']
        if hashlib.file_digest(file.open('rb'),'sha256').hexdigest()!=chunk['sha256']:
            raise ValueError('Raw archive changed')
        with np.load(file,allow_pickle=False) as raw:
            for j,t in enumerate(raw['interval_start_s']):
                local=t-offset;k=int(np.argmin(abs(times-local)))
                if abs(times[k]-local)>1e-7:continue
                row=rows[k];indices=row['chain_motor_indices']
                # The stance intentionally replaces ten leg forces only.
                consumed=raw['actuator_force'][j,indices]
                assembled=np.asarray(row['assembled_motor_forces'])[indices]
                force_error=max(force_error,float(np.max(abs(consumed-assembled))))
                geometry_clock_error=max(geometry_clock_error,abs(row['pose_time_s']-local))
                matched+=1
    if matched!=len(rows):raise ValueError('Missing physical intervals for recorded controller output')
    samples=[]
    for t in np.arange(update_time[0],update_time[-1]+.001,.1):
        i=int(np.argmin(abs(update_time-t)));row=updates[i]
        samples.append(dict(episode_time_s=float(update_time[i]+offset),
            consumed_joint_targets=target[i].tolist(),target_velocity_rad_s=velocity[max(0,i-1)].tolist(),
            weighted_jacobian_condition=float(condition[i]),left_info=row['left_info'],
            actual_joint_velocity_rad_s=row['measured_chain_joint_velocities']))
    result=dict(scope='Privileged actual command/target diagnostic. No force filtering, new gates, or changed pass/fail classification.',
        run=str(run),original_passed=report['passed'],source_trace_sha256=hashlib.file_digest((run/'panel-targets.jsonl.gz').open('rb'),'sha256').hexdigest(),
        physical_intervals_verified=matched,maximum_actual_force_assembly_error=force_error,
        maximum_pose_clock_error_s=float(geometry_clock_error),
        joint_names=rows[0]['nominal_joint_names'][:8],
        maximum_target_step_rad=np.max(abs(np.diff(target,axis=0)),axis=0).tolist(),
        maximum_target_velocity_rad_s=np.max(abs(velocity),axis=0).tolist(),
        maximum_target_acceleration_rad_s2=np.max(abs(acceleration),axis=0).tolist(),
        maximum_weighted_task_jacobian_condition=float(max(condition)),samples=samples)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise ValueError('Use a new diagnostic output')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))


if __name__=='__main__':main()
