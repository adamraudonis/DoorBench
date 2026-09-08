"""Measure finite burn-in error on recorded student observations, without physics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from doorbench.dexterous.correction_demonstrations import CorrectionDemonstration
from doorbench.dexterous.offline_teacher_queries import sha
from doorbench.dexterous.sensor_policy_controller import SensorPolicyController


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('corrections','checkpoint','output'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(1)
    if args.output.exists():raise FileExistsError('Preserve earlier recurrent audits')
    e=CorrectionDemonstration(args.corrections);motors=json.loads((e.path/'motor-contract.json').read_text())
    actor=SensorPolicyController(args.checkpoint,motor_contract=motors,sensor_layout=e.layout,physics_dt_s=.002)
    task=json.loads((e.path/'report.json').read_text())
    if actor.checkpoint_sha256!=task.get('checkpoint_sha256'):
        raise ValueError('History audit requires the exact checkpoint that executed the recorded actor trial')
    if len(e)<96:raise ValueError('History audit requires at least96 consecutive eligible decisions')
    body=np.array([i for i,n in enumerate(actor.action_order) if not n.startswith(('rh_','lh_'))])
    legs=np.array([i for i,n in enumerate(actor.action_order) if any(part in n for part in ('hip','knee','ankle'))])
    torso=actor.action_order.index('torso')
    actor.reset_episode();full=np.array([actor.force(e.packet(i),float(e.times[i])) for i in range(len(e))])
    with np.load(e.path/'acquisition-physics.npz',allow_pickle=False) as z:actual=z['motor_forces'][:len(e)].copy()
    rows=[]
    for burn in (0,32,64):
        differences=[];starts=[]
        for label_start in range(64,len(e)-31,16):
            start=label_start-burn;actor.reset_episode();predicted=[]
            for i in range(start,label_start+32):
                force=actor.force(e.packet(i),float(e.times[i]))
                if i>=label_start:predicted.append(force)
            differences.extend(np.array(predicted)-full[label_start:label_start+32]);starts.append(label_start)
        differences=np.asarray(differences)
        rows.append(dict(burn_in_steps=burn,burn_in_s=burn*.002,windows=len(starts),supervised_steps_per_window=32,
            label_start_indices=starts,body_force_rmse_vs_full_history_Nm=float(np.sqrt(np.mean(differences[:,body]**2))),
            leg_force_rmse_vs_full_history_Nm=float(np.sqrt(np.mean(differences[:,legs]**2))),
            torso_force_rmse_vs_full_history_Nm=float(np.sqrt(np.mean(differences[:,torso]**2))),
            maximum_absolute_force_difference_Nm=float(abs(differences).max())))
    r=dict(scope=__doc__,source=e.metadata,checkpoint_sha256=sha(args.checkpoint),script_sha256=sha(__file__),
        actual_run_checkpoint_sha256=task['checkpoint_sha256'],
        examples=len(e),full_history_vs_actual_recorded_body_rmse_Nm=float(np.sqrt(np.mean((full[:,body]-actual[:,body])**2))),
        full_history_vs_actual_maximum_force_difference_Nm=float(abs(full-actual).max()),burn_in_comparison=rows,
        physics_steps=0,observations='Same actual recorded student sensors and previous commands; only the recurrent initialization changes',
        limitation='This measures history truncation, not physical recovery or task success')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('examples','full_history_vs_actual_recorded_body_rmse_Nm','full_history_vs_actual_maximum_force_difference_Nm','burn_in_comparison')}))


if __name__=='__main__':main()
