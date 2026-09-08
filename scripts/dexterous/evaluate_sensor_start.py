"""Compare early commands on recorded teacher states, never physical rollouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from doorbench.dexterous.sensor_actor import native_motor_forces
from doorbench.dexterous.sensor_demonstrations import SensorDemonstration
from doorbench.dexterous.sensor_policy_controller import SensorPolicyController


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('episode','legacy-teacher-receipt','reset-observation-run','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--checkpoint',action='append',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(1)
    data=SensorDemonstration(args.episode,qualification='acquisition-report.json',legacy_teacher_receipt=args.legacy_teacher_receipt,reset_observation_run=args.reset_observation_run)
    motors=json.loads((args.episode/'motor-contract.json').read_text());names=[x['name'] for x in motors['actuators']]
    caps=np.array([x['force_range'] for x in motors['actuators']]);body=np.array([i for i,n in enumerate(names) if not n.startswith(('rh_','lh_'))])
    if len(data)<251 or data.times[0]!=0 or data.metadata['physics_dt_s']!=.002:
        raise ValueError('This diagnostic requires the complete first500ms from actual reset')
    rows=[]
    for checkpoint in args.checkpoint:
        for history in ('teacher','actor'):
            actor=SensorPolicyController(checkpoint,motor_contract=motors,sensor_layout=data.layout,physics_dt_s=.002)
            actor.reset_episode();errors=[];samples=[]
            for i in range(251):
                packet=data.packet(i)
                if history=='actor':packet['previous_action']=actor.previous_action
                force=actor.force(packet,float(data.times[i]));target=native_motor_forces(data.numeric['previous_action'][i+1],caps)
                errors.append(force-target)
                if i in (0,1,5,25,50,100,250):
                    selected=('left_knee','right_knee','left_hip_pitch','right_hip_pitch')
                    samples.append(dict(time_s=float(data.times[i]),commands={n:float(force[names.index(n)]) for n in selected},teacher={n:float(target[names.index(n)]) for n in selected}))
            error=np.array(errors)[:,body]
            row=dict(checkpoint=str(checkpoint),checkpoint_sha256=actor.checkpoint_sha256,previous_action_history=history,
                body_force_rmse_Nm_first100ms=float(np.sqrt((error[:51]**2).mean())),
                body_force_rmse_Nm_80to100ms=float(np.sqrt((error[40:51]**2).mean())),
                body_force_rmse_Nm_first500ms=float(np.sqrt((error**2).mean())),samples=samples)
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='samples'}),flush=True)
    result=dict(scope='Offline early-command diagnosis on qualified teacher sensor states with actual recorded cold-start observation; no simulator or student success',rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
