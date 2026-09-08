"""Compare command-history distributions on fixed, causally recorded sensors.

Nominal diagnosis covers the first500ms; correction diagnosis covers each full
eligible prefix. No simulator advances, and predicted forces never become
labels or alter recorded camera/tactile/proprioceptive measurements.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from doorbench.dexterous.sensor_actor import native_motor_forces
from doorbench.dexterous.sensor_policy_controller import SensorPolicyController
from doorbench.dexterous.sensor_training_bundle import load_bundle,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    episodes,manifest=load_bundle(args.dataset_manifest);rows=[]
    for checkpoint in args.checkpoint:
        for definition,episode in zip(manifest['datasets'],episodes,strict=True):
            count=min(251,len(episode)) if definition['kind']=='qualified_teacher' else len(episode)
            motors=json.loads((episode.path/'motor-contract.json').read_text())
            caps=np.array([a['force_range'] for a in motors['actuators']]);names=episode.layout['action_order']
            groups=dict(all=list(range(61)),body=[i for i,n in enumerate(names) if not n.startswith(('rh_','lh_'))],
                legs=[i for i,n in enumerate(names) if any(x in n for x in ('hip','knee','ankle'))],
                torso=[names.index('torso')],right_hand=[i for i,n in enumerate(names) if n.startswith('rh_')])
            targets=episode.targets[:count] if hasattr(episode,'targets') else episode.numeric['previous_action'][1:count+1]
            for history in ('recorded','actor'):
                actor=SensorPolicyController(checkpoint,motor_contract=motors,sensor_layout=episode.layout,physics_dt_s=.002)
                actor.reset_episode();force_errors=[];normalized_errors=[]
                for i in range(count):
                    packet=episode.packet(i)
                    if history=='actor':packet['previous_action']=actor.previous_action
                    force=actor.force(packet,float(episode.times[i]))
                    force_errors.append(force-native_motor_forces(targets[i],caps))
                    normalized_errors.append(actor.previous_action-targets[i])
                errors=np.array(force_errors);normalized=np.array(normalized_errors)
                rows.append(dict(checkpoint=str(checkpoint),checkpoint_sha256=actor.checkpoint_sha256,
                    dataset=definition['name'],previous_action_history=history,examples=count,
                    first_decision_time_s=float(episode.times[0]),last_decision_time_s=float(episode.times[count-1]),
                    normalized_force_mse={name:float(np.mean(normalized[:,indices]**2)) for name,indices in groups.items()},
                    physical_motor_force_rmse_Nm={name:float(np.sqrt(np.mean(errors[:,indices]**2))) for name,indices in groups.items()}))
    report=dict(scope=__doc__,dataset_manifest_sha256=digest(args.dataset_manifest),rows=rows,
        closed_loop_evaluated=False,physical_state_steps=0,teacher_commands_delivered=False,
        limitation='Actor-owned input history on recorded states is offline autoregression; those states do not react to the new commands. These scores do not replace the six frozen fitting criteria or physical evaluation.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    for row in rows:print(json.dumps({k:v for k,v in row.items() if k not in ('checkpoint','checkpoint_sha256')}),flush=True)


if __name__=='__main__':main()
