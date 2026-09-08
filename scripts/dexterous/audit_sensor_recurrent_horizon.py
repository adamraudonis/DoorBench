"""Compare frozen full-history predictions with the exact training warm-up.

This changes no weights, labels, sensor packets, or fitting criteria. The
nominal window locations are deterministic and include the early interval that
was absent from the original sampler. No physical simulator is instantiated.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from doorbench.dexterous.autoregressive_training import actor_history_chunk,actor_history_prediction
from doorbench.dexterous.sensor_policy_controller import SensorPolicyController
from doorbench.dexterous.sensor_training_bundle import load_bundle,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    episodes,manifest=load_bundle(args.dataset_manifest);episode=episodes[0]
    if manifest['datasets'][0]['kind']!='qualified_teacher':raise ValueError('First source must be nominal teacher')
    if len(episode)<6032:raise ValueError('Diagnostic protocol needs the complete frozen nominal prefix')
    starts=[0,32,64,128,250,500,1000,2000,3000,4000,5000,6000,len(episode)-32]
    motors=json.loads((episode.path/'motor-contract.json').read_text())
    scale=np.array([(a['force_range'][1]-a['force_range'][0])*.5 for a in motors['actuators']])
    body=[i for i,n in enumerate(episode.layout['action_order']) if not n.startswith(('rh_','lh_'))]
    def inputs(start,length):
        x,y=episode.sequence(start,length)
        return {k:torch.as_tensor(v[None]) for k,v in x.items()},y
    def stats(delta):return dict(normalized_mse_all=float(np.mean(delta**2)),body_force_rmse_Nm=float(np.sqrt(np.mean((delta[:,body]*scale[body])**2))))
    rows=[]
    with torch.inference_mode():
        for checkpoint in args.checkpoint:
            loader=SensorPolicyController(checkpoint,motor_contract=motors,sensor_layout=episode.layout,physics_dt_s=.002)
            model=loader._actor
            for history in ('recorded','actor'):
                hidden=None;previous=torch.zeros(1,61);chunks=[]
                for start in range(0,len(episode),64):
                    x,_=inputs(start,min(64,len(episode)-start))
                    if history=='recorded':pred,hidden=model(**x,hidden=hidden)
                    else:pred,hidden,previous,_=actor_history_chunk(model,x,previous=previous,hidden=hidden)
                    chunks.append(pred[0].numpy())
                full=np.concatenate(chunks);windows=[];full_errors=[];short_errors=[];disagreements=[]
                for label_start in starts:
                    burn=min(label_start,64);x,target=inputs(label_start-burn,burn+32)
                    if history=='actor':short=actor_history_prediction(model,x,burn,episode_start=[label_start-burn==0])[0].numpy()
                    else:
                        h=None
                        if burn:_,h=model(**{k:v[:,:burn] for k,v in x.items()})
                        short=model(**{k:v[:,burn:] for k,v in x.items()},hidden=h)[0][0].numpy()
                    target=target[burn:];long=full[label_start:label_start+32]
                    if label_start==0:assert np.allclose(long,short,rtol=1e-5,atol=1e-6),'Reset view must reproduce full history'
                    fe=long-target;se=short-target;diff=long-short
                    full_errors.append(fe);short_errors.append(se);disagreements.append(diff)
                    windows.append(dict(label_start=label_start,label_end=label_start+31,decision_time_s=float(episode.times[label_start]),burn_in=burn,full_history=stats(fe),training_history=stats(se),prediction_difference=stats(diff)))
                row=dict(checkpoint=str(checkpoint),checkpoint_sha256=loader.checkpoint_sha256,previous_action_history=history,
                    windows=windows,aggregate_full_history=stats(np.concatenate(full_errors)),aggregate_training_history=stats(np.concatenate(short_errors)),aggregate_prediction_difference=stats(np.concatenate(disagreements)))
                rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('windows','checkpoint')}),flush=True)
    report=dict(scope=__doc__,dataset_manifest_sha256=digest(args.dataset_manifest),source_sha256=digest(Path(__file__)),window_starts=starts,rows=rows,
        physical_state_steps=0,closed_loop_evaluated=False,teacher_commands_delivered=False,
        limitation='Fixed recorded observations do not react to predicted commands. The truncated actor-owned view has the declared actual preceding-command anchor, while full actor-owned history begins once at reset. Differences do not change any original fit or physical gate.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
