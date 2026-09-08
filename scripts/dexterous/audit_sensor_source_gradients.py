"""Decompose complete-source gradients at a frozen sensor checkpoint.

No optimizer exists here. Each source uses the identical continuous dual-view
and cold-prefix objective. Gradient vectors are scaled by1/N so their sum is
the exact full-source update direction before clipping, to rounding precision.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from doorbench.dexterous.continuous_history_training import accumulate_full_sources
from doorbench.dexterous.sensor_policy_controller import SensorPolicyController
from doorbench.dexterous.sensor_training_bundle import load_bundle,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);episodes,bundle=load_bundle(args.dataset_manifest)
    first=episodes[0];motors=json.loads((first.path/'motor-contract.json').read_text())
    actor=SensorPolicyController(args.checkpoint,motor_contract=motors,sensor_layout=first.layout,physics_dt_s=first.metadata['physics_dt_s'])
    model=actor._actor.requires_grad_(True).train();versions=[v._version for v in model.parameters()]
    vectors=[];rows=[];started=time.monotonic()
    for definition,episode in zip(bundle['datasets'],episodes,strict=True):
        model.zero_grad(set_to_none=True)
        result=accumulate_full_sources(model,[episode],chunk_length=32,cold_prefix_length=32,cold_weight=.5)
        g=torch.cat([(v.grad if v.grad is not None else torch.zeros_like(v)).reshape(-1) for v in model.parameters()]).double()/len(episodes)
        vectors.append(g)
        row=dict(source=definition['name'],examples=len(episode),source_weight=1/len(episodes),
            weighted_gradient_l2_norm=float(g.norm()),mixed_loss=result['mixed_loss'],recorded_loss=result['recorded_loss'],actor_loss=result['actor_loss'])
        rows.append(row);print(json.dumps(row),flush=True)
    assert [v._version for v in model.parameters()]==versions
    grads=torch.stack(vectors);norms=torch.linalg.vector_norm(grads,dim=1);summed=grads.sum(0)
    dot=grads@grads.T;denom=norms[:,None]*norms[None,:]
    cosine=[[float(dot[i,j]/denom[i,j]) if denom[i,j]>0 else None for j in range(len(episodes))] for i in range(len(episodes))]
    result=dict(scope=__doc__,checkpoint=str(args.checkpoint),checkpoint_sha256=actor.checkpoint_sha256,
        dataset_manifest_sha256=digest(args.dataset_manifest),source_sha256=digest(Path(__file__)),sources=rows,
        pairwise_gradient_cosine=cosine,total_gradient_l2_norm=float(summed.norm()),sum_individual_gradient_norms=float(norms.sum()),
        contributions_along_total_direction={r['source']:float(torch.dot(g,summed)/(summed.norm()+1e-30)) for r,g in zip(rows,vectors,strict=True)},
        elapsed_s=time.monotonic()-started,optimizer_updates=0,physical_state_steps=0,closed_loop_evaluated=False,
        limitation='Gradient interference at selected frozen weights is a fitting diagnostic, not a physical instability result. Parameter gradients detach every32 frames exactly as protocol010.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('scope','sources')}),flush=True)


if __name__=='__main__':main()
