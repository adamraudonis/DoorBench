"""Accumulate dual-history gradients with fixed weights over complete sources.

These are recorded sensor observations. Predicted commands affect only the
actor-owned previous-command input, never the physical observations or labels.
The caller alone performs the optimizer update, after successful accumulation.
"""
import math
import time
import numpy as np
import torch

from .autoregressive_training import actor_history_chunk,previous_action_slice


class AccumulationInterrupted(RuntimeError):
    """A partial source pass must never become an optimizer update."""
    def __init__(self,logical_samples,completed_sources):
        super().__init__('Wall budget ended before the complete accumulated update')
        self.logical_samples=logical_samples
        self.completed_sources=completed_sources


def accumulate_full_sources(model,episodes,*,chunk_length=32,cold_prefix_length=32,
                            cold_weight=.5,deadline=None,progress=None):
    if type(chunk_length) is not int or chunk_length<1 or type(cold_prefix_length) is not int or cold_prefix_length<1:
        raise ValueError('Positive integer chunk and cold-prefix lengths are required')
    if isinstance(cold_weight,bool) or not math.isfinite(cold_weight) or not 0<=cold_weight<=1:
        raise ValueError('Cold-prefix loss weight must be in [0,1]')
    if not episodes:raise ValueError('At least one complete source is required')
    for e in episodes:
        if len(e)<1 or e.times[0]!=0 or np.any(e.numeric['previous_action'][0]!=0):
            raise ValueError('Each source requires its actual zero-command episode reset')
    device=next(model.parameters()).device;versions=[p._version for p in model.parameters()]
    totals=np.zeros(3);rows=[];processed=0
    for source,e in enumerate(episodes):
        recorded_hidden=None;owned_hidden=None
        previous=torch.zeros((1,model.dimensions.actions),device=device)
        source_totals=np.zeros(3);prefix=min(cold_prefix_length,len(e))
        for start in range(0,len(e),chunk_length):
            if deadline is not None and time.monotonic()>=deadline:
                raise AccumulationInterrupted(processed,rows)
            if [p._version for p in model.parameters()]!=versions:
                raise RuntimeError('Weights changed inside an accumulated source pass')
            length=min(chunk_length,len(e)-start);values,target=e.sequence(start,length)
            x={k:torch.as_tensor(v[None],device=device) for k,v in values.items()}
            y=torch.as_tensor(target[None],device=device)
            if start==0 and torch.any(x['proprio'][:,0,previous_action_slice(model.dimensions)]!=0):
                raise ValueError('Prepared source reset contains a nonzero previous command')
            recorded,recorded_hidden=model(**x,hidden=recorded_hidden)
            owned,owned_hidden,previous,_=actor_history_chunk(model,x,previous=previous,hidden=owned_hidden)
            if y.shape!=recorded.shape or owned.shape!=recorded.shape:
                raise ValueError('Both history views must supervise the same actual labels')
            weights=torch.full((length,), (1-cold_weight)/len(e),device=device,dtype=y.dtype)
            overlap=max(0,min(length,prefix-start))
            weights[:overlap]+=cold_weight/prefix
            recorded_loss=torch.sum(torch.mean((recorded-y)**2,dim=-1)[0]*weights)
            owned_loss=torch.sum(torch.mean((owned-y)**2,dim=-1)[0]*weights)
            objective=.5*(recorded_loss+owned_loss)/len(episodes)
            if not torch.isfinite(objective):raise ValueError('Nonfinite continuous-history objective')
            objective.backward()
            source_totals+=np.array([float(objective.detach())*len(episodes),float(recorded_loss.detach()),float(owned_loss.detach())])
            # Preserve the exact numeric state produced by the current fixed
            # weights, while limiting gradients to the declared chunk length.
            recorded_hidden=recorded_hidden.detach();owned_hidden=owned_hidden.detach();previous=previous.detach()
            processed+=length
            if progress:progress(dict(source=source,next_sample=start+length,examples=len(e),logical_samples=processed))
        row=dict(source=source,examples=len(e),cold_prefix_examples=prefix,
                 mixed_loss=float(source_totals[0]),recorded_loss=float(source_totals[1]),actor_loss=float(source_totals[2]))
        rows.append(row);totals+=source_totals/len(episodes)
    if [p._version for p in model.parameters()]!=versions:raise RuntimeError('Weights changed before accumulation completed')
    return dict(mixed_loss=float(totals[0]),recorded_loss=float(totals[1]),actor_loss=float(totals[2]),
                sources=rows,logical_samples=processed,source_pass_complete=True)
