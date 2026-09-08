"""Offline actor-command feedback on fixed recorded sensor observations.

This does not advance a simulator or create a physical recovery. Only the
previous-command field changes. Teacher force targets remain labels outside
this function and cannot be substituted into the scored command history.
"""
import torch


def previous_action_slice(dimensions):
    # prepare_actor_packet: q,dq,gyro,accelerometer,previous action,ages,valid.
    start=2*dimensions.joints+6
    return slice(start,start+dimensions.actions)


def _validate_inputs(model,inputs):
    if set(inputs)!={'proprio','tactile','images'}:
        raise ValueError('Only prepared actor sensor channels are permitted')
    d=model.dimensions;batch,total=inputs['proprio'].shape[:2]
    if (total<1 or inputs['proprio'].shape!=(batch,total,d.proprio_dimension)
            or inputs['tactile'].shape!=(batch,total,d.tactile)
            or inputs['images'].shape!=(batch,total,2,3,d.image_size,d.image_size)):
        raise ValueError('Invalid actual-history window dimensions')
    return d,batch,total


def actor_history_chunk(model,inputs,*,previous,hidden=None,return_history=False):
    """Continue an existing actor history over prepared observations, without reset.

    The caller explicitly owns the initial command and GRU state. Every next
    previous-command field is this model's own detached output; recorded
    command fields are ignored throughout the chunk. Useful for full-history
    evaluation without holding an entire image episode in memory.
    """
    d,batch,total=_validate_inputs(model,inputs);slot=previous_action_slice(d)
    if previous.shape!=(batch,d.actions) or not torch.isfinite(previous).all():
        raise ValueError('Explicit finite preceding actor command is required')
    previous=previous.detach()
    vision=model.vision(inputs['images'].reshape(batch*total*2,3,d.image_size,d.image_size)).reshape(batch,total,128)
    touch=model.touch(inputs['tactile']);estimates=[];used=[]
    for i in range(total):
        original=inputs['proprio'][:,i]
        proprio=torch.cat((original[:,:slot.start],previous,original[:,slot.stop:]),dim=-1)
        if return_history:used.append(previous.detach().clone())
        features=torch.cat((vision[:,i],touch[:,i],model.proprio(proprio)),dim=-1)
        sequence,hidden=model.memory(features[:,None],hidden)
        action=model.action(sequence)[:,0]
        if not torch.isfinite(action).all():raise ValueError('Nonfinite autoregressive actor command')
        estimates.append(action);previous=action.detach()
    return torch.stack(estimates,dim=1),hidden,previous,(torch.stack(used,dim=1) if return_history else None)


def actor_history_prediction(model, inputs, burn_in, *, episode_start, return_history=False):
    """Carry actor-owned, detached actions through warm-up and scored steps.

    At a true episode reset the initial previous command must be zero. At a
    truncated-window boundary, the first command is the actual recorded
    preceding command. Every subsequent command is produced by this model.
    Warm-up has no gradients. Scored GRU state retains BPTT; action feedback
    is detached, so gradients do not pass through the command-input loop.
    """
    d,batch,total=_validate_inputs(model,inputs)
    if type(burn_in) is not int or not 0<=burn_in<total:
        raise ValueError('Invalid actual-history window dimensions')
    starts=torch.as_tensor(episode_start,device=inputs['proprio'].device)
    if starts.shape!=(batch,) or starts.dtype!=torch.bool:
        raise ValueError('Explicit boolean episode boundary per window is required')
    slot=previous_action_slice(d)
    previous=inputs['proprio'][:,0,slot].detach()
    if torch.any(previous[starts]!=0):
        raise ValueError('A true episode reset must have zero previous command')
    hidden=None;warm_history=None
    if burn_in:
        with torch.no_grad():
            _,hidden,previous,warm_history=actor_history_chunk(model,{k:v[:,:burn_in] for k,v in inputs.items()},
                previous=previous,return_history=return_history)
        hidden=hidden.detach()
    prediction,_,_,history=actor_history_chunk(model,{k:v[:,burn_in:] for k,v in inputs.items()},
        previous=previous,hidden=hidden,return_history=return_history)
    if return_history:return prediction,torch.cat((warm_history,history),dim=1) if warm_history is not None else history
    return prediction


def dual_history_loss(recorded_prediction,actor_prediction,target):
    """Equal losses on two input histories; never average predictions first."""
    if recorded_prediction.shape!=actor_prediction.shape or target.shape!=actor_prediction.shape:
        raise ValueError('Both histories must supervise the same labels')
    recorded=torch.mean((recorded_prediction-target)**2)
    actor=torch.mean((actor_prediction-target)**2)
    return .5*recorded+.5*actor,recorded,actor
