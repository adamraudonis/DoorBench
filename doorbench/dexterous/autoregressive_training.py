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


def actor_history_prediction(model, inputs, burn_in, *, episode_start, return_history=False):
    """Carry actor-owned, detached actions through warm-up and scored steps.

    At a true episode reset the initial previous command must be zero. At a
    truncated-window boundary, the first command is the actual recorded
    preceding command. Every subsequent command is produced by this model.
    Warm-up has no gradients. Scored GRU state retains BPTT; action feedback
    is detached, so gradients do not pass through the command-input loop.
    """
    if set(inputs)!={'proprio','tactile','images'}:
        raise ValueError('Only prepared actor sensor channels are permitted')
    d=model.dimensions;batch,total=inputs['proprio'].shape[:2]
    if (not 0<=burn_in<total or inputs['proprio'].shape!=(batch,total,d.proprio_dimension)
            or inputs['tactile'].shape!=(batch,total,d.tactile)
            or inputs['images'].shape!=(batch,total,2,3,d.image_size,d.image_size)):
        raise ValueError('Invalid actual-history window dimensions')
    starts=torch.as_tensor(episode_start,device=inputs['proprio'].device,dtype=torch.bool)
    if starts.shape!=(batch,):raise ValueError('Explicit episode boundary per window is required')
    slot=previous_action_slice(d)
    previous=inputs['proprio'][:,0,slot].detach()
    if torch.any(previous[starts]!=0):
        raise ValueError('A true episode reset must have zero previous command')
    hidden=None;estimates=[];used=[]

    def static_features(start,end):
        images=inputs['images'][:,start:end];length=end-start
        vision=model.vision(images.reshape(batch*length*2,3,d.image_size,d.image_size)).reshape(batch,length,128)
        return vision,model.touch(inputs['tactile'][:,start:end])

    def phase(start,end,*,scored):
        nonlocal previous,hidden
        vision,touch=static_features(start,end)
        for local,i in enumerate(range(start,end)):
            original=inputs['proprio'][:,i]
            proprio=torch.cat((original[:,:slot.start],previous,original[:,slot.stop:]),dim=-1)
            if return_history:used.append(previous.detach().clone())
            features=torch.cat((vision[:,local],touch[:,local],model.proprio(proprio)),dim=-1)
            sequence,hidden=model.memory(features[:,None],hidden)
            action=model.action(sequence)[:,0]
            if not torch.isfinite(action).all():raise ValueError('Nonfinite autoregressive actor command')
            if scored:estimates.append(action)
            previous=action.detach()

    if burn_in:
        with torch.no_grad():phase(0,burn_in,scored=False)
        hidden=hidden.detach()
    phase(burn_in,total,scored=True)
    prediction=torch.stack(estimates,dim=1)
    if return_history:return prediction,torch.stack(used,dim=1)
    return prediction
