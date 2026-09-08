import numpy as np
import pytest
torch=pytest.importorskip('torch')

from doorbench.dexterous.sensor_actor import ActorDimensions,SensorActor,prepare_actor_packet
from doorbench.dexterous.autoregressive_training import actor_history_prediction,previous_action_slice
from test_sensor_actor import packet


def fixture():
    torch.manual_seed(42);torch.set_num_threads(1)
    d=ActorDimensions(joints=3,actions=4,tactile=12,image_size=32,hidden=16)
    model=SensorActor(d).train();rows=[]
    for i in range(7):
        p=packet(d);p['sensor_time_s'][:]=i*.002;p['joint_position'][:]=i*.01
        p['previous_action'][:]=0 if i==0 else .8
        rows.append(prepare_actor_packet(p,i*.002,d))
    inputs={k:torch.tensor(np.stack([r[k] for r in rows])[None]) for k in rows[0]}
    return model,inputs


def test_recorded_commands_cannot_substitute_during_warmup_or_scored_rollout():
    model,inputs=fixture();original={k:v.clone() for k,v in inputs.items()}
    result,history=actor_history_prediction(model,inputs,2,episode_start=[True],return_history=True)
    poisoned={k:v.clone() for k,v in inputs.items()};slot=previous_action_slice(model.dimensions)
    poisoned['proprio'][:,1:,slot]=-.95
    alternative,other=actor_history_prediction(model,poisoned,2,episode_start=[True],return_history=True)
    torch.testing.assert_close(result,alternative,rtol=0,atol=0)
    torch.testing.assert_close(history,other,rtol=0,atol=0)
    torch.testing.assert_close(history[:,3:],result[:,:-1].detach(),rtol=0,atol=0)
    assert all(torch.equal(inputs[k],v) for k,v in original.items())
    assert not torch.any(history[:,0])


def test_batched_static_encoding_matches_stepwise_actor_forward():
    model,inputs=fixture();slot=previous_action_slice(model.dimensions)
    got=actor_history_prediction(model,inputs,2,episode_start=[True])
    previous=torch.zeros(1,model.dimensions.actions);hidden=None;expected=[]
    with torch.no_grad():
        for i in range(7):
            now={k:v[:,i:i+1].clone() for k,v in inputs.items()}
            now['proprio'][:,:,slot]=previous[:,None]
            predicted,hidden=model(**now,hidden=hidden);previous=predicted[:,0]
            if i>=2:expected.append(previous)
    torch.testing.assert_close(got,torch.stack(expected,1),rtol=1e-5,atol=1e-6)


def test_warmup_and_action_feedback_detach_but_scored_gru_history_keeps_bptt():
    model,inputs=fixture();inputs['proprio'].requires_grad_(True)
    action_outputs=[]
    def hook(_module,_args,result):
        if result.requires_grad:result.retain_grad();action_outputs.append(result)
    handle=model.action.register_forward_hook(hook)
    result=actor_history_prediction(model,inputs,2,episode_start=[True])
    result[:,-1].sum().backward();handle.remove()
    slot=previous_action_slice(model.dimensions)
    assert torch.count_nonzero(inputs['proprio'].grad[:,:2])==0
    assert torch.count_nonzero(inputs['proprio'].grad[:,:,slot])==0
    assert torch.count_nonzero(inputs['proprio'].grad[:,2,:slot.start])>0
    # Stacking all scored outputs may deliver explicit zeros to unused slots.
    assert all(a.grad is None or torch.count_nonzero(a.grad)==0 for a in action_outputs[:-1])
    assert action_outputs[-1].grad is not None
    assert torch.isfinite(model.memory.weight_hh_l0.grad).all()


def test_true_reset_and_truncated_boundary_have_distinct_explicit_contracts():
    model,inputs=fixture();slot=previous_action_slice(model.dimensions)
    inputs['proprio'][:,0,slot]=.25
    with pytest.raises(ValueError,match='zero previous command'):
        actor_history_prediction(model,inputs,2,episode_start=[True])
    _,history=actor_history_prediction(model,inputs,2,episode_start=[False],return_history=True)
    torch.testing.assert_close(history[:,0],torch.full((1,4),.25))
    with pytest.raises(ValueError,match='Only prepared actor sensor'):
        actor_history_prediction(model,dict(inputs,teacher_command=torch.zeros(4)),2,episode_start=[False])


@pytest.mark.parametrize('boundaries',[[1],[float('nan')],[1.0]])
def test_episode_boundaries_cannot_be_silently_coerced(boundaries):
    model,inputs=fixture()
    with pytest.raises(ValueError,match='boolean episode boundary'):
        actor_history_prediction(model,inputs,2,episode_start=boundaries)


@pytest.mark.parametrize('burn',[True,2.0])
def test_warmup_requires_an_integer_sample_count(burn):
    model,inputs=fixture()
    with pytest.raises(ValueError,match='history window'):
        actor_history_prediction(model,inputs,burn,episode_start=[True])
