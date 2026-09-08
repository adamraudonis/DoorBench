import time
import copy
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from test_autoregressive_training import fixture
from doorbench.dexterous.autoregressive_training import actor_history_chunk
from doorbench.dexterous.continuous_history_training import accumulate_full_sources,AccumulationInterrupted

class Episode:
    def __init__(self,inputs,labels=None):
        self.inputs=inputs;self.times=np.arange(inputs['proprio'].shape[1])*.002
        self.numeric={'previous_action':np.zeros((len(self.times),4),np.float32)}
        self.labels=np.zeros((len(self.times),4),np.float32) if labels is None else labels
    def __len__(self):return len(self.times)
    def sequence(self,start,length):return {k:v[0,start:start+length].detach().numpy() for k,v in self.inputs.items()},self.labels[start:start+length]


def test_same_fixed_full_history_predictions_and_exact_source_prefix_weights():
    model,x=fixture();episode=Episode(x);initial=copy.deepcopy(model.state_dict())
    with torch.no_grad():
        recorded,_=model(**x);owned,_,_,_=actor_history_chunk(model,x,previous=torch.zeros(1,4))
        weights=torch.full((7,),.5/7);weights[:2]+=.5/2
        rec=torch.sum(recorded.square().mean(-1)[0]*weights).item()
        own=torch.sum(owned.square().mean(-1)[0]*weights).item()
    seen=[];result=accumulate_full_sources(model,[episode],chunk_length=3,cold_prefix_length=2,progress=seen.append)
    assert result['recorded_loss']==pytest.approx(rec,rel=1e-5,abs=1e-7)
    assert result['actor_loss']==pytest.approx(own,rel=1e-5,abs=1e-7)
    assert result['mixed_loss']==pytest.approx((rec+own)*.5,rel=1e-5)
    assert result['logical_samples']==7 and [s['next_sample'] for s in seen]==[3,6,7]
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items())
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.parameters())


def test_sources_are_equal_weighted_and_reset_independently():
    model,x=fixture();e=Episode(x);short=Episode({k:v[:,:4] for k,v in x.items()})
    first=accumulate_full_sources(model,[e],chunk_length=3,cold_prefix_length=2)
    second=accumulate_full_sources(model,[short],chunk_length=3,cold_prefix_length=2)
    model.zero_grad();both=accumulate_full_sources(model,[e,short],chunk_length=3,cold_prefix_length=2)
    for k in ('mixed_loss','recorded_loss','actor_loss'):assert both[k]==pytest.approx((first[k]+second[k])*.5,rel=1e-6)
    assert both['logical_samples']==11


def test_deadline_during_source_never_changes_weights_or_completes_update():
    model,x=fixture();initial=copy.deepcopy(model.state_dict());stop=[False]
    def progress(_):stop[0]=True
    # Stop only after the first actual chunk, retaining a partial gradient as
    # explicit discarded work; there is no optimizer inside the accumulator.
    import doorbench.dexterous.continuous_history_training as module
    original=module.time.monotonic
    try:
        module.time.monotonic=lambda:2 if stop[0] else 0
        with pytest.raises(AccumulationInterrupted) as info:
            accumulate_full_sources(model,[Episode(x)],chunk_length=3,deadline=1,progress=progress)
        assert info.value.logical_samples==3
    finally:module.time.monotonic=original
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items())


def test_mutation_and_nonreset_inputs_fail_closed():
    model,x=fixture();e=Episode(x)
    def mutate(_):
        with torch.no_grad():next(model.parameters()).add_(.01)
    with pytest.raises(RuntimeError,match='Weights changed'):
        accumulate_full_sources(model,[e],chunk_length=3,progress=mutate)
    e.numeric['previous_action'][0]=.1
    with pytest.raises(ValueError,match='actual zero-command'):
        accumulate_full_sources(model,[e])
