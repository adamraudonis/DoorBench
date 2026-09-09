"""Completed-update recovery preserves Adam moments and rejects data drift."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
import pytest
import torch
from scripts.dexterous.train_native_continuous import restore_training_state
from doorbench.dexterous.motor_contract_identity import SENSOR_ACTOR_CHECKPOINT_SCHEMA

@dataclass
class Dimensions:
    actions: int = 2


def fixture():
    episode = SimpleNamespace(dimensions=Dimensions(), motor_contract_sha256='abc',
        layout={'pads': 5}, metadata={'physics_dt_s': .002})
    config = dict(seed=0, learning_rate=.0001, chunk_length=32, cold_prefix_length=32,
        cold_weight=.5, protocol='continuous_history_v1', dataset={'hash': 'exact'}, device='cpu')
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    update(model, optimizer)
    state = deepcopy(dict(configuration=config, actor=dict(schema=SENSOR_ACTOR_CHECKPOINT_SCHEMA,
        dimensions={'actions': 2}, motor_contract_sha256='abc', sensor_layout=episode.layout,
        physics_dt_s=.002, training_protocol='continuous_history_v1', completed_optimizer_steps=3,
        model_state=model.state_dict()), optimizer=optimizer.state_dict(), torch_rng_state=torch.get_rng_state()))
    return episode, config, model, optimizer, state


def update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(4, 3)).square().mean().backward()
    optimizer.step()


def test_resume_next_update_exact_and_missing_history_not_invented():
    episode, config, model, optimizer, state = fixture()
    recovered = torch.nn.Linear(3, 2)
    resumed = torch.optim.AdamW(recovered.parameters(), lr=.0001)
    count, history = restore_training_state(state, recovered, resumed, config, episode)
    assert count == 3 and history == []
    assert torch.equal(torch.get_rng_state(), state['torch_rng_state'])
    update(model, optimizer); update(recovered, resumed)
    for a, b in zip(model.parameters(), recovered.parameters()):
        assert torch.equal(a, b)


@pytest.mark.parametrize('key,value', [('dataset', {'hash': 'different'}), ('chunk_length', 16),
    ('learning_rate', .001), ('device', 'cuda')])
def test_resume_rejects_protocol_or_data_drift(key, value):
    episode, config, model, optimizer, state = fixture()
    config = dict(config, **{key: value})
    with pytest.raises(ValueError, match=key):
        restore_training_state(state, model, optimizer, config, episode)


def test_warm_start_contract_and_weights_without_optimizer_moments():
    from scripts.dexterous.train_native_continuous import initialize_actor_weights
    episode, config, model, optimizer, state = fixture()
    initialized = torch.nn.Linear(3, 2)
    fresh_optimizer = torch.optim.AdamW(initialized.parameters(), lr=.00001)
    initialize_actor_weights(state['actor'], initialized, episode)
    for a,b in zip(model.parameters(), initialized.parameters()):assert torch.equal(a,b)
    assert not fresh_optimizer.state
    bad=deepcopy(state['actor']);bad['sensor_layout']={'pads':6}
    with pytest.raises(ValueError,match='actor contract'):initialize_actor_weights(bad,initialized,episode)
