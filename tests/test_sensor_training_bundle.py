import json
from pathlib import Path

import pytest
pytest.importorskip('torch')

from doorbench.dexterous.sensor_training_bundle import SCHEMA, bundled_path, digest, verify_bundle


def test_bundle_relocation_preserves_content_identity_and_rejects_tampering(tmp_path):
    first=tmp_path/'original';first.mkdir()
    (first/'calibration.json').write_text('{"actuator": "knee", "force_cap": 200}')
    value=dict(schema=SCHEMA,files_sha256={'calibration.json':digest(first/'calibration.json')})
    (first/'manifest.json').write_text(json.dumps(value))
    moved=tmp_path/'moved';first.rename(moved)
    assert verify_bundle(moved/'manifest.json')==value
    (moved/'calibration.json').write_text('{"actuator": "knee", "force_cap": 201}')
    with pytest.raises(ValueError,match='hash mismatch'):verify_bundle(moved/'manifest.json')


@pytest.mark.parametrize('relative',['../outside','/absolute','', '.'])
def test_bundle_paths_cannot_escape_or_substitute_root(tmp_path,relative):
    with pytest.raises(ValueError):bundled_path(tmp_path,relative)


def test_bundle_symlink_cannot_silently_depend_on_original_cluster(tmp_path):
    root=tmp_path/'bundle';root.mkdir()
    (root/'external').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError,match='escapes'):bundled_path(root,'external')


def test_periodic_checkpoint_is_weights_only_loadable_and_atomically_replaced(tmp_path):
    torch=pytest.importorskip('torch')
    import importlib.util
    file=Path(__file__).resolve().parents[1]/'scripts/dexterous/train_sensor_imitation.py'
    spec=importlib.util.spec_from_file_location('training',file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/'training-state.pt'
    module.atomic_torch(path,{'step':1,'weights':torch.tensor([1.])})
    module.atomic_torch(path,{'step':2,'weights':torch.tensor([2.]),'settings':{'path':'relative'}})
    assert torch.load(path,weights_only=True)['step']==2
    assert not path.with_name(path.name+'.partial').exists()


def test_periodic_fit_does_not_change_weights_training_mode_or_rng(tmp_path):
    torch=pytest.importorskip('torch')
    import numpy as np
    from test_sensor_actor import packet
    from doorbench.dexterous.sensor_actor import ActorDimensions, SensorActor, prepare_actor_packet
    from doorbench.dexterous.sensor_fit_evaluation import evaluate_frozen_fit
    torch.set_num_threads(1)
    dims=ActorDimensions(joints=3,actions=4,tactile=12,image_size=32,hidden=16)
    (tmp_path/'motor-contract.json').write_text(json.dumps({'actuators':[{'force_range':[-2,3]} for _ in range(4)]}))

    class Episode:
        dimensions=dims;path=tmp_path;times=np.arange(252)*.002
        metadata={'physics_dt_s':.002};layout={'action_order':['a','b','c','d']}
        numeric={'previous_action':np.zeros((252,4),np.float32)}
        def __len__(self):return 251
        def packet(self,i):
            p=packet(dims);p['sensor_time_s'][:]=self.times[i]
            return p
        def sequence(self,start,length):
            rows=[prepare_actor_packet(self.packet(i),self.times[i],dims) for i in range(start,start+length)]
            return {k:np.stack([r[k] for r in rows]) for k in rows[0]},np.zeros((length,4),np.float32)

    model=SensorActor(dims).train()
    state={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    first=evaluate_frozen_fit(model,[Episode()],['nominal'])
    assert model.training
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    second=evaluate_frozen_fit(model,[Episode()],['nominal'])
    assert first==second
    owned=evaluate_frozen_fit(model,[Episode()],['nominal'],include_actor_history_sources=True)
    assert owned['full_source_actor_history']['nominal']['examples']==251
    assert owned['prediction_mse_normalized_force']==first['prediction_mse_normalized_force']
    assert owned['startup']==first['startup']
    assert torch.equal(rng,torch.get_rng_state())
