import json
from pathlib import Path

import pytest

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
