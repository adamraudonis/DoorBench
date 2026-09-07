import importlib.util
import json
from pathlib import Path
import tarfile
import pytest
from doorbench.dexterous.provenance import capture,sha256

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dex_launch',ROOT/'scripts/dexterous/run_experiment.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


def test_cluster_paths_overrides_and_unimplemented_backend_rejected(tmp_path):
    cfg=json.loads((ROOT/'configs/dexterous/h1-shadow-reach-v1.json').read_text())
    args=launcher.command(cfg,tmp_path,tmp_path/'output',{'envs':64,'device':'cpu'})
    assert args[args.index('--envs')+1]=='64'
    assert args[args.index('--robot')+1]==str(tmp_path/cfg['robot'])
    cfg['simulator']='isaac-sim'
    with pytest.raises(ValueError):launcher.command(cfg,tmp_path,tmp_path/'output',{})


def test_source_and_input_hashes_without_environment_secrets(tmp_path,monkeypatch):
    (tmp_path/'doorbench').mkdir();(tmp_path/'doorbench/example.py').write_text('value=1\n')
    (tmp_path/'.env').write_text('TOKEN=do-not-archive')
    monkeypatch.setenv('PRIVATE_TOKEN','do-not-archive')
    robot=tmp_path/'robot.xml';robot.write_text('<mujoco/>')
    robot.with_suffix('.audit.json').write_text('{}')
    door=tmp_path/'door';door.mkdir();(door/'spec.json').write_text('{}')
    upstream=tmp_path/'upstream';weights=upstream/'data/reach_two_hands';weights.mkdir(parents=True)
    for name in ('torch_model.pt','mean.npy','var.npy'):(weights/name).write_bytes(b'fixture')
    out=tmp_path/'output'
    report=capture(tmp_path,out,{'robot':str(robot),'door':str(door),'upstream':str(upstream)})
    assert report['inputs']['robot']['sha256']==sha256(robot)
    assert report['capture_timing']=='before_training'
    assert 'do-not-archive' not in (out/'manifest.json').read_text()
    with tarfile.open(out/'source.tar.gz') as archive:
        assert archive.getnames()==['doorbench/example.py']
