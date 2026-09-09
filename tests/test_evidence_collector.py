import hashlib
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('collector',Path(__file__).parents[1]/'scripts/isaac/collect_run.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_partial_or_changed_copy_cannot_be_verified(tmp_path):
    p=tmp_path/'raw.npz';p.write_bytes(b'complete recorded bytes')
    manifest={'raw.npz':dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())}
    assert module.verify_local(tmp_path,manifest)==1
    p.write_bytes(b'partial')
    with pytest.raises(ValueError):module.verify_local(tmp_path,manifest)
    p.write_bytes(b'x'*manifest['raw.npz']['bytes'])
    with pytest.raises(ValueError):module.verify_local(tmp_path,manifest)


def test_manifest_cannot_escape_or_follow_symlink(tmp_path):
    with pytest.raises(ValueError):module.verify_local(tmp_path,{'../escape':{}})
    with pytest.raises(ValueError):module.verify_local(tmp_path,{})
    p=tmp_path/'link';p.symlink_to('/etc/hosts')
    with pytest.raises(ValueError):module.verify_local(tmp_path,{'link':dict(bytes=0,sha256='a'*64)})


def test_final_manifest_excludes_compound_temporary_names(tmp_path, capsys):
    from scripts.isaac.collect_run import REMOTE_PROBE
    import json
    (tmp_path / 'run.pid').write_text('999999999')
    (tmp_path / 'report.json').write_text('{}')
    for name in ('pad.partial.tmp.gz', 'trace.json.pending', 'status.writing.json', 'stable.partial.json.gz'):
        (tmp_path / name).write_text('evidence')
    exec(REMOTE_PROBE, {'INPUT': {'remote': str(tmp_path), 'pid_file': 'run.pid',
                                'terminal': 'report.json', 'manifest': True}})
    result = json.loads(capsys.readouterr().out)
    assert 'stable.partial.json.gz' in result['files']
    assert not any(name in result['files'] for name in
                   ('pad.partial.tmp.gz', 'trace.json.pending', 'status.writing.json'))
