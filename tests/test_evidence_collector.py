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
