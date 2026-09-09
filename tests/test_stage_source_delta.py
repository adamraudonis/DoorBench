import hashlib,io,json,tarfile
import pytest
from scripts.isaac.stage_source_delta import stage


def bundle(files,changed):
    hashes={name:hashlib.sha256(value).hexdigest() for name,value in files.items()}
    manifest=dict(files=hashes,sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())
    entries={**changed,'source-manifest.json':json.dumps(manifest).encode()};out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w:gz') as archive:
        for name,value in entries.items():
            item=tarfile.TarInfo(name);item.size=len(value);archive.addfile(item,io.BytesIO(value))
    out.seek(0);return out


def test_new_tree_copies_verified_files_and_omits_removed_and_runtime_files(tmp_path):
    base=tmp_path/'old';base.mkdir();(base/'same.py').write_bytes(b'same');(base/'removed.py').write_text('removed');(base/'runtime.log').write_text('runtime')
    new=tmp_path/'new';result=stage(base,new,bundle({'same.py':b'same','changed.py':b'new'},{'changed.py':b'new'}))
    assert result['all_source_hashes_verified'] and result['copied_files']==1
    assert not (new/'runtime.log').exists() and not (new/'removed.py').exists()
    assert (base/'removed.py').read_text()=='removed'


def test_changed_base_and_unsafe_paths_are_rejected(tmp_path):
    base=tmp_path/'base';base.mkdir();(base/'same.py').write_text('changed')
    with pytest.raises(ValueError,match='Base source'):stage(base,tmp_path/'bad',bundle({'same.py':b'expected'},{}))
    with pytest.raises(ValueError,match='Unsafe'):stage(base,tmp_path/'unsafe',bundle({'../escape':b'x'},{'../escape':b'x'}))
    assert not (tmp_path/'escape').exists()
