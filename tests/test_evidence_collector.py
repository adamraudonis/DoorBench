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


def test_final_sync_repairs_same_size_same_mtime_progress(tmp_path):
    import os, shutil, subprocess
    if not shutil.which('rsync'):pytest.skip('rsync unavailable')
    source=tmp_path/'source';destination=tmp_path/'destination'
    source.mkdir();destination.mkdir()
    (source/'progress.json').write_bytes(b'final');(destination/'progress.json').write_bytes(b'stale')
    for folder in (source,destination):os.utime(folder/'progress.json',(1700000000,1700000000))
    manifest={'progress.json':dict(bytes=5,sha256=hashlib.sha256(b'final').hexdigest())}
    subprocess.run(['rsync','-a',*module.final_transfer_options(None),str(source)+'/',str(destination)+'/'],check=True)
    with pytest.raises(ValueError):module.verify_local(destination,manifest)
    subprocess.run(['rsync','-a',*module.final_transfer_options(manifest),str(source)+'/',str(destination)+'/'],check=True)
    assert module.verify_local(destination,manifest)==1


@pytest.mark.parametrize('free_mib,incoming_mib',[(64,0),(1500,1000)])
def test_collector_waits_without_starting_rsync_when_storage_is_low(tmp_path,monkeypatch,free_mib,incoming_mib):
    import json
    from types import SimpleNamespace
    now=[1.];calls=[]
    monkeypatch.setattr(module.time,'time',lambda:now[0])
    monkeypatch.setattr(module.time,'sleep',lambda _:now.__setitem__(0,101.))
    monkeypatch.setattr(module.signal,'signal',lambda *_:None)
    monkeypatch.setattr(module.shutil,'disk_usage',lambda _:SimpleNamespace(free=free_mib*1024**2))
    def run(argv,**kwargs):
        calls.append(argv[0]);assert argv[0]=='ssh'
        return SimpleNamespace(stdout=json.dumps(dict(exists=True,pid=42,alive=True,terminal=False,transfer_bytes=incoming_mib*1024**2)))
    monkeypatch.setattr(module.subprocess,'run',run)
    args=SimpleNamespace(host='example.test',port=22,remote='/owned/run',deadline=100.,interval=5.,
        destination=tmp_path/'evidence',resume=False,key=tmp_path/'key',terminal='report.json',pid_file='run.pid',minimum_free_mib=1024)
    module.collect(args)
    receipt=json.loads((tmp_path/'evidence-collector.json').read_text())
    assert calls==['ssh'] and receipt['storage_waits']==1
    assert not receipt['final_bytes_verified'] and receipt['copies']==0


def test_remote_size_inventory_includes_stable_partial_before_completion(tmp_path, capsys):
    import json
    (tmp_path/'stable.partial.json.gz').write_bytes(b'x'*4096)
    (tmp_path/'unfinished.tmp').write_bytes(b'x'*100)
    exec(module.REMOTE_PROBE, {'INPUT':dict(remote=str(tmp_path),pid_file='run.pid',terminal='report.json',manifest=True)})
    result=json.loads(capsys.readouterr().out)
    assert result['transfer_bytes']==4096 and 'files' not in result


@pytest.mark.parametrize('mode',['matching','changed','sparse','still_remote','bad_export'])
def test_redundant_chunks_require_verified_matching_final_records(tmp_path,mode):
    import gzip,json
    folder=tmp_path/'native/physics-chunks';folder.mkdir(parents=True)
    chunk=folder/('000001.jsonl.gz' if mode=='sparse' else '000000.jsonl.gz')
    with gzip.open(chunk,'wt') as f:f.write(json.dumps({'time':1})+'\n')
    export=folder.parent/'physics-steps.json.gz'
    with gzip.open(export,'wt') as f:json.dump([{'time':2 if mode=='changed' else 1},{'time':3}],f)
    relative=str(export.relative_to(tmp_path))
    manifest={relative:dict(bytes=export.stat().st_size,sha256=hashlib.sha256(export.read_bytes()).hexdigest())}
    if mode=='still_remote':manifest[str(chunk.relative_to(tmp_path))]={}
    if mode=='bad_export':manifest[relative]['sha256']='changed'
    if mode=='bad_export':
        with pytest.raises(ValueError):module.prune_redundant_chunks(tmp_path,manifest)
        assert chunk.exists()
    else:
        removed=module.prune_redundant_chunks(tmp_path,manifest)
        assert bool(removed)==(mode=='matching')
        assert chunk.exists()==(mode!='matching')
    assert export.exists()
