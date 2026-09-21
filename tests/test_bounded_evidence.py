import gzip
import hashlib
import io
import json
import pytest
from doorbench.dexterous.bounded_evidence import BoundedEvidence


def test_lossless_repeated_audit_iteration_and_export(tmp_path):
    rows = [{'time': i * .002, 'contacts': [{'force': i / 7}], 'valid': i % 2 == 0}
            for i in range(19)]
    evidence = BoundedEvidence(tmp_path / 'chunks', chunk_size=3)
    for row in rows:
        evidence.append(row)
    assert len(evidence.pending) < 3
    assert list(evidence) == rows
    assert list(evidence) == rows
    assert [evidence[i] for i in range(len(evidence))] == rows
    assert evidence[-1] == rows[-1]
    output = tmp_path / 'physics-steps.json.gz'
    evidence.export(output)
    with gzip.open(output, 'rt') as stream:
        encoded=stream.read()
        assert json.loads(encoded) == rows
        assert encoded==json.dumps(rows,separators=(',', ':'))
    assert list(evidence) == rows
    assert not list((tmp_path / "chunks").glob("*.gz"))
    assert evidence[4] == rows[4]
    with pytest.raises(ValueError, match="finalized"):
        evidence.append({})
    with pytest.raises(ValueError):
        evidence.export(output)


def test_append_is_an_immutable_observation_and_missing_indices_fail(tmp_path):
    evidence = BoundedEvidence(tmp_path / 'chunks', chunk_size=2)
    row = {'contacts': [1, 2]}
    evidence.append(row)
    row['contacts'].clear()
    retrieved = evidence[-1]
    retrieved['contacts'].clear()
    assert evidence[0] == {'contacts': [1, 2]}
    with pytest.raises(IndexError):
        evidence[1]
    with pytest.raises(TypeError):
        evidence[:]


def test_failed_export_keeps_recoverable_chunks(tmp_path, monkeypatch):
    evidence=BoundedEvidence(tmp_path/'chunks',chunk_size=2)
    rows=[{'time': n} for n in range(4)]
    for row in rows: evidence.append(row)
    original=gzip.open
    def fail_verify(path, mode='rb', *args, **kwargs):
        if str(path).endswith('.pending') and mode=='rb':
            raise OSError('simulated truncated final export')
        return original(path,mode,*args,**kwargs)
    monkeypatch.setattr(gzip,'open',fail_verify)
    with pytest.raises(OSError): evidence.export(tmp_path/'final.json.gz')
    assert len(list((tmp_path/'chunks').glob('*.gz')))==2
    assert list(evidence)==rows
    assert not (tmp_path/'final.json.gz').exists()


def test_first_and_latest_records_do_not_read_closed_chunks(tmp_path, monkeypatch):
    import gzip
    from doorbench.dexterous.bounded_evidence import BoundedEvidence
    records=BoundedEvidence(tmp_path/'cache',chunk_size=1)
    records.append({'time':0,'contacts':[{'force':2.}]})
    records.append({'time':1,'contacts':[{'force':3.}]})
    def fail(*args,**kwargs):raise AssertionError('Control access must not read old chunks')
    monkeypatch.setattr(gzip,'open',fail)
    assert records[0]['time']==0
    assert records[-1]['time']==1
    changed=records[0];changed['contacts'][0]['force']=99
    assert records[0]['contacts'][0]['force']==2.


def test_checkpoint_recovers_exact_prefix_without_decoding_old_chunks(tmp_path,monkeypatch):
    from doorbench.dexterous.bounded_evidence import iter_checkpoint
    evidence=BoundedEvidence(tmp_path/'chunks',chunk_size=2)
    rows=[{'time':i,'value':[i]} for i in range(5)]
    for row in rows:evidence.append(row)
    target=tmp_path/'checkpoint.json'
    original=gzip.open
    def only_writes(path,mode='rb',*args,**kwargs):
        assert 'w' in mode, 'Checkpoint must not decode previous chunks'
        return original(path,mode,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(gzip,'open',only_writes)
        manifest=evidence.checkpoint(target)
    assert manifest['complete'] is False and manifest['passed'] is False
    assert not evidence.pending
    evidence.append({'time':5})
    evidence.flush()
    assert list(iter_checkpoint(target))==rows
    assert len(evidence)==6
    evidence.export(tmp_path/'final.json.gz')
    with gzip.open(tmp_path/'final.json.gz','rt') as f:assert json.load(f)==rows+[{'time':5}]


@pytest.mark.parametrize('change',['bytes','hash','count','duplicate','escape','complete'])
def test_checkpoint_rejects_changed_or_unsafe_evidence(tmp_path,change):
    from doorbench.dexterous.bounded_evidence import iter_checkpoint
    evidence=BoundedEvidence(tmp_path/'chunks');evidence.append({'time':0})
    target=tmp_path/'checkpoint.json';m=evidence.checkpoint(target)
    if change=='bytes':m['chunks'][0]['bytes']+=1
    if change=='hash':m['chunks'][0]['sha256']='wrong'
    if change=='count':m['records']+=1
    if change=='duplicate':m['chunks']*=2
    if change=='escape':m['chunks'][0]['file']='../other'
    if change=='complete':m['complete']=True
    target.write_text(json.dumps(m))
    with pytest.raises(ValueError):list(iter_checkpoint(target))
    assert list(evidence)==[{'time':0}]


@pytest.mark.parametrize('level', range(1, 10))
def test_all_compression_levels_preserve_exact_final_bytes_counts_and_checkpoints(tmp_path, level):
    from doorbench.dexterous.bounded_evidence import iter_checkpoint
    rows = [{'time_s': i*.002, 'value': -0. if i == 0 else i/7,
             'contacts': [{'label': 'thumb λ\\quoted\"\n', 'force': [1e-12, i+.2]}],
             'valid': bool(i % 2), 'missing': None} for i in range(7)]
    encoded = json.dumps(rows, separators=(',', ':')).encode('utf-8')
    evidence = BoundedEvidence(tmp_path/'chunks', chunk_size=2, compression_level=level)
    for row in rows:evidence.append(row)
    assert len(evidence) == 7 and len(evidence.pending) == 1
    checkpoint = tmp_path/'checkpoint.json'; manifest = evidence.checkpoint(checkpoint)
    assert [part['records'] for part in manifest['chunks']] == [2, 2, 2, 1]
    assert manifest['complete'] is False and manifest['passed'] is False
    assert list(iter_checkpoint(checkpoint)) == rows
    before = [evidence[i] for i in range(len(evidence))]
    output = tmp_path/'final.json.gz'; evidence.export(output)
    actual = gzip.decompress(output.read_bytes())
    assert actual == encoded and hashlib.sha256(actual).digest() == hashlib.sha256(encoded).digest()
    assert list(evidence) == before == rows and len(evidence) == 7
    assert not list((tmp_path/'chunks').iterdir())


@pytest.mark.parametrize('level', [0, 10, -1, True, False, 3., '3', None, float('nan')])
def test_invalid_compression_level_rejects_before_creating_storage(tmp_path, level):
    path = tmp_path/'not-created'
    with pytest.raises(ValueError, match='integer from 1 through 9'):
        BoundedEvidence(path, compression_level=level)
    assert not path.exists()


def test_default_three_applies_to_both_chunk_and_final_writes_only(tmp_path, monkeypatch):
    calls = []; original = gzip.open
    def observed(path, mode='rb', *args, **kwargs):
        calls.append((mode, kwargs.get('compresslevel')))
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(gzip, 'open', observed)
    evidence = BoundedEvidence(tmp_path/'chunks', 1)
    evidence.append({'first': True}); evidence.append({'first': False})
    evidence.export(tmp_path/'final.json.gz')
    assert evidence.compression_level == 3
    assert [level for mode, level in calls if 'w' in mode] == [3, 3, 3]
    assert all(level is None for mode, level in calls if 'r' in mode)
    with pytest.raises(TypeError):BoundedEvidence(tmp_path/'positional-level', 1, 9)


@pytest.mark.parametrize('level', [1, 3, 9])
def test_wrong_decoded_export_preserves_exact_recovery_chunks_then_can_retry(tmp_path, monkeypatch, level):
    from doorbench.dexterous.bounded_evidence import iter_checkpoint
    evidence = BoundedEvidence(tmp_path/'chunks', chunk_size=2, compression_level=level)
    rows = [{'time_s': i*.002, 'load_N': i/3} for i in range(5)]
    for row in rows:evidence.append(row)
    checkpoint = tmp_path/'checkpoint.json'; evidence.checkpoint(checkpoint)
    original_chunks = {p:p.read_bytes() for p in evidence.chunks}
    original = gzip.open
    def wrong_decoded_export(path, mode='rb', *args, **kwargs):
        if str(path).endswith('.pending') and mode == 'rb':return io.BytesIO(b'[]')
        return original(path, mode, *args, **kwargs)
    target = tmp_path/'final.json.gz'
    with monkeypatch.context() as patch:
        patch.setattr(gzip, 'open', wrong_decoded_export)
        with pytest.raises(ValueError, match='Final evidence bytes differ'):
            evidence.export(target)
    assert evidence.exported_path is None and not target.exists()
    assert len(evidence) == 5 and all(p.read_bytes() == encoded for p, encoded in original_chunks.items())
    assert list(iter_checkpoint(checkpoint)) == list(evidence) == rows
    evidence.export(target)
    assert gzip.decompress(target.read_bytes()) == json.dumps(rows, separators=(',', ':')).encode('utf-8')
    assert list(evidence) == rows and not list((tmp_path/'chunks').iterdir())
