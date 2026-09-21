import gzip
import json
from pathlib import Path

import pytest

from doorbench.dexterous.bounded_evidence import BoundedEvidence,iter_checkpoint
from doorbench.dexterous import isaac_live_evidence_snapshot as module


def test_pending_tail_snapshot_survives_later_append_export_and_chunk_deletion(tmp_path):
    pad=BoundedEvidence(tmp_path/'pad-chunks');transfer=BoundedEvidence(tmp_path/'transfer-chunks')
    for i in range(263):pad.append(dict(time_s=i*.002,value=i))
    for i in range(262):transfer.append(dict(time_s=(i+1)*.002,value=i))
    result=module.snapshot_live_evidence({'pad':pad,'transfer':transfer},tmp_path/'pause-evidence',pause_token='a'*64)
    for name,writer,count in (('pad',pad,263),('transfer',transfer,262)):
        row=result['streams'][name]
        assert writer.exported_path is None and len(writer)==count and not writer.pending
        checkpoint=Path(row['checkpoint_path']);manifest=json.loads(checkpoint.read_text())
        assert manifest['complete'] is False and manifest['passed'] is False and manifest['records']==count
        with gzip.open(row['rows_path'],'rt') as stream:assert len(json.load(stream))==count
        writer.append(dict(time_s=(count+1)*.002,value='after pause'))
        writer.export(tmp_path/(name+'-final.json.gz'))
        assert len(list(iter_checkpoint(checkpoint)))==count
        assert list(writer)[-1]['value']=='after pause'
    assert result['phase_qualified'] is False and result['authorized_stages']==0


def test_copy_failure_preserves_working_chunks_and_appendability(tmp_path,monkeypatch):
    writer=BoundedEvidence(tmp_path/'chunks')
    for i in range(260):writer.append({'value':i})
    real=module._sha;calls=0
    def corrupt(path):
        nonlocal calls
        calls+=1
        return 'f'*64 if calls==3 else real(path)
    monkeypatch.setattr(module,'_sha',corrupt)
    with pytest.raises(ValueError,match='changed during'):module.snapshot_live_evidence({'pad':writer},tmp_path/'partial',pause_token='b'*64)
    assert len(list(writer))==260 and writer.exported_path is None
    assert all(path.exists() for path in writer.chunks)
    writer.append({'value':260});assert len(writer)==261
    assert (tmp_path/'partial').exists()


def test_existing_work_checkpoint_is_never_overwritten(tmp_path):
    writer=BoundedEvidence(tmp_path/'chunks');writer.append({'v':1})
    target=tmp_path/('live-pause-'+'c'*64+'-pad-checkpoint.json');target.write_text('unrelated')
    with pytest.raises(ValueError,match='Fresh working'):module.snapshot_live_evidence({'pad':writer},tmp_path/'new',pause_token='c'*64)
    assert target.read_text()=='unrelated' and writer.pending


def test_finalized_or_duplicate_writers_and_unsafe_names_reject(tmp_path):
    writer=BoundedEvidence(tmp_path/'chunks');writer.append({'v':1})
    for names in ({'../bad':writer},{'pad':writer,'again':writer}):
        with pytest.raises(ValueError):module.snapshot_live_evidence(names,tmp_path/'unused',pause_token='d'*64)
    assert not (tmp_path/'unused').exists()
    writer.export(tmp_path/'final.json.gz')
    with pytest.raises(ValueError):module.snapshot_live_evidence({'pad':writer},tmp_path/'unused',pause_token='d'*64)


def test_two_distinct_pauses_have_independent_immutable_prefixes(tmp_path):
    writer=BoundedEvidence(tmp_path/'chunks');writer.append({'v':1})
    first=module.snapshot_live_evidence({'pad':writer},tmp_path/'first',pause_token='e'*64)
    writer.append({'v':2})
    second=module.snapshot_live_evidence({'pad':writer},tmp_path/'second',pause_token='f'*64)
    assert len(list(iter_checkpoint(first['streams']['pad']['checkpoint_path'])))==1
    assert len(list(iter_checkpoint(second['streams']['pad']['checkpoint_path'])))==2
