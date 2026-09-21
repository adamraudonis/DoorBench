"""Copy closed evidence prefixes without finalizing the active writers."""
import copy
import gzip
import json
from pathlib import Path

from .bounded_evidence import BoundedEvidence, iter_checkpoint
from .isaac_live_planning_pause import _sha, _hex


def snapshot_live_evidence(writers,directory,*,pause_token,compression_level=3):
    """Checkpoint and copy each writer, leaving subsequent append/export valid.

    Every checkpoint remains complete=false/passed=false. Its copied chunks
    survive later live export/deletion. A separate phase auditor interprets the
    copied records; this file-only operation grants no source qualification.
    Failure preserves all live chunks and any partial new snapshot files.
    """
    if (type(writers) is not dict or not writers or not _hex(pause_token)
            or type(compression_level) is not int or not 1<=compression_level<=9):
        raise ValueError('Explicit writers, pause token and gzip level required')
    for name,writer in writers.items():
        if (type(name) is not str or not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name)
                or not isinstance(writer,BoundedEvidence) or writer.exported_path is not None):
            raise ValueError('Named non-finalized BoundedEvidence writers required')
    if len({id(v) for v in writers.values()})!=len(writers):raise ValueError('Each live writer must have one snapshot role')
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=False);entries={}
    for name,writer in writers.items():
        logical=(len(writer),writer.first,writer.latest)
        work_checkpoint=writer.path.parent/f'live-pause-{pause_token}-{name}-checkpoint.json'
        if work_checkpoint.exists() or work_checkpoint.with_suffix('.json.pending').exists():
            raise ValueError('Fresh working checkpoint required')
        manifest=writer.checkpoint(work_checkpoint)
        destination=root/name;destination.mkdir()
        copied=destination/'checkpoint.json'
        for chunk in manifest['chunks']:
            relative=Path(chunk['file']);source=work_checkpoint.parent/relative
            if (relative.is_absolute() or '..' in relative.parts or source.is_symlink()
                    or not source.resolve().is_relative_to(work_checkpoint.parent.resolve())
                    or _sha(source)!=chunk['sha256']):raise ValueError('Working checkpoint chunk changed or escaped')
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
            with source.open('rb') as inp,target.open('xb') as out:
                for block in iter(lambda:inp.read(1024*1024),b''):out.write(block)
            if _sha(source)!=chunk['sha256'] or _sha(target)!=chunk['sha256'] or target.stat().st_size!=chunk['bytes']:
                raise ValueError('Evidence changed during immutable snapshot copy')
        with copied.open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
        array_path=destination/'rows.json.gz';count=0
        with gzip.open(array_path,'xt',encoding='utf-8',compresslevel=compression_level) as stream:
            stream.write('[')
            for count,row in enumerate(iter_checkpoint(copied),1):
                if count>1:stream.write(',')
                stream.write(json.dumps(row,separators=(',',':'),allow_nan=False))
            stream.write(']')
        if count!=logical[0] or (len(writer),writer.first,writer.latest)!=logical or writer.exported_path is not None:
            raise ValueError('Live evidence changed while snapshot was being copied')
        entries[name]=dict(records=count,checkpoint_path=str(copied),checkpoint_sha256=_sha(copied),
            rows_path=str(array_path),rows_sha256=_sha(array_path),working_checkpoint_path=str(work_checkpoint),
            complete=False,passed=False)
    return dict(schema='doorbench.live-evidence-snapshot.v1',pause_token=pause_token,
        streams=copy.deepcopy(entries),episode_complete=False,phase_qualified=False,
        authorized_stages=0,live_writers_finalized=False)
