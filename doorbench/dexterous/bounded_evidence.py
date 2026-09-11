"""Lossless disk-backed JSON records for long diagnostic episodes.

Only evidence storage changes. Records are serialized on append, with at most
one small chunk and the latest record held in memory. Repeated audit passes read
closed chunks; exported files retain the existing gzip JSON-array contract.
"""
import bisect
import gzip
import hashlib
import json
from pathlib import Path


class BoundedEvidence:
    def __init__(self, path, chunk_size=250):
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('Positive chunk size required')
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.chunk_size = chunk_size
        self.chunks = []
        self.ends = []
        self.pending = []
        self.count = 0
        self.first = None
        self.latest = None
        self.exported_path = None

    def append(self, row):
        if self.exported_path is not None:
            raise ValueError("Cannot append to finalized evidence")
        encoded = json.dumps(row, separators=(',', ':'))
        if self.first is None:
            self.first = encoded
        self.pending.append(encoded)
        self.latest = encoded
        self.count += 1
        if len(self.pending) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        path = self.path / f'{len(self.chunks):06d}.jsonl.gz'
        temporary = path.with_suffix('.pending')
        with gzip.open(temporary, 'wt') as stream:
            for row in self.pending:
                stream.write(row + '\n')
        temporary.replace(path)
        self.chunks.append(path)
        self.ends.append(self.count)
        self.pending.clear()

    def checkpoint(self, target):
        """Publish an incomplete, hash-bound prefix without rewriting its data."""
        if self.exported_path is not None:
            raise ValueError('Cannot checkpoint finalized evidence')
        target = Path(target)
        self.flush()
        records = []
        previous = 0
        for path, end in zip(self.chunks, self.ends):
            relative = path.resolve().relative_to(target.parent.resolve())
            with path.open('rb') as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            records.append(dict(file=str(relative), records=end-previous,
                                bytes=path.stat().st_size, sha256=digest.hexdigest()))
            previous = end
        manifest = dict(schema='doorbench.bounded-evidence.checkpoint.v1',
                        complete=False, passed=False, records=self.count, chunks=records)
        temporary = target.with_suffix(target.suffix + '.pending')
        temporary.write_text(json.dumps(manifest, indent=2)+'\n')
        temporary.replace(target)
        return manifest

    def __len__(self):
        return self.count

    def __iter__(self):
        if self.exported_path is not None:
            from .json_record_stream import iter_json_object_array
            with gzip.open(self.exported_path, "rt") as stream:
                yield from iter_json_object_array(stream)
            return
        for path in self.chunks:
            with gzip.open(path, 'rt') as stream:
                for line in stream:
                    yield json.loads(line)
        for row in self.pending:
            yield json.loads(row)

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError('Use iteration instead of materializing evidence slices')
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        if index == 0:
            return json.loads(self.first)
        if index == self.count - 1:
            return json.loads(self.latest)
        if self.exported_path is not None:
            for offset, row in enumerate(self):
                if offset == index:
                    return row
            raise ValueError("Truncated finalized evidence")
        chunk = bisect.bisect_right(self.ends, index)
        start = self.ends[chunk - 1] if chunk else 0
        if chunk == len(self.chunks):
            return json.loads(self.pending[index - start])
        with gzip.open(self.chunks[chunk], 'rt') as stream:
            for offset, line in enumerate(stream):
                if offset == index - start:
                    return json.loads(line)
        raise ValueError('Truncated evidence chunk')

    def export(self, target):
        """Verify final decoded bytes before releasing redundant working chunks.

        Failed export or verification preserves all chunks for recovery. Audit
        iteration remains available from the finalized lossless JSON array.
        """
        self.flush()
        target = Path(target)
        if target.exists():
            raise ValueError('Fresh evidence export required')
        temporary = target.with_suffix(target.suffix + '.pending')
        expected = hashlib.sha256()
        def write(stream, value):
            expected.update(value.encode('utf-8'))
            stream.write(value)
        with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
            write(stream, '[')
            for index, row in enumerate(self):
                if index:
                    write(stream, ',')
                write(stream, json.dumps(row, separators=(',', ':')))
            write(stream, ']')
        actual = hashlib.sha256()
        with gzip.open(temporary, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                actual.update(block)
        if actual.digest() != expected.digest():
            raise ValueError('Final evidence bytes differ; working chunks retained')
        temporary.replace(target)
        old_chunks = list(self.chunks)
        self.exported_path = target
        self.chunks.clear()
        self.ends.clear()
        for path in old_chunks:
            path.unlink()


def iter_checkpoint(target):
    """Recover only the exact recorded prefix; never interpret it as a pass."""
    target = Path(target)
    manifest = json.loads(target.read_text())
    if (manifest.get('schema') != 'doorbench.bounded-evidence.checkpoint.v1'
            or manifest.get('complete') is not False or manifest.get('passed') is not False):
        raise ValueError('Explicit incomplete checkpoint required')
    chunks = manifest['chunks']
    if (type(manifest['records']) is not int or manifest['records'] < 0
            or not isinstance(chunks, list)):
        raise ValueError('Invalid checkpoint record counts')
    count = 0
    seen = set()
    for entry in chunks:
        relative = Path(entry['file'])
        path = target.parent / relative
        if (relative.is_absolute() or '..' in relative.parts or str(relative) in seen
                or path.is_symlink() or not path.resolve().is_relative_to(target.parent.resolve())
                or type(entry['records']) is not int or entry['records'] <= 0):
            raise ValueError('Unsafe or duplicate checkpoint chunk')
        seen.add(str(relative))
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if path.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
            raise ValueError('Checkpoint chunk changed')
        actual = 0
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                actual += 1
                yield json.loads(line)
        if actual != entry['records']:
            raise ValueError('Checkpoint chunk record count differs')
        count += actual
    if count != manifest['records']:
        raise ValueError('Checkpoint total record count differs')
