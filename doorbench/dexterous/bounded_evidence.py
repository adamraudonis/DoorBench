"""Lossless disk-backed JSON records for long diagnostic episodes.

Only evidence storage changes. Records are serialized on append, with at most
one small chunk and the latest record held in memory. Repeated audit passes read
closed chunks; exported files retain the existing gzip JSON-array contract.
"""
import bisect
import gzip
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
        self.latest = None

    def append(self, row):
        encoded = json.dumps(row, separators=(',', ':'))
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

    def __len__(self):
        return self.count

    def __iter__(self):
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
        if index == self.count - 1:
            return json.loads(self.latest)
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
        """Atomically publish the original JSON array; keep chunks for recovery."""
        self.flush()
        target = Path(target)
        if target.exists():
            raise ValueError('Fresh evidence export required')
        temporary = target.with_suffix(target.suffix + '.pending')
        with gzip.open(temporary, 'wt') as stream:
            stream.write('[')
            for index, row in enumerate(self):
                if index:
                    stream.write(',')
                stream.write(json.dumps(row, separators=(',', ':')))
            stream.write(']')
        temporary.replace(target)
