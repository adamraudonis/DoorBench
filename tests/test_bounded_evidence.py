import gzip
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
