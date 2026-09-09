import io
import json
import pytest
from doorbench.dexterous.json_record_stream import iter_json_object_array


@pytest.mark.parametrize('chunk_size',[1,2,7,65536])
def test_exact_records_across_arbitrary_chunk_boundaries(chunk_size):
    rows=[{'number':1.2345678901234567,'escape':'a\\b"\n☃','nested':[{},[1,-2.3e-12],True,None]},{}]
    data=json.dumps(rows,ensure_ascii=False)
    assert list(iter_json_object_array(io.StringIO(data),chunk_size=chunk_size))==json.loads(data)
    assert list(iter_json_object_array(io.StringIO(' [ ] \n'),chunk_size=chunk_size))==[]


@pytest.mark.parametrize('data',['','{}','[', '[{},]','[{} {}]','[{}','[1]','[{"x":','[{}]false','[{}] []'])
def test_incomplete_or_wrong_framing_is_rejected(data):
    with pytest.raises(ValueError):list(iter_json_object_array(io.StringIO(data),chunk_size=2))


def test_first_record_does_not_require_whole_recording_and_record_bound_applies():
    data=json.dumps([{'sample':n} for n in range(10000)])
    stream=io.StringIO(data);rows=iter_json_object_array(stream,chunk_size=32)
    assert next(rows)=={'sample':0}
    assert stream.tell()<=32
    with pytest.raises(ValueError,match='memory bound'):
        list(iter_json_object_array(io.StringIO('[{"x":"'+'x'*100+'"}]'),chunk_size=8,max_record_chars=32))


@pytest.mark.parametrize('rows', [[], [{}], [
    {'float': 1.2345678901234567, 'negative_zero': -0.0,
     'nested': [True, False, None, {'unicode': '手', 'escape': '\\n'}]},
    {'large': 10**30, 'tiny': 1e-250}]])
def test_record_writer_matches_original_json_dump_bytes(rows):
    from doorbench.dexterous.json_record_stream import write_json_record_array
    expected = io.StringIO(); json.dump(rows, expected)
    actual = io.StringIO(); write_json_record_array(actual, iter(rows))
    assert actual.getvalue() == expected.getvalue()
    assert list(iter_json_object_array(io.StringIO(actual.getvalue()))) == rows


def test_record_writer_consumes_once_and_bounds_individual_writes():
    from doorbench.dexterous.json_record_stream import write_json_record_array
    class Sink:
        def __init__(self): self.parts = []
        def write(self, value): self.parts.append(value)
    sink = Sink()
    write_json_record_array(sink, ({'sample': n} for n in range(1000)))
    assert max(map(len, sink.parts)) < 20
    assert json.loads(''.join(sink.parts))[-1] == {'sample': 999}
    with pytest.raises(TypeError, match='record objects'):
        write_json_record_array(io.StringIO(), [1])
