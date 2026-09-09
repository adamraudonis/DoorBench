"""Stream JSON record arrays without retaining a second whole recording."""
import json


def iter_json_object_array(stream, *, chunk_size=65536, max_record_chars=16777216):
    if chunk_size<=0 or max_record_chars<=0:raise ValueError('Positive stream bounds required')
    decoder=json.JSONDecoder();buffer='';eof=False

    def more():
        nonlocal buffer,eof
        part=stream.read(chunk_size)
        if not part:eof=True
        buffer+=part

    def space():
        nonlocal buffer
        while True:
            buffer=buffer.lstrip(' \t\r\n')
            if buffer or eof:return
            more()

    space()
    if not buffer.startswith('['):raise ValueError('Expected a JSON record array')
    buffer=buffer[1:];space();first=True
    while not buffer.startswith(']') or not first:
        if not buffer.startswith('{'):raise ValueError('Expected a record object; no trailing comma')
        while True:
            try:record,end=decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:raise ValueError('Truncated or malformed JSON record') from None
                if len(buffer)>max_record_chars:raise ValueError('JSON record exceeds memory bound')
                more();continue
            if end>max_record_chars:raise ValueError('JSON record exceeds memory bound')
            break
        buffer=buffer[end:];space()
        if not buffer or buffer[0] not in ',]':raise ValueError('Expected comma or array terminator')
        yield record
        first=False
        if buffer.startswith(']'):break
        buffer=buffer[1:];space()
    buffer=buffer[1:];space()
    if buffer:raise ValueError('Unexpected data after JSON record array')


def write_json_record_array(stream, records):
    """Match default json.dump(list(records)) bytes with one encode per record.

    Encoding a nested record at once avoids one stream write per scalar/token.
    Only the current encoded record is retained; callers own atomic publication.
    """
    stream.write('[')
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError('Expected record objects')
        if index:
            stream.write(', ')
        stream.write(json.dumps(record))
    stream.write(']')
