"""Strict bounded JSON object-array decoding for new continuation evidence."""
import json
import math


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate continuation JSON field: ' + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError('Nonfinite continuation JSON: ' + value)


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result): _nonfinite(value)
    return result


def iter_continuation_records(stream, *, chunk_size=65536, max_record_chars=16777216):
    """Retain one bounded record, reject duplicate keys and nonfinite literals."""
    if (type(chunk_size) is not int or type(max_record_chars) is not int
            or not 1 <= chunk_size <= 1048576 or not 1 <= max_record_chars <= 16777216):
        raise ValueError('Explicit bounded continuation stream sizes required')
    decoder = json.JSONDecoder(object_pairs_hook=_unique, parse_constant=_nonfinite, parse_float=_finite_float)
    buffer = ''; eof = False

    def space():
        nonlocal buffer, eof
        while True:
            buffer = buffer.lstrip(' \t\r\n')
            if buffer or eof: return
            part = stream.read(chunk_size)
            if not part: eof = True
            buffer += part

    space()
    if not buffer.startswith('['): raise ValueError('Continuation object array required')
    buffer = buffer[1:]; space()
    if buffer.startswith(']'):
        buffer = buffer[1:]; space()
        if buffer: raise ValueError('Trailing continuation data')
        return
    while True:
        if not buffer.startswith('{'): raise ValueError('Continuation object required; no trailing comma')
        while True:
            try: record, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof: raise ValueError('Truncated or malformed continuation record') from None
                if len(buffer) > max_record_chars: raise ValueError('Continuation record exceeds memory bound')
                part = stream.read(chunk_size)
                if not part: eof = True
                buffer += part
                continue
            if end > max_record_chars: raise ValueError('Continuation record exceeds memory bound')
            break
        buffer = buffer[end:]; space()
        if not buffer or buffer[0] not in ',]': raise ValueError('Continuation comma or terminator required')
        yield record
        if buffer.startswith(']'):
            buffer = buffer[1:]; space()
            if buffer: raise ValueError('Trailing continuation data')
            return
        buffer = buffer[1:]; space()
