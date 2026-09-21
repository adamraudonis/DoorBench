"""Bounded read-only row iteration for numeric C-order NPZ recording fields."""
from contextlib import ExitStack
from pathlib import Path
import zipfile

import numpy as np


def iter_npz_records(path, shapes, *, expected_rows, block_rows=128):
    """Read required .npy members in fixed blocks; never materialize a run.

    ``shapes`` maps field names to the expected shape of one record. Arrays
    must contain finite float32/float64 values, with one common leading clock
    dimension. Object arrays, Fortran layout, duplicate ZIP entries and trailing
    member data are rejected. No files are extracted or written.
    """
    if (type(expected_rows) is not int or expected_rows<1 or type(block_rows) is not int
            or not 1<=block_rows<=1024 or not isinstance(shapes,dict) or not shapes):
        raise ValueError('Explicit positive stream length and bounded block required')
    with zipfile.ZipFile(Path(path),'r') as archive, ExitStack() as stack:
        entries=archive.namelist()
        if len(entries)!=len(set(entries)):raise ValueError('Duplicate NPZ member names')
        members={}
        for name,shape in shapes.items():
            if type(name) is not str or '/' in name or '\\' in name or name+'.npy' not in entries:
                raise ValueError('Complete unambiguous NPZ field required: '+str(name))
            member=stack.enter_context(archive.open(name+'.npy'))
            version=np.lib.format.read_magic(member)
            if version==(1,0):actual,fortran,dtype=np.lib.format.read_array_header_1_0(member)
            elif version==(2,0):actual,fortran,dtype=np.lib.format.read_array_header_2_0(member)
            else:raise ValueError('Unsupported numeric recording NPY format')
            shape=tuple(shape)
            if (actual!=(expected_rows,*shape) or fortran or dtype.kind!='f'
                    or dtype.itemsize not in (4,8) or dtype.hasobject):
                raise ValueError('Complete finite-float C-order archive field required: '+name)
            members[name]=(member,dtype,shape,int(np.prod(shape,dtype=int)) if shape else 1)
        for begin in range(0,expected_rows,block_rows):
            count=min(block_rows,expected_rows-begin);buffers={}
            for name,(member,dtype,shape,width) in members.items():
                needed=count*width*dtype.itemsize;raw=member.read(needed)
                if len(raw)!=needed:raise ValueError('Truncated NPZ member: '+name)
                values=np.frombuffer(raw,dtype=dtype).reshape(count,*shape)
                if not np.isfinite(values).all():raise ValueError('Nonfinite actual archive field: '+name)
                buffers[name]=values
            for index in range(count):
                yield {name:values[index].copy() for name,values in buffers.items()}
        if any(member.read(1) for member,_,_,_ in members.values()):
            raise ValueError('Unexpected trailing NPZ member data')
