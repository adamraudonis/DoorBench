"""One synchronized host copy per device for heterogeneous PhysX snapshots."""
import numpy as np


def host_snapshot(tensors):
    """Return independent NumPy arrays preserving every supported scalar value.

    PhysX can reuse its tensor buffers after the next getter. Packing and copying
    finishes before returning; no output aliases the solver or another field.
    Float64 transport represents native float32 values and bounded contact
    indices exactly. Integer values with absolute value at least 2**53 are rejected.
    """
    import torch
    tensors=list(tensors)
    if not tensors:return []
    types={torch.float32:np.float32,torch.float64:np.float64,torch.int32:np.int32,torch.int64:np.int64,torch.bool:np.bool_}
    if any(x.dtype not in types for x in tensors):raise ValueError('Unsupported snapshot dtype')
    devices=list(dict.fromkeys(x.device for x in tensors))
    if len(devices)>1:
        result=[None]*len(tensors)
        for device in devices:
            indices=[i for i,x in enumerate(tensors) if x.device==device]
            for index,value in zip(indices,host_snapshot([tensors[i] for i in indices])):result[index]=value
        return result
    sizes=[x.numel() for x in tensors]
    packed=torch.cat([x.reshape(-1).to(dtype=torch.float64) for x in tensors]).detach().cpu().numpy()
    result=[];offset=0
    for tensor,size in zip(tensors,sizes):
        values=packed[offset:offset+size];offset+=size
        if tensor.dtype in (torch.int32,torch.int64) and np.any(np.abs(values)>=2**53):raise ValueError('Integer snapshot exceeds exact transport range')
        result.append(values.astype(types[tensor.dtype],copy=True).reshape(tuple(tensor.shape)))
    return result
