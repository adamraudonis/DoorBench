"""Reject invalid robot initialization before writing any simulator state."""
import numpy as np


def check_joint_reset(names, values, limits, *, tolerance=1e-5):
    values=np.asarray(values,dtype=float);limits=np.asarray(limits,dtype=float)
    if values.shape!=(len(names),) or limits.shape!=(len(names),2):
        raise ValueError('Reset joint arrays have inconsistent dimensions')
    if not np.isfinite(values).all() or np.isnan(limits).any() or np.any(limits[:,0]>limits[:,1]):
        raise ValueError('Reset contains nonfinite positions or invalid joint limits')
    bad=[n for n,q,(lo,hi) in zip(names,values,limits) if q<lo-tolerance or q>hi+tolerance]
    if bad:raise ValueError('Reset exceeds physical joint limits: '+', '.join(bad))
