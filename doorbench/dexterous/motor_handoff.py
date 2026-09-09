"""Bumpless controller switching inside the original motor force limits."""
import numpy as np
from .operation_teacher import smooth_phase


class MotorHandoff:
    def __init__(self, previous, first, caps, seconds):
        previous=np.asarray(previous,float);first=np.asarray(first,float)
        self.caps=np.asarray(caps,float).copy()
        if (previous.ndim!=1 or first.shape!=previous.shape or self.caps.shape!=(len(first),2)
            or not np.isfinite(np.r_[previous,first,self.caps.ravel(),seconds]).all()
            or not 0<seconds<=2 or np.any(self.caps[:,0]>=self.caps[:,1])
            or np.any(previous<self.caps[:,0]) or np.any(previous>self.caps[:,1])):
            raise ValueError('Finite previous motor command, original caps and bounded handoff required')
        self.offset=previous-first;self.seconds=seconds;self.last_elapsed=0.

    def force(self, command, elapsed):
        command=np.asarray(command,float)
        if command.shape!=self.offset.shape or not np.isfinite(np.r_[command,elapsed]).all() or elapsed<self.last_elapsed:
            raise ValueError('Finite motor command and monotonic handoff clock required')
        self.last_elapsed=elapsed
        return np.clip(command+(1-float(smooth_phase(elapsed/self.seconds)))*self.offset,
                       self.caps[:,0],self.caps[:,1])
