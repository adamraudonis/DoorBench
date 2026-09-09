"""Explicit bounded prefix of a separately admitted complete native episode."""
import numpy as np


class NativeDemonstrationPrefix:
    def __init__(self,source,seconds):
        if not np.isfinite(seconds) or seconds<=0:
            raise ValueError('Finite positive curriculum duration required')
        dt=source.metadata['physics_dt_s'];count=round(seconds/dt)
        if abs(count*dt-seconds)>1e-8 or count>len(source):
            raise ValueError('Prefix must cover consecutive whole steps inside the admitted episode')
        self.source=source;self.count=count
        for key in ('dimensions','layout','motor_contract_sha256'):
            setattr(self,key,getattr(source,key))
        self.times=source.times[:count+1]
        self.numeric={k:v[:count+1] for k,v in source.numeric.items()}
        self.metadata=dict(source.metadata,examples=count,admitted_prefix_seconds=float(seconds),
            qualification='bounded_prefix_of_qualified_native_task',
            complete_task_training_coverage=False,source_complete_examples=len(source))

    def __len__(self):return self.count

    def sequence(self,start,length):
        if start<0 or length<=0 or start+length>self.count:
            raise ValueError('Sequence crosses declared native curriculum boundary')
        return self.source.sequence(start,length)
