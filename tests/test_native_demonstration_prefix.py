import numpy as np
import pytest
from doorbench.dexterous.native_demonstration_prefix import NativeDemonstrationPrefix


class Source:
    dimensions=None;layout={};motor_contract_sha256='original'
    times=np.arange(101)*.002;numeric={'previous_action':np.zeros((101,61))}
    metadata={'physics_dt_s':.002,'examples':100,'qualification':'full'}
    def __len__(self):return 100
    def sequence(self,start,length):return start,length


def test_prefix_preserves_actual_zero_and_explicit_coverage():
    source=Source();p=NativeDemonstrationPrefix(source,.1)
    assert len(p)==50 and len(p.times)==51 and p.times[0]==0
    assert p.sequence(49,1)==(49,1)
    assert p.metadata['complete_task_training_coverage'] is False
    assert source.metadata['qualification']=='full'
    with pytest.raises(ValueError,match='boundary'):p.sequence(49,2)


@pytest.mark.parametrize('seconds',[.201,.5,0.,float('nan'),float('inf')])
def test_invalid_curriculum_boundary(seconds):
    with pytest.raises(ValueError):NativeDemonstrationPrefix(Source(),seconds)
