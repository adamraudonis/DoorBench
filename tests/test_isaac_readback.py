"""PhysX host snapshots must survive solver buffer reuse without value changes."""
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from doorbench.dexterous.isaac_readback import host_snapshot


def test_snapshot_preserves_mixed_solver_values_and_does_not_alias():
    original=[torch.tensor([[.1,-3.5],[float('inf'),float('nan')]],dtype=torch.float32).T,
              torch.tensor([0,8191,2**53-1],dtype=torch.int64),
              torch.tensor([True,False]),torch.tensor(.1,dtype=torch.float64),
              torch.empty((0,3),dtype=torch.float32)]
    expected=[x.numpy().copy() for x in original]
    result=host_snapshot(original)
    for before,after in zip(expected,result):
        np.testing.assert_array_equal(before,after)
        assert before.shape==after.shape and before.dtype==after.dtype
    for value in original:value.zero_()
    for before,after in zip(expected,result):np.testing.assert_array_equal(before,after)


@pytest.mark.parametrize('value',[2**53,2**53+1,-2**53-1])
def test_snapshot_rejects_integer_rounding(value):
    with pytest.raises(ValueError,match='exact transport'):
        host_snapshot([torch.tensor([value],dtype=torch.int64)])


def test_snapshot_rejects_unsupported_scalar_type():
    with pytest.raises(ValueError,match='dtype'):
        host_snapshot([torch.tensor([1.],dtype=torch.float16)])


def test_empty_snapshot():
    assert host_snapshot([])==[]
