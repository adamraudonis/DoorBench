import numpy as np
import pytest
from scripts.dexterous.audit_native_handle_assembly import extra_handle_patch_indices


def test_good_lever_load_cannot_hide_simultaneous_hub_load():
    pair=[[1,2],[1,2],[2,1],[1,3],[4,2],[1,2]]
    geoms=[[10,20],[11,20],[20,11],[11,30],[40,20],[11,20]]
    bad=extra_handle_patch_indices(pair,geoms,[5,2,3,4,5,0],handle=1,lever=10,hands=[2])
    np.testing.assert_array_equal(bad,[1,2])


def test_nonfinite_or_missing_patch_load_cannot_disappear():
    for force in ([float('nan')],[-1],[]):
        with pytest.raises(ValueError):extra_handle_patch_indices([[1,2]],[[11,20]],force,handle=1,lever=10,hands=[2])
