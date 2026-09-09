import importlib.util
from pathlib import Path
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('screen',Path(__file__).resolve().parents[1]/'scripts/dexterous/screen_whole_body_panel.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_barrier_preserves_interior_and_normalizes_foot_weight():
    hand=np.zeros(12);foot=np.zeros(12)
    np.testing.assert_array_equal(m.pose_constraint_barrier(hand,foot,20),np.zeros(8))
    hand[3]=10*.0009;foot[3]=20*.0009
    b=m.pose_constraint_barrier(hand,foot,20)
    assert b[1]==pytest.approx(.1)
    assert b[5]==pytest.approx(b[1])
    hand[0]=100*.00009
    assert m.pose_constraint_barrier(hand,foot,20)[0]==pytest.approx(.1)


def test_barrier_rejects_nonfinite_pose_evidence():
    h=np.zeros(12);h[0]=float('nan')
    with pytest.raises(ValueError):m.pose_constraint_barrier(h,np.zeros(12),20)
