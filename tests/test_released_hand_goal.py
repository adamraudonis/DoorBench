import numpy as np
import pytest
from doorbench.dexterous.released_hand_goal import released_hand_goal


def test_legacy_world_goal_is_fixed_and_root_goal_moves_rigidly():
    p=np.array([1.,0.,1.]);r=np.eye(3);root=np.array([0.,0.,1.]);delta=np.array([.1,.2,0.,0.,0.,np.pi/2]);out=np.array([0.,-1.,0.])
    actual,rotation=released_hand_goal(p,r,root,delta,out,1.)
    np.testing.assert_array_equal(actual,p);np.testing.assert_array_equal(rotation,r)
    actual,rotation=released_hand_goal(p,r,root,delta,out,1.,frame='root',retreat_m=.1)
    np.testing.assert_allclose(actual,[.1,1.1,1.],atol=1e-12)
    np.testing.assert_allclose(rotation@np.array([1.,0.,0.]),[0.,1.,0.],atol=1e-12)


def test_root_goal_has_exact_initial_position_and_rejects_ambiguous_retreat():
    p=np.array([1.,2.,3.]);r=np.eye(3);root=np.array([.5,1.,1.]);out=np.array([0.,-1.,0.])
    q,_=released_hand_goal(p,r,root,np.zeros(6),out,0.,frame='root',retreat_m=.1)
    np.testing.assert_array_equal(p,q)
    with pytest.raises(ValueError):released_hand_goal(p,r,root,np.zeros(6),out,1.,retreat_m=.1)
