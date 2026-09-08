"""Portable continuation rejects stale measurements and malformed contracts."""
import numpy as np
import pytest
from doorbench.dexterous.post_opening_teacher import contact_contract,numeric_mapping,pose_parts


def evidence():
    return dict(physics_qualified=True,left_hand_contacts=0,left_hand_load_N=0.,right_environment_contacts=0)


def test_nonzero_global_start_accepts_only_the_actual_previous_interval():
    interval,feet=contact_contract(42.,42.,[41.998,42.],evidence(),[240.,250.],None,.002)
    np.testing.assert_array_equal(interval,[41.998,42.]);np.testing.assert_array_equal(feet,[240,250])
    with pytest.raises(ValueError,match='preceding'):
        contact_contract(42.,42.,[42.,42.],evidence(),[240,250],None,.002)


@pytest.mark.parametrize('clock,pose,interval,last',[
    (1.,.998,[.998,1.],.998),
    (1.,1.,[.996,.998],.998),
    (1.,1.,[.998,1.],1.),
    (1.,1.,[.998,1.],.996),
    (float('nan'),1.,[.998,1.],.998),
])
def test_mixed_pose_contact_or_controller_clocks_are_rejected(clock,pose,interval,last):
    with pytest.raises(ValueError):contact_contract(clock,pose,interval,evidence(),[240,250],last,.002)


@pytest.mark.parametrize('change',[
    {'physics_qualified':False},{'physics_qualified':1},
    {'left_hand_contacts':True},{'left_hand_contacts':-1},
    {'right_environment_contacts':1.2},{'left_hand_load_N':float('nan')},
    {'left_hand_load_N':-1.}, {'unrelated_oracle_field':0},
])
def test_explicit_active_safety_and_count_contract_is_required(change):
    data=evidence();data.update(change)
    with pytest.raises(ValueError):contact_contract(0.,0.,[0.,0.],data,[240,250],None,.002)


def test_names_determine_order_and_missing_extra_values_fail_closed():
    np.testing.assert_array_equal(numeric_mapping({'b':2.,'a':1.},['a','b'],'test'),[1,2])
    for values in ({'a':1.},{'a':1.,'b':2.,'c':3.},{'a':1.,'b':np.nan}):
        with pytest.raises(ValueError):numeric_mapping(values,['a','b'],'test')


def test_nonunit_or_malformed_pose_never_silently_normalizes():
    pos,rotation=pose_parts([1,2,3,1,0,0,0]);np.testing.assert_array_equal(pos,[1,2,3]);np.testing.assert_array_equal(rotation,np.eye(3))
    for pose in ([1,2,3,2,0,0,0],[1,2,3,1.000003,0,0,0],[1,2,3,0,0,0,0],[1,2,3,1,0,0,float('nan')],[1,2,3]):
        with pytest.raises(ValueError):pose_parts(pose)
