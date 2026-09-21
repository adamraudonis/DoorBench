"""Reference-only thumb correction; no contact or physical success is inferred."""
from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous.operation_teacher import DoorOperationTeacher
from test_operation_teacher import Acquisition, GEOMETRY, tick


def fixture(**options):
    acq=Acquisition()
    acq.names=['rh_THJ5','rh_THJ4','rh_THJ3','rh_THJ2','rh_THJ1','rh_FFJ3']
    acq.path=np.array([[.2,.3,.1,-.2,-.1,.5]])
    acq.m=SimpleNamespace(joint=lambda name:SimpleNamespace(range=np.array([-.5,.5])))
    wrapper=DoorOperationTeacher(acq,GEOMETRY,wait_for_press_completion=False,**options)
    for t in np.arange(0.,.51,.01):tick(wrapper,t)
    return acq,wrapper


def test_thumb_waits_for_measured_rest_then_moves_only_named_reference_smoothly():
    acq,wrapper=fixture(thumb_reference_joint='rh_THJ5',thumb_reference_offset_rad=-.04)
    original=acq.path.copy()
    for t,angles in [(1.,dict(operator=.81,latch=.012,leaf=.08)),
                     (1.1,dict(operator=.04,latch=.0005,leaf=.07)),
                     (1.2,dict(operator=.06,latch=.0005,leaf=.08)),
                     (1.3,dict(operator=.04,latch=.002,leaf=.08))]:
        tick(wrapper,t,**angles)
        assert wrapper.thumb_reference_started is None
        np.testing.assert_array_equal(acq.path,original)
    tick(wrapper,1.4,operator=.04,latch=.0005,leaf=.08)
    assert wrapper.thumb_reference_started==1.4
    np.testing.assert_array_equal(acq.path,original)
    tick(wrapper,1.9,operator=0.,latch=0.,leaf=.08)
    assert acq.path[0,0]==pytest.approx(.18)
    np.testing.assert_array_equal(acq.path[0,1:],original[0,1:])
    force,_=tick(wrapper,2.4,operator=0.,latch=0.,leaf=.08)
    assert acq.path[0,0]==pytest.approx(.16)
    np.testing.assert_array_equal(force,np.arange(6.))
    tick(wrapper,4.,operator=0.,latch=0.,leaf=.08)
    assert acq.path[0,0]==pytest.approx(.16)


@pytest.mark.parametrize('joint,offset',[(None,.01),('left_ankle',.01),('rh_THJ5',float('nan')),
                                       ('rh_THJ5',float('inf')),('rh_THJ5',.101)])
def test_thumb_requires_bounded_finite_named_reference(joint,offset):
    with pytest.raises(ValueError,match='right-thumb'):
        fixture(thumb_reference_joint=joint,thumb_reference_offset_rad=offset)


def test_thumb_target_cannot_exceed_authored_joint_range():
    acq,_=fixture()
    acq.path[0,0]=.49
    with pytest.raises(ValueError,match='original authored joint limits'):
        DoorOperationTeacher(acq,GEOMETRY,thumb_reference_joint='rh_THJ5',thumb_reference_offset_rad=.02)


def test_default_retains_all_existing_reference_targets():
    acq,wrapper=fixture()
    original=acq.path.copy()
    tick(wrapper,1.,operator=.81,latch=.012,leaf=.08)
    tick(wrapper,3.,operator=0.,latch=0.,leaf=.08)
    np.testing.assert_array_equal(acq.path,original)
    assert wrapper.thumb_reference_started is None
