"""Opposite-swing panic hardware retains its physical per-leaf access side."""
import pytest
from copy import deepcopy

from doorbench.build import build_model
from doorbench.spec import generate_all


@pytest.mark.parametrize('door',('db0112_swing_double','db0608_swing_double'))
def test_double_egress_panic_access_is_per_leaf_without_mirroring_hardware(door):
    spec=next(s for s in generate_all() if s['id']==door)
    model=build_model(spec)
    assert spec['kinematics']['double_egress']
    assert spec['robot']['approach_side']=='-y'
    a=model.body('leaf_a_exit_device');b=model.body('leaf_b_exit_device')
    assert a.joint.robot_interactive and not b.joint.robot_interactive
    # The bars remain on their respective pushing faces. B's real fixed pull
    # is accessible from the approach, but is not an actuator or latch link.
    assert next(s for s in a.sites if s.role=='push').pos[1]<0
    assert next(s for s in b.sites if s.role=='push').pos[1]>0
    leaf_b=model.body('leaf_b')
    pull=next(s for s in leaf_b.sites if s.name=='leaf_b_far_pull_grip_n')
    assert pull.pos[1]<0 and pull.role=='grip'
    assert model.meta['primary_joint']=='leaf_a_hinge'
    assert model.meta['operator_joint']=='leaf_a_exit_device_slide'
    opposite=deepcopy(spec);opposite['robot']['approach_side']='+y'
    reverse=build_model(opposite)
    assert not reverse.body('leaf_a_exit_device').joint.robot_interactive
    assert reverse.body('leaf_b_exit_device').joint.robot_interactive
