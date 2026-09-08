"""Small native geometry fixtures for acquisition route screening regressions."""
import mujoco
import numpy as np
import pytest

from doorbench.dexterous.grasp_route import (
    approach_clearance_requirement,
    hand_lever_clearance_failures,
)


def slider():
    m = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
<worldbody><geom name="lever" type="capsule" size=".01 .05"/>
<body name="finger"><joint type="slide" axis="1 0 0"/>
<geom name="finger" type="sphere" size=".01" mass=".1"/></body></worldbody></mujoco>''')
    return m, mujoco.MjData(m)


def failures(m, d, fraction=.5, terminal_gap=-.0001):
    required = approach_clearance_requirement(fraction, .006, terminal_gap)
    return hand_lever_clearance_failures(m, d, [m.geom('finger').id], m.geom('lever').id, required)


def test_shallow_precontact_must_fail_even_when_old_depth_gate_passes():
    m, d = slider()
    d.qpos[0] = .0199
    mujoco.mj_forward(m, d)
    # Exactly the former false acceptance: 0.1mm penetration is below the old
    # 1mm rejection depth, but this is far from intended terminal seating.
    assert d.ncon and all(c.dist >= -.001 for c in d.contact[:d.ncon])
    result = failures(m, d)
    assert len(result) == 1
    assert result[0]['gap_m'] == pytest.approx(-.0001, abs=1e-8)


def test_terminal_soft_contact_is_anchored_without_an_unchecked_interval():
    m, d = slider()
    d.qpos[0] = .0199
    mujoco.mj_forward(m, d)
    assert not failures(m, d, fraction=0.)
    assert approach_clearance_requirement(0., .006, -.0001) == -.0001
    assert approach_clearance_requirement(.1, .006, -.0001) == pytest.approx(.00295)
    assert approach_clearance_requirement(1., .006, -.0001) == .006
    assert failures(m, d, fraction=.1)
    # Even extremely close to the terminal pose, an additional overlap must
    # not be accepted by skipping the first path interval.
    d.qpos[0] = .0195
    mujoco.mj_forward(m, d)
    assert failures(m, d, fraction=.0001)


def test_clear_endpoints_can_hide_an_unsafe_but_nonpenetrating_interpolation():
    m = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
<worldbody><geom name="lever" type="capsule" size=".01 .05"/>
<body name="finger" pos=".071 0 0"><joint type="hinge" axis="0 0 1"/>
<geom name="finger" type="sphere" pos="-.05 0 0" size=".01" mass=".1"/>
</body></worldbody></mujoco>''')
    d = mujoco.MjData(m)
    for q in (-.6, .6):
        d.qpos[0] = q
        mujoco.mj_forward(m, d)
        assert not failures(m, d)
    d.qpos[0] = 0.
    mujoco.mj_forward(m, d)
    assert all(c.dist >= 0. for c in d.contact[:d.ncon])
    result = failures(m, d)
    assert result and result[0]['gap_m'] == pytest.approx(.001, abs=1e-8)


def test_exact_touch_cannot_satisfy_positive_approach_clearance():
    m, d = slider()
    d.qpos[0] = .02
    mujoco.mj_forward(m, d)
    result = failures(m, d)
    assert result and result[0]['gap_m'] == pytest.approx(0., abs=1e-8)


def test_nonfinite_requirements_and_empty_surfaces_cannot_pass():
    m, d = slider()
    with pytest.raises(ValueError, match='finite'):
        approach_clearance_requirement(.5, np.nan)
    with pytest.raises(ValueError, match='geometry'):
        hand_lever_clearance_failures(m, d, [], m.geom('lever').id, .006)
