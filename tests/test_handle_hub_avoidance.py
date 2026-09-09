import numpy as np
import pytest
from doorbench.dexterous.handle_hub_avoidance import avoidance_force


def test_avoidance_is_outward_bounded_and_vanishes_with_clearance():
    np.testing.assert_allclose(avoidance_force(.01,[1,0,0],[0,0,0]),0)
    np.testing.assert_allclose(avoidance_force(.003,[1,0,0],[0,0,0]),[.8,0,0])
    np.testing.assert_allclose(avoidance_force(-.001,[1,0,0],[-1,0,0]),[3,0,0])
    np.testing.assert_allclose(avoidance_force(.003,[1,0,0],[1,0,0]),0)
    with pytest.raises(ValueError):avoidance_force(0,[2,0,0],[0,0,0])


def test_distal_segment_cannot_hide_behind_clear_middle_segment():
    import mujoco
    from types import SimpleNamespace
    from doorbench.dexterous.handle_hub_avoidance import HandleHubAvoidance
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="analytic_handle_hub" type="sphere" size=".01"/>
      <body name="rh_lfmiddle" pos=".04 0 0"><joint type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".003"/>
        <body name="rh_lfdistal" pos="-.025 0 0"><geom type="sphere" size=".003"/></body>
      </body></worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    t=SimpleNamespace(m=m,d=d,fingers=np.array([0]),va=np.array([0]),finger_inverse=np.eye(1),caps=np.array([[-2.,2.]]))
    old=HandleHubAvoidance(t);whole=HandleHubAvoidance(t,include_distal=True)
    force,_=old.force(np.zeros(1),1.);np.testing.assert_array_equal(force,[0.])
    force,info=whole.force(np.zeros(1),1.)
    assert 0<force[0]<=2.
    assert info['hub_gap_m']==pytest.approx(.002)
    assert info['hub_avoidance_profile']=='whole-little-finger-3N-v1'
