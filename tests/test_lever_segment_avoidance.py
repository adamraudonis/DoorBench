import mujoco
import numpy as np
import pytest
from types import SimpleNamespace
from doorbench.dexterous.lever_segment_avoidance import LeverSegmentAvoidance


def test_clearance_uses_capped_motors_without_writing_physics_state():
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="analytic_lever_capsule" type="sphere" size=".01"/>
      <body name="rh_rfmiddle" pos=".015 0 0">
        <joint type="slide" axis="1 0 0"/><geom type="sphere" size=".003"/>
      </body></worldbody></mujoco>''')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    teacher = SimpleNamespace(m=m, d=d, fingers=np.array([0]), va=np.array([0]),
                              finger_inverse=np.eye(1), caps=np.array([[-.5, .5], [-1., 1.]]))
    controller = LeverSegmentAvoidance(teacher)
    before = [x.copy() for x in (d.qpos, d.qvel, d.qfrc_applied, d.xfrc_applied)]
    initial = np.array([0., .3])
    zero, _ = controller.force(initial, 0.)
    np.testing.assert_array_equal(zero, initial)
    force, info = controller.force(initial, 1.)
    assert force[0] == .5 and force[1] == .3
    assert info['release_segment_gap_m'] == pytest.approx(.002)
    assert 0 < info['release_segment_force_N'] <= 3
    for old, current in zip(before, (d.qpos, d.qvel, d.qfrc_applied, d.xfrc_applied)):
        np.testing.assert_array_equal(old, current)
    d.qpos[0] = .01
    mujoco.mj_forward(m, d)
    clear, info = controller.force(initial, 1.)
    np.testing.assert_array_equal(clear, initial)
    assert info['release_segment_force_N'] == 0
    with pytest.raises(ValueError): controller.force(initial, 1.1)
