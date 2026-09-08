import numpy as np
import pytest

from doorbench.dexterous.isaac_opening_measurements import (
    OpeningGeometryMeasurements, contact_force_pairs, panel_surface_loads,
)


def test_surface_load_uses_exact_panel_and_actual_palm_separately():
    paths = ['/World/H1/lh_palm', '/World/H1/lh_ffdistal', '/World/H1/rh_palm']
    filters = np.tile(['/World/Door/Articulation/leaf', '/World/floor'], (3, 1))
    forces = np.zeros((3, 2, 3))
    forces[0, 0] = [1, -2, 3]
    forces[1, 0, 1] = -4
    forces[0, 1, 1] = -100
    forces[2, 0, 1] = -100
    r = panel_surface_loads(paths, filters, forces, [0, 0, 0, 1, 0, 0, 0])
    assert r['palm_normal_load_N'] == 2
    assert r['total_normal_load_N'] == 6
    forces[0, 0, 1] = 2
    assert panel_surface_loads(paths, filters, forces, [0, 0, 0, 1, 0, 0, 0])['palm_normal_load_N'] == 0


def test_panel_rotates_with_actual_door_pose():
    q = [0, 0, 0, np.sqrt(.5), 0, 0, np.sqrt(.5)]
    r = panel_surface_loads(['/World/H1/lh_palm'], [['/World/Door/Articulation/leaf']],
                            [[[3, 0, 0]]], q)
    assert r['palm_normal_load_N'] == pytest.approx(3)


def test_missing_or_duplicate_surface_cannot_supply_qualification():
    with pytest.raises(ValueError, match='exactly once'):
        panel_surface_loads(['/World/H1/lh_palm'], [['floor']], [[[0, -3, 0]]], [0, 0, 0, 1, 0, 0, 0])
    with pytest.raises(ValueError, match='palm'):
        panel_surface_loads(['/World/H1/lh_ffdistal'], [['/World/Door/Articulation/leaf']],
                            [[[0, -3, 0]]], [0, 0, 0, 1, 0, 0, 0])


def test_separate_friction_buffers_preserve_surface_pairs():
    normal = np.zeros((2, 2, 3))
    friction = np.array([[1., 2, 3], [4, 5, 6], [7, 8, 9], [np.nan]*3])
    result = contact_force_pairs(normal, friction, np.array([[2, 0], [0, 1]]),
                                 np.array([[0, 0], [0, 2]]), capacity=4)
    np.testing.assert_array_equal(result[0, 0], [5, 7, 9])
    np.testing.assert_array_equal(result[1, 1], [7, 8, 9])
    assert not normal.any()
    with pytest.raises(ValueError, match='Overlapping'):
        contact_force_pairs(normal, friction, np.array([[2, 0], [0, 1]]),
                            np.array([[0, 0], [0, 1]]), capacity=4)
    with pytest.raises(ValueError, match='truncated'):
        contact_force_pairs(normal, friction, np.array([[2, 0], [0, 2]]),
                            np.array([[0, 0], [0, 2]]), capacity=4)


@pytest.fixture
def calculator(tmp_path):
    pytest.importorskip('mujoco')
    door = tmp_path / 'door.xml'
    robot = tmp_path / 'robot.xml'
    door.write_text('''<mujoco><worldbody><body name="leaf">
      <joint name="leaf_hinge" type="hinge" axis="0 0 1"/>
      <geom type="box" size=".4 .02 1"/>
      <body name="leaf_handle" pos=".25 -.05 0">
       <joint name="leaf_handle_hinge" axis="0 1 0"/>
       <geom name="leaf_handle_lever_col_n" type="capsule" size=".01 .07"/>
      </body><body name="bolt"><joint name="leaf_latch_bolt_slide" type="slide"/>
       <geom size=".01"/></body></body></worldbody></mujoco>''')
    robot.write_text('''<mujoco><worldbody><body name="pelvis"><freejoint name="free_base"/>
      <geom size=".1"/><body name="rh_palm" pos=".2 0 .5">
      <joint name="wrist"/><geom name="palm" type="box" size=".03 .02 .05" contype="0" conaffinity="1"/>
      <geom name="visual" type="box" size=".01 .01 .01" contype="0" conaffinity="0"/>
      <site name="rh_palm_touch" pos="0 0 .05"/>
      </body></body></worldbody></mujoco>''')
    return OpeningGeometryMeasurements(door, robot, ['wrist'])


def measured_fixture(c):
    m, d, mj = c.m, c.d, c.mujoco
    d.qpos[c.root:c.root+7] = [0, -1.5, 1, 1, 0, 0, 0]
    mj.mj_kinematics(m, d)
    poses = {m.body(b).name.removeprefix('robot/'): np.r_[d.xpos[b], d.xquat[b]]
             for b in c.bodies + [c.site_body]}
    hp, lp = [np.r_[d.xpos[m.body(n).id], d.xquat[m.body(n).id]] for n in ('leaf_handle', 'leaf')]
    return dict(time_s=.002, pose_time_s=.002, root=d.qpos[c.root:c.root+7].copy(),
                joints={'wrist': 0.}, angles={'operator': 0., 'leaf': 0., 'latch': 0.},
                body_poses=poses, handle_pose=hp, leaf_pose=lp)


def test_geometry_does_not_step_and_uses_actual_site_body_pose(calculator):
    c = calculator
    args = measured_fixture(c)
    result = c.read(**args)
    assert c.d.time == 0
    assert result['native_mirror_steps'] == 0
    assert result['right_lever_clearance_m'] > .02
    np.testing.assert_allclose(result['right_palm_pose'][:3], [.2, -1.5, 1.55])
    assert result['maximum_pose_position_error_m'] < 1e-10
    assert [c.m.geom(g).name for g in c.geoms]==['robot/palm']


def test_geometry_rejects_stale_or_inconsistent_measurements(calculator):
    c = calculator
    args = measured_fixture(c)
    with pytest.raises(ValueError, match='clock'):
        c.read(**{**args, 'pose_time_s': 0.})
    args['body_poses']['rh_palm'][0] += .01
    with pytest.raises(ValueError, match='disagree'):
        c.read(**args)
