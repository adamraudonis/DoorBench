from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous.sensor_contract import (
    ACTOR_KEYS, SENSOR_KEYS, ActorObservationBuilder, AngularTaxelGrid,
    SensorEffects, rotation_xyzw, validate_actor_packet,
)
from doorbench.dexterous.isaac_sensors import (
    PhysXTaxelAdapter, TactileMount, _collapse_pairs, enqueue_robot_sensors,
)


def builder(**kw):
    return ActorObservationBuilder(joint_count=2, action_count=2, tactile_dimension=24, image_shape=(2, 3, 3), **kw)


def fill(sensor_builder, time=0):
    for key in SENSOR_KEYS:
        sensor_builder.push(key, np.ones(sensor_builder.shapes[key]), capture_s=time)


def test_tactile_local_frame_is_invariant_to_global_pose_and_sums_force():
    grid = AngularTaxelGrid()
    points = np.array([[.2, .3, -1], [.3, .4, -1], [-.3, -.2, -1]])
    forces = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
    baseline = grid.bin_forces(points, forces, [0, 0, 0], np.eye(3))
    rot = rotation_xyzw([0, 0, np.sin(.6), np.cos(.6)])
    translation = np.array([98., -4., 7.])
    moved = grid.bin_forces(points @ rot.T + translation, forces @ rot.T, translation, rot)
    np.testing.assert_allclose(baseline, moved)
    np.testing.assert_allclose(baseline.reshape(3, -1).sum(axis=1), forces.sum(axis=0)[[2, 0, 1]])
    assert np.count_nonzero(baseline.reshape(3, -1)[0]) == 2


def test_finite_angular_bins_discard_range_and_have_native_boundary_convention():
    grid = AngularTaxelGrid(2, 2, (45, 45))
    a = grid.bin_forces([[0, 0, -1]], [[1, 2, 3]], [0, 0, 0], np.eye(3))
    b = grid.bin_forces([[0, 0, -100]], [[1, 2, 3]], [0, 0, 0], np.eye(3))
    np.testing.assert_array_equal(a, b)
    assert a.reshape(3, 2, 2)[0, 0, 0] == 3
    assert not grid.bin_forces([[1, 0, 0]], [[1, 1, 1]], [0, 0, 0], np.eye(3)).any()


def test_native_mujoco_touch_grid_force_sign_and_bin_parity():
    """Compare against real solved native contact/plugin output, not our formula."""
    import mujoco
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <extension><plugin plugin="mujoco.sensor.touch_grid"/></extension>
      <worldbody><geom name="floor" type="plane" size="1 1 .1" friction="1 .005 .0001"/>
        <body name="finger" pos="0 0 .049"><freejoint/><geom name="pad" type="box" size=".05 .05 .05" mass="1"/>
          <site name="touch" pos=".01 .015 0" quat=".9238795 0 0 .3826834"/>
        </body>
      </worldbody>
      <sensor><plugin name="pad_touch" plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch">
        <config key="nchannel" value="3"/><config key="size" value="4 2"/>
        <config key="fov" value="180 90"/><config key="gamma" value="0"/>
      </plugin></sensor></mujoco>''')
    data = mujoco.MjData(model)
    for _ in range(250):
        mujoco.mj_step(model, data)
    # Add shear and solve at this state; test force channels as well as support.
    data.qvel[0] = .1
    mujoco.mj_forward(model, data)
    points, forces = [], []
    finger = model.body('finger').id
    for i, contact in enumerate(data.contact):
        b1, b2 = (model.geom_bodyid[g] for g in contact.geom)
        if finger not in (b1, b2):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        force_world = contact.frame.reshape(3, 3).T @ force[:3]
        if b1 == finger:
            force_world *= -1
        points.append(contact.pos.copy())
        forces.append(force_world)
    assert len(points) >= 1
    actual = AngularTaxelGrid().bin_forces(points, forces, data.site_xpos[0], data.site_xmat[0].reshape(3, 3))
    np.testing.assert_allclose(actual, data.sensordata, rtol=1e-6, atol=1e-6)
    assert np.linalg.norm(actual) > 1


def test_sensor_latency_camera_capture_age_dropout_and_episode_reset():
    b = builder(effects={'tactile': SensorEffects(5, delay_s=.05, max_age_s=.2)})
    b.push('tactile', np.full(24, 8.), capture_s=0.)
    b.push('rgb_left', np.full((2, 3, 3), 120, np.uint8), capture_s=0., available_s=.04)
    early = b.observe(now_s=.03, previous_action=[0, 0])
    assert not early['sensor_valid'].any()
    late = b.observe(now_s=.05, previous_action=[2, -2])
    assert late['tactile'].max() == 5
    assert late['rgb_left'].max() == 120
    assert late['sensor_time_s'][SENSOR_KEYS.index('rgb_left')] == 0
    np.testing.assert_array_equal(late['previous_action'], [1, -1])
    stale = b.observe(now_s=.4, previous_action=[0, 0])
    assert not stale['sensor_valid'].any() and not stale['tactile'].any()
    b.reset(seed=4)
    reset = b.observe(now_s=0, previous_action=[0, 0])
    assert not reset['sensor_valid'].any() and np.all(reset['sensor_time_s'] == -1)


def test_noise_reproducibility_clipping_and_dropout():
    effects = {'tactile': SensorEffects(2, noise_std=100), 'imu_gyro': SensorEffects(10, dropout_probability=1)}
    a, b = builder(effects=effects, seed=123), builder(effects=effects, seed=123)
    fill(a); fill(b)
    aa = a.observe(now_s=0, previous_action=[0, 0])
    bb = b.observe(now_s=0, previous_action=[0, 0])
    np.testing.assert_array_equal(aa['tactile'], bb['tactile'])
    assert abs(aa['tactile']).max() <= 2
    assert not aa['imu_gyro'].any() and not aa['sensor_valid'][SENSOR_KEYS.index('imu_gyro')]


@pytest.mark.parametrize('forbidden', ['door_q', 'handle_pose', 'object_id', 'contact_labels', 'world_contacts', 'gravity_body', 'goal', 'critic'])
def test_actor_boundary_rejects_privileged_fields(forbidden):
    b = builder()
    with pytest.raises(ValueError):
        b.push(forbidden, [1], capture_s=0)
    packet = b.observe(now_s=0, previous_action=[0, 0])
    packet[forbidden] = np.zeros(1)
    with pytest.raises(ValueError):
        validate_actor_packet(packet, b.shapes, 2)


def test_actor_arrays_are_detached_and_do_not_alias_simulation_or_internal_buffers():
    b = builder()
    source = np.array([.1, .2])
    b.push('joint_position', source, capture_s=0)
    source[:] = 9
    packet = b.observe(now_s=0, previous_action=[0, 0])
    assert set(packet) == ACTOR_KEYS
    assert all(type(value) is np.ndarray and value.dtype != object for value in packet.values())
    np.testing.assert_allclose(packet['joint_position'], [.1, .2])
    packet['joint_position'][:] = 8
    np.testing.assert_allclose(b.observe(now_s=0, previous_action=[0, 0])['joint_position'], [.1, .2])


def test_robot_adapter_never_reads_pose_gravity_or_privileged_state():
    class RestrictedData:
        def __getattr__(self, key):
            raise AssertionError(f'Forbidden simulator channel read: {key}')
    robot, imu = RestrictedData(), RestrictedData()
    robot.joint_pos, robot.joint_vel = np.zeros((1, 2)), np.ones((1, 2))
    imu.ang_vel_b, imu.lin_acc_b = np.zeros((1, 3)), np.array([[0, 0, 9.81]])
    b = builder()
    enqueue_robot_sensors(b, robot, imu, np.ones(24), joint_indices=[1, 0], capture_s=.01)
    packet = b.observe(now_s=.01, previous_action=[0, 0])
    np.testing.assert_allclose(packet['imu_accelerometer'], [0, 0, 9.81])


def test_invalid_times_nonfinite_and_malformed_contact_buffers_fail_closed():
    b = builder()
    with pytest.raises(ValueError):
        b.push('joint_position', [0, np.nan], capture_s=0)
    with pytest.raises(ValueError):
        b.push('joint_position', [0, 0], capture_s=1, available_s=.5)
    b.push('joint_position', [0, 0], capture_s=0)
    with pytest.raises(ValueError):
        b.push('joint_position', [0, 0], capture_s=0)
    with pytest.raises(ValueError):
        rotation_xyzw([1, 2, 3, 4])
    with pytest.raises(RuntimeError):
        _collapse_pairs(np.zeros((2, 3)), np.zeros((2, 3)), [[2]], [[0]], capacity=2)


def test_physx_counterparts_collapse_and_shared_getter_buffers_are_copied(monkeypatch):
    import doorbench.dexterous.isaac_sensors as module
    monkeypatch.setattr(module, 'all_scene_contact_paths', lambda stage: ('/Robot/finger', '/random_a', '/random_b'))
    shared_counts = np.array([[0, 1, 1]], dtype=np.int32)
    shared_starts = np.array([[0, 0, 1]], dtype=np.int32)
    class ContactView:
        sensor_count, filter_count = 1, 3
        sensor_paths = ['/Robot/finger']
        filter_paths = [['/Robot/finger', '/random_a', '/random_b']]
        def get_contact_data(self, dt):
            assert dt == .002
            shared_counts[:] = [[0, 1, 1]]
            shared_starts[:] = [[0, 0, 1]]
            return (np.array([[2], [3], [0], [0.]]),
                    np.array([[.1, .1, -1], [.1, .1, -1], [0, 0, 0], [0, 0, 0]]),
                    np.tile([0, 0, 1.], (4, 1)), np.zeros((4, 1)), shared_counts, shared_starts)
        def get_friction_data(self, dt):
            # Reuse counts/starts, deliberately changing their contents.
            shared_counts[:] = [[0, 1, 0]]
            shared_starts[:] = [[0, 0, 0]]
            return (np.array([[1., 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                    np.array([[.1, .1, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]), shared_counts, shared_starts)
    class PhysicsView:
        def create_rigid_body_view(self, paths):
            return SimpleNamespace(prim_paths=paths, get_transforms=lambda: np.array([[0, 0, 0, 0, 0, 0, 1.]]))
        def create_rigid_contact_view(self, paths, *, filter_patterns, max_contact_data_count):
            assert filter_patterns == [['/Robot/finger', '/random_a', '/random_b']]
            return ContactView()
    adapter = PhysXTaxelAdapter(PhysicsView(), None,
        [TactileMount('/Robot/finger', (0, 0, 0), (0, 0, 0, 1), AngularTaxelGrid())], capacity=4)
    result = adapter.read(physics_dt=.002).reshape(3, -1)
    np.testing.assert_allclose(result.sum(axis=1), [5, 1, 0])


def test_scene_contact_selection_includes_self_dynamic_and_static_without_roles():
    Usd = pytest.importorskip('pxr.Usd')
    from pxr import UsdGeom, UsdPhysics
    from doorbench.dexterous.isaac_sensors import all_scene_contact_paths
    stage = Usd.Stage.CreateInMemory()
    for name in ('x', 'y'):
        body = UsdGeom.Xform.Define(stage, '/World/'+name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        for shape in ('first', 'second'):
            prim = UsdGeom.Cube.Define(stage, '/World/'+name+'/'+shape).GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
    floor = UsdGeom.Cube.Define(stage, '/World/z').GetPrim()
    UsdPhysics.CollisionAPI.Apply(floor)
    disabled = UsdGeom.Cube.Define(stage, '/World/disabled').GetPrim()
    UsdPhysics.CollisionAPI.Apply(disabled).CreateCollisionEnabledAttr(False)
    assert all_scene_contact_paths(stage) == ('/World/x', '/World/y', '/World/z')


def test_camera_adapter_discards_rgba_and_does_not_access_pose_or_semantic_data():
    from doorbench.dexterous.isaac_sensors import enqueue_camera
    class Camera:
        output = {'rgb': np.full((1, 2, 3, 4), 37, dtype=np.uint8)}
        def __getattr__(self, key):
            raise AssertionError(f'Forbidden camera field: {key}')
    b = builder()
    enqueue_camera(b, 'rgb_right', Camera(), capture_s=.01, available_s=.03)
    early = b.observe(now_s=.02, previous_action=[0, 0])
    assert not early['rgb_right'].any()
    late = b.observe(now_s=.03, previous_action=[0, 0])
    assert np.all(late['rgb_right'] == 37)
    assert late['sensor_time_s'][SENSOR_KEYS.index('rgb_right')] == .01
