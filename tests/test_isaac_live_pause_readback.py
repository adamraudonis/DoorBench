from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from doorbench.dexterous.isaac_live_pause_readback import (
    RAW_ARTICULATION_GETTERS, ARTICULATION_PROPERTY_GETTERS, PhysicsStepCounter, articulation_inventory,
    contact_inventory, controller_inventory, capture_native_pause_anchor,
)


def test_exact_controller_inventory_retains_aliases_identity_and_float_bits():
    array = np.array([0., 1.])
    controller = SimpleNamespace(previous=array, alias=array, counter=4, recursive=None)
    controller.recursive = controller
    roots = dict(controller=controller)
    initial = controller_inventory(roots)
    assert initial == controller_inventory(roots)
    array[0] = -0.
    assert initial != controller_inventory(roots)
    array[0] = 0.
    assert initial == controller_inventory(roots)
    controller.previous = array.copy()
    assert initial != controller_inventory(roots)
    controller.previous = array
    controller.counter += 1
    assert initial != controller_inventory(roots)


def test_native_mujoco_state_is_read_only_and_changed_native_values_detected():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    roots = dict(model=model, data=data)
    qpos = data.qpos.copy()
    original = controller_inventory(roots)
    assert original == controller_inventory(roots)
    assert data.time == 0.
    assert np.array_equal(qpos, data.qpos)
    data.qpos[0] += .001
    assert original != controller_inventory(roots)
    data.qpos[:] = qpos
    assert original == controller_inventory(roots)
    model.body_mass[1] += .001
    assert original != controller_inventory(roots)


def test_actual_scipy_interpolator_is_fingerprinted_without_evaluation():
    rotations = Rotation.from_euler('z', [0., .2])
    slerp = Slerp([0., 1.], rotations)
    roots = dict(slerp=slerp, rotation=rotations)
    assert controller_inventory(roots) == controller_inventory(roots)
    original = controller_inventory(roots)
    slerp.times[1] = 2.
    assert original != controller_inventory(roots)


@pytest.mark.parametrize('value', [object(), lambda:None, np.array([object()], dtype=object)])
def test_unknown_state_is_not_silently_excluded(value):
    with pytest.raises(ValueError):
        controller_inventory(dict(value=value))


class Backend:
    def __init__(self):
        self.values = {name:np.zeros((1, 3), dtype=np.float32) for name in
            (*RAW_ARTICULATION_GETTERS,*ARTICULATION_PROPERTY_GETTERS)}
        self.calls = []

    def __getattr__(self, name):
        if name not in self.values:
            raise AttributeError(name)
        def read():
            self.calls.append(name)
            return self.values[name]
        return read


TimestampedBuffer = type('TimestampedBuffer', (), {'__module__':'isaaclab.utils.buffers.timestamped_buffer'})


class LazyData:
    @property
    def root_state_w(self):
        raise AssertionError('No lazy property may be requested while paused')


def articulation():
    buffer = TimestampedBuffer()
    buffer.timestamp = .002
    buffer.data = np.array([[1., 2.]])
    data = LazyData()
    data._sim_timestamp = .002
    data._root_state_w = buffer
    data._root_physx_view = object()
    data._physics_sim_view = object()
    data.joint_names = ['a', 'b']
    value = SimpleNamespace(root_physx_view=Backend(), _data=data)
    for name in ('_joint_pos_target_sim', '_joint_vel_target_sim', '_joint_effort_target_sim'):
        setattr(value, name, np.array([1., 2.]))
    for name in ('_instantaneous_wrench_composer', '_permanent_wrench_composer'):
        setattr(value, name, SimpleNamespace(_active=False, _link_poses_updated=False,
            _composed_force_b_torch=np.zeros((1, 2, 3)), _composed_torque_b_torch=np.zeros((1, 2, 3))))
    return value


def test_articulation_reads_backend_and_existing_caches_without_lazy_update():
    value = articulation()
    result = articulation_inventory(value)
    assert value.root_physx_view.calls == [*RAW_ARTICULATION_GETTERS,*ARTICULATION_PROPERTY_GETTERS]
    value.root_physx_view.values['get_dof_positions'][0, 0] = 2.
    value._data._root_state_w.data[0, 0] = 7.
    value._joint_effort_target_sim[0] = 8.
    assert result['backend']['get_dof_positions'][0, 0] == 0.
    assert result['cache']['_root_state_w']['data'][0, 0] == 1.
    assert result['submission_targets']['_joint_effort_target_sim'][0] == 1.


@pytest.mark.parametrize('getter',ARTICULATION_PROPERTY_GETTERS)
def test_physical_property_changes_are_copied_and_detected(getter):
    value=articulation();before=articulation_inventory(value)
    value.root_physx_view.values[getter][0,0]=1.
    after=articulation_inventory(value)
    assert before['backend'][getter][0,0]==0.
    assert after['backend'][getter][0,0]==1.


def test_new_unsupported_cache_and_missing_submission_buffer_reject():
    value = articulation()
    value._data.new_state = object()
    with pytest.raises(ValueError, match='new_state'):
        articulation_inventory(value)
    del value._data.new_state
    del value._joint_effort_target_sim
    with pytest.raises(ValueError, match='submission targets'):
        articulation_inventory(value)


@pytest.mark.parametrize('field', ['_active', '_link_poses_updated', '_composed_force_b_torch', '_composed_torque_b_torch'])
def test_pause_cannot_hide_pending_direct_wrenches(field):
    value = articulation()
    composer = value._permanent_wrench_composer
    if field.endswith('_torch'):
        getattr(composer, field)[0, 0, 0] = .01
    else:
        setattr(composer, field, True)
    with pytest.raises(ValueError, match='zero-wrench'):
        articulation_inventory(value)


class ContactView:
    def __init__(self):
        self.counts = np.array([[0, 2]], dtype=np.int32)
        self.starts = np.array([[0, 1]], dtype=np.int32)
        self.values = np.arange(12, dtype=np.float32).reshape(4, 3)
        self.calls = []

    def get_contact_force_matrix(self, dt):
        self.calls.append(('matrix', dt))
        return np.zeros((1, 2, 3))

    def get_contact_data(self, dt):
        self.calls.append(('normal', dt))
        return self.values[:, :1], self.values, self.values, self.values[:, :1], self.counts, self.starts

    def get_friction_data(self, dt):
        self.calls.append(('friction', dt))
        return self.values, self.values, self.counts, self.starts


def test_contact_inventory_copies_only_occupied_slots_once_in_actual_order():
    view = ContactView()
    result = contact_inventory(view, .002)
    assert view.calls == [('matrix', .002), ('normal', .002), ('friction', .002)]
    assert result['normal']['occupied_indices'].tolist() == [1, 2]
    assert result['friction']['force'].tolist() == [[3., 4., 5.], [6., 7., 8.]]
    view.values[:] = -1.
    assert result['friction']['force'][0, 0] == 3.


@pytest.mark.parametrize('first,count', [(-1, 1), (3, 2), (0, -1)])
def test_malformed_contact_accounting_rejects(first, count):
    view = ContactView()
    view.starts[0, 1] = first
    view.counts[0, 1] = count
    with pytest.raises(ValueError, match='[Cc]ontact'):
        contact_inventory(view, .002)


def test_counter_tracks_contact_free_steps_and_unexpected_interval():
    counter = PhysicsStepCounter()
    counter(.002)
    counter(.002)
    assert counter.receipt() == dict(count=2, elapsed=.004, invalid_interval=False)
    counter(.004)
    assert counter.receipt() == dict(count=3, elapsed=.008, invalid_interval=True)


def test_callback_count_and_raw_clock_guard_preserve_original_clock_tolerance(monkeypatch):
    import doorbench.dexterous.isaac_live_pause_readback as module
    monkeypatch.setattr(module, 'articulation_inventory', lambda value:dict(raw=np.zeros(1)))
    monkeypatch.setattr(module, 'contact_inventory', lambda value, dt:dict(raw=np.zeros(1)))
    counter=PhysicsStepCounter()
    for _ in range(1000):counter(float(np.float32(.002)))
    sim=SimpleNamespace(current_time=counter.elapsed,current_time_step_index=1000,is_playing=lambda:True)
    controllers=dict(controller=SimpleNamespace(value=np.zeros(1)))
    args=dict(episode_id='probe',step_index=1000,epoch_s=2.,sim=sim,time_origin=0.,counter=counter,
        robot=object(),door=object(),contacts=object(),controllers=controllers,invariant_getters={},
        evidence_counts=dict(physical=1000,pad=1001))
    before=capture_native_pause_anchor(**args)
    assert before==capture_native_pause_anchor(**args)
    sim.current_time=np.nextafter(sim.current_time, np.inf)
    assert before!=capture_native_pause_anchor(**args)
    sim.current_time=2.001
    with pytest.raises(ValueError,match='backend clock'):
        capture_native_pause_anchor(**args)
    sim.current_time=counter.elapsed
    counter(.002)
    with pytest.raises(ValueError,match='callbacks'):
        capture_native_pause_anchor(**args)
