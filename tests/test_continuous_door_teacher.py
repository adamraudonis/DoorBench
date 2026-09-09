"""Composition tests use numeric fakes; they do not qualify robot physics."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import doorbench.dexterous.continuous_door_teacher as module


NAMES = tuple(f'joint_{i}' for i in range(69))
MOTORS = {'actuators': [{'name': f'motor_{i}'} for i in range(61)]}
CAPS = np.tile([-10., 10.], (61, 1))
EVENTS = {'qualified_grasp': 400, 'latch_released': 450, 'left_approach': 750,
          'right_release': 1050, 'right_clearance': 1600, 'panel_continuation': 1600}
CROSSING = 1900


class Walking:
    def __init__(self, *args, **kwargs):
        self.names, self.caps = NAMES, CAPS.copy()
        self.opening = SimpleNamespace(physics_dt=.002, qualification_seconds=.5,
            target_aperture=1.2, initial_contract=True, all_physics_qualified=True,
            all_pad_patches_qualified=True)
        self.acquisition_started = None
        self.readiness_screen = {'passed': True}
        self.blocked_reason = None
        self.handoffs = {}
        self.calls = []
        self.change = lambda step, info, force: None

    def force(self, t, root, joints, velocities, feet, handle, leaf, angles,
              forces, **kwargs):
        step = round(t/.002)
        self.calls.append((t, copy.deepcopy(kwargs)))
        if step >= 100:
            self.acquisition_started = .2
            self.handoffs['acquisition'] = {'hand_contact_count': 0}
        info = dict(phase='arm preparation' if step < 100 else 'loaded_panel_opening',
            body_stance_solver_failures=0,
            handoffs={key: tick*.002-.2 for key, tick in EVENTS.items() if step >= tick},
            release={'release_fraction': 1. if step >= 1600 else 0.})
        if step >= CROSSING:
            info['aperture_crossing'] = dict(time_s=CROSSING*.002-.2,
                aperture_rad=angles['leaf'], final_palm_hold=True,
                all_physics_qualified=True, all_pad_patches_qualified=True,
                right_release_complete=True)
        # Deliberately different at crossing: it is never delivered to the plant.
        force = np.full(61, .2 if step < CROSSING else .8)
        self.change(step, info, force)
        return force, info


class Post:
    feet = ('left_ankle_link', 'right_ankle_link')
    pose_names = (*feet, 'lh_palm', 'rh_palm', 'leaf', 'leaf_handle')
    door_names = ('leaf_hinge', 'leaf_handle_hinge', 'leaf_latch_bolt_slide')

    def __init__(self, *args, **kwargs):
        self.names, self.caps = NAMES, CAPS.copy()
        self.motor_names = tuple(a['name'] for a in MOTORS['actuators'])
        self.constructor_options = kwargs
        self.calls = []
        self.initialization = None
        self.change = lambda step, info, force: None

    def force(self, t, root, joints, velocities, feet, forces, **kwargs):
        self.calls.append((t, copy.deepcopy(root), copy.deepcopy(joints),
            copy.deepcopy(velocities), copy.deepcopy(kwargs)))
        if self.initialization is None:
            self.initialization = {'time_s': t, 'root': root.tolist(),
                'prior': kwargs['previous_motor_forces'].tolist()}
        info = dict(stance_solver_failures=0, minimum_body_y_m=1.,
            passage_completed=t >= 4.)
        force = np.full(61, .4)
        self.change(round(t/.002), info, force)
        return force, info


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr(module, 'WalkingOpeningTeacher', Walking)
    monkeypatch.setattr(module, 'PostOpeningTeacher', Post)
    return module.ContinuousDoorTeacher('robot.xml', MOTORS, {}, {},
        {'goal_xy': [0., 0.]}, 'motion.pt', {}, door_xml='door.xml',
        left_targets={}, release_screen={})


def packet(step, last_force=None):
    t = step*.002
    pose = np.array([0., 0., 0., 1., 0., 0., 0.])
    root = np.array([0., 1. if step >= 2000 else -1., 1., 1., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.])
    bodies = {name: pose.copy() for name in Post.pose_names}
    if step >= 20:
        for name in Post.feet:
            bodies[name][2] = .02
    palm_site = pose.copy()
    palm_site[0] = .03  # Distinct site offset; never substitute a BODY origin.
    angles = dict(leaf=1.21 if step >= CROSSING else .08 if step >= 500 else 0.,
                  operator=.85 if step >= 450 else 0., latch=.012 if step >= 450 else 0.)
    return dict(t=t, root=root, joints=dict.fromkeys(NAMES, 0.),
        velocities={name: i*.0001 for i, name in enumerate(NAMES)},
        foot_loads=[240., 240.], handle_pose=pose.copy(), leaf_pose=pose.copy(),
        angles=angles, hand_forces={},
        evidence=dict(grasp_qualified=125 <= step <= 1050, physics_qualified=True,
            right_pad_patches_valid=True, hand_contact_count=0 if step < 125 else 1,
            left_panel_load_N=3. if step >= 800 else 0.,
            left_palm_load_N=3. if step >= 800 else 0., right_lever_clearance_m=.03),
        right_palm_pose=palm_site, pose_time_s=t,
        contact_interval_s=[max(0., t-.002), t], body_poses=bodies,
        door_velocities=dict(leaf_hinge=.5, leaf_handle_hinge=.001, leaf_latch_bolt_slide=.0001),
        continuation_evidence=dict(physics_qualified=True, left_hand_contacts=0,
            left_hand_load_N=0., right_environment_contacts=0),
        applied_motor_forces=np.zeros(61) if last_force is None else last_force.copy(),
        release_normal_world=[0., -1., 0.])


def advance(controller, end, modify=None):
    start = 0 if controller.last_time is None else round(controller.last_time/.002)+1
    for step in range(start, end+1):
        measured = packet(step, controller.last_force)
        if modify is not None:
            modify(step, measured)
        result = controller.force(**measured)
    return result


def test_complete_contract_keeps_global_clock_actual_state_and_delivered_force(controller):
    force, info = advance(controller, CROSSING)
    assert not info['completed']
    assert len(controller.post.calls) == 1
    call = controller.post.calls[0]
    assert call[0] == CROSSING*.002
    np.testing.assert_allclose(call[4]['contact_interval_s'], [3.798, 3.8], rtol=0., atol=1e-14)
    assert call[4]['pose_time_s'] == call[0]
    np.testing.assert_array_equal(call[4]['previous_motor_forces'], np.full(61, .2))
    np.testing.assert_array_equal(force, np.full(61, .4))
    assert call[4]['door_positions']['leaf_hinge'] == 1.21
    assert call[4]['door_velocities']['leaf_hinge'] == .5
    assert call[3] == packet(CROSSING)['velocities']
    assert controller.walking.calls[-1][1]['right_palm_pose'][0] == .03
    assert call[4]['body_poses']['rh_palm'][0] == 0.
    assert controller.post.constructor_options == dict(door_xml='door.xml',
        stow_profile='sequential-v2', phase_seconds=5., inward_roll=.07, passage=True)
    opening_calls = len(controller.walking.calls)
    advance(controller, CROSSING+1)
    assert len(controller.walking.calls) == opening_calls
    assert 'previous_motor_forces' not in controller.post.calls[-1][4]
    assert 'release_normal_world' not in controller.post.calls[-1][4]
    for key, tick in EVENTS.items():
        assert controller.handoff['independently_qualified_events'][key] == pytest.approx(tick*.002, abs=1e-14)
    advance(controller, 2499)
    assert not controller.completed
    _, info = advance(controller, 2500)
    assert info['completed'] and info['completed_time_s'] == 5.
    assert controller.done and controller.blocked_reason is None
    receipt = controller.opening_audit
    assert receipt['passed'] and all(receipt['checks'].values())
    assert receipt['time_s'] == CROSSING*.002
    assert len(receipt['qualification_window']) == 251
    assert receipt['handoff_state_sha256'] == module.handoff_state_sha256(controller.handoff['state'])
    changed_state = copy.deepcopy(controller.handoff['state'])
    changed_state['door_velocities']['leaf_hinge'] += .001
    assert module.handoff_state_sha256(changed_state) != receipt['handoff_state_sha256']
    receipt['passed'] = False
    assert controller.opening_audit['passed']
    chronology = controller.handoffs
    chronology['opening'].clear()
    assert len(controller.handoffs['opening']) == 6


@pytest.mark.parametrize('step', [0, 1900, 1901, 2501])
def test_invalid_pad_cannot_be_redeemed_by_later_opening_or_traversal(controller, step):
    if step:
        advance(controller, step-1)
    data = packet(step, controller.last_force)
    data['evidence']['right_pad_patches_valid'] = False
    with pytest.raises(module.ContinuousDoorFailure, match='right-pad'):
        controller.force(**data)
    original = copy.deepcopy(controller.failure)
    data['evidence']['right_pad_patches_valid'] = True
    with pytest.raises(module.ContinuousDoorFailure, match='cannot resume'):
        controller.force(**data)
    assert controller.failure == original
    assert not controller.completed


@pytest.mark.parametrize('flag', ['final_palm_hold', 'all_physics_qualified',
    'all_pad_patches_qualified', 'right_release_complete'])
def test_false_crossing_receipt_never_initializes_continuation(controller, flag):
    def change(step, info, force):
        if step == CROSSING:
            info['aperture_crossing'][flag] = False
    controller.walking.change = change
    with pytest.raises(module.ContinuousDoorFailure, match=flag):
        advance(controller, CROSSING)
    assert not controller.post.calls


@pytest.mark.parametrize('break_step,event,field,value', [
    (350, 400, 'grasp_qualified', False),
    (650, 750, 'grasp_qualified', False),
    (950, 1050, 'left_panel_load_N', 1.9),
])
def test_actual_hold_break_rejects_component_declaration(controller, break_step, event, field, value):
    def change(step, data):
        if step == break_step:
            data['evidence'][field] = value
    with pytest.raises(module.ContinuousDoorFailure, match='actual-history'):
        advance(controller, event, change)


@pytest.mark.parametrize('change', ['stale_event', 'rewrite_event', 'missing_event',
                                  'readiness', 'stance_solver', 'stale_crossing'])
def test_upstream_clock_and_qualification_failures_are_sticky(controller, change):
    def alter(step, info, force):
        if change == 'stale_event' and step == 400:
            info['handoffs']['qualified_grasp'] -= .002
        if change == 'rewrite_event' and step == 401:
            info['handoffs']['qualified_grasp'] += .002
        if change == 'missing_event':
            info['handoffs'].pop('qualified_grasp', None)
        if change == 'readiness':
            controller.walking.readiness_screen = {'passed': False}
        if change == 'stance_solver':
            info['body_stance_solver_failures'] = 1
        if change == 'stale_crossing' and step == CROSSING:
            info['aperture_crossing']['time_s'] -= .002
    controller.walking.change = alter
    with pytest.raises(module.ContinuousDoorFailure):
        advance(controller, CROSSING)
    assert not controller.post.calls


def test_open_door_without_qualified_event_is_not_a_shortcut(controller):
    advance(controller, 30)
    data = packet(31, controller.last_force)
    data['angles']['leaf'] = 1.21
    with pytest.raises(module.ContinuousDoorFailure, match='without an actual qualified'):
        controller.force(**data)
    assert not controller.post.calls


def test_real_foot_motion_and_contact_free_preparation_are_required(controller):
    def no_lift(step, data):
        data['body_poses']['right_ankle_link'][2] = 0.
    with pytest.raises(module.ContinuousDoorFailure, match='walking/preparation'):
        advance(controller, CROSSING, no_lift)


def test_preparation_touch_rejected_on_actual_tick(controller):
    advance(controller, 1)
    data = packet(2, controller.last_force)
    data['evidence']['hand_contact_count'] = 1
    with pytest.raises(module.ContinuousDoorFailure, match='during preparation'):
        controller.force(**data)


@pytest.mark.parametrize('change', [
    lambda x: x.update(t=.002, pose_time_s=.002, contact_interval_s=[0., .002]),
    lambda x: x.update(pose_time_s=.002),
    lambda x: x.update(contact_interval_s=[0., .002]),
    lambda x: x['root'].__setitem__(1, -.49),
    lambda x: x['root'].__setitem__(3, 1.01),
    lambda x: x['body_poses']['leaf'].__setitem__(0, .01),
    lambda x: x['evidence'].update(right_pad_patches_valid=1),
    lambda x: x['evidence'].update(hand_contact_count=True),
    lambda x: x['continuation_evidence'].update(physics_qualified=False),
    lambda x: x['velocities'].pop(NAMES[0]),
    lambda x: x['door_velocities'].update(extra_joint=0.),
    lambda x: x.update(applied_motor_forces=np.zeros(60)),
])
def test_bad_numeric_reset_contract_fails_before_any_controller_call(controller, change):
    data = packet(0)
    change(data)
    with pytest.raises(module.ContinuousDoorFailure):
        controller.force(**data)
    assert not controller.walking.calls
    assert not controller.post.calls


@pytest.mark.parametrize('tick', [0, 2])
def test_duplicate_or_missing_physics_tick_is_rejected(controller, tick):
    advance(controller, 0)
    with pytest.raises(module.ContinuousDoorFailure, match='uninterrupted'):
        controller.force(**packet(tick, controller.last_force))


def test_actual_delivery_must_match_preceding_command(controller):
    advance(controller, 0)
    data = packet(1, controller.last_force)
    data['applied_motor_forces'][4] += .001
    with pytest.raises(module.ContinuousDoorFailure, match='delivery differs'):
        controller.force(**data)
    assert len(controller.walking.calls) == 1


def test_even_small_command_cap_violation_is_not_silently_clipped(controller):
    controller.walking.change = lambda step, info, force: force.__setitem__(0, 10.+1e-8)
    with pytest.raises(module.ContinuousDoorFailure, match='original-capped'):
        advance(controller, 0)


def test_unexpected_component_exception_is_sticky(controller):
    def explode(step, info, force):
        raise RuntimeError('solver failure')
    controller.walking.change = explode
    with pytest.raises(module.ContinuousDoorFailure, match='solver failure'):
        advance(controller, 0)
    assert controller.failure['reason'] == 'solver failure'


def test_qualified_opening_receipt_survives_failed_continuation_initialization(controller):
    def explode(step, info, force):
        raise ValueError('new stow plan has a collision')
    controller.post.change = explode
    with pytest.raises(module.ContinuousDoorFailure, match='stow plan'):
        advance(controller, CROSSING)
    assert controller.opening_audit['passed']
    assert controller.handoff is None
    assert not controller.done
    assert controller.blocked_reason == 'new stow plan has a collision'


@pytest.mark.parametrize('change', ['trailing_body', 'root_speed', 'foot_load', 'left_contact'])
def test_post_passage_flag_alone_does_not_qualify_quiet_finish(controller, change):
    def alter(step, info, force):
        if change == 'trailing_body':
            info['minimum_body_y_m'] = .19
    controller.post.change = alter
    def measured(step, data):
        if step >= 2000:
            if change == 'root_speed':
                data['root'][7] = .04
            if change == 'foot_load':
                data['foot_loads'][0] = 9.
            if change == 'left_contact':
                data['continuation_evidence']['left_hand_contacts'] = 1
    advance(controller, 2500, measured)
    assert not controller.completed
    assert controller.post.calls[-1][0] == 5.


def test_controller_does_not_mutate_callers_measurements_or_return_aliases(controller):
    data = packet(0)
    frozen = copy.deepcopy(data)
    force, info = controller.force(**data)
    for key in ('root', 'handle_pose', 'leaf_pose', 'right_palm_pose', 'applied_motor_forces'):
        np.testing.assert_array_equal(data[key], frozen[key])
    assert data['joints'] == frozen['joints']
    assert data['velocities'] == frozen['velocities']
    force[:] = 7.
    info['component']['phase'] = 'rewritten'
    assert np.all(controller.last_force == .2)
    assert controller.info['component']['phase'] != 'rewritten'


def test_declared_time_limit_stops_incomplete_episode(controller):
    controller.maximum_seconds = .002
    advance(controller, 1)
    with pytest.raises(module.ContinuousDoorFailure, match='declared bound'):
        advance(controller, 2)


def test_loaded_hold_policy_preserves_failed_first_crossing_and_waits(controller):
    controller.handoff_policy='loaded-hold-v2'
    def change(step,info,force):
        if step>=CROSSING:info['aperture_crossing']['final_palm_hold']=False
    controller.walking.change=change
    def missing_contact(step,data):
        if step==CROSSING:data['evidence']['left_palm_load_N']=0.
    advance(controller,CROSSING+250,missing_contact)
    assert controller.handoff is None and not controller.post.calls
    advance(controller,CROSSING+251)
    audit=controller.opening_audit
    assert audit['passed'] and audit['handoff_policy']=='loaded-hold-v2'
    assert audit['first_aperture_crossing']['final_palm_hold'] is False
    assert audit['first_aperture_crossing']['time_s']==CROSSING*.002-.2
    assert audit['crossing']['final_palm_hold'] is True
    assert audit['time_s']==(CROSSING+251)*.002
    assert 'first_actual_aperture_crossing' not in audit['checks']


def test_loaded_hold_policy_cannot_redeem_invalid_physics(controller):
    controller.handoff_policy='loaded-hold-v2'
    def unsafe(step,data):
        if step==CROSSING:data['evidence']['physics_qualified']=False
    with pytest.raises(module.ContinuousDoorFailure,match='physics'):
        advance(controller,CROSSING,unsafe)
    assert not controller.post.calls
