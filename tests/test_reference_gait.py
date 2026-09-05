"""World-contact and route invariants for original reference walking."""
import math

import numpy as np
import pytest

from doorbench.reference.gait import _ease, _lift, plan_walk


def assert_contacts(result):
    positions, quaternions, contacts = (result[k] for k in ('foot_pos', 'foot_quat', 'foot_contact'))
    assert contacts.any(axis=1).all(), 'both feet left the floor'
    assert contacts[[0, -1]].all()
    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=-1), 1.0, atol=1e-14)
    for foot in range(2):
        active = contacts[:, foot]
        starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
        ends = np.flatnonzero(active & np.r_[~active[1:], True])
        for start, end in zip(starts, ends):
            np.testing.assert_array_equal(positions[start:end+1, foot], np.tile(positions[start, foot], (end-start+1, 1)))
            np.testing.assert_array_equal(quaternions[start:end+1, foot], np.tile(quaternions[start, foot], (end-start+1, 1)))
    assert result['style_metrics']['max_stance_slip_m'] == 0
    assert result['style_metrics']['max_stance_rotation_deg'] == 0


@pytest.mark.parametrize('yaw,points', [
    (0., [[0., 1.]]),
    (0., [[0., .6], [.6, .6], [.6, -.1]]),
    (math.pi, [[0., .3]]),
    (math.radians(179), [[.002, -.7], [-.002, -1.4]]),
    (0., [[0., 0.], [0., .001], [0., .001]]),
])
def test_route_contacts_step_bounds_and_final_stance(yaw, points):
    result = plan_walk([0., 0.], yaw, points)
    assert_contacts(result)
    assert result['style_metrics']['max_step_length_m'] <= .24 + 1e-12
    assert result['style_metrics']['max_step_yaw_deg'] <= 20 + 1e-12
    assert np.diff(result['time']).min() == pytest.approx(1/30)
    for point, sample in zip(points, result['waypoint_samples']):
        np.testing.assert_allclose(result['pelvis_xyz'][sample], [*point, .94], atol=1e-13)
        np.testing.assert_allclose(result['foot_pos'][sample, :, :2].mean(axis=0), point, atol=1e-13)
        assert result['foot_contact'][sample].all()
    final_yaw = result['pelvis_yaw'][-1]
    left = np.array([-math.cos(final_yaw), -math.sin(final_yaw)])
    np.testing.assert_allclose(result['foot_pos'][-1, 0, :2] - result['foot_pos'][-1, 1, :2], .22 * left, atol=1e-13)


def test_exact_original_stance_and_empty_hold():
    result = plan_walk([2., -3.], 0., [])
    assert_contacts(result)
    np.testing.assert_array_equal(result['foot_pos'][0], [[1.89, -3., .055], [2.11, -3., .055]])
    np.testing.assert_array_equal(result['pelvis_xyz'], np.tile([2., -3., .94], (len(result['time']), 1)))
    assert result['style_metrics']['step_count'] == 0
    assert result['waypoint_samples'].size == 0
    assert len(result['time']) > 1


def test_turn_occurs_at_corner_before_translating_next_segment():
    result = plan_walk([0., 0.], 0., [[0., .6], [.6, .6]])
    corner = result['waypoint_samples'][0]
    turn_frames = np.flatnonzero(np.abs(np.diff(result['pelvis_yaw'])) > 1e-10)
    assert turn_frames.min() > corner
    assert turn_frames.size
    # During the isolated turn every ankle remains inside the local stance
    # radius of this corner; no curved route skips across the waypoint.
    for frame in turn_frames:
        assert np.max(np.linalg.norm(result['foot_pos'][frame, :, :2] - [0., .6], axis=-1)) <= .1100001


def test_swing_endpoints_are_flat_and_contacts_change_only_at_floor():
    result = plan_walk([0., 0.], 0., [[0., .48]], fps=240)
    assert_contacts(result)
    # Check the actual sampled contact transitions, not just reported metrics.
    for foot in range(2):
        contacts = result['foot_contact'][:, foot]
        airborne = ~contacts
        starts = np.flatnonzero(airborne & np.r_[False, ~airborne[:-1]])
        ends = np.flatnonzero(airborne & np.r_[~airborne[1:], False])
        for begin, end in zip(starts, ends):
            assert result['foot_pos'][begin-1, foot, 2] == .055
            assert result['foot_pos'][end+1, foot, 2] == .055
            assert result['foot_pos'][begin:end+1, foot, 2].min() > .055
            speed_start = np.linalg.norm(result['foot_pos'][begin, foot] - result['foot_pos'][begin-1, foot]) * 240
            speed_end = np.linalg.norm(result['foot_pos'][end+1, foot] - result['foot_pos'][end, foot]) * 240
            assert max(speed_start, speed_end) < .004
    # Verify derivatives of the polynomial used for lift/placement at joins.
    h = 1e-4
    for curve, joins in ((_ease, (0., 1.)), (_lift, (0., .5, 1.))):
        for u in joins:
            derivative = (curve(u+h)-curve(u-h))/(2*h)
            acceleration = (curve(u+h)-2*curve(u)+curve(u-h))/(h*h)
            assert abs(derivative) < 3e-6
            assert abs(acceleration) < .02


def test_small_step_bound_also_limits_in_place_turn_foot_displacement():
    result = plan_walk([0., 0.], math.pi, [[0., .035]], step_length=.015, stance_width=.30)
    assert_contacts(result)
    assert result['style_metrics']['max_step_length_m'] <= .015 + 1e-12
    assert result['style_metrics']['max_step_yaw_deg'] < 6


@pytest.mark.parametrize('blend_turns', [False, True])
def test_backward_pull_keeps_facing_door_and_stance_contacts(blend_turns):
    result = plan_walk([0., 0.], 0., [[0., -.3], [0., -.6]], waypoint_yaws=[0., 0.], blend_turns=blend_turns)
    assert_contacts(result)
    np.testing.assert_array_equal(result['pelvis_yaw'], 0.)
    np.testing.assert_array_equal(result['foot_quat'], np.broadcast_to([1., 0., 0., 0.], result['foot_quat'].shape))
    np.testing.assert_allclose(result['pelvis_xyz'][-1], [0., -.6, .94], atol=1e-14)


@pytest.mark.parametrize('direction', [-1., 1.])
@pytest.mark.parametrize('blend_turns', [False, True])
def test_lateral_slide_does_not_cross_feet_even_with_narrow_stance(direction, blend_turns):
    result = plan_walk([0., 0.], 0., [[direction*.6, 0.]], waypoint_yaws=[0.], stance_width=.10, step_length=.40, blend_turns=blend_turns)
    assert_contacts(result)
    np.testing.assert_array_equal(result['pelvis_yaw'], 0.)
    assert np.min(result['foot_pos'][:, 1, 0]-result['foot_pos'][:, 0, 0]) >= .055-1e-12


@pytest.mark.parametrize('blend_turns', [False, True])
def test_zero_distance_heading_change_has_real_turn_contacts(blend_turns):
    result = plan_walk([.4, -.2], 0., [[.4, -.2]], waypoint_yaws=[math.pi/2], blend_turns=blend_turns)
    assert_contacts(result)
    assert result['style_metrics']['step_count'] > 0
    assert result['pelvis_yaw'][-1] == pytest.approx(math.pi/2)
    np.testing.assert_array_equal(result['pelvis_xyz'][-1], [.4, -.2, .94])
    assert np.max(np.linalg.norm(result['foot_pos'][:, :, :2] - [.4, -.2], axis=-1)) <= .1100001


@pytest.mark.parametrize('blend_turns', [False, True])
def test_reported_step_bounds_match_actual_swing_endpoints(blend_turns):
    result = plan_walk([0., 0.], -.6, [[0., .7], [.3, .7]], blend_turns=blend_turns)
    lengths, angles = [], []
    for foot in range(2):
        airborne = ~result['foot_contact'][:, foot]
        starts = np.flatnonzero(airborne & np.r_[False, ~airborne[:-1]])
        ends = np.flatnonzero(airborne & np.r_[~airborne[1:], False])
        for start, end in zip(starts, ends):
            lengths.append(np.linalg.norm(result['foot_pos'][end+1, foot, :2] - result['foot_pos'][start-1, foot, :2]))
            a, b = result['foot_quat'][[start-1, end+1], foot]
            angles.append(math.degrees(2*math.acos(np.clip(abs(np.dot(a, b)), 0., 1.))))
    assert max(lengths) <= .24+1e-12
    assert max(angles) <= 20+1e-10
    assert result['style_metrics']['max_step_length_m'] == pytest.approx(max(lengths))
    assert result['style_metrics']['max_step_yaw_deg'] == pytest.approx(max(angles))


def test_blended_turn_advances_during_turn_and_preserves_every_route_vertex():
    points = np.array([[0., .6], [.5, 1.1], [.5, 1.8]])
    options = dict(fps=30, step_length=.42, stance_width=.21)
    separated = plan_walk([0., 0.], -.3, points, **options)
    result = plan_walk([0., 0.], -.3, points, **options, blend_turns=True)
    assert_contacts(result)
    assert result['time'][-1] < .8 * separated['time'][-1]
    assert result['style_metrics']['step_count'] < separated['style_metrics']['step_count']
    # No cadence speedup: each nonempty swing retains exactly the same duration.
    for candidate in (separated, result):
        for foot in range(2):
            airborne = ~candidate['foot_contact'][:, foot]
            starts = np.flatnonzero(airborne & np.r_[False, ~airborne[:-1]])
            ends = np.flatnonzero(airborne & np.r_[~airborne[1:], False])
            np.testing.assert_array_equal(ends-starts+2, math.ceil(.65*.72*30))
    for point, sample in zip(points, result['waypoint_samples']):
        np.testing.assert_allclose(result['pelvis_xyz'][sample], [*point, .94], atol=1e-13)
        np.testing.assert_allclose(result['foot_pos'][sample, :, :2].mean(axis=0), point, atol=1e-13)
        assert result['foot_contact'][sample].all()
    # Inspect landing centers by subtracting each foot's own lateral offset.
    # All landings stay on the current straight route segment, including turns.
    start, first_frame = np.zeros(2), 0
    for point, last_frame in zip(points, result['waypoint_samples']):
        for foot in range(2):
            contact = result['foot_contact'][:, foot]
            landings = np.flatnonzero(contact[1:] & ~contact[:-1]) + 1
            for frame in landings[(landings > first_frame) & (landings <= last_frame)]:
                quat = result['foot_quat'][frame, foot]
                angle = 2 * math.atan2(quat[3], quat[0])
                sign = 1 if foot == 0 else -1
                center = result['foot_pos'][frame, foot, :2] + sign*.105*np.array([math.cos(angle), math.sin(angle)])
                direction = point-start
                fraction = np.dot(center-start, direction) / np.dot(direction, direction)
                np.testing.assert_allclose(center, start+fraction*direction, atol=1e-13)
                assert -1e-12 <= fraction <= 1+1e-12
        first_frame, start = last_frame, point
    turning = np.abs(np.diff(result['pelvis_yaw'])) > 1e-9
    advancing = np.linalg.norm(np.diff(result['pelvis_xyz'][:, :2], axis=0), axis=1) > 1e-7
    assert (turning & advancing).sum() > 10


@pytest.mark.parametrize('heading,point,width,stride', [
    (-math.pi/3, [.2, -.3], .21, .42),  # retreat while facing the moving door
    (math.pi/2, [.6, 0.], .08, .45),   # narrow stance plus a large turn
    (math.pi, [0., .3], .21, .42),     # full reversal of facing
    (.7, [.03, -.02], .30, .015),      # combined rotation obeys a tiny stride
])
def test_blended_footsteps_bound_continuous_separation_and_actual_stride(heading, point, width, stride):
    result = plan_walk([0., 0.], 0., [point], waypoint_yaws=[heading],
                       stance_width=width, step_length=stride, blend_turns=True, fps=90)
    assert_contacts(result)
    lateral = -np.column_stack((np.cos(result['pelvis_yaw']), np.sin(result['pelvis_yaw'])))
    separation = np.sum((result['foot_pos'][:, 0, :2]-result['foot_pos'][:, 1, :2])*lateral, axis=1)
    assert separation.min() >= .55*width-1e-12
    assert result['pelvis_yaw'][-1] == pytest.approx(heading)
    for foot in range(2):
        airborne = ~result['foot_contact'][:, foot]
        starts = np.flatnonzero(airborne & np.r_[False, ~airborne[:-1]])
        ends = np.flatnonzero(airborne & np.r_[~airborne[1:], False])
        for first, last in zip(starts, ends):
            assert np.linalg.norm(result['foot_pos'][last+1, foot, :2]-result['foot_pos'][first-1, foot, :2]) <= stride+1e-12
            a, b = result['foot_quat'][[first-1, last+1], foot]
            assert 2*math.acos(np.clip(abs(np.dot(a, b)), 0., 1.)) <= math.radians(20)+1e-12


def test_deterministic_output_and_no_input_mutation():
    start = np.array([.1, .2]); points = np.array([[.1, .5], [.4, .5]])
    original = points.copy()
    a = plan_walk(start, .25, points)
    b = plan_walk(start, .25, points)
    for key in a:
        if key == 'style_metrics':
            assert a[key] == b[key]
        else:
            np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(points, original)
    np.testing.assert_array_equal(start, [.1, .2])


@pytest.mark.parametrize('kwargs', [
    {'fps': 0}, {'step_length': -1}, {'step_duration': 0}, {'stance_width': 0},
    {'ankle_height': -1}, {'fps': float('nan')}, {'step_length': True},
    {'blend_turns': 'yes'},
])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        plan_walk([0, 0], 0, [[0, 1]], **kwargs)


@pytest.mark.parametrize('start,yaw,points', [
    ([0], 0, [[0, 1]]), ([0, 0], float('inf'), []),
    ([0, 0], 0, [[0, 1, 2]]), ([0, 0], 0, [[0, float('nan')]]),
])
def test_invalid_routes_rejected(start, yaw, points):
    with pytest.raises(ValueError):
        plan_walk(start, yaw, points)


@pytest.mark.parametrize('headings', [[], [0., 1.], [float('nan')], [[0.]]])
def test_invalid_waypoint_headings_rejected(headings):
    with pytest.raises(ValueError):
        plan_walk([0., 0.], 0., [[0., .2]], waypoint_yaws=headings)
