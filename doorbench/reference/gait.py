"""Original walking references with explicit, immobile stance contacts.

This is a kinematic schedule, not a dynamically balanced gait controller. It
does not choose a collision-free route: the caller supplies ordered waypoints
and must check the resulting body/feet against the scene. Coordinates are
metres, seconds, Z-up. Yaw rotates about +Z; zero faces +Y, anatomical left -X.
Foot positions locate the ankle frame, with no toe/heel roll during contact.
"""
from __future__ import annotations

import math

import numpy as np


def _ease(u):
    """Quintic interpolation: position, velocity and acceleration join at rest."""
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def _lift(u):
    """A C2 lift with flat takeoff, apex, and landing; unit peak at u=1/2."""
    return _ease(2.0 * u) if u <= 0.5 else _ease(2.0 * (1.0 - u))


def _left(yaw):
    return np.array([-math.cos(yaw), -math.sin(yaw)])


def _quat(yaw):
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _angle_delta(target, current):
    # Deterministically choose positive rotation for an exact half-turn.
    delta = (target - current + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if abs(delta + math.pi) < 1e-12 else delta


def plan_walk(start_xy, start_yaw, waypoints_xy, *, fps=30, step_length=.24,
              step_duration=.65, stance_width=.22, ankle_height=.055,
              waypoint_yaws=None, blend_turns=False, max_step_yaw_deg=20.,
              pelvis_acceleration_m_s2=None):
    """Return a sampled, contact-aware walk through every ordered waypoint.

    Arrays: time (N,), pelvis_xyz (N,3), pelvis_yaw (N,), foot_pos (N,2,3),
    foot_quat (N,2,4; WXYZ), foot_contact (N,2; bool). Foot 0 is left.
    Additional waypoint_samples indices identify a centered, both-planted sample
    for each input waypoint. style_metrics reports analytical step bounds and
    measured sampled contact slip. No weights or external motion data are used.

    step_length bounds each foot's horizontal displacement per swing, not the
    distance between successive left/right footprints. Turn steps also obey this
    bound and rotate at most max_step_yaw_deg (20 by default). By default each segment begins with
    in-place turns. With blend_turns=True, translating steps gradually change
    heading while staying on that segment; every waypoint still ends with a
    closing step and neutral double support. This does not round route corners.
    Optional
    waypoint_yaws (K,) specifies the facing for travel to each waypoint, allowing
    backward and lateral steps. The default faces travel. A duplicate waypoint
    with a changed explicit yaw is an in-place turn; otherwise it adds no motion.

    Timing is quantized upward to whole sample intervals. At least two intervals
    represent a support transfer and four represent a swing, even at low fps.
    Empty routes return a short planted hold. The pelvis stays near 0.94 m; the
    caller's whole-body solver must enforce reach, collision, and joint limits.
    """
    start = np.asarray(start_xy, dtype=float)
    points = np.asarray(waypoints_xy, dtype=float)
    if points.size == 0:
        points = np.empty((0, 2), dtype=float)
    if start.shape != (2,) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('start_xy must be (2,) and waypoints_xy must be (K,2)')
    if not np.isfinite(start).all() or not np.isfinite(points).all():
        raise ValueError('Waypoints must be finite')
    headings = None if waypoint_yaws is None else np.asarray(waypoint_yaws, dtype=float)
    if headings is not None and (headings.shape != (len(points),) or not np.isfinite(headings).all()):
        raise ValueError('waypoint_yaws must be finite and have one heading per waypoint')
    values = [start_yaw, fps, step_length, step_duration, stance_width, ankle_height]
    if any(isinstance(v, (bool, np.bool_)) or not np.isscalar(v) for v in values):
        raise ValueError('Gait parameters must be finite numeric scalars')
    try:
        yaw, fps, step_length, step_duration, width, ankle = map(float, values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Gait parameters must be finite numeric scalars') from exc
    if not all(math.isfinite(v) for v in (yaw, fps, step_length, step_duration, width, ankle)):
        raise ValueError('Gait parameters must be finite')
    if min(fps, step_length, step_duration, width) <= 0 or ankle < 0:
        raise ValueError('fps, step_length, step_duration and stance_width must be positive; ankle_height nonnegative')
    if not isinstance(blend_turns, (bool, np.bool_)):
        raise ValueError('blend_turns must be a boolean')
    max_yaw=math.radians(float(max_step_yaw_deg))
    if not math.isfinite(max_yaw) or not 0<max_yaw<=math.pi/3:
        raise ValueError('max_step_yaw_deg must be in (0,60]')
    if pelvis_acceleration_m_s2 is not None:
        pelvis_acceleration_m_s2=float(pelvis_acceleration_m_s2)
        if not math.isfinite(pelvis_acceleration_m_s2) or pelvis_acceleration_m_s2<=0:
            raise ValueError('pelvis_acceleration_m_s2 must be finite and positive')

    center = start.copy()
    feet = np.array([np.r_[center + sign * width / 2.0 * _left(yaw), ankle]
                     for sign in (1.0, -1.0)])
    foot_yaws = np.array([yaw, yaw])
    pelvis = np.r_[center, .94]
    contact = np.ones(2, dtype=bool)
    rows = []
    waypoint_samples = []
    step_lengths, step_angles = [], []
    next_foot = 0
    n_shift = max(2, math.ceil(step_duration * .28 * fps))
    n_swing = max(4, math.ceil(step_duration * .72 * fps))
    # Pelvis is not the whole-body COM. A restrained lateral bias invites the
    # downstream solver to transfer weight without a full ankle-to-ankle lurch.
    support_bias = .38

    def motion_intervals(origin,destination,minimum,yaw_delta=0.):
        if pelvis_acceleration_m_s2 is None:return minimum
        # max |quintic easing second derivative| = 10/sqrt(3).
        # This bounds the authored root translation, not articulated-body
        # acceleration; the independent validator still checks the solved rig.
        distance=float(np.linalg.norm(destination-origin))
        duration=math.sqrt((10/math.sqrt(3))*distance/pelvis_acceleration_m_s2)
        duration=max(duration,math.sqrt((10/math.sqrt(3))*abs(yaw_delta)/6.))
        return max(minimum,math.ceil(duration*fps))

    def append():
        rows.append((pelvis.copy(), yaw, feet.copy(),
                     np.stack([_quat(a) for a in foot_yaws]), contact.copy()))

    def shift(destination, destination_yaw=None):
        nonlocal pelvis, yaw
        origin, initial_yaw = pelvis.copy(), yaw
        angle = initial_yaw if destination_yaw is None else destination_yaw
        intervals=motion_intervals(origin,destination,n_shift,angle-initial_yaw)
        for frame in range(1, intervals + 1):
            amount = _ease(frame / intervals)
            pelvis = origin + amount * (destination - origin)
            yaw = initial_yaw + amount * (angle - initial_yaw)
            if frame == intervals:
                pelvis, yaw = destination.copy(), angle
            append()
        # Exact phase endpoints avoid tiny residuals accumulating over a route.
        pelvis = destination.copy()
        yaw = angle

    def support_pelvis(foot_positions, stance, body_yaw):
        midpoint = foot_positions[:, :2].mean(axis=0)
        lateral = _left(body_yaw)
        bias = support_bias * np.dot(foot_positions[stance, :2] - midpoint, lateral)
        return np.r_[midpoint + bias * lateral, .94]

    def swing(foot, destination_xy, destination_yaw, body_yaw):
        nonlocal pelvis, yaw, next_foot
        stance = 1 - foot
        origin = feet[foot].copy()
        destination = np.r_[destination_xy, ankle]
        length = float(np.linalg.norm(destination[:2] - origin[:2]))
        angle = float(destination_yaw - foot_yaws[foot])
        if length > step_length + 1e-9 or abs(angle) > max_yaw + 1e-9:
            raise ValueError('Internal gait planner exceeded a step bound')
        shift(support_pelvis(feet, stance, yaw))
        initial_pelvis, initial_yaw, initial_foot_yaw = pelvis.copy(), yaw, foot_yaws[foot]
        future_feet = feet.copy()
        future_feet[foot] = destination
        final_pelvis = support_pelvis(future_feet, stance, body_yaw)
        # Short finishing/turn steps should not become exaggerated high-knee steps.
        clearance = min(.065, .025 + .15 * length)
        intervals=motion_intervals(initial_pelvis,final_pelvis,n_swing,body_yaw-initial_yaw)
        if pelvis_acceleration_m_s2 is not None:
            # Lift uses two quintics over half a swing each. Leave room for
            # the downstream knee/ankle motion instead of snapping the foot up.
            intervals=max(intervals,math.ceil(math.sqrt(4*(10/math.sqrt(3))*clearance/2.)*fps))
        for frame in range(1, intervals + 1):
            u = frame / intervals
            amount, lift = _ease(u), _lift(u)
            feet[foot] = origin + amount * (destination - origin)
            feet[foot, 2] += clearance * lift
            foot_yaws[foot] = initial_foot_yaw + amount * angle
            pelvis = initial_pelvis + amount * (final_pelvis - initial_pelvis)
            pelvis[2] += .008 * lift
            yaw = initial_yaw + amount * (body_yaw - initial_yaw)
            contact[foot] = frame == intervals
            if frame == intervals:
                feet[foot] = destination
                foot_yaws[foot] = destination_yaw
                pelvis, yaw = final_pelvis.copy(), body_yaw
            append()
        step_lengths.append(length)
        step_angles.append(abs(angle))
        next_foot = stance

    def settle():
        shift(np.r_[center, .94])

    def blended_footsteps(vector, target_yaw):
        """Plan before moving, bounding each foot and continuous ankle spacing.

        The heading belongs to the stance frame, independently of travel, so
        explicit headings retain backward pulls and lateral sliding steps.
        Small increments resolve rotation plus translation jointly instead of
        assuming their individually bounded displacements remain bounded when
        added. A segment always starts and ends with both feet centered.
        """
        turn = target_yaw - yaw
        distance = float(np.linalg.norm(vector))
        # Each foot moves every second placement, hence half of the per-foot
        # angular/linear bound per alternating center increment.
        count = max(1, math.ceil(2 * distance / step_length),
                    math.ceil(abs(turn) / (max_yaw/2)))
        while True:
            trial_feet = feet[:, :2].copy()
            trial_yaws = foot_yaws.copy()
            body_yaw = yaw
            foot = next_foot
            planned = []
            valid = True
            # The last item closes the trailing foot at the final stance.
            for step in range(1, count + 2):
                fraction = min(step / count, 1.0)
                angle = yaw + turn * fraction
                sign = 1.0 if foot == 0 else -1.0
                destination = center + vector * fraction + sign * width / 2 * _left(angle)
                next_body_yaw = (trial_yaws[1 - foot] + angle) / 2
                length = np.linalg.norm(destination - trial_feet[foot])
                if (length > step_length + 1e-12 or
                        abs(angle - trial_yaws[foot]) > max_yaw + 1e-12):
                    valid = False
                    break
                before = trial_feet[0] - trial_feet[1]
                trial_feet[foot] = destination
                after = trial_feet[0] - trial_feet[1]
                # During swing, separation interpolates between these vectors
                # and body heading spans these two axes. Positive projections
                # at all four endpoint combinations bound the entire small
                # angular arc, not just the discrete animation samples. This
                # prevents crossing during narrow-stance sidesteps and turns.
                separation = min(float(np.dot(relative, lateral))
                                 for relative in (before, after)
                                 for lateral in (_left(body_yaw), _left(next_body_yaw)))
                if separation < .55 * width - 1e-12:
                    valid = False
                    break
                planned.append((foot, destination, angle, next_body_yaw))
                trial_yaws[foot] = angle
                body_yaw = next_body_yaw
                foot = 1 - foot
            if valid:
                return planned
            # Refining only the count preserves the route and step cadence.
            # Usually zero or a few refinements suffice; multiplicative growth
            # also handles deliberately tiny strides or narrow stances.
            count = max(count + 1, math.ceil(count * 1.1))

    append()
    for index, point in enumerate(points):
        vector = point - center
        distance = float(np.linalg.norm(vector))
        if headings is not None:
            direction_yaw = float(headings[index])
        else:
            direction_yaw = math.atan2(-vector[0], vector[1]) if distance > 1e-12 else yaw
        target_yaw = yaw + _angle_delta(direction_yaw, yaw)
        turn = target_yaw - yaw
        if blend_turns and distance > 1e-12:
            for foot, destination, foot_angle, body_angle in blended_footsteps(vector, target_yaw):
                swing(foot, destination, foot_angle, body_angle)
            center = point.copy()
            settle()
            waypoint_samples.append(len(rows) - 1)
            continue
        if abs(turn) > 1e-12:
            # A wide stance or deliberately tiny step bound also restricts the
            # chord travelled by the ankle during an in-place turn.
            max_turn = min(max_yaw, 2.0 * math.asin(min(1.0, step_length / width)))
            pieces = max(1, math.ceil(abs(turn) / max_turn))
            begin_yaw = yaw
            for piece in range(1, pieces + 1):
                end_yaw = begin_yaw + turn * piece / pieces
                mid_yaw = (yaw + end_yaw) / 2.0
                leading = 0 if turn > 0 else 1
                for foot, body_angle in ((leading, mid_yaw), (1 - leading, end_yaw)):
                    sign = 1.0 if foot == 0 else -1.0
                    swing(foot, center + sign * width / 2.0 * _left(end_yaw), end_yaw, body_angle)
            settle()

        if distance <= 1e-12:
            waypoint_samples.append(len(rows) - 1)
            continue

        # Each alternate foot moves at most twice the center increment. Both
        # feet start centered at this waypoint, including after every turn.
        increment_limit = step_length / 2.0
        lateral_fraction = abs(float(np.dot(vector / distance, _left(yaw))))
        if lateral_fraction > 1e-12:
            # Lateral travel must not make the trailing ankle cross the stance
            # ankle, even with a narrow stance or a large user step_length.
            increment_limit = min(increment_limit, .45 * width / lateral_fraction)
        count = max(1, math.ceil(distance / increment_limit))
        start_center = center.copy()
        for step in range(1, count + 1):
            foot = next_foot
            sign = 1.0 if foot == 0 else -1.0
            foot_center = start_center + vector * step / count
            swing(foot, foot_center + sign * width / 2.0 * _left(yaw), yaw, yaw)
        center = point.copy()
        # Bring the trailing foot alongside the last footprint rather than
        # declaring a staggered stance to be the requested final destination.
        foot = next_foot
        sign = 1.0 if foot == 0 else -1.0
        destination = center + sign * width / 2.0 * _left(yaw)
        if np.linalg.norm(feet[foot, :2] - destination) > 1e-12:
            swing(foot, destination, yaw, yaw)
        settle()
        waypoint_samples.append(len(rows) - 1)

    if len(rows) == 1:
        settle()
    pelvis_xyz = np.stack([row[0] for row in rows])
    pelvis_yaw = np.array([row[1] for row in rows])
    foot_pos = np.stack([row[2] for row in rows])
    foot_quat = np.stack([row[3] for row in rows])
    foot_contact = np.stack([row[4] for row in rows])
    planted = foot_contact[1:] & foot_contact[:-1]
    slip = np.linalg.norm(np.diff(foot_pos, axis=0), axis=-1)
    # Compare quaternion differences instead of acos(dot) near identity, where
    # floating-point rounding could falsely report rotation of a fixed foot.
    rotation_changed = np.any(foot_quat[1:] != foot_quat[:-1], axis=-1) & planted
    return {
        'time': np.arange(len(rows), dtype=float) / fps,
        'pelvis_xyz': pelvis_xyz, 'pelvis_yaw': pelvis_yaw,
        'foot_pos': foot_pos, 'foot_quat': foot_quat, 'foot_contact': foot_contact,
        'waypoint_samples': np.array(waypoint_samples, dtype=np.int64),
        'style_metrics': {
            'step_count': len(step_lengths),
            'max_step_length_m': max(step_lengths, default=0.0),
            'max_step_yaw_deg': math.degrees(max(step_angles, default=0.0)),
            'max_stance_slip_m': float(np.max(slip[planted], initial=0.0)),
            'max_stance_rotation_deg': 0.0 if not rotation_changed.any() else None,
            'swing_endpoint_velocity_m_s': 0.0,
            'swing_endpoint_acceleration_m_s2': 0.0,
            'both_feet_airborne_frames': int(np.sum(~foot_contact.any(axis=1))),
            'double_support_fraction': float(np.mean(foot_contact.all(axis=1))),
            'kinematic_schedule_only': True,
            'blend_turns': bool(blend_turns),
            'max_step_yaw_limit_deg': float(max_step_yaw_deg),
            'pelvis_translation_acceleration_limit_m_s2': pelvis_acceleration_m_s2,
        },
    }
