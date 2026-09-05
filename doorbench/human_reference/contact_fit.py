"""Explicit leg-only adaptation of a captured motion to different human proportions.

This is a visual contact candidate, not recorded support, force, balance or a
dynamics solution. It preserves time, pelvis, upper body and absolute foot/toe
rotations. The input arrays are never modified.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def two_bone_knee(hip, knee, ankle, target, *, fallback_axis=None, bend_direction=None):
    """Fixed-length two-bone IK with the knee nearest its captured bend plane.

    Reject unreachable targets instead of stretching limbs or silently moving
    the pelvis. A straight captured leg needs a documented transverse fallback.
    """
    hip, knee, ankle, target = [np.asarray(x, dtype=float) for x in
                                (hip, knee, ankle, target)]
    if any(x.shape != (3,) or not np.isfinite(x).all()
           for x in (hip, knee, ankle, target)):
        raise ValueError('Expected finite XYZ leg landmarks')
    upper, lower = np.linalg.norm(knee - hip), np.linalg.norm(ankle - knee)
    distance = np.linalg.norm(target - hip)
    if min(upper, lower, distance) < 1e-10:
        raise ValueError('Degenerate leg')
    if distance > upper + lower + 1e-9 or distance < abs(upper - lower) - 1e-9:
        raise ValueError('Ankle target is unreachable with the fixed limb lengths')
    direction = (target - hip) / distance
    along = (upper * upper - lower * lower + distance * distance) / (2 * distance)
    height = np.sqrt(max(0., upper * upper - along * along))
    pole = knee-hip if bend_direction is None else np.asarray(bend_direction, dtype=float)
    bend = pole - direction * np.dot(pole, direction)
    if np.linalg.norm(bend) < 1e-9:
        if fallback_axis is None:
            raise ValueError('Straight leg needs a captured transverse fallback')
        axis = np.asarray(fallback_axis, dtype=float)
        bend = axis - direction * np.dot(axis, direction)
        if np.linalg.norm(bend) < 1e-9:
            raise ValueError('Fallback axis is parallel to the leg')
    return hip + along * direction + height * bend / np.linalg.norm(bend)


def _align_rotation(before, after):
    a, b = before / np.linalg.norm(before), after / np.linalg.norm(after)
    cross, dot = np.cross(a, b), float(np.clip(np.dot(a, b), -1., 1.))
    if np.linalg.norm(cross) < 1e-12:
        if dot < 0:
            raise ValueError('A leg correction would reverse a segment')
        return np.eye(3)
    return Rotation.from_rotvec(cross / np.linalg.norm(cross)
                               * np.arctan2(np.linalg.norm(cross), dot)).as_matrix()


def transported_bend_direction(hip, knee, ankle, target, thigh_rotation,
                               previous_local_pole, *, low_m=.005, high_m=.020):
    """Transport captured bend orientation, with continuity near a straight leg.

    Near extension, projecting the old knee onto a newly tilted leg axis can
    choose the opposite knee. Instead minimally rotate the *old bend plane*
    from old hip→ankle to new hip→ankle. Below5mm bend height, use the last
    reliable local pole carried by the captured thigh frame. Blend5–20mm.
    """
    old_axis = ankle-hip
    old_axis = old_axis/np.linalg.norm(old_axis)
    raw = knee-hip-old_axis*np.dot(knee-hip, old_axis)
    height = np.linalg.norm(raw)
    fallback = thigh_rotation@previous_local_pole
    fallback -= old_axis*np.dot(fallback, old_axis)
    fallback /= np.linalg.norm(fallback)
    current = raw/height if height > 1e-12 else fallback
    flipped = bool(np.dot(current,fallback) < 0)
    if flipped:
        current = -current
    weight = np.clip((height-low_m)/(high_m-low_m),0.,1.)
    weight = weight*weight*(3-2*weight)
    pole = (1-weight)*fallback+weight*current
    pole /= np.linalg.norm(pole)
    new_history = thigh_rotation.T@current if height >= high_m else previous_local_pole.copy()
    return (_align_rotation(ankle-hip,target-hip)@pole, new_history,
            {'bend_height_m':float(height),'history_blend_weight':float(weight),
             'pole_sign_aligned':flipped})


def smooth_clearance_lift(minimum_z, times, *, clearance_m=.001, smoothing_s=.06):
    """Nonnegative smooth upper envelope of the measured shoe penetration.

    A convex quadratic balances squared lift and its second differences, with
    a hard pointwise lower bound. It changes vertical translation only; it does
    not flatten the captured heel/toe rotation or claim a planted-foot phase.
    """
    z, times = np.asarray(minimum_z, float), np.asarray(times, float)
    if (z.ndim != 1 or times.shape != z.shape or len(z) < 3
            or not np.isfinite(z).all() or not np.isfinite(times).all()
            or clearance_m < 0 or smoothing_s < 0):
        raise ValueError('Invalid shoe surface or smoothing parameters')
    dt = np.diff(times)
    if np.any(dt <= 0) or not np.allclose(dt, dt[0], atol=1e-10, rtol=1e-8):
        raise ValueError('Clearance fit requires the unchanged uniform capture clock')
    required = np.maximum(0., clearance_m - z)
    if not np.any(required):
        return required
    weight = (smoothing_s / dt[0]) ** 4

    def quadratic(x):
        second = np.diff(x, n=2)
        gradient = x.copy()
        gradient[:-2] += weight * second
        gradient[1:-1] -= 2 * weight * second
        gradient[2:] += weight * second
        return .5 * (x @ x + weight * (second @ second)), gradient

    result = minimize(quadratic, required, jac=True, method='L-BFGS-B',
                      bounds=[(float(a), None) for a in required],
                      options={'maxiter':10000, 'ftol':1e-15, 'gtol':1e-9})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f'Clearance smoothing did not converge: {result.message}')
    lift = np.maximum(required, result.x)
    if np.any(z + lift < clearance_m - 1e-10):
        raise AssertionError('Shoe clearance lower bound was lost')
    return lift


def fit_capture_legs(arrays, shoe_min_z, *, source_xy_translation=(0., 0.),
                     clearance_m=.001, smoothing_s=.06):
    """Return adapted arrays and diagnostics, preserving immutable source arrays.

    Ankle XY follows the actual source LeftFoot/RightFoot path plus one global
    fixed translation. There is no per-frame world-side offset or foot lock.
    Each thigh/shin's two twist subdivisions receive the same absolute rotation
    correction. New downstream positions use the original calibrated offsets.
    Re-evaluate deformed shoes in Blender: mixed skin weights can differ from
    the rigid-foot surface translation predicted here.
    """
    output = {k:np.array(v, copy=True) for k, v in arrays.items()}
    names = output['bone_names'].tolist()
    source_names = output['source_joint_names'].tolist()
    parents = np.asarray(output['parent_index'])
    old_pos, old_rot = output['bone_pos'].copy(), output['bone_rot'].copy()
    old_matrix, old_basis = output['bone_matrix'].copy(), output['bone_basis'].copy()
    count, bones = old_pos.shape[:2]
    offsets = output['calibration_parent_offsets']
    shoe_min_z = np.asarray(shoe_min_z, float)
    translation = np.asarray(source_xy_translation, float)
    if (shoe_min_z.shape != (count, 2) or not np.isfinite(shoe_min_z).all()
            or translation.shape != (2,) or not np.isfinite(translation).all()):
        raise ValueError('Expected per-frame left/right shoe minima and fixed XY translation')
    if any(parent >= j for j, parent in enumerate(parents)):
        raise ValueError('Bones must be ordered parent before child')
    lengths = np.linalg.norm(output['bone_tail'][0] - old_pos[0], axis=1)
    affected, report = set(), {}
    ankle_targets = np.empty((count, 2, 3))
    lift_series = np.empty((count, 2))
    for side_index, (side, source_side) in enumerate([('L', 'Left'), ('R', 'Right')]):
        thigh = [names.index(f'upperleg{i:02d}.{side}') for i in [1, 2]]
        shin = [names.index(f'lowerleg{i:02d}.{side}') for i in [1, 2]]
        foot = names.index('foot.' + side)
        source_foot = source_names.index('Character1_' + source_side + 'Foot')
        captured_xy = output['source_joint_pos'][:, source_foot, :2] + translation
        hip_pos, knee_pos, ankle_pos = [old_pos[:, j] for j in (thigh[0], shin[0], foot)]
        reach = (np.linalg.norm(knee_pos-hip_pos, axis=1)
                 + np.linalg.norm(ankle_pos-knee_pos, axis=1) - .0001)
        horizontal = np.linalg.norm(captured_xy-hip_pos[:, :2], axis=1)
        if np.any(horizontal >= reach):
            raise ValueError('Source ankle XY exceeds fixed leg reach even at hip height')
        # A shorter target limb can need a few millimetres of extra elevation
        # at full extension. Include that analytical bound before smoothing;
        # do not clamp the IK endpoint after the fact or stretch the skeleton.
        reach_min_z = hip_pos[:, 2] - np.sqrt(reach*reach-horizontal*horizontal)
        reach_lift = np.maximum(0., reach_min_z-ankle_pos[:, 2])
        effective_minimum = np.minimum(shoe_min_z[:, side_index], clearance_m-reach_lift)
        lift = smooth_clearance_lift(effective_minimum, output['time'],
                                    clearance_m=clearance_m, smoothing_s=smoothing_s)
        desired = old_pos[:, foot].copy()
        desired[:, :2] = captured_xy
        desired[:, 2] += lift
        ankle_targets[:, side_index], lift_series[:, side_index] = desired, lift
        angles, margins, pole_records = [], [], []
        old_axes = ankle_pos-hip_pos
        old_axes /= np.linalg.norm(old_axes,axis=1,keepdims=True)
        old_bends = knee_pos-hip_pos-old_axes*np.sum((knee_pos-hip_pos)*old_axes,axis=1)[:,None]
        bend_height = np.linalg.norm(old_bends,axis=1)
        reliable = bend_height >= .020
        if not np.any(reliable):
            raise ValueError('Capture provides no reliable anatomical leg bend plane')
        local_poles = np.einsum('nji,nj->ni',old_rot[reliable,thigh[0]],
                               old_bends[reliable]/bend_height[reliable,None])
        history_pole = np.median(local_poles,axis=0)
        history_pole /= np.linalg.norm(history_pole)
        for k in range(count):
            h, knee, ankle = old_pos[k, [thigh[0], shin[0], foot]]
            margins.append(float(np.linalg.norm(knee-h) + np.linalg.norm(ankle-knee)
                                 - np.linalg.norm(desired[k]-h)))
            try:
                bend, history_pole, pole_record = transported_bend_direction(
                    h,knee,ankle,desired[k],old_rot[k,thigh[0]],history_pole)
                new_knee = two_bone_knee(h, knee, ankle, desired[k],
                                         bend_direction=bend,
                                         fallback_axis=old_rot[k, thigh[0], :, 0])
            except ValueError as exc:
                raise ValueError(f'{side} source frame {output["source_frame"][k]}: {exc}') from exc
            upper_q = _align_rotation(knee-h, new_knee-h)
            lower_q = _align_rotation(ankle-knee, desired[k]-new_knee)
            for j in thigh:
                output['bone_rot'][k, j] = upper_q @ old_rot[k, j]
            for j in shin:
                output['bone_rot'][k, j] = lower_q @ old_rot[k, j]
            angles.append([np.linalg.norm(Rotation.from_matrix(q).as_rotvec())
                           for q in (upper_q, lower_q)])
            pole_records.append(pole_record)
        descendants = {thigh[0]}
        for j, parent in enumerate(parents):
            if parent in descendants:
                descendants.add(j)
        affected.update(descendants)
        # Infer support only for reporting, never as a hidden contact constraint.
        source_end = source_names.index('Character1_' + source_side + 'Foot__end')
        end_z = output['source_joint_pos'][:, source_end, 2]
        speed = np.linalg.norm(np.gradient(output['source_joint_pos'][:, source_foot, :2],
                                          output['time'], axis=0), axis=1)
        support = (speed < .12) & (end_z < np.percentile(end_z, 10) + .025)
        segments = []
        for group in np.split(np.where(support)[0], np.where(np.diff(np.where(support)[0]) > 1)[0]+1):
            if len(group) >= 8:
                segments.append([float(output['source_time'][group[0]]),
                                 float(output['source_time'][group[-1]])])
        report[side] = {
            'maximum_ankle_correction_m':float(np.linalg.norm(desired-old_pos[:, foot], axis=1).max()),
            'maximum_vertical_lift_m':float(lift.max()),
            'maximum_fixed_length_reach_lift_m':float(reach_lift.max()),
            'reach_extension_margin_m':.0001,
            'maximum_thigh_shin_rotation_correction_deg':np.degrees(np.max(angles, axis=0)).tolist(),
            'minimum_extension_margin_m':float(min(margins)),
            'near_extension_plane_blend_frames':sum(x['history_blend_weight']<1 for x in pole_records),
            'pole_sign_alignment_frames':sum(x['pole_sign_aligned'] for x in pole_records),
            'bend_plane_method':'Minimal old-leg-axis→new-leg-axis transport; last reliable pole in captured thigh frame below5mm bend height, cubic blend5–20mm. Initial pole from median captured reliable local bends. No arbitrary world-axis pole.',
            'inferred_support_source_time_intervals_s':segments,
        }
    for j, parent in enumerate(parents):
        if j in affected:
            output['bone_pos'][:, j] = (output['bone_pos'][:, parent]
                + np.einsum('tij,j->ti', output['bone_rot'][:, parent], offsets[j]))
            output['bone_tail'][:, j] = (output['bone_pos'][:, j]
                + output['bone_rot'][:, j, :, 1] * lengths[j])
            output['bone_matrix'][:, j, :3, :3] = output['bone_rot'][:, j]
            output['bone_matrix'][:, j, :3, 3] = output['bone_pos'][:, j]
            output['bone_basis'][:, j] = (old_basis[:, j] @ np.linalg.inv(old_matrix[:, j])
                @ old_matrix[:, parent] @ np.linalg.inv(output['bone_matrix'][:, parent])
                @ output['bone_matrix'][:, j])
            if 'bone_local_quat_wxyz' in output:
                quats = Rotation.from_matrix(output['bone_basis'][:, j, :3, :3]).as_quat(scalar_first=True)
                for k in range(1, count):
                    if quats[k] @ quats[k-1] < 0:
                        quats[k] *= -1
                output['bone_local_quat_wxyz'][:, j] = quats
    actual = output['bone_pos'][:, [names.index('foot.L'), names.index('foot.R')]]
    endpoint_error = float(np.max(np.abs(actual-ankle_targets)))
    if endpoint_error > 1e-8:
        raise ValueError('Rig subdivision offsets disagree with the two-bone leg model')
    untouched = [j for j in range(bones) if j not in affected]
    for field in ['bone_pos','bone_rot','bone_tail','bone_matrix','bone_basis']:
        if not np.array_equal(output[field][:, untouched], arrays[field][:, untouched]):
            raise AssertionError('Non-leg transforms changed')
    for field in ['time','source_time','source_frame','pelvis_pos','source_joint_pos','source_joint_rot']:
        if not np.array_equal(output[field], arrays[field]):
            raise AssertionError('Capture clock, pelvis or source data changed')
    report.update({
        'schema':'doorbench.human-leg-contact-fit.v1',
        'status':'visual_contact_candidate_requires_deformed_surface_recheck',
        'source_xy_translation_m':translation.tolist(), 'clearance_m':clearance_m,
        'smoothing_s':smoothing_s, 'ankle_endpoint_max_coordinate_error_m':endpoint_error,
        'affected_bones':[names[j] for j in sorted(affected)],
        'unchanged_clock_pelvis_upperbody_source_arrays':True,
        'foot_and_toe_world_rotations_unchanged':all(np.array_equal(
            output['bone_rot'][:, j], old_rot[:, j]) for j in affected
            if names[j].startswith(('foot.', 'toe'))),
        'support_inference':'Source ankle XY speed<.12m/s and source foot-end Z<10thpercentile+25mm, runs>=8samples. Reporting only, not measured contact or a foot lock.',
        'limitations':['No recorded force, balance or dynamics solution.',
                       'Restoring source ankle XY adapts capture to target proportions; original rotations of thigh/shin change.',
                       'Predicted floor lift assumes rigidly translated feet; Blender skin weights require fresh surface checks.',
                       'No shoe-to-shoe collision constraint is hidden in this leg fit; test the resulting surfaces independently.'],
    })
    output['contact_fit_ankle_target'] = ankle_targets
    output['contact_fit_vertical_lift'] = lift_series
    return output, report
