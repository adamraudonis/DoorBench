"""Phase-specific Isaac withdrawal checks with original native thresholds."""
import copy
import math
import numpy as np

from .isaac_opening_measurements import pose_parts
from .standing_withdrawal_audit import withdrawal_checks as original_withdrawal_checks


def pack_withdrawal_row(time_s, angles, pad, surface, teacher, geometry=None, *, leaf_pose):
    if abs(float(pad['sim_time_s'])-time_s) > 1e-8:
        raise ValueError('Withdrawal and measured contact epochs differ')
    return dict(sim_time_s=float(time_s), door_q=float(angles['leaf']),
        handle_angle_rad=float(angles['operator']), bolt_slide_m=float(angles['latch']),
        pad_grasp=copy.deepcopy({k: pad[k] for k in ('valid_pad_grasp', 'contacts')}),
        left_surface=copy.deepcopy(surface), leaf_pose=np.asarray(leaf_pose, float).tolist(),
        geometry=copy.deepcopy(geometry),
        right_environment_clearance_m=None if geometry is None else geometry['right_environment_clearance_m'],
        teacher=copy.deepcopy({k: teacher.get(k) for k in (
            'phase', 'stance_status', 'withdrawal_started_s', 'release_started_s',
            'withdrawal_progress', 'withdrawal_clock_s', 'grip_preload_scale',
            'support_load_target_N', 'filtered_palm_load_N', 'requested_normal_force_N',
            'inherited_support', 'withdrawal_motor_capture', 'captured_command_time_s',
            'maximum_cache_disagreement_Nm', 'captured_command_Nm')}))


def withdrawal_checks(original, rows, *, dt, duration, started, release_started, completed):
    if not all(math.isfinite(x) and x > 0 for x in (dt, duration)) or dt != .002:
        raise ValueError('Original 500 Hz finite withdrawal clock required')
    metrics = dict(count=0, clock=True, palm=True, geometry=True, tail=0,
        final_support=True, entry_count=0, entry_support=True)

    def checked_rows():
        for row in rows:
            metrics['count'] += 1
            t = row['sim_time_s']
            metrics['clock'] &= math.isfinite(t) and abs(t-metrics['count']*dt) <= 1e-8
            surface = row['left_surface']
            _, rotation = pose_parts(row['leaf_pose'])
            vector = np.asarray(surface['body_panel_forces_world_N']['lh_palm'], float)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError('Actual finite palm force vector required')
            load = max(0., float(-rotation[:, 1]@vector))
            reported = surface['palm_normal_load_N']
            metrics['palm'] &= math.isfinite(reported) and abs(load-reported) <= 1e-8
            if t >= duration-.5-1e-8:
                metrics['tail'] += 1
                metrics['final_support'] &= load >= 2.
            if started is not None and started-.5-1e-8 <= t <= started+1e-8:
                metrics['entry_count'] += 1
                metrics['entry_support'] &= load >= 2.
            geometry = row.get('geometry')
            if started is not None and t > started+1e-8:
                metrics['geometry'] &= bool(geometry is not None
                    and geometry.get('native_mirror_steps') == 0
                    and abs(geometry.get('time_s', -1.)-t) <= 1e-8
                    and abs(geometry.get('geometry_time_s', -1.)-t) <= 1e-8
                    and math.isfinite(geometry.get('right_environment_clearance_m', float('nan')))
                    and geometry.get('maximum_pose_position_error_m', float('inf')) <= .003
                    and geometry.get('maximum_pose_rotation_error', float('inf')) <= .02
                    and bool(geometry.get('body_poses_xyz_wxyz')))
            # Before the stage there is deliberately no extra geometry query.
            # The original final-clearance predicate only consumes the tail.
            yield {**row, 'right_environment_clearance_m': row.get('right_environment_clearance_m')
                if row.get('right_environment_clearance_m') is not None else -1.}

    checks = original_withdrawal_checks(original, checked_rows(), dt=dt, duration=duration,
        started=started, release_started=release_started, completed=completed)
    checks.update(complete_withdrawal_clock=bool(metrics['clock'] and metrics['count'] == round(duration/dt)),
        measured_palm_load_accounting=bool(metrics['palm']),
        synchronized_withdrawal_geometry=bool(started is not None and metrics['geometry']),
        resting_palm_support_before_withdrawal=bool(metrics['entry_count'] == round(.5/dt)+1 and metrics['entry_support']),
        final_left_palm_support=bool(metrics['tail'] == round(.5/dt)+1 and metrics['final_support']))
    return checks
