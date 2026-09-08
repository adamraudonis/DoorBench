"""Independent reduction of measured full-opening development evidence."""
import numpy as np


def full_opening_checks(base_checks, *, steps, pad_steps, physics_dt, maximum_seconds,
                        target_aperture, operation_started, release_started):
    """Do not reuse a partial-opening/final-RH-grasp gate after intentional release.

    The replacement requires qualified acquisition and left support before
    release, then a loaded left palm through the final aperture crossing.
    Historical partial-opening reports and their failures are never modified.
    """
    if not np.isfinite([physics_dt, maximum_seconds, target_aperture]).all() or min(physics_dt, maximum_seconds, target_aperture)<=0:
        raise ValueError('Opening audit requires finite positive timing and aperture')
    if not isinstance(base_checks,dict) or any(not isinstance(v,(bool,np.bool_)) for v in base_checks.values()):
        raise ValueError('Actual physical check outcomes must be explicit booleans')
    for value in (operation_started,release_started):
        if value is not None and (not np.isfinite(value) or value<0):
            raise ValueError('Invalid actual handoff time')
    for row in steps:
        values=[row['time_s'],row['geometry']['time_s'],row['geometry']['right_lever_clearance_m'],
                row['surface']['total_normal_load_N'],row['surface']['palm_normal_load_N'],
                *(row['angles'][name] for name in ('leaf','operator','latch')),
                row['teacher'].get('release',{}).get('release_fraction',0)]
        if not np.isfinite(values).all() or min(values[3:5])<0:
            raise ValueError('Nonfinite or negative measured opening evidence')
    for row in pad_steps:
        if not np.isfinite(row['sim_time_s']) or not isinstance(row['valid_pad_grasp'],(bool,np.bool_)):
            raise ValueError('Invalid timestamped grasp evidence')
        if any(not isinstance(c['pad_qualified'],(bool,np.bool_)) for c in row['contacts']):
            raise ValueError('Pad qualification must be an actual boolean')
    checks = {k:bool(v) for k,v in base_checks.items()}
    checks.pop('sustained_pad_grasp', None)
    times = np.array([row['time_s'] for row in steps])
    end = float(times[-1]) if len(times) else 0.
    expected = round(end / physics_dt)
    complete = bool(expected > 0 and len(times) == expected and len(pad_steps) == expected + 1
                    and np.allclose(times, np.arange(1, expected + 1)*physics_dt, atol=1e-8, rtol=0)
                    and np.allclose([r['sim_time_s'] for r in pad_steps],
                                    np.arange(expected + 1)*physics_dt, atol=1e-8, rtol=0))
    crossing = bool(steps and steps[-1]['angles']['leaf'] >= target_aperture)
    if crossing and any(row['angles']['leaf'] >= target_aperture for row in steps[:-1]):
        complete = False  # The declared terminal event was skipped.
    checks['complete_physics_steps'] = bool(complete and end <= maximum_seconds + 1e-8
        and (crossing or abs(end-maximum_seconds) < 1e-8))

    def held(rows, at, time_key, predicate):
        if at is None or not np.isfinite(at):
            return False
        tail = [r for r in rows if at-.5-1e-8 <= r[time_key] <= at+1e-8]
        return bool(len(tail) == round(.5/physics_dt)+1
                    and abs(tail[-1][time_key]-at) < 1e-8
                    and abs(tail[0][time_key]-(at-.5)) < 1e-8
                    and all(predicate(r) for r in tail))

    checks.update(
        acquisition_precedes_operation=held(pad_steps, operation_started, 'sim_time_s', lambda r:r['valid_pad_grasp']),
        operator_driven_to_release=bool(steps and max(r['angles']['operator'] for r in steps)>=.8
                                       and max(r['angles']['latch'] for r in steps)>=.011),
        qualified_grasp_before_intentional_release=held(pad_steps, release_started, 'sim_time_s', lambda r:r['valid_pad_grasp']),
        left_support_precedes_right_release=held(steps, release_started, 'time_s', lambda r:r['surface']['total_normal_load_N']>=2.),
        sustained_left_palm_load=held(steps, end, 'time_s', lambda r:r['surface']['palm_normal_load_N']>=2.),
        usable_aperture_under_palm_load=bool(crossing and steps[-1]['surface']['palm_normal_load_N']>=2.),
        right_release_completed=bool(steps and release_started is not None
            and steps[-1]['geometry']['right_lever_clearance_m']>=.02
            and steps[-1]['teacher'].get('release',{}).get('release_fraction',0)>=.999),
        no_invalid_right_pad_patch=all(c['pad_qualified'] for row in pad_steps for c in row['contacts']),
        pose_joint_clock_aligned=bool(steps and all(abs(r['time_s']-r['geometry']['time_s'])<1e-8 for r in steps)),
        no_native_mirror_steps=bool(steps and all(r['geometry']['native_mirror_steps']==0 for r in steps)))
    return checks
