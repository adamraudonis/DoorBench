"""Opt-in static panel-normal clearance for an actual-state left-hand path.

This changes only geometric planning targets. It never steps or writes an active
plant, and its result requires the independent dense path audit before use.
"""
import copy
import numpy as np
from scipy.optimize import least_squares

from .landed_left_planner import read_targets


def clearance_envelope(u, distance):
    if not np.isfinite([u, distance]).all() or not 0 <= u <= 1 or not 0 < distance <= .025:
        raise ValueError('Finite path coordinate and explicit 0..25 mm clearance required')
    return float(distance*np.sin(np.pi*u))


def add_left_approach_clearance(scene, config, state, *, distance_m):
    """Move intermediate palm goals outward while preserving both endpoints.

    The attained torso, fingers, legs, opposite arm and door remain frozen.
    Seven bounded arm/wrist joints solve each new goal. The original normalized
    path timing is retained; its controller supplies endpoint velocity easing.
    """
    source = read_targets(dict(config, passed=True))
    frozen = scene.freeze(state)
    q = frozen.copy()
    m, qa = scene.m, scene.qa
    initial, _ = scene.palm_local(frozen)
    if abs(initial[1]) < .001:
        raise ValueError('Measured hand must identify an unambiguous panel side')
    side = float(np.sign(initial[1]))
    lower, upper = m.jnt_range[scene.arm[1:], 0]+.01, m.jnt_range[scene.arm[1:], 1]-.01
    result = copy.deepcopy(config)
    reports = []
    for i, row in enumerate(result['targets']):
        u = i/(len(result['targets'])-1)
        offset = side*clearance_envelope(u, distance_m)
        position = source['position'][i].copy()
        position[1] += offset
        normal = source['normal'][i]
        nominal = source['nominal'][i]
        q[:] = frozen
        q[qa] = nominal
        if i in (0, len(result['targets'])-1):
            reports.append(dict(index=i, converged=True, position_error_m=0., normal_error_rad=0.))
            continue

        def residual(arm):
            q[qa[1:]] = arm
            pos, orientation = scene.palm_local(q)
            return np.r_[100*(pos-position), 10*(orientation-normal), .03*(arm-nominal[1:])]

        fit = least_squares(residual, np.clip(nominal[1:], lower, upper),
                            bounds=(lower, upper), max_nfev=200)
        q[qa[1:]] = fit.x
        pos, orientation = scene.palm_local(q)
        row.update(position=pos.tolist(), normal=orientation.tolist(), nominal=q[qa].tolist())
        reports.append(dict(index=i, converged=bool(fit.success),
            position_error_m=float(np.linalg.norm(pos-position)),
            normal_error_rad=float(np.arccos(np.clip(orientation@normal, -1., 1.)))))
    checks = dict(solvers_converged=all(r['converged'] for r in reports),
        bounded_position_error=all(r['position_error_m'] <= .001 for r in reports),
        bounded_normal_error=all(r['normal_error_rad'] <= np.deg2rad(.5) for r in reports),
        endpoints_unchanged=all(result['targets'][i] == config['targets'][i] for i in (0,-1)))
    result['passed'] = False
    result['source']['approach_clearance'] = dict(distance_m=float(distance_m), side=side,
        profile='panel-normal-sine-v1', frozen_coordinates='root, torso, fingers, legs, right arm, door')
    return result, dict(passed=all(checks.values()), checks=checks, samples=reports,
        physics_steps=0, scope='Geometric clearance candidate; dense collision audit and physical trial still required')
