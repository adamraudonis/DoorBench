"""Geometric pregrasp clearance checks, never physical acquisition scores.

The path planner fits backwards from a measured terminal grasp. Positive
clearance before terminal seating prevents a shallow, wrong-side contact from
being mistaken for a collision-free approach. Native execution remains required.
"""
import numpy as np
import mujoco


def approach_clearance_requirement(reverse_fraction, maximum, terminal_gap=0., *, taper_fraction=.2):
    """Taper from measured terminal overlap at 0 to approach clearance at 1."""
    values = (reverse_fraction, maximum, terminal_gap, taper_fraction)
    if not all(np.isfinite(v) for v in values):
        raise ValueError('Clearance parameters must be finite')
    if not 0. <= reverse_fraction <= 1. or not 0. < taper_fraction <= 1.:
        raise ValueError('Path and taper fractions must lie within the path')
    if maximum < 0. or terminal_gap > 0.:
        raise ValueError('Use nonnegative approach clearance and nonpositive terminal overlap')
    blend = min(1., reverse_fraction / taper_fraction)
    return float(terminal_gap * (1. - blend) + maximum * blend)


def hand_lever_clearance_failures(model, data, hand_geoms, lever_geom, required_clearance, *, tolerance=.0002):
    """Check all supplied hand surfaces, including interpolated path poses.

A negative requirement allows only the measured terminal soft-contact overlap.
Callers choose when to apply the check; this function has no unchecked terminal
interval. The tolerance is geometric screening tolerance, not a physics limit.
    """
    if not np.isfinite(required_clearance) or not np.isfinite(tolerance) or tolerance < 0.:
        raise ValueError('Use finite clearance and nonnegative finite tolerance')
    if len(hand_geoms) == 0:
        raise ValueError('At least one hand collision geometry is required')
    # Keep the distance query valid even for a negative terminal requirement.
    query_limit = max(1e-6, required_clearance + tolerance + .001)
    gap = min(float(mujoco.mj_geomDistance(model, data, int(g), int(lever_geom), query_limit, None))
              for g in hand_geoms)
    if not np.isfinite(gap):
        raise ValueError('Nonfinite hand/lever distance cannot pass route screening')
    if gap < required_clearance - tolerance:
        return [dict(reason='precontact clearance', required_m=float(required_clearance), gap_m=gap)]
    return []
