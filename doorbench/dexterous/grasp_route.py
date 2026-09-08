"""Geometric pregrasp clearance checks, never physical acquisition scores.

The path planner fits backwards from a measured terminal grasp. Positive
clearance before terminal seating prevents a shallow, wrong-side contact from
being mistaken for a collision-free approach. Native execution remains required.
"""
import numpy as np
import mujoco


def reproject_attached_pose(position, rotation, reference_position, reference_rotation,
                            actual_position, actual_rotation):
    """Preserve a proposed hand pose relative to a moving physical operator.

    All transforms are privileged teacher reference/state, never sensor actor
    inputs. This only returns a target; the robot must execute it with motors.
    """
    positions=[np.asarray(x,dtype=float) for x in (position,reference_position,actual_position)]
    rotations=[np.asarray(x,dtype=float) for x in (rotation,reference_rotation,actual_rotation)]
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in positions):raise ValueError('Invalid position')
    if any(x.shape!=(3,3) or not np.isfinite(x).all() or not np.allclose(x.T@x,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(x),1,atol=1e-6) for x in rotations):raise ValueError('Invalid rotation')
    target,origin,current=positions;target_rotation,source_rotation,current_rotation=rotations
    correction=current_rotation@source_rotation.T
    return current+correction@(target-origin),correction@target_rotation


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
