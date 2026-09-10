"""Geometric receiving-palm reach check; never substitutes for measured load."""
import mujoco
import numpy as np


def receiving_palm_gap(model, data, *, maximum_gap_m=.008):
    """Require a palm collider within the controller's bounded capture distance.

    Fingers and hardware cannot stand in for the receiving palm and panel.
    The dense route audit separately checks penetration and all other contacts.
    """
    if not np.isfinite(maximum_gap_m) or not 0 < maximum_gap_m <= .008:
        raise ValueError('Capture distance must not exceed the 8 mm controller bound')
    palms=[g for g in range(model.ngeom)
           if model.body(int(model.geom_bodyid[g])).name=='robot/lh_palm'
           and (model.geom_contype[g] or model.geom_conaffinity[g])]
    panels=[g for g in range(model.ngeom)
            if model.body(int(model.geom_bodyid[g])).name=='leaf'
            and (model.geom(g).name or '').startswith('leaf_slab')
            and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not palms or not panels:
        raise ValueError('Original palm and slab collision geometries required')
    mujoco.mj_kinematics(model,data)
    gap=min(float(mujoco.mj_geomDistance(model,data,a,b,1.,None))
            for a in palms for b in panels)
    return dict(passed=bool(np.isfinite(gap) and -.003<=gap<=maximum_gap_m),
                minimum_palm_slab_distance_m=gap,maximum_capture_distance_m=maximum_gap_m,
                scope='Unstepped geometric reach only; actual sustained palm force still required')
