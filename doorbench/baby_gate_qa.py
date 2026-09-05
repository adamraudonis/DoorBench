"""Independent compiled-geometry check for an open passage above a baby gate.

This tests the static reference scene, including visible non-colliding geometry.
It is a regression gate for the erroneous low lintel, not an accessibility,
structural, or child-safety certification. The surrounding room ceiling may
remain above the tested 2.7 m passage; side walls and gate hardware remain.
"""
from __future__ import annotations

import numpy as np


def run_baby_gate_qa(model, spec):
    import mujoco

    if spec.get("family") != "baby_gate":
        return {"ok": True, "applicable": False, "failures": []}
    opening, leaf = spec["opening"], spec["leaf"]
    # Stay 40 mm inside the clear opening and 100 mm above the leaf/hardware.
    # Test the full wall thickness at its actual offset from the leaf plane.
    width, thickness = float(opening["width"]), float(opening["wall_thickness"])
    lower = max(float(opening["height"]), float(leaf["height"]) +
                float(opening.get("ground_clearance", 0.03))) + 0.1
    direction = 1.0 if spec["robot"]["is_push"] else -1.0
    from .geometry.common import LEAF_FACE_INSET
    wall_y = -direction * max(0.0, thickness / 2 - float(leaf["thickness"]) / 2 - LEAF_FACE_INSET)
    lo = np.array([-width / 2 + 0.04, wall_y - thickness / 2, lower])
    hi = np.array([width / 2 - 0.04, wall_y + thickness / 2, 2.7])
    if np.any(hi <= lo):
        return {"ok": False, "applicable": True, "failures": [{"reason": "Invalid overhead test volume"}]}
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    failures = []
    for gid in range(model.ngeom):
        # MuJoCo supplies local bounding boxes for primitives and mesh vertices;
        # include both visual and collision geometry, independent of names.
        rotation = data.geom_xmat[gid].reshape(3, 3)
        center = data.geom_xpos[gid] + rotation @ model.geom_aabb[gid, :3]
        extent = np.abs(rotation) @ model.geom_aabb[gid, 3:]
        overlap = np.minimum(center + extent, hi) - np.maximum(center - extent, lo)
        if np.all(overlap > 1e-6):
            failures.append({"geom": model.geom(gid).name, "overlap_m": overlap.tolist()})
    return {"ok": not failures, "applicable": True, "volume_min_m": lo.tolist(),
            "volume_max_m": hi.tolist(), "n_geoms_checked": model.ngeom,
            "scope": "Reference-scene geometry in the clear passage above the gate, up to 2.7 m",
            "failures": failures}
