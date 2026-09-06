"""Deterministic collision-checked motor-reference path for withdrawing an arm.

Pose queries use a separate MjData. Execution still uses native joint motors.
The planner checks every 0.035 rad along edges, including the open finger skin.
"""

import mujoco
import numpy as np


def plan_release(model, data, arm_names, destination, seed=17):
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    qids = np.array([model.joint(name).qposadr[0] for name in arm_names])
    start = data.qpos[qids].copy()
    destination = np.asarray(destination)
    bodies = {
        model.body(n).id for n in ["actor_shoulder_l", "actor_elbow_l", "actor_wrist_l"]
    }
    arm = [
        g
        for g in range(model.ngeom)
        if model.geom_contype[g]
        and (
            model.geom_bodyid[g] in bodies
            or (model.body(model.geom_bodyid[g]).name or "").startswith("hand_l_")
        )
    ]
    environment = [
        g
        for g in range(model.ngeom)
        if model.geom_contype[g]
        and model.geom(g).name != "floor"
        and not (model.body(model.geom_bodyid[g]).name or "").startswith(
            ("actor_", "hand_")
        )
    ]
    checks = 0

    def clearance(q):
        nonlocal checks
        checks += 1
        scratch.qpos[qids] = q
        mujoco.mj_forward(model, scratch)
        return min(
            mujoco.mj_geomDistance(model, scratch, a, b, 0.20, None)
            for a in arm
            for b in environment
        )

    def edge(a, b):
        return all(
            clearance(a + (b - a) * t) > 0.004
            for t in np.linspace(0, 1, max(2, int(np.max(abs(b - a)) / 0.035) + 1))
        )

    if not clearance(start) > 0.004:
        raise RuntimeError("Release must begin with at least 4 mm clearance")
    if edge(start, destination):
        via = (start + destination) / 2
    else:
        rng = np.random.default_rng(seed)
        limits = model.jnt_range[[model.joint(n).id for n in arm_names]]
        candidates = []
        for k in range(4000):
            via = start.copy()
            via[:3] = rng.uniform(limits[:3, 0], limits[:3, 1])
            via[3] = rng.uniform(1.6, 2.5)
            via[4:] = start[4:] * rng.uniform(0.5, 1)
            if clearance(via) > 0.004 and edge(start, via) and edge(via, destination):
                candidates.append(
                    (
                        np.linalg.norm(via - start) + np.linalg.norm(destination - via),
                        via.copy(),
                    )
                )
                if len(candidates) >= 8:
                    break
        if not candidates:
            raise RuntimeError("No collision-free arm withdrawal path found")
        via = min(candidates, key=lambda item: item[0])[1]
    path = np.array([start, via, destination])
    return path, {
        "seed": seed,
        "native_pose_queries": checks,
        "edge_resolution_rad": 0.035,
        "minimum_required_clearance_m": 0.004,
        "waypoints_rad": path.tolist(),
    }
