"""Privileged release-only clearance feedback through original finger motors."""
import mujoco
import numpy as np
from .handle_hub_avoidance import avoidance_force


class LeverSegmentAvoidance:
    def __init__(self, teacher):
        self.teacher = teacher
        m = teacher.m
        self.lever = m.geom('analytic_lever_capsule').id
        self.geoms = [g for g in range(m.ngeom) if m.geom_contype[g]
                      and m.body(m.geom_bodyid[g]).name in ('rh_rfmiddle', 'rh_rfproximal')]
        if not self.geoms:
            raise ValueError('Original ring-finger segment geometry required')
        self.jac = np.zeros((3, m.nv))

    def force(self, forces, blend):
        if not np.isfinite(blend) or not 0 <= blend <= 1:
            raise ValueError('Bounded release blend required')
        t = self.teacher
        nearest = []
        for g in self.geoms:
            pair = np.zeros(6)
            gap = mujoco.mj_geomDistance(t.m, t.d, g, self.lever, .02, pair)
            nearest.append((gap, g, pair.copy()))
        gap, g, pair = min(nearest, key=lambda item: item[0])
        delta = pair[:3] - pair[3:]
        distance = np.linalg.norm(delta)
        force = np.zeros(3)
        result = np.asarray(forces, float).copy()
        if distance > 1e-7 and gap < .004:
            direction = delta / distance * (1 if gap >= 0 else -1)
            mujoco.mj_jac(t.m, t.d, self.jac, None, pair[:3], int(t.m.geom_bodyid[g]))
            force = blend * avoidance_force(gap, direction, self.jac @ t.d.qvel)
            generalized = self.jac.T @ force
            result[t.fingers] += t.finger_inverse @ generalized[t.va]
        result = np.clip(result, t.caps[:, 0], t.caps[:, 1])
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite segment-clearance command')
        return result, dict(release_segment_avoidance_profile='ring-nonpad-lever-3N-v1',
                            release_segment_gap_m=float(gap),
                            release_segment_force_N=float(np.linalg.norm(force)))
