"""Read actual articulated-lift travel; never prescribe a native configuration."""
from __future__ import annotations

import numpy as np


def lift_state(model, data, meta, with_velocity=False):
    """Measure the bottom of a sectional or winding curtain in metres."""
    if meta.get('sectional_track'):
        return sectional_state(model, data, meta, with_velocity)
    record = meta.get('rollup_curtain')
    if not record:
        return None
    bounds = record['progress']
    sid = model.site(bounds['site']).id
    span = bounds['open_z_m'] - bounds['closed_z_m']
    travel = float(data.site_xpos[sid, 2]) - bounds['closed_z_m']
    result = {'travel_m': travel, 'progress': travel/span, 'span_m': span,
              'tangent': [0., 0., 1.], 'point': data.site_xpos[sid].tolist(),
              'carried_mass_kg': float(model.body_subtreemass[model.body('curtain_barrel').id])}
    if with_velocity:
        import mujoco
        from ..rollup import rollup_grip_dynamics
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac, None, sid)
        result['velocity'] = (jac @ data.qvel).tolist()
        result['speed_m_s'] = result['velocity'][2]
        if not meta.get('rollup_hoist'):
            result.update(rollup_grip_dynamics(model, data, record))
    return result


def sectional_state(model, data, meta, with_velocity=False):
    record = meta.get('sectional_track')
    if not record:
        return None
    from ..geometry.sectional import track_progress, track_path
    sid = model.site(record['progress']['site']).id
    s, residual = track_progress(data.site_xpos[sid, 1:3], record['path'])
    _, tangent = track_path(s, record['path'])
    bounds = record['progress']
    travel = s - bounds['closed_s_m']
    span = bounds['open_s_m'] - bounds['closed_s_m']
    result = {'travel_m': travel, 'progress': travel/span, 'span_m': span,
              'track_error_m': residual, 'tangent': [0., *tangent.tolist()],
              'point': data.site_xpos[sid].tolist()}
    if with_velocity:
        import mujoco
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac, None, sid)
        result['speed_m_s'] = float(np.dot(jac @ data.qvel, result['tangent']))
        result['velocity'] = (jac @ data.qvel).tolist()
    return result
