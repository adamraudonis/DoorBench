"""Bounded inputs on actual lockset surfaces; no hand or grasp certificate."""
from __future__ import annotations

import numpy as np


def surface_action(env, rows, action):
    """Replace commanded trim torques with lever or opposed knob forces."""
    result=dict(action)
    out=dict(result.get('torques',{}))
    commands=dict(result.get('site_forces',{}))
    first=None
    m,d=env.m,env.d
    for row in rows:
        for name,sites in row['input_sites'].items():
            effort=out.pop(name,0.)
            if abs(effort)<1e-8:continue
            j=m.joint(name).id;dof=int(m.jnt_dofadr[j]);bid=int(m.jnt_bodyid[j])
            ids=[m.site(site).id for site in sites]
            if not ids or any(m.site_bodyid[sid]!=bid for sid in ids):
                raise ValueError('Rotary input must touch its own articulated trim')
            tangents=[np.cross(d.xaxis[j],d.site_xpos[sid]-d.xanchor[j]) for sid in ids]
            radii=[float(np.linalg.norm(tangent)) for tangent in tangents]
            if min(radii)<.01:raise ValueError('Rotary input has no useful moment arm')
            # The operator phase selects turn or return. The cap is per
            # actual surface, matching the declared component test fixture.
            target=row['operator_travel_rad']+.4 if effort>0 else 0.
            q=float(d.qpos[m.jnt_qposadr[j]])
            force=float(np.clip((12*(target-q)-.2*d.qvel[dof])/sum(radii),
                                -row['operator_force_cap_N'],row['operator_force_cap_N']))
            for site,tangent,radius in zip(sites,tangents,radii):
                if site in commands:raise ValueError('Duplicate force source on rotary grip')
                commands[site]=(tangent/radius*force).tolist()
            if first is None:first=(name,sites[0])
    result['torques']=out
    if commands:result['site_forces']=commands
    if first:result['contact_joint'],result['contact_site']=first
    return result
