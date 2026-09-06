"""Exact primitive mounting/contact checks for authored rotating-door hardware.

The checks do not certify credential control, ratchets, strength or traversal.
"""
from __future__ import annotations

import math
import numpy as np


def _surface_distance(kind, size, local):
    import mujoco
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        q=np.array([np.linalg.norm(local[:2])-size[0],abs(local[2])-size[1]])
        return float(np.linalg.norm(np.maximum(q,0))+min(float(q.max()),0))
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        x=local.copy();x[2]-=np.clip(x[2],-size[1],size[1])
        return float(np.linalg.norm(x)-size[0])
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        q=np.abs(local)-size
        return float(np.linalg.norm(np.maximum(q,0))+min(float(q.max()),0))
    raise ValueError('Rotating contact must bind an authored primitive')


def run_rotating_hardware_qa(model, metadata):
    import mujoco
    if metadata.get('family') not in ('revolving','turnstile_tripod','turnstile_fullheight'):
        return {'ok':True,'applicable':False,'failures':[]}
    d=mujoco.MjData(model);mujoco.mj_kinematics(model,d)
    failures=[];contacts=[];attachments=[];journals=[]
    def gap(a,b):return float(mujoco.mj_geomDistance(model,d,model.geom(a).id,model.geom(b).id,1.,None))
    if not metadata.get('rotating_contacts'):failures.append('Missing real rotating contact metadata')
    for c in metadata.get('rotating_contacts',[]):
        try:
            sid=model.site(c['site']).id;gid=model.geom(c['geom']).id;jid=model.joint(c['joint']).id
        except KeyError:
            failures.append({'missing_contact_binding':c});continue
        p=d.site_xpos[sid];rotation=d.geom_xmat[gid].reshape(3,3)
        local=rotation.T@(p-d.geom_xpos[gid])
        distance=_surface_distance(model.geom_type[gid],model.geom_size[gid],local)
        normal=d.site_xmat[sid].reshape(3,3)[:,2]
        # Outside/inside witnesses prove the authored normal points outward.
        outside=_surface_distance(model.geom_type[gid],model.geom_size[gid],rotation.T@(p+normal*.001-d.geom_xpos[gid]))
        inside=_surface_distance(model.geom_type[gid],model.geom_size[gid],rotation.T@(p-normal*.001-d.geom_xpos[gid]))
        moment=float(np.dot(np.cross(p-d.xanchor[jid],-normal),d.xaxis[jid]))
        row={'site':c['site'],'geom':c['geom'],'surface_residual_m':distance,'outward_1mm_m':outside,
             'inward_1mm_m':inside,'positive_torque_per_N_m':moment}
        contacts.append(row)
        if model.site_bodyid[sid]!=model.geom_bodyid[gid]:failures.append({'contact_body_mismatch':row})
        if abs(distance)>2e-6 or outside<.00099 or inside>-.00099:failures.append({'invalid_contact_surface':row})
        if moment<=.1:failures.append({'invalid_push_moment':row})
    for row in metadata.get('rotating_hardware',[]):
        for a,b in row.get('attachments',[]):
            value=gap(a,b);attachments.append({'pair':[a,b],'gap_m':value})
            if value>.0005:failures.append({'detached_mount':[a,b],'gap_m':value})
        if row['kind']=='tripod_journal':
            shaft=model.geom(row['shaft']).id;jid=model.joint(row['joint']).id
            axis=d.xaxis[jid];shaft_axis=d.geom_xmat[shaft].reshape(3,3)[:,2]
            alignment=float(abs(axis@shaft_axis));eccentricity=float(np.linalg.norm(np.cross(d.geom_xpos[shaft]-d.xanchor[jid],axis)))
            least=1.
            for angle in np.linspace(0,2*math.pi,81):
                d.qpos[model.jnt_qposadr[jid]]=angle;mujoco.mj_kinematics(model,d)
                least=min(least,*(gap(row['shaft'],n) for n in row['bearing_geoms']))
            journals.append({'axis_alignment':alignment,'eccentricity_m':eccentricity,'minimum_radial_gap_m':least})
            if alignment<1-1e-6 or eccentricity>1e-6:failures.append({'noncoaxial_journal':journals[-1]})
            if least<.00049:failures.append({'journal_obstruction':journals[-1]})
    return {'ok':not failures,'applicable':True,'failures':failures,'contacts':contacts,'attachments':attachments,'journals':journals,
            'scope':'Exact declared primitive surfaces, normals, torque directions, attachments and journal sweep; no ratchet or traversal certificate'}
