"""Privileged contact evidence for cylindrical lever grasps.

This is a diagnostic, not a universal grasp criterion: knobs, pulls and buttons
need separate task-specific contact requirements. Touch proximity is not force.
"""
import re
import numpy as np
import mujoco

DIGITS=('ff','mf','rf','lf','th')


def opposition(contacts, center, axis, *, minimum_force=.2):
    """Require four finger contact centroids on one radial side, thumb opposite."""
    center=np.asarray(center,dtype=float);axis=np.asarray(axis,dtype=float)
    norm=np.linalg.norm(axis)
    if norm<1e-8:raise ValueError('Lever axis must be nonzero')
    axis=axis/norm;directions={};forces={}
    for digit in DIGITS:
        points=[c for c in contacts if c['digit']==digit and c['normal_force_N']>0]
        force=sum(c['normal_force_N'] for c in points);forces[digit]=float(force)
        if force<minimum_force:continue
        centroid=sum(np.asarray(c['position'])*c['normal_force_N'] for c in points)/force
        radial=centroid-center;radial-=np.dot(radial,axis)*axis
        length=np.linalg.norm(radial)
        if length>1e-6:directions[digit]=radial/length
    if len(directions)!=5:
        return {'opposed':False,'digit_forces_N':forces,'reason':'missing loaded digit contact'}
    finger_mean=sum(directions[d] for d in DIGITS[:-1])
    length=np.linalg.norm(finger_mean)
    if length<1e-6:
        return {'opposed':False,'digit_forces_N':forces,'reason':'fingers span opposing sides'}
    finger_mean/=length
    alignment=min(float(np.dot(directions[d],finger_mean)) for d in DIGITS[:-1])
    thumb_dot=float(np.dot(directions['th'],finger_mean))
    return {'opposed':bool(alignment>.5 and thumb_dot<-.5), 'digit_forces_N':forces,
            'minimum_finger_alignment':alignment,'thumb_opposition_dot':thumb_dot,
            'reason':'geometry and loaded contacts checked'}


def lever_contacts(model,data,lever_geom,*,side='rh'):
    """Read solved contacts against precisely the named lever collision shape."""
    if side not in ('rh','lh'):raise ValueError('Unknown hand side')
    target=model.geom(lever_geom).id
    if model.geom_type[target] not in (mujoco.mjtGeom.mjGEOM_CAPSULE,mujoco.mjtGeom.mjGEOM_CYLINDER):
        raise ValueError('Lever opposition audit requires a cylindrical collision shape')
    contacts=[]
    for i in range(data.ncon):
        contact=data.contact[i]
        if target not in contact.geom:continue
        other=int(contact.geom[1] if contact.geom[0]==target else contact.geom[0])
        name=model.body(model.geom_bodyid[other]).name
        match=re.match(r'robot/'+side+r'_(ff|mf|rf|lf|th)',name)
        if not match:continue
        force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
        contacts.append({'digit':match.group(1),'position':contact.pos.copy().tolist(),
                         'normal_force_N':float(max(0,force[0])),'distance_m':float(contact.dist)})
    result=opposition(contacts,data.geom_xpos[target],data.geom_xmat[target].reshape(3,3)[:,2])
    return {**result,'contacts':contacts,'hand':side,'lever_geom':lever_geom}
