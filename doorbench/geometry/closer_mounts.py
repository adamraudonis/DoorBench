"""Original dimensioned supports for the generic surface-closer linkage.

These are not OEM mounting templates. The pinion spring transmits its force
through these arms; housing internals remain an ideal torsional spring.
"""
from __future__ import annotations
import numpy as np
from ..ir import QUAT_ID, ALL_TIERS
from . import common as C


def frame_face(world, x, z, face, pad=.023):
    candidates=[]
    for geom in world.geoms:
        if geom.type!='box' or not np.allclose(geom.quat,QUAT_ID):continue
        if not geom.name.startswith(('jamb_head','head_stud','casing_h')):continue
        if abs(x-geom.pos[0])<=geom.size[0] and abs(z-geom.pos[2])<=geom.size[2]+pad:
            candidates.append((face*geom.pos[1]+geom.size[1],geom.name))
    if not candidates:raise ValueError('Closer shoe has no actual head/frame mounting surface')
    depth,name=max(candidates)
    return face*depth,name


def shoe_ring(geoms,center,material,*,name='closer_shoe_block',half_width=.0195,half_depth=.010):
    """Four solids around an actual 14 mm bore for a 10 mm pivot pin."""
    x,y,z=center;result=[]
    for axis,outer in ((0,half_width),(1,half_depth)):
        for side in (-1,1):
            p=[x,y,z];p[axis]+=side*(outer+.007)/2
            half=[.007,half_depth,.008] if axis==0 else [.007,.007,.008]
            half[axis]=(outer-.007)/2
            n=f'{name}_{axis}_{side}'
            geoms.append(C.box(n,tuple(p),tuple(half),material,2700,False,True,ALL_TIERS,'closer','Open shoe pivot bearing'))
            result.append(n)
    return result


def fixed_shoe(world, x, y, z, face, surface, material):
    # The bearing sits below the arm plane, so the forearm sweeps above it.
    block=shoe_ring(world.geoms,(x,y,z-.018),material)
    mount_y=surface+face*.004
    world.geoms.append(C.box('closer_bracket',(x,mount_y,z),(.032,.004,.024),
        material,2700,False,True,ALL_TIERS,'closer','Frame-mounted shoe backplate'))
    # Two side gussets carry the bearing to the screw plate without filling its bore.
    for side in (-1,1):
        world.geoms.append(C.box(f'closer_shoe_support_{side}',(x+side*.015,(mount_y+y-face*.010)/2,z-.018),
            (.0045,abs(y-face*.010-mount_y)/2,.008),material,2700,False,True,ALL_TIERS,'closer','Shoe support to frame backplate'))
    return block


def resolve_closer_configuration(model, qpos, metadata):
    """Resolve only the authored planar closer loops for inspection.

    The leaf state is prescribed. Both passive arm hinges and a cam-lift shoe
    follow the same native connect anchors; no contact or driver is waived.
    """
    import mujoco
    q=np.asarray(qpos)
    if not metadata.get('closer_mounts'):return qpos
    d=mujoco.MjData(model);rest=mujoco.MjData(model);mujoco.mj_kinematics(model,rest)
    for row in metadata['closer_mounts']:
        if row.get('kind')=='single_arm_track':
            from .closer_track import resolve_track_configuration
            resolve_track_configuration(model,q,row)
            continue
        main=model.joint(row['main_joint']).id;elbow=model.joint(row['elbow_joint']).id
        a1=int(model.jnt_qposadr[main]);a2=int(model.jnt_qposadr[elbow])
        bmain=int(model.jnt_bodyid[main]);bfore=int(model.jnt_bodyid[elbow]);eq=model.equality(row['connect']).id
        anchor=np.asarray(model.eq_data[eq,:3]);target_local=np.asarray(model.eq_data[eq,3:6]);btarget=int(model.eq_obj2id[eq])
        d.qpos[:]=q;mujoco.mj_kinematics(model,d)
        P=d.xanchor[main].copy()
        def target(data):return data.xpos[btarget]+data.xmat[btarget].reshape(3,3)@target_local
        B=target(d)
        if row.get('shoe_joint'):
            sj=model.joint(row['shoe_joint']).id;sa=int(model.jnt_qposadr[sj]);delta=float(P[2]-B[2])
            q[sa]+=delta
            if model.jnt_limited[sj] and not model.jnt_range[sj,0]-1e-6<=q[sa]<=model.jnt_range[sj,1]+1e-6:
                raise ValueError('Closer shoe cannot reach its required native loop height')
            d.qpos[:]=q;mujoco.mj_kinematics(model,d);B=target(d)
        if abs(P[2]-B[2])>2e-6:raise ValueError('Closer fixed shoe is outside the arm plane')
        L1=float(np.linalg.norm(model.body_pos[bfore]));L2=float(np.linalg.norm(anchor))
        vector=B-P;distance=float(np.linalg.norm(vector))
        if not abs(L1-L2)+1e-8<distance<L1+L2-1e-8:raise ValueError('Closer arm lengths cannot reach actual frame shoe')
        along=vector/distance;axis=d.xaxis[main];across=np.cross(axis,along)
        length=(L1*L1-L2*L2+distance*distance)/(2*distance)
        altitude=np.sqrt(max(0.,L1*L1-length*length))
        oldP=rest.xanchor[main];oldE=rest.xanchor[elbow];oldB=target(rest)
        handed=np.sign(np.dot(np.cross(oldE-oldP,oldB-oldE),rest.xaxis[main]))
        candidates=[P+length*along+sign*altitude*across for sign in (-1,1)]
        E=next(e for e in candidates if np.sign(np.dot(np.cross(e-P,B-e),axis))==handed)
        heading1=float(np.arctan2(E[1]-P[1],E[0]-P[0]))
        base1=float(np.arctan2(d.xmat[bmain].reshape(3,3)[1,0],d.xmat[bmain].reshape(3,3)[0,0]))-float(q[a1])
        heading2=float(np.arctan2(B[1]-E[1],B[0]-E[0]))
        # Initial forearm orientation relative to the main arm is unchanged.
        initial_relative=rest.xmat[bmain].reshape(3,3).T@rest.xmat[bfore].reshape(3,3)
        relative=float(np.arctan2(initial_relative[1,0],initial_relative[0,0]))
        q[a1]=(heading1-base1+np.pi)%(2*np.pi)-np.pi
        q[a2]=(heading2-heading1-relative+np.pi)%(2*np.pi)-np.pi
        d.qpos[:]=q;mujoco.mj_kinematics(model,d)
        endpoint=d.xpos[bfore]+d.xmat[bfore].reshape(3,3)@anchor
        if np.linalg.norm(endpoint-target(d))>2e-6:raise ValueError('Closer solved pose fails its exact native connect anchor')
    return qpos


def frame_backing(world, x, z, half_height, face, surface, material):
    actual, name = frame_face(world,x,z,face,pad=0.)
    depth=face*(surface-actual)
    if depth>1e-6:
        world.geoms.append(C.box('closer_frame_spacer',(x,(surface+actual)/2,z),(.032,depth/2,half_height),
            material,2700,False,True,ALL_TIERS,'closer','Rigid shoe spacer to structural frame face'))
        return name,'closer_frame_spacer'
    return name,None


def finish_deferred_closers(model):
    """Complete paired mechanisms only after their real frame is authored."""
    pending=getattr(model,'_deferred_closers',[])
    if hasattr(model,'_deferred_closers'):delattr(model,'_deferred_closers')
    for args in pending:C.add_closer(model,*args,tier_full_arms=True)
