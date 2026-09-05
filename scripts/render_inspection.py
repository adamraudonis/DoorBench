#!/usr/bin/env python
"""Render fully framed, coupled-pose inspection images; no external model/API.

Images are evidence, not physical sign-off. Leaf travel is prescribed; released
locks are shown as hypothetical inspection poses, explicitly recorded. Connect
loops are numerically solved using only their mechanism joints; residuals are
recorded so an impossible linkage is never silently approved.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from multiprocessing import Pool
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_one(job):
    import mujoco
    from scipy.optimize import least_squares
    from doorbench.clearance import Clearance
    assets, out, sid = job
    source = Path(assets) / 'doors' / sid
    dest = Path(out) / sid
    dest.mkdir(parents=True, exist_ok=True)
    source_bytes = {kind: (source / filename).read_bytes() for kind, filename in
                    [('model', 'model.json'), ('spec', 'spec.json'), ('xml', 'door.xml')]}
    source_hashes = {f'source_{kind}_sha256': hashlib.sha256(blob).hexdigest()
                     for kind, blob in source_bytes.items()}
    spec = json.loads(source_bytes['spec'])
    model = json.loads(source_bytes['model'])
    ms = mujoco.MjSpec.from_file(str(source / 'door.xml'))
    ms.visual.global_.offwidth = 640
    ms.visual.global_.offheight = 480
    m = ms.compile()
    d = mujoco.MjData(m)
    gate = Clearance(str(source), 'full')
    roles = {b['joint']['name']: b['joint'].get('role') for b in model['bodies'] if b.get('joint')}
    sem = {g['name']: g.get('semantic', '') for b in model['bodies'] for g in b['geoms']}
    visible = [g for g in range(m.ngeom) if m.geom_group[g] != 3 and sem.get(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,g)) not in ('floor','wall')]
    hw = [g for g in visible if sem.get(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g)) in ('operator','latch','lock')]
    connects = [i for i in range(m.neq) if m.eq_active0[i] and int(m.eq_type[i]) == int(mujoco.mjtEq.mjEQ_CONNECT)]
    equations = [i for i in range(m.neq) if m.eq_active0[i] and int(m.eq_type[i]) == int(mujoco.mjtEq.mjEQ_JOINT)]
    controlled = {m.joint(int(m.eq_obj1id[e])).name for e in equations}
    def resolve_equalities(q):
        # Match native MuJoCo polynomial equalities, which are relative to ref.
        for _ in range(max(2, len(equations))):
            for e in equations:
                a,b=int(m.eq_obj1id[e]),int(m.eq_obj2id[e])
                adr=int(m.jnt_qposadr[a])
                x=q[m.jnt_qposadr[b]]-m.qpos0[m.jnt_qposadr[b]] if b >= 0 else 0.0
                q[adr]=m.qpos0[adr]+sum(m.eq_data[e,k]*x**k for k in range(5))
        return q
    chains = set()
    for eq in connects:
        for body in (int(m.eq_obj1id[eq]), int(m.eq_obj2id[eq])):
            while body > 0:
                for j in range(int(m.body_jntadr[body]), int(m.body_jntadr[body]) + int(m.body_jntnum[body])):
                    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT,j)
                    if roles.get(name) not in ('primary','secondary') and name not in controlled:
                        chains.add(j)
                body = int(m.body_parentid[body])
    movable = sorted(chains)
    adrs = [int(m.jnt_qposadr[j]) for j in movable]
    def residual():
        rs=[]
        for e in connects:
            a,b=int(m.eq_obj1id[e]),int(m.eq_obj2id[e])
            pa=d.xpos[a]+d.xmat[a].reshape(3,3) @ m.eq_data[e,:3]
            pb=d.xpos[b]+d.xmat[b].reshape(3,3) @ m.eq_data[e,3:6]
            rs.extend(pa-pb)
        return np.array(rs)
    def pose(frac):
        q = gate.released_qpos() if frac else gate.resolve(m.qpos0.copy())
        forced=False
        for j in range(m.njnt):
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
            if roles.get(name) not in ('primary','secondary') or name in controlled:
                continue
            adr=int(m.jnt_qposadr[j]); q0=m.qpos0[adr]
            if m.jnt_limited[j]:
                lo,hi=m.jnt_range[j]
                if hi-lo < .006 and frac:
                    kin=spec['kinematics']
                    target=math.radians(kin.get('max_open_deg') or 90) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE) else float(kin.get('travel_m') or .9)
                    forced=True
                else:
                    target=hi if abs(hi-q0)>=abs(lo-q0) else lo
            else: target=q0+1.2
            q[adr]=q0+frac*(target-q0)
        q=resolve_equalities(gate.resolve(q))
        d.qpos[:]=q; mujoco.mj_forward(m,d)
        if adrs:
            def fun(values):
                d.qpos[:]=q; d.qpos[adrs]=values
                d.qpos[:]=resolve_equalities(d.qpos.copy())
                mujoco.mj_forward(m,d)
                return residual()
            lo=[float(m.jnt_range[j,0]) if m.jnt_limited[j] else -np.inf for j in movable]
            hi=[float(m.jnt_range[j,1]) if m.jnt_limited[j] else np.inf for j in movable]
            x=np.clip(q[adrs],np.array(lo)+1e-9,np.array(hi)-1e-9)
            solved=least_squares(fun,x,bounds=(lo,hi),ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=100)
            q[adrs]=solved.x
            q=resolve_equalities(q)
        d.qpos[:]=q; mujoco.mj_forward(m,d)
        rr=residual()
        return q, (float(np.max(np.linalg.norm(rr.reshape(-1,3),axis=1))) if len(rr) else 0), forced
    def bbox(ids):
        points=[]
        for g in ids:
            R=d.geom_xmat[g].reshape(3,3)
            c=d.geom_xpos[g]+R @ m.geom_aabb[g,:3]
            h=np.abs(R) @ m.geom_aabb[g,3:]
            points.extend([c-h,c+h])
        if not points: return np.array([0,0,1.]),1.2
        lo,hi=np.min(points,axis=0),np.max(points,axis=0)
        return (lo+hi)/2,max(float(np.linalg.norm(hi-lo)/2),.1)
    opt=mujoco.MjvOption(); opt.geomgroup[:]=0; opt.geomgroup[:3]=1
    renderer=mujoco.Renderer(m,height=360,width=480)
    # Keep material data intact; use restrained inspection lighting to preserve surface detail.
    m.light_diffuse[:]=np.minimum(m.light_diffuse,.55)
    m.vis.headlight.diffuse[:]=.35
    results=[]
    try:
        for view,frac,az,el,detail in [('front',0,65,-18,False),('reverse',0,-65,-18,False),('open',1,65,-28,False),('hardware',0,65,-14,True),('mid',.5,65,-25,False)]:
            q,res,forced=pose(frac)
            horizontal=bool(model['meta'].get('horizontal')) or spec['family'] in ('hatch_floor','hatch_ceiling')
            if horizontal: el=-55 if view != 'reverse' else 35
            c,r=bbox(hw if detail and hw else visible)
            cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:]=c;cam.distance=max(.3,1.12*r/math.sin(math.radians(float(m.vis.global_.fovy)/2)))
            cam.azimuth=az;cam.elevation=el
            renderer.update_scene(d,camera=cam,scene_option=opt)
            Image.fromarray(renderer.render()).save(dest/f'{view}.jpg',quality=85)
            results.append({'view':view,'loop_residual_m':res,'forced_locked_pose':forced})
        for kind, filename in [('model', 'model.json'), ('spec', 'spec.json'), ('xml', 'door.xml')]:
            if (source / filename).read_bytes() != source_bytes[kind]:
                raise RuntimeError(f'{sid}: {filename} changed during rendering; rerender this door')
        record={'door_id':sid,'family':spec['family'],'views':results, **source_hashes}
        (dest/'render.json').write_text(json.dumps(record,indent=2)+'\n')
        return record
    finally: renderer.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',default='assets');ap.add_argument('--out',default='out/inspection')
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--ids',default='')
    a=ap.parse_args(); man=json.loads((Path(a.assets)/'manifest.json').read_text())
    rows=[d for d in man['doors'] if not a.ids or d['id'] in a.ids.split(',')]
    jobs=[(str(Path(a.assets).resolve()),str(Path(a.out).resolve()),d['id']) for d in rows]
    with Pool(a.workers,maxtasksperchild=50) as p:
        records=[]
        for r in p.imap_unordered(render_one,jobs):
            records.append(r)
            if len(records)%50==0:print(f'{len(records)}/{len(rows)} rendered',flush=True)
    records.sort(key=lambda x:x['door_id'])
    (Path(a.out)/'renders.json').write_text(json.dumps(records,indent=2)+'\n')
    print(f'Done: {len(records)} doors, {sum(len(x["views"]) for x in records)} views',flush=True)


if __name__=='__main__':main()
