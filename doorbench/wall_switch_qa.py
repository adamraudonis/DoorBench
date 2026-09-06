"""Prepared switch stock, physical press surfaces and spring-return cycles."""
import math
import numpy as np


def run_wall_switch_qa(model,meta,*,step=None,forward=None):
    import mujoco as mj
    step=step or mj.mj_step;forward=forward or mj.mj_forward
    rows=[];failures=[]
    for r in meta.get('wall_switches',[]):
        d=mj.MjData(model);forward(model,d)
        j=model.joint(r['joint']).id;a=model.jnt_qposadr[j];v=model.jnt_dofadr[j]
        cap=model.geom(r['cap_geom']).id;stem=model.geom(r['stem_geom']).id;sid=model.site(r['site']).id
        plates=[model.geom(n).id for n in r['plate_geoms']]
        local=d.geom_xmat[cap].reshape(3,3).T@(d.site_xpos[sid]-d.geom_xpos[cap])
        surface_error=abs(abs(float(local[2]))-float(model.geom_size[cap,1]))
        if surface_error>1e-6 or np.linalg.norm(local[:2])>model.geom_size[cap,0]:failures.append(r['joint']+': press site off cap')
        gap=math.inf
        for q in np.linspace(0,r['travel_m'],17):
            d.qpos[a]=q;mj.mj_kinematics(model,d)
            for g in (cap,stem):
                for plate in plates:gap=min(gap,float(mj.mj_geomDistance(model,d,g,plate,.05,None)))
        if gap<.00045:failures.append(r['joint']+': insufficient prepared-stock gap')
        walls=[g for g in range(model.ngeom) if (model.geom(g).name or '').startswith('wall_') and g not in plates]
        mount_gap=max((min((abs(float(mj.mj_geomDistance(model,d,g,w,.1,None))) for w in walls),default=math.inf) for g in plates),default=math.inf)
        if mount_gap>1e-5:failures.append(r['joint']+': plate has no wall attachment')
        d=mj.MjData(model);forward(model,d);cycles=[];depth=0.
        for cycle in range(2):
            positions=[]
            for pressed in (True,False):
                for _ in range(math.ceil(.3/model.opt.timestep)):
                    d.qfrc_applied[:]=0
                    if pressed:
                        mj.mj_applyFT(model,d,np.array([0,-r['face']*20.,0]),np.zeros(3),d.site_xpos[sid],int(model.site_bodyid[sid]),d.qfrc_applied)
                    step(model,d)
                    for c in d.contact:
                        if cap in c.geom or stem in c.geom:depth=max(depth,-float(c.dist))
                positions.append(float(d.qpos[a]))
            if positions[0]<r['travel_m']-.0003 or abs(positions[1])>.0005:failures.append(r['joint']+': incomplete press/return')
            cycles.append({'pressed_m':positions[0],'released_m':positions[1]})
        if depth>.001 or any(w.number for w in d.warning):failures.append(r['joint']+': native collision/warning')
        rows.append({'joint':r['joint'],'face':r['face'],'site':r['site'],'surface_error_m':surface_error,
            'minimum_stock_gap_m':gap,'maximum_mount_gap_m':mount_gap,'max_contact_penetration_m':depth,'cycles':cycles})
    return {'ok':not failures,'applicable':bool(rows),'failures':failures,'measurements':rows,
        'scope':'17 prepared-stock poses and two 20 N physical-site press/return cycles; electrical contact block and switch durability are not simulated.'}
