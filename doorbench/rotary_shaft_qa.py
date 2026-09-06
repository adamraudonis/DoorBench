"""Direct native stock/journal inspection, including parent-filtered geometry."""
from __future__ import annotations
import numpy as np


def run_rotary_shaft_mount_qa(native,metadata,*,samples=13):
    import mujoco
    rows=metadata.get('rotary_shafts',[]);failures=[];measurements=[]
    for row in rows:
        m=native;d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
        joint=m.joint(row['joint']).id;body=m.body(row['body']).id;support=m.body(row['support_body']).id
        shaft=m.geom(row['shaft_geom']).id;stock=[m.geom(n).id for n in row['leaf_stock_geoms'] if mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n)>=0]
        fixed=[m.geom(n).id for n in row['support_geoms']]
        parent_stock=[m.geom(n).id for n in row.get('fixed_parent_geoms',[]) if mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n)>=0]
        if m.geom_bodyid[shaft]!=body or m.body_jntnum[support] or m.body_parentid[support]!=m.body_parentid[body]:
            raise ValueError('Prepared shaft/support body binding differs from authored assembly')
        def gap(a,b,limit=.05):return float(mujoco.mj_geomDistance(m,d,a,b,limit,None))
        # The complete fixed assembly must be connected to actual retained
        # stock; a journal floating beside its old visual rose is insufficient.
        reachable={g for g in fixed if any(gap(g,s)<=1e-6 for s in stock)};remaining=set(fixed)-reachable
        while remaining:
            joined={g for g in remaining if any(gap(g,s)<=1e-6 for s in reachable)}
            if not joined:break
            reachable.update(joined);remaining-=joined
        if remaining:failures.append({'joint':row['joint'],'check':'unsupported_fixed_stock','geoms':[m.geom(g).name for g in sorted(remaining)]})
        moving=[g for g in range(m.ngeom) if m.geom_bodyid[g]==body]
        worst_shaft=worst_support=worst_parent=float('inf');shaft_pair=support_pair=parent_pair=None
        for q in np.linspace(*m.jnt_range[joint],samples):
            d.qpos[:]=m.qpos0;d.qpos[m.jnt_qposadr[joint]]=q;mujoco.mj_kinematics(m,d)
            for g in stock:
                value=gap(shaft,g,.003)
                if value<worst_shaft:worst_shaft=value;shaft_pair=[row['shaft_geom'],m.geom(g).name]
            for g in moving:
                for h in fixed:
                    if np.linalg.norm(d.geom_xpos[g]-d.geom_xpos[h])>m.geom_rbound[g]+m.geom_rbound[h]+.003:continue
                    value=gap(g,h,.003)
                    if value<worst_support:worst_support=value;support_pair=[m.geom(g).name,m.geom(h).name]
                for h in parent_stock:
                    if np.linalg.norm(d.geom_xpos[g]-d.geom_xpos[h])>m.geom_rbound[g]+m.geom_rbound[h]+.003:continue
                    value=gap(g,h,.003)
                    if value<worst_parent:worst_parent=value;parent_pair=[m.geom(g).name,m.geom(h).name]
        if worst_shaft<.0005:failures.append({'joint':row['joint'],'check':'shaft_vs_prepared_stock','pair':shaft_pair,'gap_m':worst_shaft})
        if worst_support<-1e-6:failures.append({'joint':row['joint'],'check':'moving_trim_vs_fixed_support','pair':support_pair,'gap_m':worst_support})
        if worst_parent<-1e-6:failures.append({'joint':row['joint'],'check':'moving_trim_vs_fixed_parent_stock','pair':parent_pair,'gap_m':worst_parent})
        for name,surface in row['input_surfaces'].items():
            sid=m.site(name).id;gid=m.geom(surface).id
            if m.site_bodyid[sid]!=body or m.geom_bodyid[gid]!=body:raise ValueError('Rotary input surface belongs to another body')
            if m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_SPHERE:
                error=abs(float(np.linalg.norm(d.site_xpos[sid]-d.geom_xpos[gid]))-m.geom_size[gid,0])
                if error>1e-6:failures.append({'joint':row['joint'],'check':'knob_surface_site','site':name,'error_m':error})
        measurements.append({'joint':row['joint'],'samples':samples,'minimum_shaft_stock_gap_m':worst_shaft,
            'shaft_stock_pair':shaft_pair,'minimum_moving_support_gap_m':None if worst_support==float('inf') else worst_support,'moving_support_pair':support_pair,
            'minimum_moving_parent_gap_m':None if worst_parent==float('inf') else worst_parent,'moving_parent_pair':parent_pair,
            'fixed_support_count':len(fixed),'connected_support_count':len(reachable)})
    return {'ok':not failures,'applicable':bool(rows),'failures':failures,'measurements':measurements,
        'scope':'Exact native distance queries and attachment graph for shaft/stock/journal geometry through sampled operator travel. Parent filtering bypassed. No force cycle, bearing strength, locking-function or whole-door-task certificate.'}
