"""Direct native distance queries for internal lock stock, bypassing filtering."""
from __future__ import annotations
import numpy as np


def run_lock_stock_qa(model,metadata,*,tier='full',samples=17):
    import mujoco
    if tier not in ('full','simple','minimal') or isinstance(samples,bool) or not isinstance(samples,int) or samples<2:
        raise ValueError('Lock-stock QA needs a supported tier and at least two travel samples')
    rows=metadata.get('lock_stock',[])
    failures=[];measurements=[]
    for row in rows:
        if tier not in row.get('tiers',('full','simple')):continue
        try:
            leaf=model.body(row['leaf_body']).id
            bolts=[model.geom(n).id for n in row['bolt_geoms']]
            parent=[i for i in range(model.ngeom) if model.geom_bodyid[i]==leaf and i not in bolts
                    and (model.geom_contype[i] or model.geom_conaffinity[i])]
            body=model.body(row['bolt_body']).id
            joints=[j for j in range(model.njnt) if model.jnt_bodyid[j]==body and model.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE]
            # A fixed key-only bolt may have been fused into its leaf. Never
            # mistake the primary leaf slider for that absent lock joint.
            joints=[j for j in joints if model.joint(j).name==row['name']+'_slide']
            joint=joints[0] if joints else None
            groups=[('bolt',bolts,joint,float(row['clearance_m']))]
            if row.get('thumbturn_joint'):
                groups.append(('thumbturn',[model.geom(n).id for n in row['thumbturn_geoms']],
                               model.joint(row['thumbturn_joint']).id,float(row['thumbturn_clearance_m'])))
            for mount in row.get('operator_standoffs',[]):
                # Prepared journals can live in their own exact-BOM rigid
                # support body. Follow the explicit source binding rather
                # than silently dropping them from a direct-leaf-only query.
                for name in mount['spacer_geoms']:
                    fixed=model.geom(name).id;ancestor=int(model.geom_bodyid[fixed])
                    while ancestor!=leaf and ancestor:
                        if model.body_jntnum[ancestor]:raise ValueError('Operator spacer is not fixed to its owning leaf')
                        ancestor=int(model.body_parentid[ancestor])
                    if ancestor!=leaf:raise ValueError('Operator spacer belongs to another leaf')
                    if fixed not in parent:parent.append(fixed)
                moving=[model.geom(n).id for n in mount['moving_geoms']+[mount['shaft_geom']]]
                moving=[g for g in moving if model.geom_contype[g] or model.geom_conaffinity[g]]
                groups.append(('operator_standoff',moving,model.joint(mount['joint']).id,.0005))
            for label,moving,joint,required in groups:
                values=np.linspace(*model.jnt_range[joint],samples) if joint is not None else [None]
                if not .0005<=required<=.001:raise ValueError('Unsupported internal lock guide clearance')
                d=mujoco.MjData(model);worst=float('inf');pair=None
                for value in values:
                    d.qpos[:]=model.qpos0
                    if value is not None:d.qpos[model.jnt_qposadr[joint]]=value
                    mujoco.mj_kinematics(model,d)
                    for a in moving:
                        if not(model.geom_contype[a] or model.geom_conaffinity[a]):raise ValueError('Lock collider is disabled')
                        for b in parent:
                            if np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b])>model.geom_rbound[a]+model.geom_rbound[b]+required+.001:continue
                            gap=float(mujoco.mj_geomDistance(model,d,a,b,required+.001,None))
                            if gap<worst:worst=gap;pair=[model.geom(a).name,model.geom(b).name]
                            if gap<required-1e-5:
                                failures.append({'mechanism':row['name'],'part':label,'geoms':[model.geom(a).name,model.geom(b).name],
                                                 'q':None if value is None else float(value),'gap_m':gap,'required_m':required})
                measurements.append({'name':row['name'],'part':label,'minimum_gap_m':None if worst==float('inf') else worst,'pair':pair,'samples':len(values)})
        except (ValueError,KeyError) as exc:
            failures.append({'mechanism':row.get('name'),'error':str(exc)})
    return {'ok':not failures,'applicable':bool(rows),'failures':failures,'measurements':measurements,
            'scope':'Direct bolt/spindle/thumbturn-to-parent collider distances throughout native travel; parent-child filtering cannot hide filled mortises. Native force cycles remain separate.'}
