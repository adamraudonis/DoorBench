"""Physical surface-closer mounts, independent of hydraulic torque calibration."""
from __future__ import annotations
import numpy as np

from .geometry.closer_mounts import resolve_closer_configuration


def run_closer_mount_qa(model, metadata, samples=25):
    import mujoco
    rows=metadata.get('closer_mounts',[])
    if not rows:return {'ok':True,'applicable':False,'failures':[]}
    d=mujoco.MjData(model);mujoco.mj_kinematics(model,d);failures=[];mounts=[]
    geom_names={model.geom(i).name for i in range(model.ngeom)}
    def gap(a,b):return float(mujoco.mj_geomDistance(model,d,model.geom(a).id,model.geom(b).id,1.,None))
    for row in rows:
        shaft=model.geom(row['shaft_geom']).id;body=model.geom(row['body_geom']).id
        distance=gap(row['shaft_geom'],row['body_geom'])
        mount_gap=gap(row['frame_plate'],row.get('frame_spacer') or row['frame_geom'])
        if row.get('frame_spacer'):mount_gap=max(mount_gap,gap(row['frame_spacer'],row['frame_geom']))
        native_leaf=int(model.geom_bodyid[body])
        support=row.get('housing_spacer_geom') or row['body_geom']
        leaf_geoms=row.get('leaf_support_geoms') or [model.geom(i).name for i in range(model.ngeom) if model.geom_bodyid[i]==native_leaf and
                    model.geom(i).name.startswith(('leaf_slab','leaf_glass','leaf_stile','leaf_picket','leaf_rail','leaf_frame'))]
        leaf_geoms=[n for n in leaf_geoms if n in geom_names]
        if not leaf_geoms:
            prefix=row.get('leaf_body','leaf')
            leaf_geoms=[model.geom(i).name for i in range(model.ngeom) if model.geom_bodyid[i]==native_leaf and model.geom(i).name.startswith(tuple(prefix+'_'+k for k in ('slab','glass','stile','picket','rail','frame')))]
        leaf_gap=min((gap(support,n) for n in leaf_geoms),default=1.)
        item={'shaft_to_housing_gap_m':distance,'frame_mount_gap_m':mount_gap,'housing_mount_to_leaf_gap_m':leaf_gap}
        mounts.append(item)
        if distance>0.0005:failures.append({'detached_pinion':item})
        if mount_gap>0.0005:failures.append({'detached_frame_shoe':item})
        if leaf_gap>0.0005:failures.append({'detached_closer_body':item})
        for a,b in [(row.get('fore_geom','closer_arm_fore_geom'),row.get('neck_geom','closer_shoe_neck')),(row.get('neck_geom','closer_shoe_neck'),row['pivot_geom'])]:
            if gap(a,b)>.0005:failures.append({'detached_forearm_end':[a,b]})
    min_gap=1.;worst=None
    for driven in rows:
      primary=model.joint(driven.get('leaf_joint',metadata['primary_joint'])).id
      lo,hi=model.jnt_range[primary] if model.jnt_limited[primary] else [0.,np.pi]
      for angle in np.linspace(lo,hi,max(2,samples)):
        q=model.qpos0.copy();q[model.jnt_qposadr[primary]]=angle
        for _ in range(2):
            for e in range(model.neq):
                if model.eq_type[e]==mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[e]>=0:
                    j1,j2=int(model.eq_obj1id[e]),int(model.eq_obj2id[e]);x=q[model.jnt_qposadr[j2]]
                    q[model.jnt_qposadr[j1]]=sum(model.eq_data[e,k]*x**k for k in range(5))
        try:resolve_closer_configuration(model,q,metadata)
        except (ValueError,StopIteration) as exc:
            failures.append({'unreachable_native_loop':str(exc),'angle_rad':float(angle)});continue
        d.qpos[:]=q;mujoco.mj_kinematics(model,d)
        for row in rows:
            for n in row['shoe_geoms']:
                value=gap(row['pivot_geom'],n)
                if value<min_gap:min_gap=value;worst=[row['pivot_geom'],n,float(angle)]
    if min_gap<.0005:failures.append({'shoe_pivot_obstruction':worst,'gap_m':min_gap})
    return {'ok':not failures,'applicable':True,'failures':failures,'mounts':mounts,'samples':max(2,samples),
            'minimum_shoe_bore_gap_m':min_gap,'scope':'Exact support distances and connected planar-linkage/bore sweep; physical pinion force element is separate; ideal bearings are not internally simulated hydraulics'}
