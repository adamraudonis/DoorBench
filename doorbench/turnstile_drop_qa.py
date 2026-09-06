"""Native indexed drop, power restoration and hand-force recapture proof.

The root angle is initialized only to select an indexed starting configuration.
All later motion comes from gravity, native contacts and bounded surface force.
This is a mechanism control, not a fire/egress or structural certificate.
"""
from __future__ import annotations
import copy
import math
import numpy as np


def run_turnstile_drop_mount_qa(model, metadata):
    """Exact native distances within welded parts and to their support anchor."""
    import mujoco
    row=metadata.get('turnstile_drop_arm')
    if not row:return {'ok':True,'applicable':False,'failures':[]}
    d=mujoco.MjData(model);mujoco.mj_kinematics(model,d)
    def body_geoms(name):
        b=model.body(name).id
        return [model.geom(g).name for g in range(model.ngeom) if model.geom_bodyid[g]==b]
    groups=[(row['anchor_geom'],row['fixed_support_geoms']),
            ('tripod_hub',body_geoms('turnstile_drop_carrier'))]
    for k,arm in enumerate(row['arms']):
        groups.append((f'arm_{k}_col',body_geoms(f'turnstile_drop_arm_hardware_{k}')))
        names=body_geoms(arm['catch_body']);groups.append((arm['catch_geom'],[n for n in names if n!=arm['catch_geom']]))
    groups.append((row['release_nose_geom'],[n for n in body_geoms('turnstile_drop_release') if n!=row['release_nose_geom']]))
    failures=[];reports=[]
    for anchor,names in groups:
        names=[anchor,*names];ids=[model.geom(n).id for n in names];connected={0};edges=[]
        while True:
            extra=set()
            for i in connected:
                for k in range(len(ids)):
                    if k in connected or k in extra:continue
                    gap=float(mujoco.mj_geomDistance(model,d,ids[i],ids[k],.001,None))
                    if gap<=.0005:extra.add(k);edges.append({'geoms':[names[i],names[k]],'gap_m':gap})
            if not extra:break
            connected|=extra
        missing=[n for k,n in enumerate(names) if k not in connected]
        result={'anchor':anchor,'connected_count':len(connected)-1,'detached_parts':missing,'edges':edges};reports.append(result)
        if missing:failures.append(result)
    return {'ok':not failures,'applicable':True,'failures':failures,'groups':reports,
        'scope':'Native shape-distance continuity within welded components and structural supports; hinge/journal constraints represent their bearings, not a stress or fastener certificate'}


def run_turnstile_drop_qa(model, metadata, *, indices=(0, 1, 2), cycles=2):
    import mujoco
    from .turnstile_locks import compile_turnstile_locks, apply_turnstile_locks
    from .turnstile_drop import compile_turnstile_drop, apply_turnstile_drop
    row=metadata.get('turnstile_drop_arm')
    if not row:return {'ok':True,'applicable':False,'failures':[]}
    m=copy.copy(model)
    locks=compile_turnstile_locks(m,metadata);drops=compile_turnstile_drop(m,metadata)
    rid=m.joint(row['rotor_joint']).id;ra=int(m.jnt_qposadr[rid])
    mount=run_turnstile_drop_mount_qa(m,metadata)
    previous=mujoco.get_mjcb_passive();powered=True;failures=list(mount['failures']);probes=[];held_configuration={}
    arm_joints=[m.joint(r['arm_joint']).id for r in row['arms']]
    arm_qa=[int(m.jnt_qposadr[j]) for j in arm_joints]
    catch_qa=[int(m.jnt_qposadr[m.joint(r['catch_joint']).id]) for r in row['arms']]
    def callback(native,data):
        if native is m:
            apply_turnstile_locks(native,data,locks,False)
            apply_turnstile_drop(native,data,drops,powered)
    try:
        mujoco.set_mjcb_passive(callback)
        for index in indices:
            arm=row['arms'][index];d=mujoco.MjData(m)
            d.qpos[ra]=float(arm['indexed_rotor_angle_rad'])
            mujoco.mj_forward(m,d)
            aj=arm_joints[index];av=int(m.jnt_dofadr[aj]);aa=arm_qa[index]
            site=m.site(arm['reset_site']).id;body=int(m.site_bodyid[site])
            load_pair=tuple(sorted((m.geom(arm['toe_geom']).id,m.geom(arm['catch_geom']).id)))
            stop_pair=tuple(sorted(m.geom(n).id for n in arm['fold_stop_geoms']))
            release_pair=tuple(sorted((m.geom(arm['tail_geom']).id,m.geom(row['release_nose_geom']).id)))
            phases=[('powered_load',.8,True)]+[
                (name,duration,on) for _ in range(cycles)
                for name,duration,on in [('power_loss',2.5,False),('restored',.8,True),('manual_reset',2.5,True),('reset_hold',.8,True)]]
            for number,(phase,duration,powered) in enumerate(phases):
                worst=0.;worst_pair=None;pairs={k:0. for k in (load_pair,stop_pair,release_pair)};peak_force=0.;peak_catch=0.;peak_arm=0.
                for _ in range(round(duration/m.opt.timestep)):
                    d.qfrc_applied[:]=0.
                    if phase in ('powered_load','manual_reset'):
                        point=d.site_xpos[site];tangent=np.cross(d.xaxis[aj],point-d.xanchor[aj]);tangent/=np.linalg.norm(tangent)
                        magnitude=15. if phase=='powered_load' else float(np.clip(160.*(-.025-d.qpos[aa])-8.*d.qvel[av],-30.,30.))
                        mujoco.mj_applyFT(m,d,magnitude*tangent,np.zeros(3),point,body,d.qfrc_applied)
                        peak_force=max(peak_force,abs(magnitude))
                    mujoco.mj_step(m,d)
                    for ci,c in enumerate(d.contact):
                        pair=tuple(sorted((int(c.geom1),int(c.geom2))))
                        if pair in pairs:
                            force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                            pairs[pair]=max(pairs[pair],float(np.linalg.norm(force[:3])))
                        if -c.dist>worst:worst=-float(c.dist);worst_pair=[m.geom(g).name for g in pair]
                    peak_catch=max(peak_catch,float(d.qpos[catch_qa[index]]));peak_arm=max(peak_arm,float(d.qpos[aa]))
                result={'index':index,'phase':phase,'phase_number':number,'duration_s':duration,
                    'arm_angles_rad':[float(d.qpos[a]) for a in arm_qa],
                    'catch_travel_m':[float(d.qpos[a]) for a in catch_qa],
                    'rotor_angle_rad':float(d.qpos[ra]),'peak_arm_angle_rad':peak_arm,'peak_catch_travel_m':peak_catch,
                    'release_travel_m':float(d.qpos[drops[0].qpos]),'max_manual_surface_force_N':peak_force,
                    'catch_load_contact_N':pairs.get(load_pair,0.),'fold_stop_contact_N':pairs.get(stop_pair,0.),
                    'release_contact_N':pairs.get(release_pair,0.),'max_penetration_m':worst,'worst_pair':worst_pair,
                    'native_warnings':d.warning.number.tolist()}
                probes.append(result)
                if index==0 and phase=='reset_hold':
                    names=[row['release_joint']]+[n for r in row['arms'] for n in (r['arm_joint'],r['catch_joint'])]
                    held_configuration={n:float(d.qpos[m.jnt_qposadr[m.joint(n).id]]) for n in names}
                def fail(code):failures.append({'code':code,'index':index,'phase_number':number,'phase':phase})
                if worst>.001 or np.any(d.warning.number) or not np.isfinite(d.qpos).all():fail('native_penetration_or_warning')
                if any(abs(d.qpos[a])>.015 for k,a in enumerate(arm_qa) if k!=index):fail('nonindexed_arm_released')
                if abs(d.qpos[ra]-arm['indexed_rotor_angle_rad'])>.06:fail('rotation_not_physically_arrested')
                if phase=='powered_load':
                    if abs(d.qpos[aa])>.015 or pairs.get(load_pair,0.)<1.:fail('powered_catch_did_not_hold_load')
                if phase=='power_loss':
                    if d.qpos[aa]<math.radians(85) or d.qpos[aa]>math.radians(97):fail('gravity_drop_not_at_physical_stop')
                    if pairs.get(stop_pair,0.)<.1 or pairs.get(release_pair,0.)<1.:fail('missing_drop_load_path')
                if phase=='restored' and d.qpos[aa]<math.radians(85):fail('power_restoration_automatically_reset_arm')
                if phase=='manual_reset' and (abs(d.qpos[aa])>.015 or peak_catch<.014):fail('manual_cam_recapture_failed')
                if phase=='reset_hold' and (abs(d.qpos[aa])>.015 or pairs.get(load_pair,0.)<.1):fail('reset_catch_did_not_retain_arm')
        return {'ok':not failures,'applicable':True,'failures':failures,'probes':probes,'mount':mount,'held_configuration':held_configuration,
            'scope':'Native indexed gravity drop, power restoration without automatic reset, and <=30 N surface-force cam recapture; no pose writes during cycles, structural/fire/egress or traversal certificate'}
    finally:mujoco.set_mjcb_passive(previous)
