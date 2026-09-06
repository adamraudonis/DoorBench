"""Finite surface-force independent-trim/catch proof, separate from credentials."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import numpy as np
import mujoco
from .rotary_lockset import compile_rotary_catches,apply_rotary_catches
from .native_warnings import capture_native_warnings


def rotary_release_snapshot(native,metadata):
    """Private native preparation for explicitly unlocked inspection only."""
    rows=metadata.get('rotary_locksets',[])
    if not rows:return {'ok':True,'positions':{},'probes':[]}
    m=copy.copy(native);d=mujoco.MjData(m);rules=compile_rotary_catches(m,metadata)
    previous=mujoco.get_mjcb_passive();depth=0.;limits=0.;loads={};warnings=np.zeros(len(d.warning.number),int)
    def callback(model,data):
        if model is m:apply_rotary_catches(model,data,rules,True)
        elif previous:previous(model,data)
    try:
        mujoco.set_mjcb_passive(callback)
        with capture_native_warnings() as messages:
            mujoco.mj_forward(m,d)
            for _ in range(round(.4/m.opt.timestep)):
                mujoco.mj_step(m,d);warnings=np.maximum(warnings,d.warning.number)
                for i,c in enumerate(d.contact):
                    depth=max(depth,-float(c.dist));pair=(m.geom(c.geom1).name,m.geom(c.geom2).name)
                    if any('catch_rear_stop' in n for n in pair) and any('catch_collar' in n for n in pair):
                        f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);loads[pair]=max(loads.get(pair,0.),float(np.linalg.norm(f[:3])))
                for r in rows:
                    j=m.joint(r['catch_joint']).id
                    mask=(d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))&(d.efc_id==j)
                    limits=max(limits,max(abs(d.efc_force[mask]),default=0.))
            positions={r.name:float(d.qpos[r.qpos]) for r in rules}
            ok=(depth<=.001 and limits<=.01 and not messages and not np.any(warnings) and np.isfinite(d.qpos).all()
                and all(r.release_threshold<=positions[r.name]<=r.stroke+.0005 for r in rules)
                and all(any(r['catch_collar_geom'] in pair and any(n in r['native_stop_geoms'] and 'rear_stop' in n for n in pair) and force>.1
                            for pair,force in loads.items()) for r in rows))
            return {'ok':bool(ok),'positions':positions,'max_penetration_m':depth,'max_catch_limit_force_N':float(limits),
                    'global_native_warnings':list(messages),'native_warning_counters':warnings.tolist(),
                    'stop_loads':[{'pair':list(k),'force_N':v} for k,v in loads.items()],
                    'force_cap_N':max(r.force for r in rules),'scope':'Native catch preparation only; no pose, collision or joint-range edits. Not a credential or complete door-service test.'}
    finally:mujoco.set_mjcb_passive(previous)


def run_rotary_lockset_qa(native,metadata,cycles=2,*,source_xml=None):
    rows=metadata.get('rotary_locksets',[])
    if not rows:return {'ok':True,'applicable':False}
    fixture={'kind':'unrestrained_source','scope':'Leaf is not held; relatch can require an additional door-closing action.'}
    if source_xml is not None:
        source=Path(source_xml);spec=mujoco.MjSpec.from_file(str(source));base=spec.compile()
        for attr in ('qpos0','body_mass','body_inertia','body_pos','body_quat','geom_pos','geom_quat','geom_size','geom_type','geom_contype','geom_conaffinity','jnt_axis','jnt_pos','jnt_range','jnt_stiffness'):
            if not np.array_equal(getattr(base,attr),getattr(native,attr)):
                raise ValueError('Rotary bench source/native mismatch: '+attr)
        names=[]
        for leaf in sorted({r['leaf'] for r in rows}):
            name='rotary_bench_'+leaf;names.append(name)
            spec.add_equality(name=name,type=mujoco.mjtEq.mjEQ_WELD,name1=leaf,objtype=mujoco.mjtObj.mjOBJ_BODY,
                              solref=[.002,1.],solimp=[.99,.999,.0001,.5,2.])
        native=spec.compile()
        fixture={'kind':'recompiled_rigid_leaf_bench','source_xml_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                 'equality_names':names,'scope':'External bench holds the authored closed leaf only. Original children, contacts, input forces and springs retained. Not a whole-door closing test.'}
    m=copy.copy(native);d=mujoco.MjData(m);rules=compile_rotary_catches(m,metadata)
    failures=[];probes=[];commands={r.name:False for r in rules}
    previous=mujoco.get_mjcb_passive()
    def passive(model,data):
        if model is m:apply_rotary_catches(model,data,rules,commands)
        elif previous:previous(model,data)
    def value(name):return float(d.qpos[m.joint(name).qposadr[0]])
    def phase(name,row,seconds,drives=()):
        peak=0.;total_peak=0.;torque_peak=0.;resultant_peak=0.;depth=0.;contacts={};warnings=np.zeros(len(d.warning.number),int);limits=0.;bench_residual=0.
        for _ in range(round(seconds/m.opt.timestep)):
            d.qfrc_applied[:]=0.
            for joint in drives:
                j=m.joint(joint).id;bid=int(m.jnt_bodyid[j]);dof=m.jnt_dofadr[j]
                candidates=[m.site(name).id for name in row['input_sites'][joint]]
                if not candidates:raise ValueError('Independent trim has no actual grip site')
                if any(m.site_bodyid[sid]!=bid for sid in candidates):raise ValueError('Independent trim input site belongs to another body')
                tangents=[np.cross(d.xaxis[j],d.site_xpos[sid]-d.xanchor[j]) for sid in candidates]
                radii=[float(np.linalg.norm(tangent)) for tangent in tangents]
                if min(radii)<.01:raise ValueError('Independent trim grip has no useful moment arm')
                force=float(np.clip((12*(row['operator_travel_rad']+.4-value(joint))-.2*d.qvel[dof])/sum(radii),
                                    -row['operator_force_cap_N'],row['operator_force_cap_N']))
                forces=[tangent/radius*force for tangent,radius in zip(tangents,radii)]
                peak=max(peak,abs(force));total_peak=max(total_peak,len(candidates)*abs(force))
                torque_peak=max(torque_peak,sum(radii)*abs(force));resultant_peak=max(resultant_peak,float(np.linalg.norm(np.sum(forces,axis=0))))
                for sid,applied in zip(candidates,forces):
                    mujoco.mj_applyFT(m,d,applied,np.zeros(3),d.site_xpos[sid],bid,d.qfrc_applied)
            mujoco.mj_step(m,d);warnings=np.maximum(warnings,d.warning.number)
            for name in fixture.get('equality_names',[]):
                mask=(d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))&(d.efc_id==m.equality(name).id)
                bench_residual=max(bench_residual,float(np.linalg.norm(d.efc_pos[mask])))
            for ci,c in enumerate(d.contact):
                depth=max(depth,-float(c.dist));names=(m.geom(c.geom1).name,m.geom(c.geom2).name)
                if any(n in names for n in (row['catch_geom'],row['cam_geom'])) or any('catch_collar' in n for n in names):
                    force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force);pair=tuple(sorted(names))
                    contacts[pair]=max(contacts.get(pair,0.),float(np.linalg.norm(force[:3])))
            catch=m.joint(row['catch_joint']).id
            mask=(d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))&(d.efc_id==catch)
            limits=max(limits,max(abs(d.efc_force[mask]),default=0.))
            if np.any(warnings) or not np.isfinite(d.qpos).all():break
        p={'phase':name,'inside_rad':value(row['inside_joint']),'outside_rad':value(row['outside_joint']),
           'catch_m':value(row['catch_joint']),'latch_m':value(row['latch_joint']),
           'max_input_force_N':peak,'max_total_absolute_surface_force_per_input_N':total_peak,
           'max_applied_surface_torque_per_input_Nm':torque_peak,'max_resultant_force_per_input_N':resultant_peak,
           'max_penetration_m':depth,'max_catch_joint_limit_force_N':float(limits),
           'max_bench_weld_residual':bench_residual,'leaf_rad':value(metadata['primary_joint']),
           'native_warning_counters':warnings.tolist(),'contacts':[{'pair':list(k),'force_N':v} for k,v in contacts.items() if v>.01]}
        probes.append(p)
        if depth>.001 or limits>.01 or bench_residual>.0001 or np.any(warnings) or not np.isfinite(d.qpos).all():failures.append({'check':'native_geometry_warning_or_limit','probe':p})
        return p
    def require(ok,check,p):
        if not ok:failures.append({'check':check,'probe':p})
    def force(p,*names):return max((c['force_N'] for c in p['contacts'] if all(n in c['pair'] for n in names)),default=0.)
    def released(row,p):
        throw=float(m.jnt_range[m.joint(row['latch_joint']).id,1])
        require(p['latch_m']>=throw-.0005,'full_actual_latch_stroke',p)
        require(p['latch_m']<=throw+.0005,'no_additive_overtravel',p)
    try:
        mujoco.set_mjcb_passive(passive)
        with capture_native_warnings() as messages:
            for row in rows:
                # Source-unlocked catch is moved into its engaged state by its
                # own return spring. No initial pose writes are needed.
                mujoco.mj_forward(m,d);phase('unpowered_settle',row,.6)
                for cycle in range(cycles):
                    p=phase(f'{cycle}:outside_locked',row,.5,[row['outside_joint']])
                    require(p['outside_rad']<.06 and p['latch_m']<.002 and force(p,row['catch_geom'],row['cam_geom'])>1.,'exterior_physical_arrest',p)
                    phase(f'{cycle}:unload',row,.4)
                    p=phase(f'{cycle}:inside_locked',row,.7,[row['inside_joint']]);released(row,p)
                    require(abs(p['outside_rad'])<.01 and p['catch_m']<.0005,'inside_does_not_backdrive_exterior',p)
                    p=phase(f'{cycle}:inside_return',row,.7);require(abs(p['latch_m'])<.0005,'inside_return_relatched',p)
                    commands[row['catch_joint']]=True;p=phase(f'{cycle}:catch_release',row,.4)
                    require(p['catch_m']>=row['released_threshold_m'] and p['catch_m']<=row['catch_stroke_m']+.0005,'physical_catch_withdrawal',p)
                    require(any('rear_stop' in n and c['force_N']>.1 for c in p['contacts'] for n in c['pair']),'actual_release_stop_load',p)
                    p=phase(f'{cycle}:outside_released',row,.7,[row['outside_joint']]);released(row,p)
                    require(abs(p['inside_rad'])<.01,'exterior_does_not_backdrive_inside',p)
                    phase(f'{cycle}:outside_return',row,.7)
                    p=phase(f'{cycle}:both_inputs',row,.7,[row['inside_joint'],row['outside_joint']]);released(row,p)
                    phase(f'{cycle}:both_return',row,.7)
                    commands[row['catch_joint']]=False;p=phase(f'{cycle}:catch_reengage',row,.6)
                    require(p['catch_m']<.0005 and abs(p['latch_m'])<.0005,'spring_return_catch_and_latch',p)
                # Remove only actual locking pin collision. The independent
                # outside cam must then operate despite the seated pin qpos.
                gid=m.geom(row['catch_geom']).id;m.geom_contype[gid]=m.geom_conaffinity[gid]=0
                p=phase('removed_locking_pin_negative',row,.7,[row['outside_joint']]);released(row,p)
            if messages:failures.append({'check':'global_native_warning','messages':list(messages)})
        return {'ok':not failures,'applicable':True,'cycles':cycles,'failures':failures,'probes':probes,
                'fixture':fixture,
                'global_native_warnings':list(messages),'scope':'Independent cam inputs, real catch/contact/end stops, finite source-limit surface forces; no key insertion, electronic credential or humanoid task certificate.'}
    finally:mujoco.set_mjcb_passive(previous)
