"""Actual surface-input and hands-free return proof for ordinary rotary trim.

The declared fixture holds the source leaf rigidly. It is a component bench,
not a whole-door task, credential workflow or hidden cam construction proof.
"""
from __future__ import annotations
import copy,hashlib
from pathlib import Path
import numpy as np
import mujoco
from .native_warnings import capture_native_warnings


def run_rotary_shaft_return_qa(native,metadata,*,source_xml,cycles=2,release_seconds=4.):
    rows=metadata.get('rotary_shafts',[])
    if not rows:return {'ok':True,'applicable':False,'failures':[]}
    special={r['lever_joint'] for r in metadata.get('multipoint_locks',[])}
    ordinary=[r for r in rows if r['joint'] not in special]
    delegated=[{'joint':r['joint'],'gate':'run_multipoint_qa','reason':'Contact-driven lift/key/depress sequence required'} for r in rows if r['joint'] in special]
    if not ordinary:return {'ok':True,'applicable':False,'delegated':delegated,'failures':[]}
    source=Path(source_xml);spec=mujoco.MjSpec.from_file(str(source));base=spec.compile()
    for field in ('qpos0','body_mass','body_inertia','geom_pos','geom_quat','geom_size','jnt_range','jnt_stiffness','dof_damping','dof_frictionloss'):
        if not np.array_equal(getattr(native,field),getattr(base,field)):raise ValueError('Rotary return bench input differs from source: '+field)
    welds=[]
    for leaf in sorted({r['leaf'] for r in rows}):
        name='rotary_return_bench_'+leaf;welds.append(name)
        spec.add_equality(name=name,type=mujoco.mjtEq.mjEQ_WELD,name1=leaf,objtype=mujoco.mjtObj.mjOBJ_BODY,
            solref=[.002,1.],solimp=[.99,.999,.0001,.5,2.])
    m=spec.compile();d=mujoco.MjData(m);failures=[];probes=[]
    previous=mujoco.get_mjcb_passive()
    # The bench has no powered credential or closer service commands. The
    # native operator/latch springs, damping, gravity and couplings remain.
    def passive(model,data):
        if model is not m and previous:previous(model,data)
    def q(j):return float(d.qpos[m.jnt_qposadr[j]])
    try:
        mujoco.set_mjcb_passive(passive)
        with capture_native_warnings() as messages:
            mujoco.mj_forward(m,d)
            for row in ordinary:
                joint=m.joint(row['joint']).id;dof=m.jnt_dofadr[joint];body=int(m.jnt_bodyid[joint])
                lower,upper=map(float,m.jnt_range[joint]);rest=float(m.qpos0[m.jnt_qposadr[joint]])
                direction=1. if upper-rest>=rest-lower else -1.
                nominal=float(row['nominal_operator_travel_rad']);target=rest+direction*nominal;span=nominal
                if not lower-.000001<=target<=upper+.000001:
                    failures.append({'check':'nominal_stroke_requires_locked_or_other_control','joint':row['joint'],'range_rad':[lower,upper],'nominal_target_rad':target});continue
                if span<.15:
                    failures.append({'check':'insufficient_available_operator_travel','joint':row['joint'],'range_rad':[lower,upper]});continue
                tag='p' if row['faces'][0]>0 else 'n';names=row['input_sites_by_face'][tag]
                sites=[m.site(name).id for name in names];cap=float(row['operator_force_cap_N'])
                if not sites or any(m.site_bodyid[s]!=body for s in sites):raise ValueError('Return bench lacks correctly bound surface inputs')
                for cycle in range(cycles):
                    for phase,seconds in (('turn',.8),('release',release_seconds)):
                        peak=total=moment=depth=residual=0.;warning=np.zeros(len(d.warning.number),int)
                        for _ in range(round(seconds/m.opt.timestep)):
                            d.qfrc_applied[:]=0.
                            if phase=='turn':
                                tangents=[np.cross(d.xaxis[joint],d.site_xpos[s]-d.xanchor[joint]) for s in sites]
                                radii=np.array([np.linalg.norm(v) for v in tangents])
                                if min(radii)<.005:raise ValueError('Rotary surface has no usable turning radius')
                                command=float(np.clip((12*(target+np.sign(target-rest)*.15-q(joint))-.25*d.qvel[dof])/sum(radii),-cap,cap))
                                for sid,tangent,radius in zip(sites,tangents,radii):
                                    mujoco.mj_applyFT(m,d,tangent/radius*command,np.zeros(3),d.site_xpos[sid],body,d.qfrc_applied)
                                peak=max(peak,abs(command));total=max(total,len(sites)*abs(command));moment=max(moment,sum(radii)*abs(command))
                            mujoco.mj_step(m,d);warning=np.maximum(warning,d.warning.number)
                            depth=max(depth,max((-float(c.dist) for c in d.contact),default=0.))
                            for name in welds:
                                mask=(d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))&(d.efc_id==m.equality(name).id)
                                residual=max(residual,float(np.linalg.norm(d.efc_pos[mask])))
                            if np.any(warning) or not np.isfinite(d.qpos).all():break
                        probe={'joint':row['joint'],'cycle':cycle,'phase':phase,'q_rad':q(joint),'qvel_rad_s':float(d.qvel[dof]),
                            'rest_rad':rest,'target_rad':target,'surface_sites':names,'max_per_point_force_N':peak,
                            'max_total_absolute_force_N':total,'max_surface_moment_Nm':moment,'max_penetration_m':depth,
                            'max_bench_residual':residual,'native_warning_counters':warning.tolist()}
                        probe.update(duration_s=seconds,joint_positions={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt)},
                            native_spring_damping_torque_Nm=float(d.qfrc_passive[dof]),gravity_bias_torque_Nm=float(d.qfrc_bias[dof]))
                        probes.append(probe)
                        if depth>.001 or residual>.0001 or np.any(warning) or not np.isfinite(d.qpos).all():failures.append({'check':'native_contact_warning_or_bench_residual','probe':probe})
                        if phase=='turn' and abs(q(joint)-rest)<.90*span:failures.append({'check':'force_limited_turn_did_not_reach_stroke','probe':probe})
                        if phase=='release' and (abs(q(joint)-rest)>.04 or abs(d.qvel[dof])>.05):failures.append({'check':'hands_free_return_failed','probe':probe})
            if messages:failures.append({'check':'global_native_warning','messages':list(messages)})
        return {'ok':not failures,'applicable':True,'cycles':cycles,'failures':failures,'probes':probes,'delegated':delegated,
            'global_native_warnings':list(messages),'fixture':{'kind':'recompiled_rigid_leaf_component_bench',
            'source_xml_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'welds':welds,
            'scope':'Only source leaf is held. Original trim/cam/latch/contact/spring coefficients retained; other powered/passive service callbacks are not exercised.'},
            'scope':'Finite actual-surface turn and zero-input return. Opposed knobs use the declared cap per point, twice that total absolute force. No free joint torque, pose reset between cycles, credential or whole-door success claim.'}
    finally:mujoco.set_mjcb_passive(previous)
