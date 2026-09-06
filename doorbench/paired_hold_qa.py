"""Native inactive-leaf hold/release fixture, conditional on the active leaf open.

The active leaf is clamped open once at fixture setup. This tests neither its
credentials nor robot reach. There are no configuration writes during cycles.
"""
from __future__ import annotations

import copy
import math
import numpy as np


def _exercise(m,meta):
    import mujoco as mj
    rows=meta.get('paired_leaf_holds',[])
    if not rows:return {'ok':True,'applicable':False,'failures':[]}
    failures=[];phases=[];samples=[];worst=0.;worst_pair=None;max_grip=0.;max_pull=0.;max_clamp=0.;max_limit=0.
    try:
        if m.opt.timestep>.0005+1e-12:raise ValueError('Inactive bolts require a native step of at most 0.5 ms')
        if len({r['leaf_joint']for r in rows})!=1 or len({r['primary_joint']for r in rows})!=1:
            raise ValueError('Fixture requires one inactive leaf and its active partner')
        kinds={r['kind']for r in rows}
        expected=2 if kinds=={'flush_bolt'}else 1 if kinds=={'cane_bolt'}else 0
        if len(rows)!=expected or len({r['joint']for r in rows})!=expected:
            raise ValueError('Complete distinct top/bottom flush bolts or one cane bolt must be declared')
        jA=m.joint(rows[0]['primary_joint']).id;jB=m.joint(rows[0]['leaf_joint']).id
        qA,vA=int(m.jnt_qposadr[jA]),int(m.jnt_dofadr[jA]);qB,vB=int(m.jnt_qposadr[jB]),int(m.jnt_dofadr[jB])
        if any(m.jnt_limited[j] and m.jnt_range[j,1]<.8 for j in(jA,jB)):
            raise ValueError('Inactive-leaf proof requires full independent hinge ranges')
        info=[];engage_sites={}
        inspection=mj.MjData(m)
        for r in rows:
            if r['force_cap_N']!=20.:raise ValueError('Inactive service fixture requires its documented 20 N release cap')
            j=m.joint(r['joint']).id;sid=m.site(r['site']).id;rod=m.geom(r['rod_geom']).id;grip=m.geom(r['grip_geom']).id
            guides=[m.geom(n).id for n in r['guide_geoms']];keepers=[m.geom(n).id for n in r['keeper_geoms']]
            stops=[m.geom(n).id for n in r['stop_geoms']]
            if not guides or not keepers or not stops:raise ValueError('Missing physical bolt guide, receiver or travel stops')
            if any(not(m.geom_contype[g] or m.geom_conaffinity[g])for g in[rod,grip,*guides,*keepers,*stops]):
                raise ValueError('Inactive holding hardware has disabled contacts')
            if m.site_bodyid[sid]!=m.geom_bodyid[grip] or m.geom_bodyid[rod]!=m.geom_bodyid[grip]:
                raise ValueError('Release site and grip must belong to the actual bolt')
            q,v=int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j]);minimum=math.inf
            for value in np.linspace(0.,r['travel_m'],31):
                inspection.qpos[q]=value;mj.mj_kinematics(m,inspection)
                minimum=min(minimum,*(float(mj.mj_geomDistance(m,inspection,rod,g,.02,None))for g in guides))
            if minimum<r['guide_clearance_m']-1e-6:raise ValueError('Bolt lacks full-stroke prepared-guide clearance')
            local=inspection.geom_xmat[grip].reshape(3,3).T@(inspection.site_xpos[sid]-inspection.geom_xpos[grip])
            if m.geom_type[grip]==int(mj.mjtGeom.mjGEOM_SPHERE):distance=np.linalg.norm(local)
            elif m.geom_type[grip]==int(mj.mjtGeom.mjGEOM_CYLINDER):
                if abs(local[2])>m.geom_size[grip,1]:raise ValueError('Release site lies beyond the real grip')
                distance=np.linalg.norm(local[:2])
            else:raise ValueError('Unsupported release grip surface')
            if abs(distance-m.geom_size[grip,0])>1e-6:raise ValueError('Release site is not on its actual grip surface')
            esid=m.site(r['engage_site']).id;engage_sites[r['joint']]=esid
            elocal=inspection.geom_xmat[grip].reshape(3,3).T@(inspection.site_xpos[esid]-inspection.geom_xpos[grip])
            eradius=np.linalg.norm(elocal)if m.geom_type[grip]==int(mj.mjtGeom.mjGEOM_SPHERE)else np.linalg.norm(elocal[:2])
            if m.site_bodyid[esid]!=m.geom_bodyid[grip]or abs(eradius-m.geom_size[grip,0])>1e-6:
                raise ValueError('Re-engagement site is not on its actual grip surface')
            info.append((r,j,q,v,sid,rod,set(keepers),set(stops),int(m.site_bodyid[sid])))
        pull=next(p for p in meta['inactive_leaf_pulls'] if p['face']==1)
        pull_sid=m.site(pull['site']).id;pull_body=int(m.site_bodyid[pull_sid])
        d=mj.MjData(m);d.qpos[qA]=.8
        if meta.get('closer_mounts'):
            from .geometry.closer_mounts import resolve_closer_configuration
            resolve_closer_configuration(m,d.qpos,meta)
        mj.mj_forward(m,d)
        samples.append({'time':float(d.time),'phase':'fixture_initial','qpos':d.qpos.tolist()})
        next_sample=.1
        def phase(name,duration,leaf_target,bolt_target=None,load=False):
            nonlocal worst,worst_pair,max_grip,max_pull,max_clamp,max_limit,next_sample
            contacts=set();stop_contacts=set();receiver_forces={};stop_forces={};max_angle=0.;limit_force=0.
            for _ in range(math.ceil(duration/m.opt.timestep)):
                d.qfrc_applied[:]=0.
                clamp=float(np.clip(600*(.8-d.qpos[qA])-45*d.qvel[vA],-100.,100.));d.qfrc_applied[vA]=clamp;max_clamp=max(max_clamp,abs(clamp))
                q=float(d.qpos[qB]);speed=float(d.qvel[vB])
                if load:effort=30.
                elif leaf_target==0.:
                    desired=-min(.2,8*max(q-.00015,0.));effort=100*(desired-speed)
                    effort-=float(m.dof_frictionloss[vB])+2.
                else:
                    error=leaf_target-q;effort=100*error-20*speed
                    if abs(error)>.001:effort+=np.sign(error)*float(m.dof_frictionloss[vB])
                jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jacSite(m,d,jp,jr,pull_sid)
                direction=d.xmat[m.body(rows[0]['leaf_body']).id].reshape(3,3)[:,1]
                leverage=float(jp[:,vB]@direction)
                if abs(leverage)<.1:raise ValueError('Inactive pull has insufficient real moment arm')
                force=float(np.clip(effort/leverage,-50.,50.));max_pull=max(max_pull,abs(force))
                mj.mj_applyFT(m,d,direction*force,np.zeros(3),d.site_xpos[pull_sid],pull_body,d.qfrc_applied)
                for r,j,qadr,vadr,sid,rod,keepers,stops,bid in info:
                    if bolt_target is not None:
                        axis=d.xaxis[j];mass=float(m.body_mass[bid]);target=r['travel_m']+.001 if bolt_target else -.001
                        force=float(np.clip(2500*(target-d.qpos[qadr])-20*d.qvel[vadr]+mass*9.81*axis[2],-r['force_cap_N'],r['force_cap_N']))
                        if r['kind']=='flush_bolt':
                            # A finger pressing one face can push inward only.
                            # Physical bearing damping handles deceleration;
                            # do not invent suction on the finger contact.
                            force=max(0.,force)if bolt_target else min(0.,force)
                        contact_sid=sid if bolt_target else engage_sites[r['joint']]
                        max_grip=max(max_grip,abs(force));mj.mj_applyFT(m,d,axis*force,np.zeros(3),d.site_xpos[contact_sid],bid,d.qfrc_applied)
                mj.mj_step(m,d)
                if d.time>=next_sample:
                    samples.append({'time':float(d.time),'phase':name,'qpos':d.qpos.tolist()});next_sample+=.1
                max_angle=max(max_angle,abs(float(d.qpos[qB])))
                for contact_index,c in enumerate(d.contact):
                    if -c.dist>worst:worst=-float(c.dist);worst_pair=[m.geom(g).name for g in c.geom]
                    for r,j,qadr,vadr,sid,rod,keepers,stops,bid in info:
                        receiver=rod in c.geom and any(g in keepers for g in c.geom)
                        stop=any(m.geom_bodyid[g]==bid for g in c.geom)and any(g in stops for g in c.geom)
                        if receiver or stop:
                            pair=tuple(m.geom(g).name for g in c.geom);wrench=np.zeros(6)
                            mj.mj_contactForce(m,d,contact_index,wrench);normal=max(0.,float(wrench[0]))
                            if receiver:
                                contacts.add(pair);receiver_forces[pair]=max(normal,receiver_forces.get(pair,0.))
                            if stop:
                                stop_contacts.add(pair);stop_forces[pair]=max(normal,stop_forces.get(pair,0.))
                for e in range(d.nefc):
                    if d.efc_type[e]==int(mj.mjtConstraint.mjCNSTR_LIMIT_JOINT) and d.efc_id[e] in {x[1]for x in info}:
                        limit_force=max(limit_force,abs(float(d.efc_force[e])))
                if any(w.number for w in d.warning)or not np.isfinite(d.qpos).all():raise ValueError(name+': native warning or nonfinite state')
                if worst>.001:raise ValueError(name+': native penetration exceeded 1 mm')
            max_limit=max(max_limit,limit_force)
            samples.append({'time':float(d.time),'phase':name+':end','qpos':d.qpos.tolist()})
            row={'phase':name,'inactive_angle_rad':float(d.qpos[qB]),'active_angle_rad':float(d.qpos[qA]),
                'max_inactive_angle_rad':max_angle,'bolt_q_m':{r['joint']:float(d.qpos[qadr])for r,j,qadr,*_ in info},
                'receiver_contacts':[list(p)for p in sorted(contacts)],'stroke_stop_contacts':[list(p)for p in sorted(stop_contacts)],
                'receiver_max_normal_force_N':[{'pair':list(p),'force_N':f}for p,f in sorted(receiver_forces.items())],
                'stop_max_normal_force_N':[{'pair':list(p),'force_N':f}for p,f in sorted(stop_forces.items())],
                'bolt_joint_limit_force_N':limit_force}
            phases.append(row);return row
        for cycle in range(2):
            phase('seat',.5,0.)
            row=phase('retained_opening_load',.5,.5,load=True)
            if row['max_inactive_angle_rad']>.004 or any(not any(r['rod_geom']in p['pair']and p['force_N']>.05 for p in row['receiver_max_normal_force_N'])for r in rows):
                raise ValueError('Physical receivers do not each carry the inactive-leaf load')
            phase('unload',1.,0.)
            row=phase('withdraw',1.2,0.,True)
            if any(row['bolt_q_m'][r['joint']]<r['withdrawn_threshold_m']for r in rows):raise ValueError('Bolts do not withdraw clear of their receivers')
            if any(not any(set(p['pair'])&set(r['stop_geoms'])and p['force_N']>.05 for p in row['stop_max_normal_force_N'])for r in rows):
                raise ValueError('Withdrawal did not reach every actual collar stop')
            row=phase('open',2.3,.5)
            if row['inactive_angle_rad']<.4:raise ValueError('Released inactive leaf cannot open independently')
            row=phase('close',4.,0.)
            if abs(row['inactive_angle_rad'])>.0006:raise ValueError('Inactive leaf did not seat for reinsertion')
            row=phase('reinsert',1.2,0.,False)
            if any(abs(q)>.002 for q in row['bolt_q_m'].values()):raise ValueError('Inactive bolts did not re-enter their receivers')
            if any(not any(set(p['pair'])&set(r['stop_geoms'])and p['force_N']>.05 for p in row['stop_max_normal_force_N'])for r in rows):
                raise ValueError('Insertion did not reach every actual collar stop')
        if max_limit>.01:raise ValueError('Native joint range rather than physical collars stopped a bolt')
    except (ValueError,KeyError)as exc:failures.append(str(exc))
    return {'ok':not failures,'applicable':True,'failures':failures,'phases':phases,'inspection_samples':samples,'max_penetration_m':worst,
        'worst_pair':worst_pair,'max_release_force_N':max_grip,'max_pull_force_N':max_pull,
        'max_active_fixture_torque_Nm':max_clamp,'max_bolt_joint_limit_force_N':max_limit,
        'scope':'Two inside-service cycles, actual release-site and fixed-pull forces. Active leaf prescribed open once then clamped; primary credential/access and fastener strength are not tested. No intermediate pose updates.'}


def run_paired_hold_qa(model,metadata):
    """Private model, authored closer fields, captured warning strings/counters."""
    import mujoco as mj
    from .native_warnings import capture_native_warnings
    from .closer_pinion import compile_pinion_closers,apply_pinion_closers
    from .closer_track_hold import compile_track_holds,apply_track_holds
    private=copy.copy(model);previous=mj.get_mjcb_passive()
    pinions=compile_pinion_closers(private,metadata);tracks=compile_track_holds(private,metadata)
    def callback(m,d):
        if m is private:apply_pinion_closers(m,d,pinions);apply_track_holds(m,d,tracks)
        elif previous is not None:previous(m,d)
    try:
        mj.set_mjcb_passive(callback)
        with capture_native_warnings()as warnings:result=_exercise(private,metadata)
    finally:mj.set_mjcb_passive(previous)
    result['native_warning_messages']=warnings
    if warnings:result['ok']=False;result['failures'].append('Native warning callback emitted a message')
    return result
