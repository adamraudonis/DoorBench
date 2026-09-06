"""Native service fixtures for independently moving paired-door hardware."""
from __future__ import annotations

import math
import numpy as np


def run_dutch_join_qa(model,metadata):
    import mujoco as mj
    r=metadata.get('dutch_joining_bolt')
    if not r:return {'ok':True,'applicable':False,'failures':[]}
    failures=[];rows=[];worst=0.;pair=None;max_force=0.;max_torque=0.
    try:
        if model.opt.timestep>.0005+1e-12:raise ValueError('Dutch bolt needs 0.5 ms or smaller native step')
        joint={k:model.joint(r[k]).id for k in ('joint','upper_joint','lower_joint')}
        qa={k:int(model.jnt_qposadr[j]) for k,j in joint.items()}
        va={k:int(model.jnt_dofadr[j]) for k,j in joint.items()}
        if any(model.jnt_range[joint[k],1]<.75 for k in ('upper_joint','lower_joint')):
            raise ValueError('Dutch leaves require independent full hinge ranges')
        rod=model.geom(r['rod_geom']).id;grip=model.geom(r['grip_geom']).id;sid=model.site(r['site']).id
        guides=[model.geom(n).id for n in r['guide_geoms']]
        keepers=[model.geom(n).id for n in r['keeper_geoms']]
        if any(not(model.geom_contype[g] or model.geom_conaffinity[g]) for g in [rod,*guides,*keepers]):
            raise ValueError('Dutch rod, guide or keeper contact disabled')
        # Separate inspection state: primitive distances check stock even when
        # native parent filtering would otherwise hide a prepared-guide defect.
        inspection=mj.MjData(model);minimum=math.inf
        for value in np.linspace(0.,r['travel_m'],25):
            inspection.qpos[qa['joint']]=value;mj.mj_kinematics(model,inspection)
            minimum=min(minimum,*(float(mj.mj_geomDistance(model,inspection,rod,g,.02,None)) for g in guides))
        if minimum<r['guide_clearance_m']-1e-6:raise ValueError('Dutch rod lacks complete prepared-guide clearance')
        if abs(np.linalg.norm(inspection.site_xpos[sid]-inspection.geom_xpos[grip])-model.geom_size[grip,0])>1e-6:
            raise ValueError('Dutch joining-bolt grip is not on its physical knob')
        d=mj.MjData(model);mj.mj_forward(model,d)
        latches=[j for j in range(model.njnt) if 'latch_bolt' in model.joint(j).name or model.joint(j).name=='lower_handle_hinge']
        def phase(name,duration,lower,upper,bolt=None):
            nonlocal worst,pair,max_force,max_torque
            max_difference=0.;contacts=set()
            for _ in range(math.ceil(duration/model.opt.timestep)):
                d.qfrc_applied[:]=0.
                for key,target in (('upper_joint',upper),('lower_joint',lower)):
                    if target is None:effort=0.
                    elif target==0.:
                        q=float(d.qpos[qa[key]]);speed=float(d.qvel[va[key]])
                        desired=-min(.20,8*max(q-.0002,0.))
                        effort=120*(desired-speed)
                        if q<.002 and speed>-.025:effort-=5.
                    else:effort=180*(target-d.qpos[qa[key]])-25*d.qvel[va[key]]
                    effort=float(np.clip(effort,-40.,40.))
                    d.qfrc_applied[va[key]]=effort;max_torque=max(max_torque,abs(effort))
                # Isolate joining hardware from the ordinary lower-leaf latch.
                for j in latches:
                    q,v=int(model.jnt_qposadr[j]),int(model.jnt_dofadr[j])
                    d.qfrc_applied[v]=np.clip(500*(model.jnt_range[j,1]-d.qpos[q])-10*d.qvel[v],-20.,20.)
                if bolt is not None:
                    force=float(np.clip(2500*(bolt-d.qpos[qa['joint']])-20*d.qvel[va['joint']],-20.,20.))
                    mj.mj_applyFT(model,d,np.array([0.,0.,force]),np.zeros(3),d.site_xpos[sid],int(model.site_bodyid[sid]),d.qfrc_applied)
                    max_force=max(max_force,abs(force))
                mj.mj_step(model,d)
                max_difference=max(max_difference,abs(float(d.qpos[qa['upper_joint']]-d.qpos[qa['lower_joint']])))
                for contact in d.contact:
                    if -contact.dist>worst:worst=-float(contact.dist);pair=[model.geom(g).name for g in contact.geom]
                    if rod in contact.geom and any(g in keepers for g in contact.geom):contacts.add(tuple(model.geom(g).name for g in contact.geom))
                if any(w.number for w in d.warning) or not np.isfinite(d.qpos).all():raise ValueError(name+': native warning or nonfinite state')
            row={'phase':name,'joint_q_m':float(d.qpos[qa['joint']]),'upper_angle_rad':float(d.qpos[qa['upper_joint']]),
                'lower_angle_rad':float(d.qpos[qa['lower_joint']]),'max_leaf_difference_rad':max_difference,
                'keeper_contacts':[list(p) for p in sorted(contacts)]}
            rows.append(row);return row
        for cycle in range(2):
            phase('seat_leaves',1.2,0.,0.,float(d.qpos[qa['joint']]))
            if max(abs(float(d.qpos[qa[k]])) for k in ('upper_joint','lower_joint'))>.0005:
                raise ValueError('Leaves not seated closely enough for bolt insertion')
            row=phase('engage',.7,0.,0.,0.)
            if row['joint_q_m']>.002:raise ValueError('Joining rod did not enter keeper')
            row=phase('joined_upper_load',.6,0.,.35)
            if row['max_leaf_difference_rad']>.004 or not row['keeper_contacts']:
                raise ValueError('Keeper does not transfer upper-leaf load to lower leaf')
            phase('close_joined',2.,0.,0.)
            row=phase('withdraw',.7,0.,0.,r['travel_m'])
            if row['joint_q_m']<r['withdrawn_threshold_m']:raise ValueError('Joining rod did not clear keeper')
            row=phase('upper_only_open',1.5,0.,.55)
            if row['upper_angle_rad']<.45 or abs(row['lower_angle_rad'])>.01:
                raise ValueError('Upper leaf cannot open independently after withdrawal')
            phase('upper_close',4.,0.,0.)
            if max(abs(float(d.qpos[qa[k]])) for k in ('upper_joint','lower_joint'))>.0005:
                raise ValueError('Leaves not seated for re-engagement')
            row=phase('reengage',.7,0.,0.,0.)
            if row['joint_q_m']>.002:raise ValueError('Joining rod did not re-enter keeper')
            row=phase('joined_lower_open',1.5,.5,None)
            if min(row['upper_angle_rad'],row['lower_angle_rad'])<.2 or row['max_leaf_difference_rad']>.004 or not row['keeper_contacts']:
                raise ValueError('Lower leaf does not carry upper leaf after re-engagement')
            phase('both_close',4.,0.,0.)
        if worst>.001:failures.append('Dutch native contact penetration exceeded 1 mm')
    except (ValueError,KeyError) as exc:failures.append(str(exc))
    return {'ok':not failures,'applicable':True,'failures':failures,'phases':rows,
        'max_penetration_m':worst,'worst_pair':pair,'max_grip_force_N':max_force,'max_leaf_fixture_torque_Nm':max_torque,
        'scope':'Two inside-service joining/release/load cycles using actual knob forces and bounded leaf fixture torques. Ordinary lower latch isolated; not robot access, bolt strength or single-hand certification.'}
