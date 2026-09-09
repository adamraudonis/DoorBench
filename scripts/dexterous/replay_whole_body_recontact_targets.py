#!/usr/bin/env python3
"""Unstepped replay of measured leaf motion through whole-body target planning.

This explicitly accepts an incomplete failed source as a finite diagnostic
input. It never qualifies missing physics or applies the resulting commands.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prediction-seconds',type=float,default=0.,help='Additional unstepped constant-final-leaf-rate diagnostic; all other measured coordinates stay at the archived terminal value')
    args=parser.parse_args();run=args.run.resolve();source=run.with_name(run.name+'-source')
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from doorbench.dexterous.environment import DexterousDoorEnv
    from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
    from doorbench.dexterous.screened_panel_path import ScreenedPanelPath
    from doorbench.dexterous.palm_recontact_teacher import reproject_palm
    from doorbench.dexterous.whole_body_contact_targets import WholeBodyContactTargets
    from screen_actual_base_palm import enclosing_collision_radius
    cfg=json.loads((run/'manifest.json').read_text())['configuration'];robot=Path(cfg['robot'])
    if hashlib.file_digest(robot.open('rb'),'sha256').hexdigest()!=hashlib.file_digest((run/'robot-input.xml').open('rb'),'sha256').hexdigest():
        raise ValueError('The original robot input changed')
    sim=DexterousDoorEnv(cfg['door'],robot,json.loads(robot.with_suffix('.audit.json').read_text()))
    m,d=sim.m,sim.d;rm=mujoco.MjModel.from_xml_path(str(robot));rd=mujoco.MjData(rm)
    plan=json.loads((source/'whole-body-panel-plan.json').read_text());names=plan['joint_names']
    planner=WholeBodyContactTargets(rm,names);path=ScreenedPanelPath(plan['progress'],plan['coordinates'],plan['duration_s'])
    with gzip.open(run/'panel-phase.jsonl.gz','rt') as file:phase=[json.loads(line) for line in file]
    act_names=[a['name'] for a in json.loads((run/'motors-input.json').read_text())['actuators']]
    qadr=np.array([m.jnt_qposadr[m.joint('robot/'+n).id] for n in planner.scalar])
    root_joint=m.joint('robot/free_base').id;rq=m.jnt_qposadr[root_joint];rv=m.jnt_dofadr[root_joint]
    body=m.jnt_bodyid[root_joint];leaf=m.body('leaf').id;lh=rm.site('lh_palm_touch').id
    if not np.isfinite(args.prediction_seconds) or not 0<=args.prediction_seconds<=6.:
        raise ValueError('Require a bounded zero-to-six-second geometric prediction')
    actual_count=len(phase)
    def observations():
        last=None
        for raw in NativeTransitionArchive.read(run/'raw-transitions',allow_incomplete=True):
            if float(raw['interval_start_s'])<phase[0]['episode_time_s']-1e-9:continue
            last=raw
            yield raw,False
        q=m.jnt_qposadr[m.joint('leaf_hinge').id];v=m.jnt_dofadr[m.joint('leaf_hinge').id]
        for step in range(1,round(args.prediction_seconds/.002)+1):
            raw=dict(last);raw['qpos_before']=last['qpos_before'].copy()
            raw['qpos_before'][q]+=last['qvel_before'][v]*step*.002
            raw['interval_start_s']=float(last['interval_start_s'])+step*.002
            old=dict(phase[actual_count-1]);old['episode_time_s']=raw['interval_start_s'];old['actual_aperture_rad']=float(raw['qpos_before'][q]);phase.append(old)
            yield raw,True
    rows=[];clock_error=0.;frame_error=0.;initial_leaf=None;index=0;start=phase[0]['episode_time_s']
    cd=mujoco.MjData(m);floor=m.geom('floor').id;torso=m.body('robot/torso_link').id
    hand=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
    scene=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and not m.body(m.geom_bodyid[g]).name.startswith('robot/')]
    radii=np.zeros(m.ngeom)
    for g in hand+scene:radii[g]=enclosing_collision_radius(m,g)
    robotq=np.array([m.jnt_qposadr[m.joint('robot/'+n).id] for n in names])
    minimum_right_clearance=.040001;exact_pairs=0;failures=[]
    terminal_error=None
    for raw,predicted in observations():
        t=float(raw['interval_start_s'])
        if t<start-1e-9:continue
        if index>=len(phase) or abs(phase[index]['episode_time_s']-t)>1e-9:raise ValueError('Require matching actual force/pose/target epochs')
        old=phase[index];index+=1;d.qpos[:]=raw['qpos_before'];d.qvel[:]=raw['qvel_before'];mujoco.mj_kinematics(m,d)
        ids=raw['body_ids'];error=max(float(np.max(abs(d.xpos[ids]-raw['body_positions_world_m']))),float(np.max(abs(d.xmat[ids].reshape(-1,3,3)-raw['body_rotations_world']))))
        if not predicted:
            frame_error=max(frame_error,error)
            if error>1e-9:raise ValueError('Actual source frames disagree with FK')
        joints=dict(zip(planner.scalar,d.qpos[qadr]));rotation=d.xmat[body].reshape(3,3)
        root=np.r_[d.qpos[rq:rq+7],d.qvel[rv:rv+3],rotation@d.qvel[rv+3:rv+6]]
        leaf_pose=np.r_[d.xpos[leaf],d.xquat[leaf]]
        if initial_leaf is None:
            initial_leaf=leaf_pose.copy();initial_root=root[:7].copy();initial_rotation=Rotation.from_quat([*root[4:7],root[3]])
            previous=json.loads((source/'ungrip-path.json').read_text())['rows'][-1]
            position=np.array(previous['root'][:3]);quat=previous['root'][3:]
            body_rotation=Rotation.from_quat([*quat[1:],quat[0]])
            targets=[]
            for n in names:
                if any(n==side+'_'+j for side in ('left','right') for j in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')):
                    targets.append(previous['joints'][n]);continue
                jid=rm.joint(n).id;acts=[a for a in range(rm.nu) if rm.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_JOINT and rm.actuator_trnid[a,0]==jid]
                if len(acts)!=1 or not np.array_equal(rm.actuator_gear[acts[0]],[1,0,0,0,0,0]):raise ValueError('Require an original direct body motor')
                targets.append(old['latched_motor_targets'][act_names.index(rm.actuator(acts[0]).name)])
            for n,v in zip(old['corrected_chain_joint_names'],old['corrected_chain_targets']):targets[names.index(n)]=v
            initial_command=np.r_[position-root[:3],(body_rotation*initial_rotation.inv()).as_rotvec(),targets]
            planner.begin(t,root,joints,coordinate=initial_command)
        sample=path.sample(t-start);x=sample['position'];rd.qpos[:]=rm.qpos0
        rd.qpos[:3]=initial_root[:3]+x[:3];quat=(Rotation.from_rotvec(x[3:6])*initial_rotation).as_quat();rd.qpos[3:7]=np.r_[quat[3],quat[:3]]
        for n,v in plan['initial_robot_joints'].items():rd.qpos[rm.jnt_qposadr[rm.joint(n).id]]=v
        for n,v in zip(names,x[6:]):rd.qpos[rm.jnt_qposadr[rm.joint(n).id]]=v
        mujoco.mj_kinematics(rm,rd)
        p,r=reproject_palm(rd.site_xpos[lh],rd.site_xmat[lh].reshape(3,3),initial_leaf,leaf_pose)
        normal=Rotation.from_quat([*leaf_pose[4:],leaf_pose[3]]).as_matrix()[:,1]
        p+=normal*old['normal_admittance']['normal_offset_m']
        try:
            target,velocity,acceleration,info=planner.update(t,joints,p,r,x)
        except ValueError as error:
            terminal_error=dict(time_s=t,predicted=predicted,message=str(error))
            break
        cd.qpos[:]=raw['qpos_before'];cd.qpos[rq:rq+3]=initial_root[:3]+target[:3]
        quat=(Rotation.from_rotvec(target[3:6])*initial_rotation).as_quat();cd.qpos[rq+3:rq+7]=np.r_[quat[3],quat[:3]]
        cd.qpos[robotq]=target[6:];mujoco.mj_kinematics(m,cd);mujoco.mj_comPos(m,cd);mujoco.mj_collision(m,cd)
        center=np.linalg.norm(cd.geom_xpos[hand,None,:]-cd.geom_xpos[None,scene,:],axis=2)
        candidates=np.argwhere(center-radii[hand,None]-radii[None,scene]<.040001);nearest=.040001
        for hi,si in candidates:
            nearest=min(nearest,float(mujoco.mj_geomDistance(m,cd,hand[hi],scene[si],.040001,None)));exact_pairs+=1
        minimum_right_clearance=min(minimum_right_clearance,nearest)
        penetration=0.
        for contact in cd.contact[:cd.ncon]:
            bodies=[m.body(m.geom_bodyid[g]).name for g in contact.geom]
            allowed=floor in contact.geom and any(b.endswith('_ankle_link') for b in bodies)
            if not allowed and any(b.startswith('robot/') for b in bodies):penetration=max(penetration,-float(contact.dist))
        info.update(right_scene_clearance_capped_m=nearest,nonfoot_robot_penetration_m=penetration,
                    torso_tilt_deg=float(np.degrees(np.arccos(np.clip(cd.xmat[torso].reshape(3,3)[2,2],-1,1)))),
                    com_xy_motion_m=float(np.linalg.norm(cd.subtree_com[body,:2]-planner.com_position)),
                    maximum_combined_robot_fk_disagreement=max(float(np.max(abs(cd.site_xpos[m.site('robot/lh_palm_touch').id]-planner.data.site_xpos[planner.palms[0]]))),float(np.max(abs(cd.site_xmat[m.site('robot/lh_palm_touch').id]-planner.data.site_xmat[planner.palms[0]])))))
        if (nearest<.04 or penetration>.003 or info['torso_tilt_deg']>12 or info['com_xy_motion_m']>.015
                or info['maximum_combined_robot_fk_disagreement']>1e-9
                or info['maximum_commanded_foot_position_error_m']>.0001):
            failures.append(dict(time_s=t,info=info.copy()))
        rows.append(dict(time_s=t,predicted=predicted,source_or_predicted_leaf_rad=old['actual_aperture_rad'],target=target.tolist(),velocity=velocity.tolist(),acceleration=acceleration.tolist(),info=info))
    if terminal_error is None and index!=len(phase):raise ValueError('The actual source ends before its target evidence')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'diagnostic-source.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'target-planner-source.py').write_bytes((Path(__file__).resolve().parents[2]/'doorbench/dexterous/whole_body_contact_targets.py').read_bytes())
    report=dict(scope='Unstepped counterfactual targets driven by actual archived source leaf motion; no new motor/force/contact qualification.',
                passed=not failures and terminal_error is None,failures=failures,terminal_error=terminal_error,prediction_seconds=args.prediction_seconds,prediction_contract='After the actual prefix: constant final measured leaf rate; all other source state and normal-admittance offset held fixed. These are synthetic kinematics, never physics.',right_collision_shapes=len(hand),scene_collision_shapes=len(scene),exact_distance_pairs=exact_pairs,minimum_right_scene_clearance_capped_m=minimum_right_clearance,
                source_run=str(run),source_archive_complete=json.loads((run/'raw-transitions/manifest.json').read_text())['complete'],
                samples=len(rows),maximum_actual_source_frame_error=frame_error,initial_command=initial_command.tolist(),
                maximums={key:max(row['info'][key] for row in rows) for key in rows[0]['info'] if key!='time_s'},
                final=rows[-1],rows=rows)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    sim.close();print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
