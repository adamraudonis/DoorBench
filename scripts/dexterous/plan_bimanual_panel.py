#!/usr/bin/env python3
"""Independent FK, collision and static arm-strength screen for contact transfer.

This writes candidate poses only. It never advances or controls a physical
robot. Passing does not establish grasp loading, balance, a motion policy or
hardware fidelity beyond the explicitly checked constraints.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.stance import StanceController
from doorbench.dexterous.bimanual_screen import screen_checks, validate_grasp_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--radius',type=float,default=.18)
    p.add_argument('--height',type=float,default=1.05)
    p.add_argument('--push-force',type=float,default=15.)
    p.add_argument('--offset',type=float,default=-.048)
    p.add_argument('--transfer-angle',type=float,default=.08)
    p.add_argument('--reach-path',choices=('cartesian','joint'),default='joint')
    p.add_argument('--reach-roll-bump',type=float,default=0.)
    p.add_argument('--reach-pitch-bump',type=float,default=.8)
    p.add_argument('--release-slide',type=float,default=0.)
    p.add_argument('--endpoint-only',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Use a new output directory')
    a.output.mkdir(parents=True)
    (a.output/'planner-source.py').write_text(Path(__file__).read_text())
    (a.output/'reference-input.json').write_bytes(a.reference.read_bytes())
    (a.output/'robot-input.xml').write_bytes(a.robot.read_bytes())
    (a.output/'robot-input.audit.json').write_bytes(a.robot.with_suffix('.audit.json').read_bytes())
    sim=DexterousDoorEnv(a.door,a.robot,json.loads(a.robot.with_suffix('.audit.json').read_text()))
    m,d=sim.m,sim.d;sim.reset(images=False,randomize=False)
    ref=json.loads(a.reference.read_text())
    validate_grasp_reference(ref)
    d.qpos[sim.root_qadr:sim.root_qadr+7]=ref['initial_root']
    for n,q in ref['initial_joints'].items():d.qpos[m.jnt_qposadr[m.joint('robot/'+n).id]]=q
    projection=[]
    for side in ('lh','rh'):
        for digit in ('FF','MF','RF','LF'):
            ids=[m.joint(f'robot/{side}_{digit}J{k}').id for k in (1,2)];qa=m.jnt_qposadr[ids]
            if d.qpos[qa[0]]>d.qpos[qa[1]]:
                before=d.qpos[qa].copy();d.qpos[qa]=before.mean()
                projection.append(dict(joints=[m.joint(int(j)).name for j in ids],before=before.tolist(),after=d.qpos[qa].tolist()))
    mujoco.mj_forward(m,d);base=d.qpos.copy()
    arms={side:[side+'_'+j for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+[prefix+'_WRJ2',prefix+'_WRJ1'] for side,prefix in [('left','lh'),('right','rh')]}
    names=['torso']+arms['left']+arms['right'];ids=[m.joint('robot/'+n).id for n in names];qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids]
    low=m.jnt_range[ids,0]+.02;high=m.jnt_range[ids,1]-.02
    palms={s:m.site('robot/'+s+'_palm_touch').id for s in ('lh','rh')}
    handle=m.body('leaf_handle').id;leaf=m.body('leaf').id
    door_qa={n:m.jnt_qposadr[m.joint(n).id] for n in ('leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide')}
    hrot=d.xmat[handle].reshape(3,3);rh_relative=hrot.T@(d.site_xpos[palms['rh']]-d.xpos[handle]);rh_rotation=hrot.T@d.site_xmat[palms['rh']].reshape(3,3)
    initial_left=d.site_xpos[palms['lh']].copy();initial_left_z=d.site_xmat[palms['lh']].reshape(3,3)[:,2].copy()
    left_geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/lh_')]
    slab=m.geom('leaf_slab').id
    actuators=[]
    for n in names:
        actuators.append(m.actuator('robot/'+('lh_A_'+n[3:] if n.startswith('lh_') else 'rh_A_'+n[3:] if n.startswith('rh_') else n)).id)
    force_limits=m.actuator_forcerange[actuators]
    initial_arm=d.qpos[qa].copy()
    endpoint_goal=d.xpos[leaf]+d.xmat[leaf].reshape(3,3)@np.array([a.radius,a.offset,a.height])
    endpoint_normal=-d.xmat[leaf].reshape(3,3)[:,1].copy()
    right_initial=d.site_xpos[palms['rh']].copy();right_initial_rotation=d.site_xmat[palms['rh']].reshape(3,3).copy()
    def endpoint_residual(q):
        d.qpos[qa]=q;mujoco.mj_kinematics(m,d)
        return np.r_[100*(d.site_xpos[palms['lh']]-endpoint_goal),10*(d.site_xmat[palms['lh']].reshape(3,3)[:,2]-endpoint_normal),
            100*(d.site_xpos[palms['rh']]-right_initial),10*Rotation.from_matrix(right_initial_rotation@d.site_xmat[palms['rh']].reshape(3,3).T).as_rotvec(),.005*(q-initial_arm)]
    rng=np.random.default_rng(31415);best=None
    for seed in range(12):
        guess=initial_arm.copy()
        if seed:
            guess[1:8]=rng.uniform([-1.,.1,-1.5,.2,-1.5,-.4,-.5],[1.,1.6,1.5,2.5,1.5,.1,.4])
            guess[0]+=rng.normal(0,.3)
        fit=least_squares(endpoint_residual,np.clip(guess,low,high),bounds=(low,high),max_nfev=350)
        score=float(np.max(np.abs(fit.fun[:12])))
        if best is None or score<best[0]:best=(score,fit.x.copy())
        print('endpoint',seed,score,flush=True)
        if score<.001:break
    endpoint=best[1];d.qpos[:]=base;mujoco.mj_kinematics(m,d)
    if a.endpoint_only:
        endpoint_residual(endpoint);mujoco.mj_forward(m,d)
        report=dict(scope='Closed-door bimanual endpoint FK only; no contact or physics success',weighted_pose_error=best[0],qpos=d.qpos.tolist(),
            parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'endpoint_weighted_error':best[0]}));sim.close();return
    schedule=[('left_reach',0.,0.,float(x)) for x in np.linspace(0,1,41)]
    schedule += [('operator',float(h),0.,1.) for h in np.linspace(.1,.87,9)]
    schedule += [('both_push',.87,float(t),1.) for t in np.linspace(.02,a.transfer_angle,9)]
    schedule += [('right_release',.87,a.transfer_angle,float(x)) for x in np.linspace(0,1,41)]
    schedule += [('left_push',0.,float(t),1.) for t in np.linspace(a.transfer_angle+.02,1.2,56)]
    all_rows=[];previous=np.clip(d.qpos[qa],low,high);left_offset=a.offset;release_position=None;release_rotation=None
    jp=np.zeros((3,m.nv));jr=jp.copy()
    for frame,(phase,h,angle,u) in enumerate(schedule):
        d.qpos[:]=base;d.qpos[qa]=previous
        for n,q in [('leaf_hinge',angle),('leaf_handle_hinge',h),('leaf_latch_bolt_slide',.0127*h/.87)]:d.qpos[door_qa[n]]=q
        if phase in ('right_release','left_push'):
            for j in sim.joints:
                n=m.joint(int(j)).name
                if n.startswith('robot/rh_') and 'WRJ' not in n:
                    if phase=='left_push':fade=1.
                    elif a.release_slide:fade=float(np.clip((u-.55)/.3,0,1))
                    elif any(n.endswith(f'{digit}J{k}') for digit in ('FF','MF','RF','LF') for k in (1,2)):
                        fade=min(1.,u/.35)
                    elif '/rh_TH' in n:fade=float(np.clip((u-.1)/.3,0,1))
                    else:fade=float(np.clip((u-.6)/.4,0,1))
                    d.qpos[m.jnt_qposadr[j]]=base[m.jnt_qposadr[j]]*(1-fade)
        mujoco.mj_kinematics(m,d)
        R=d.xmat[leaf].reshape(3,3).copy();normal=R[:,1];hrot=d.xmat[handle].reshape(3,3).copy()
        right_goal=d.xpos[handle]+hrot@rh_relative;right_rotation=hrot@rh_rotation
        if phase=='right_release' and release_position is None:
            release_position=d.site_xpos[palms['rh']].copy();release_rotation=d.site_xmat[palms['rh']].reshape(3,3).copy();release_normal=normal.copy();release_axis=hrot[:,0].copy()
        if phase in ('right_release','left_push'):
            blend=1. if phase=='left_push' else float(np.clip((u-.3)/.4,0,1))
            right_goal=release_position-.16*blend*release_normal;right_rotation=release_rotation
            if a.release_slide:
                slide=1. if phase=='left_push' else float(np.clip(u/.5,0,1))
                retreat=1. if phase=='left_push' else float(np.clip((u-.65)/.35,0,1))
                right_goal=release_position-a.release_slide*slide*release_axis-.16*retreat*release_normal
        left_goal=d.xpos[leaf]+R@np.array([a.radius,left_offset,a.height])
        desired_z=-normal
        if phase=='left_reach':
            # Translate across the aperture before approaching the wall plane.
            # This is an explicit Cartesian hypothesis, checked against geometry.
            stage=float(np.clip(u/.55,0,1));stage=stage*stage*(3-2*stage)
            midway=endpoint_goal.copy();midway[1]=min(initial_left[1],endpoint_goal[1]-.25)
            if u<=.55:left_goal=initial_left+stage*(midway-initial_left)
            else:
                stage=(u-.55)/.45;stage=stage*stage*(3-2*stage)
                left_goal=midway+stage*(endpoint_goal-midway)
            orient=float(np.clip(u/.8,0,1));orient=orient*orient*(3-2*orient)
            desired_z=(1-orient)*initial_left_z+orient*endpoint_normal
            desired_z/=np.linalg.norm(desired_z)
            blend=u*u*(3-2*u)
            if a.reach_path=='joint':
                reach_seed=initial_arm+blend*(endpoint-initial_arm)
                reach_seed[2]+=a.reach_roll_bump*np.sin(np.pi*u)**2
                reach_seed[1]+=a.reach_pitch_bump*np.sin(np.pi*u)**2
                reach_seed=np.clip(reach_seed,low,high)
                d.qpos[qa]=reach_seed;mujoco.mj_kinematics(m,d)
                left_goal=d.site_xpos[palms['lh']].copy();desired_z=d.site_xmat[palms['lh']].reshape(3,3)[:,2].copy()
        def residual(q):
            d.qpos[qa]=q;mujoco.mj_kinematics(m,d)
            right_weight=max(0.,1-(angle-(a.transfer_angle+.02))/.15) if phase=='left_push' else 1.
            return np.r_[100*(d.site_xpos[palms['lh']]-left_goal),10*(d.site_xmat[palms['lh']].reshape(3,3)[:,2]-desired_z),
                right_weight*100*(d.site_xpos[palms['rh']]-right_goal),right_weight*10*Rotation.from_matrix(right_rotation@d.site_xmat[palms['rh']].reshape(3,3).T).as_rotvec(),.025*(q-previous), (10*(q[1:8]-reach_seed[1:8]) if phase=='left_reach' and a.reach_path=='joint' else np.zeros(7))]
        fit=least_squares(residual,previous,bounds=(low,high),max_nfev=180,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        if phase=='left_reach' and np.max(np.abs(fit.fun[:12]))>.001:
            seeded=least_squares(residual,(reach_seed if a.reach_path=='joint' else np.clip(initial_arm+blend*(endpoint-initial_arm),low,high)),bounds=(low,high),max_nfev=300,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            if np.linalg.norm(seeded.fun[:12])<np.linalg.norm(fit.fun[:12]):fit=seeded
        previous=fit.x.copy();residual(previous);mujoco.mj_forward(m,d)
        errors=[float(np.linalg.norm(d.site_xpos[palms[s]]-goal)) for s,goal in [('lh',left_goal),('rh',right_goal)]]
        right_constrained=phase!='left_push' or angle<=a.transfer_angle+.02
        normal_error=float(np.linalg.norm(d.site_xmat[palms['lh']].reshape(3,3)[:,2]-desired_z))
        nearest=min(float(mujoco.mj_geomDistance(m,d,g,slab,1.,None)) for g in left_geoms)
        collisions=[]
        for c in d.contact[:d.ncon]:
            bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom];geoms=[m.geom(int(g)).name for g in c.geom]
            if not any(n.startswith('robot/') for n in bodies):continue
            foot='floor' in geoms and any(n.endswith('_ankle_link') for n in bodies)
            if c.dist < (-.003 if foot else -.001):collisions.append(dict(depth_m=-float(c.dist),bodies=bodies,geoms=geoms))
        gravity=d.qfrc_bias[va].copy();mujoco.mj_jacSite(m,d,jp,jr,palms['lh']);required=jp.T@(normal*a.push_force);torque=gravity+required[va]
        if phase in ('operator','both_push'):
            mujoco.mj_jacSite(m,d,jp,jr,palms['rh'])
            right_required=jp.T@(normal*18.+np.array([0.,0.,-12.]))+jr.T@(-normal*1.2)
            required+=right_required;torque+=right_required[va]
        utilization=float(np.max(np.maximum(torque/force_limits[:,1],torque/force_limits[:,0])))
        # Independent finite-support inverse-dynamics screen. These forces are
        # planning inputs only; no step, constraint or wrench enters the plant.
        sim.external_generalized_force=-required
        stance=StanceController(sim);_,stance_status=stance.command()
        root_acceleration=stance.last[:6].tolist() if stance.last is not None else None
        loops={f'{s}_{f}':float(d.qpos[m.jnt_qposadr[m.joint(f'robot/{s}_{f}J1').id]]-d.qpos[m.jnt_qposadr[m.joint(f'robot/{s}_{f}J2').id]]) for s in ('lh','rh') for f in ('FF','MF','RF','LF')}
        row=dict(frame=frame,phase=phase,handle_rad=h,leaf_rad=angle,left_position_error_m=errors[0],right_position_error_m=errors[1],right_pose_constrained=right_constrained,left_normal_error=normal_error,
            left_hand_slab_distance_m=nearest,arm_static_motor_utilization=utilization,stance_status=stance_status,planned_root_acceleration=root_acceleration,
            max_loopback_violation_rad=max(loops.values()),collisions=collisions,qpos=d.qpos.tolist())
        all_rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('qpos','collisions')})+' collisions='+str(len(collisions)),flush=True)
    arm_steps=np.diff(np.asarray([r['qpos'] for r in all_rows])[:,qa],axis=0)
    checks=screen_checks(all_rows,float(np.abs(arm_steps).max()))
    report=dict(scope=__doc__,passed=all(checks.values()),checks=checks,projection=projection,robot_sha256=hashlib.sha256(a.robot.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},max_arm_waypoint_step_rad=float(np.abs(arm_steps).max()),limitations=['FK and static arm-load screen only; no right-grasp force qualification, dynamic stance or contact-transfer proof.','Use the recorded versioned v2 robot and qualified grasp; no physical controller has executed these candidates.'],rows=all_rows)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(a.output/'poses.npz',qpos=[r['qpos'] for r in all_rows]);print(json.dumps({'passed':report['passed'],'checks':checks}))
    sim.close()


if __name__=='__main__':main()
