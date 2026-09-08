#!/usr/bin/env python3
"""Development screen: recontact the panel and keep the cleared right arm in joint posture.

Foot frames and the cleared RH joint posture are planning objectives only. This
script owns a fresh unstepped model and never receives an active plant handle.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--at',type=float,default=69.8)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--height-drop-m',type=float,default=0.)
    p.add_argument('--radius-shift-m',type=float,default=-.04)
    p.add_argument('--root-extent-m',type=float,default=.05)
    p.add_argument('--root-rotation-rad',type=float,default=.15)
    p.add_argument('--root-rotation-norm-rad',type=float)
    p.add_argument('--target-aperture-rad',type=float,default=1.2)
    p.add_argument('--nodes',type=int,default=61)
    p.add_argument('--flatten-over-rad',type=float,default=.2)
    p.add_argument('--normal-recontact-m',type=float,default=0.)
    p.add_argument('--flatten-palm',action='store_true',help='Rotate the actual palm face toward the panel over0.2rad while preserving its collision support plane')
    p.add_argument('--admit-exact-soft-limit-start',action='store_true',help='Retain only the measured initial solver-limit excursion, then smoothly regain the 1mm/rad numeric joint margin within 0.1rad aperture')
    a=p.parse_args()
    if not 0<=a.normal_recontact_m<=.006:raise ValueError('Recontact is a declared bounded geometric target, not contact evidence')
    if not -.04<=a.radius_shift_m<=.04 or not .1<=a.flatten_over_rad<=.6 or not 0<=a.height_drop_m<=.15 or not 0<=a.root_extent_m<=.08 or not 0<=a.root_rotation_rad<=.2 or a.nodes<21:
        raise ValueError('Require bounded declared geometry settings')
    if a.root_rotation_norm_rad is not None and not .01<=a.root_rotation_norm_rad<=.05:raise ValueError('Rotation norm must remain inside the original0.05rad screen')
    run=a.source_run.resolve();config=json.loads((run/'manifest.json').read_text())['configuration']
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from doorbench.dexterous.environment import DexterousDoorEnv
    from doorbench.dexterous.palm_panel_geometry import original_palm_vertices,flatten_palm_goal
    robot=Path(config['robot']);sim=DexterousDoorEnv(config['door'],robot,json.loads(robot.with_suffix('.audit.json').read_text()))
    m,d=sim.m,sim.d
    manifest=json.loads((run/'raw-transitions/manifest.json').read_text())
    if not manifest['complete']:raise ValueError('Require a complete measured source')
    chunk=next(c for c in manifest['chunks'] if c['interval_start_s']-1e-8<=a.at<c['interval_end_s']-1e-8)
    path=run/'raw-transitions'/chunk['file']
    if hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=chunk['sha256']:raise ValueError('Changed actual source')
    with np.load(path,allow_pickle=False) as raw:
        i=int(np.argmin(abs(raw['interval_start_s']-a.at)))
        base=raw['qpos_before'][i].copy();velocity=raw['qvel_before'][i].copy();time=float(raw['interval_start_s'][i])
        start,end=raw['body_offsets'][i:i+2];ids=raw['body_ids'][start:end]
        expected_p=raw['body_positions_world_m'][start:end];expected_r=raw['body_rotations_world'][start:end]
    d.qpos[:]=base;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    error=max(float(np.max(abs(d.xpos[ids]-expected_p))),float(np.max(abs(d.xmat[ids].reshape(-1,3,3)-expected_r))))
    if error>1e-9:raise ValueError('Destination FK differs from measured source; requalify geometry')
    rh,lh=[m.site('robot/'+s+'_palm_touch').id for s in ('rh','lh')]
    root_joint=m.joint('robot/free_base').id;rq=m.jnt_qposadr[root_joint]
    rp=base[rq:rq+3].copy();rr=Rotation.from_quat([*base[rq+4:rq+7],base[rq+3]])
    robot_body=m.jnt_bodyid[root_joint];com=d.subtree_com[robot_body].copy()
    leaf=m.body('leaf').id;leafj=m.joint('leaf_hinge').id;leafq=m.jnt_qposadr[leafj]
    leaf_p=d.xpos[leaf].copy();leaf_r=d.xmat[leaf].reshape(3,3).copy()
    local_p=leaf_r.T@(d.site_xpos[lh]-leaf_p);local_r=leaf_r.T@d.site_xmat[lh].reshape(3,3)
    palm_vertices=original_palm_vertices(m,d,lh) if a.flatten_palm else None
    right_p=d.site_xpos[rh].copy();right_r=d.site_xmat[rh].reshape(3,3).copy()
    feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    feet_p=d.xpos[feet].copy();feet_r=d.xmat[feet].reshape(2,3,3).copy()
    names=[side+'_'+n for side in ('left','right') for n in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')]+['torso']
    names += [side+'_'+n for side in ('right','left') for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]
    names += ['rh_WRJ2','rh_WRJ1','lh_WRJ2','lh_WRJ1']
    js=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[js];initial=base[qa].copy()
    low=np.r_[np.full(3,-max(1e-12,a.root_extent_m)),np.full(3,-max(1e-12,a.root_rotation_rad)),m.jnt_range[js,0]+.001]
    high=np.r_[np.full(3,max(1e-12,a.root_extent_m)),np.full(3,max(1e-12,a.root_rotation_rad)),m.jnt_range[js,1]-.001]
    right_indices=np.array([i for i,n in enumerate(names) if n.startswith('right_') and any(k in n for k in ('shoulder','elbow','wrist')) or n.startswith('rh_WRJ')])
    if len(right_indices)!=7:raise ValueError('Require exactly the original seven cleared RH arm joints')
    previous=np.r_[np.zeros(6),initial];rows=[]
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'screen-source.py').write_bytes(Path(__file__).read_bytes())
    (a.output/'palm-panel-geometry-source.py').write_bytes((Path(__file__).resolve().parents[2]/'doorbench/dexterous/palm_panel_geometry.py').read_bytes())
    for angle in np.linspace(base[leafq],a.target_aperture_rad,a.nodes):
        if a.admit_exact_soft_limit_start:
            progress=float(np.clip((angle-base[leafq])/.1,0,1));ramp=progress**3*(10+progress*(-15+6*progress))
            low[6:]=(1-ramp)*np.minimum(m.jnt_range[js,0]+.001,initial-1e-8)+ramp*(m.jnt_range[js,0]+.001)
            high[6:]=(1-ramp)*np.maximum(m.jnt_range[js,1]-.001,initial+1e-8)+ramp*(m.jnt_range[js,1]-.001)
        low[6+right_indices]=initial[right_indices]-1e-10
        high[6+right_indices]=initial[right_indices]+1e-10
        state=base.copy();state[leafq]=angle;d.qpos[:]=state;mujoco.mj_kinematics(m,d)
        lr=d.xmat[leaf].reshape(3,3).copy();lp=d.xpos[leaf].copy()
        u=float(np.clip((angle-base[leafq])/.35,0,1));blend=u**3*(10+u*(-15+6*u))
        local=local_p.copy();local[0]+=a.radius_shift_m*blend;local[2]-=a.height_drop_m*blend
        goal_rotation=local_r
        if a.flatten_palm:
            fu=float(np.clip((angle-base[leafq])/a.flatten_over_rad,0,1));fb=fu**3*(10+fu*(-15+6*fu))
            local,goal_rotation,_=flatten_palm_goal(local,local_r,palm_vertices,fb)
            local[1]+=a.normal_recontact_m*fb
        goal_p=lp+lr@local;goal_r=lr@goal_rotation
        def evaluate(x):
            d.qpos[:]=state;d.qpos[rq:rq+3]=rp+x[:3]
            quat=(Rotation.from_rotvec(x[3:6])*rr).as_quat();d.qpos[rq+3:rq+7]=np.r_[quat[3],quat[:3]]
            d.qpos[qa]=x[6:];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            hands=np.r_[100*(d.site_xpos[lh]-goal_p),10*Rotation.from_matrix(goal_r@d.site_xmat[lh].reshape(3,3).T).as_rotvec(),
                        np.zeros(6)]
            foot=np.concatenate([np.r_[100*(d.xpos[b]-feet_p[j]),10*Rotation.from_matrix(feet_r[j]@d.xmat[b].reshape(3,3).T).as_rotvec()] for j,b in enumerate(feet)])
            return np.r_[hands,foot,5*(d.subtree_com[robot_body,:2]-com[:2]),.015*(x[6:]-initial),.05*x[:6],0. if a.root_rotation_norm_rad is None else 1000.*max(0.,np.linalg.norm(x[3:6])-a.root_rotation_norm_rad)]
        fit=least_squares(evaluate,np.clip(previous,low,high),bounds=(low,high),max_nfev=800,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        previous=fit.x.copy();res=evaluate(previous);mujoco.mj_collision(m,d)
        collisions=[]
        for c in d.contact[:d.ncon]:
            bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom]
            foot_contact='floor' in [m.geom(int(g)).name for g in c.geom] and any(b.endswith('_ankle_link') for b in bodies)
            if not foot_contact and any(b.startswith('robot/') for b in bodies) and c.dist<-.003:
                collisions.append(dict(bodies=bodies,depth_m=-float(c.dist)))
        up=d.xmat[m.body('robot/torso_link').id].reshape(3,3)[:,2]
        row=dict(leaf_angle_rad=float(angle),qpos=d.qpos.tolist(),root_delta=fit.x[:6].tolist(),joint_targets=dict(zip(names,fit.x[6:].tolist())),
            left_position_error_m=float(np.linalg.norm(res[:3])/100),left_rotation_error_rad=float(np.linalg.norm(res[3:6])/10),
            right_position_error_m=float(np.linalg.norm(d.site_xpos[rh]-right_p)),right_rotation_error_rad=float(np.linalg.norm(Rotation.from_matrix(right_r@d.site_xmat[rh].reshape(3,3).T).as_rotvec())),right_joint_posture_error_rad=float(max(abs(fit.x[6+right_indices]-initial[right_indices]))),
            maximum_foot_position_error_m=max(float(np.linalg.norm(res[12+6*j:15+6*j])/100) for j in range(2)),
            torso_tilt_deg=float(np.degrees(np.arccos(np.clip(up[2],-1,1)))),com_displacement_xy_m=(d.subtree_com[robot_body,:2]-com[:2]).tolist(),
            forbidden_collisions=collisions,nfev=int(fit.nfev))
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('qpos','joint_targets')}),flush=True)
    summary=dict(scope='Fresh unstepped attained-state panel workspace screen. No physical or loaded-palm qualification; target-rate resampling and dense collision audit are still required.',
        configuration={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'right_target_mode':'fixed-attained-joints','support_prelude':'Declared normal recontact and flattening versus leaf progress; no contact qualification'},source_code_sha256=hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest(),source_time_s=time,source_chunk_sha256=chunk['sha256'],
        maximum_actual_fk_error=error,initial_qpos=base.tolist(),initial_qvel=velocity.tolist(),names=names,rows=rows,
        maximum_palm_position_error_m=max(r['left_position_error_m'] for r in rows),
        maximum_foot_position_error_m=max(r['maximum_foot_position_error_m'] for r in rows),maximum_torso_tilt_deg=max(r['torso_tilt_deg'] for r in rows),
        forbidden_collision_samples=sum(bool(r['forbidden_collisions']) for r in rows))
    (a.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n');sim.close()


if __name__=='__main__':main()
