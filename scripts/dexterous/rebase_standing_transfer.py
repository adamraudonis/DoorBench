#!/usr/bin/env python3
"""Rebuild a geometric transfer from a fully audited attained grasp state.

The old route supplies posture preferences only. Actual feet, working hand,
fingers and mechanism state are taken from the new episode. No plant is stepped.
"""
import argparse,hashlib,json,subprocess,sys,shutil
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from doorbench.dexterous.landed_left_planner import LandedLeftScene,JOINT_NAMES
from doorbench.dexterous.landed_left_audit import static_pose_check
from doorbench.dexterous.standing_transfer import validate_route_geometry


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','template','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pose-weight',type=float,default=200.)
    p.add_argument('--initial-margin-transition-nodes',type=int,default=20)
    a=p.parse_args();
    if not 100<=a.pose_weight<=1000:raise ValueError('Pose weight must be100..1000')
    if not 10<=a.initial_margin_transition_nodes<=40:raise ValueError('Initial margin transition must use10..40 nodes')
    source=a.source_run;manifest=json.loads((source/'manifest.json').read_text());cfg=manifest['configuration']
    inputs=[source/n for n in ('report.json','independent-pad-audit.json','independent-whole-handle-audit.json')]
    if not all(json.loads(f.read_text())['passed'] for f in inputs):raise ValueError('All grasp and whole-handle source checks required')
    template=json.loads(a.template.read_text());validate_route_geometry(template)
    robot=Path(cfg['robot']);door=Path(cfg['door'])/'door.xml'
    if sha(robot)!=manifest['inputs']['robot']['sha256'] or sha(door)!=manifest['inputs']['door']['door.xml'] or sha(robot)!=template['robot_xml_sha256'] or sha(door)!=template['door_xml_sha256']:raise ValueError('Exact common robot and door inputs required')
    scene=LandedLeftScene(robot,door);m,d=scene.m,scene.d
    with np.load(source/'trajectory.npz') as z:frozen=z['terminal_qpos'].copy();at=float(z['terminal_time_s'])
    scene.freeze(scene.state_from_qpos(frozen,pose_time_s=at))
    prior=np.array(json.loads(Path(template['scene_path_source']).read_text())['path_qpos'])
    if prior.shape!=(101,m.nq):raise ValueError('Complete101-node posture template required')
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,a.output/'planner-source.py')
    names=['torso']+['right_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['rh_WRJ2','rh_WRJ1']+['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1']
    legs=[n for n in scene.robot_names if any(k in n for k in ('hip','knee','ankle'))]
    ids=[m.joint('robot/'+n).id for n in names];legids=[m.joint('robot/'+n).id for n in legs]
    qa=np.r_[m.jnt_qposadr[ids],np.arange(scene.root,scene.root+3),m.jnt_qposadr[legids]]
    lower=np.r_[m.jnt_range[ids,0]+.025,frozen[scene.root:scene.root+3]+[-.08,-.08,-.047],m.jnt_range[legids,0]+.025,[-.04,-.04,-.15]]
    upper=np.r_[m.jnt_range[ids,1]-.025,frozen[scene.root:scene.root+3]+[.08,.08,.012],m.jnt_range[legids,1]-.025,[.04,.04,.15]]
    d.qpos[:]=frozen;mujoco.mj_kinematics(m,d);hand=m.site('robot/rh_palm_touch').id
    hp=d.site_xpos[hand].copy();hr=d.site_xmat[hand].reshape(3,3).copy();feet=[m.body('robot/'+n+'_ankle_link').id for n in ('left','right')];fp=d.xpos[feet].copy();fr=d.xmat[feet].reshape(2,3,3).copy()
    root=Rotation.from_quat(frozen[scene.root+3:scene.root+7][[1,2,3,0]]);oldroot=Rotation.from_quat(prior[0,scene.root+3:scene.root+7][[1,2,3,0]])
    initial=np.r_[frozen[qa],np.zeros(3)];previous=initial.copy();active=np.r_[np.arange(1,8),np.arange(15,len(initial))];path=[];rows=[]
    for i,old in enumerate(prior):
        phase=min(1.,i/a.initial_margin_transition_nodes);blend=phase**3*(10+phase*(-15+6*phase))
        node_lower=(1-blend)*np.minimum(lower,initial-1e-12)+blend*lower
        node_upper=(1-blend)*np.maximum(upper,initial+1e-12)+blend*upper
        nominal=np.r_[frozen[qa]+old[qa]-prior[0,qa],(Rotation.from_quat(old[scene.root+3:scene.root+7][[1,2,3,0]])*oldroot.inv()).as_rotvec()]
        def residual(x):
            q=frozen.copy();q[qa]=x[:-3];quat=(Rotation.from_rotvec(x[-3:])*root).as_quat();q[scene.root+3:scene.root+7]=quat[[3,0,1,2]];d.qpos[:]=q;mujoco.mj_kinematics(m,d)
            return np.r_[a.pose_weight*(d.site_xpos[hand]-hp),(a.pose_weight/10)*Rotation.from_matrix(d.site_xmat[hand].reshape(3,3)@hr.T).as_rotvec(),100*(x[15:18]-nominal[15:18]),10*(x[-3:]-nominal[-3:]),*[np.r_[a.pose_weight*(d.xpos[b]-pos),(a.pose_weight/10)*Rotation.from_matrix(d.xmat[b].reshape(3,3)@rot.T).as_rotvec()] for b,pos,rot in zip(feet,fp,fr)],.001*(x-nominal),.01*(x-previous)]
        def objective(value):
            x=nominal.copy();x[active]=value;return residual(x)
        if i==0:x=initial.copy();converged=True
        else:
            fit=least_squares(objective,np.clip(previous[active],node_lower[active],node_upper[active]),bounds=(node_lower[active],node_upper[active]),max_nfev=300,ftol=1e-9,xtol=1e-9,gtol=1e-9);x=nominal.copy();x[active]=fit.x;converged=bool(fit.success)
        errors=residual(x);previous=x.copy();q=d.qpos.copy();row=static_pose_check(m,d,coordinate=i/100)
        pe=float(max(np.linalg.norm(d.site_xpos[hand]-hp),max(np.linalg.norm(d.xpos[b]-pos) for b,pos in zip(feet,fp))))
        re=float(max(np.linalg.norm(Rotation.from_matrix(d.site_xmat[hand].reshape(3,3)@hr.T).as_rotvec()),max(np.linalg.norm(Rotation.from_matrix(d.xmat[b].reshape(3,3)@rot.T).as_rotvec()) for b,rot in zip(feet,fr))))
        row.update(index=i,max_position_error_m=pe,max_rotation_error_rad=re,solver_converged=converged);row['passed']=bool(row['passed'] and pe<=.001 and re<=.01);rows.append(row);path.append(q.tolist())
        if not row['passed']:break
    result=dict(pose_weight=a.pose_weight,initial_margin_transition_nodes=a.initial_margin_transition_nodes,passed=len(path)==101 and all(r['passed'] for r in rows),physics_steps=0,scope=__doc__,samples=rows,path_qpos=path,input_sha256={str(f):sha(f) for f in [*inputs,a.template,source/'trajectory.npz',robot,door,Path(__file__)]})
    report=a.output/'report.json';report.write_text(json.dumps(result,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
    print(json.dumps(dict(sampled_passed=result['passed'],nodes=len(path))),flush=True)
    if not result['passed']:return 1
    proof=a.output/'dense-audit.json'
    subprocess.run([sys.executable,'scripts/dexterous/audit_standing_transfer_path.py','--robot',str(robot),'--door',str(door),'--path',str(report),'--output',str(proof)],check=True)
    if not json.loads(proof.read_text())['passed']:return 1
    path=np.array(path);jointqa=[m.joint('robot/'+n).qposadr[0] for n in template['joint_names']]
    template.update(scene_path_source=str(report),scene_path_sha256=sha(report),dense_audit_path=str(proof.resolve()),dense_audit_sha256=sha(proof),root_path=path[:,scene.root:scene.root+7].tolist(),joint_path=path[:,jointqa].tolist(),left_targets=[],attained_time_s=at,attained_trial=str(source),attained_manifest_sha256=sha(source/'manifest.json'))
    for q in path:
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);rot=d.xmat[scene.leaf].reshape(3,3)
        template['left_targets'].append(dict(phase='left_reach',leaf_rad=float(q[m.joint('leaf_hinge').qposadr[0]]),position=(rot.T@(d.site_xpos[scene.palm]-d.xpos[scene.leaf])).tolist(),normal=(rot.T@d.site_xmat[scene.palm].reshape(3,3)[:,2]).tolist(),nominal=[float(q[m.joint('robot/'+n).qposadr[0]]) for n in JOINT_NAMES]))
    validate_route_geometry(template);(a.output/'transfer.json').write_text(json.dumps(template,indent=2)+'\n');return 0


if __name__=='__main__':raise SystemExit(main())
