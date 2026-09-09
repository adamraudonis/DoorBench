#!/usr/bin/env python3
"""Independently screen interpolated standing-transfer geometry, without physics."""
import argparse,hashlib,json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from doorbench.dexterous.landed_left_planner import LandedLeftScene
from doorbench.dexterous.landed_left_audit import static_pose_check


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('robot','door','path','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous audits')
    scene=LandedLeftScene(a.robot,a.door);m,d=scene.m,scene.d
    path=np.asarray(json.loads(a.path.read_text())['path_qpos'],float)
    if path.shape!=(101,m.nq) or not np.isfinite(path).all():raise ValueError('Complete finite 101-node scene route required')
    d.qpos[:]=path[0];mujoco.mj_kinematics(m,d)
    bodies=[m.body('robot/'+n).id for n in ('left_ankle_link','right_ankle_link')];rp=m.site('robot/rh_palm_touch').id
    bp=d.xpos[bodies].copy();br=d.xmat[bodies].reshape(2,3,3).copy();hp=d.site_xpos[rp].copy();hr=d.site_xmat[rp].reshape(3,3).copy()
    bad=[];peakpos=peakrot=peaktilt=0.
    for i,u in enumerate(np.linspace(0,1,1001)):
        coordinate=u*100;j=min(int(coordinate),99);f=coordinate-j;q=path[j]*(1-f)+path[j+1]*f
        quat=Slerp([0,1],Rotation.from_quat(path[j:j+2,scene.root+3:scene.root+7][:,[1,2,3,0]]))(f).as_quat();q[scene.root+3:scene.root+7]=quat[[3,0,1,2]]
        d.qpos[:]=q;check=static_pose_check(m,d,coordinate=float(u))
        pe=max(np.linalg.norm(d.site_xpos[rp]-hp),np.max(np.linalg.norm(d.xpos[bodies]-bp,axis=1)))
        re=max(np.linalg.norm(Rotation.from_matrix(d.site_xmat[rp].reshape(3,3)@hr.T).as_rotvec()),max(np.linalg.norm(Rotation.from_matrix(d.xmat[b].reshape(3,3)@r.T).as_rotvec()) for b,r in zip(bodies,br)))
        peakpos=max(peakpos,float(pe));peakrot=max(peakrot,float(re));rotation=Rotation.from_quat(q[scene.root+3:scene.root+7][[1,2,3,0]]).as_matrix();tilt=float(np.degrees(np.arccos(np.clip(rotation[2,2],-1,1))));peaktilt=max(peaktilt,tilt)
        if not check['passed'] or pe>.001 or re>.01 or tilt>4:bad.append(dict(index=i,position_error_m=float(pe),rotation_error_rad=float(re),root_tilt_deg=tilt,collision=check))
    out=dict(passed=not bad,samples=1001,maximum_fixed_hand_or_foot_position_error_m=peakpos,maximum_fixed_hand_or_foot_rotation_error_rad=peakrot,maximum_root_tilt_deg=peaktilt,bad_samples=bad,physics_steps=0,scope='Dense interpolation/FK/collision only; no physical contact/load or motor feasibility claim',input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.robot,a.door,a.path,Path(__file__))})
    a.output.write_text(json.dumps(out,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('bad_samples','input_sha256')}))


if __name__=='__main__':main()
