#!/usr/bin/env python3
"""Fit a prior grasp route at an independently qualified standing height.

Privileged geometric planning only; the robot still needs a physical acquisition
trial and a continuous approach. Torso yaw participates within original limits.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
from scripts.dexterous.evaluate_native_sensor_policy import prepare_trial
from scripts.dexterous.probe_sensor_acquisition_balance import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('stance-run','reference','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous candidates')
    proof=json.loads((a.stance_run/'independent-locomotion-audit.json').read_text())
    if not proof['passed'] or not proof['checks'].get('quiet_final_second') or not proof['checks'].get('final_two_foot_support'):raise ValueError('Independently qualified quiet standing capture required')
    for n in ('manifest.json','report.json','raw-transitions/manifest.json'):
        if proof['inputs'][n]!=sha(a.stance_run/n):raise ValueError('Standing evidence changed')
    cfg=json.loads((a.stance_run/'manifest.json').read_text())['configuration']
    sim,motors,layout,_=prepare_trial(Path(cfg['teacher_run']),Path(cfg['camera_profile']))
    last=None
    for row in NativeTransitionArchive.read(a.stance_run/'raw-transitions'):last=row
    ref=json.loads(a.reference.read_text());names=ref['acquisition']['joint_names'];path=np.asarray(ref['acquisition']['path_qpos'],float)
    m=mujoco.MjModel.from_xml_path(cfg['robot']);d=mujoco.MjData(m);original=mujoco.MjData(m)
    qa=np.array([m.joint(n).qposadr[0] for n in names]);arm=ref['workspace_fit']['joint_names'];aa=np.array([m.joint(n).qposadr[0] for n in arm]);bounds=np.array([m.joint(n).range for n in arm]);palm=m.site('rh_palm_touch').id
    original.qpos[:7]=ref['initial_root'];d.qpos[:7]=ref['initial_root'];d.qpos[2]=last['qpos_after'][sim.root_qadr+2]
    sq=np.array([sim.m.joint('robot/'+n).qposadr[0] for n in names]);standing=np.asarray(last['qpos_after'])[sq]
    for i,n in enumerate(names):standing[i]=np.clip(standing[i],*m.joint(n).range)
    for side in ('lh','rh'):
        for digit in ('FF','MF','RF','LF'):
            i=names.index(f'{side}_{digit}J1');j=names.index(f'{side}_{digit}J2');standing[i]=min(standing[i],standing[j])
    moving=[n for n in names if n=='torso' or n.startswith(('rh_','right_shoulder','right_elbow','right_wrist'))]
    results=[];solutions=[];prior=path[0,[names.index(n) for n in arm]].copy()
    for index,row in enumerate(path):
        original.qpos[qa]=row;mujoco.mj_kinematics(m,original);pos=original.site_xpos[palm].copy();rot=original.site_xmat[palm].reshape(3,3).copy()
        d.qpos[qa]=standing
        for n in moving:d.qpos[m.joint(n).qposadr[0]]=row[names.index(n)]
        def residual(x):
            d.qpos[aa]=x;mujoco.mj_kinematics(m,d)
            return np.r_[10*(d.site_xpos[palm]-pos),Rotation.from_matrix(d.site_xmat[palm].reshape(3,3)@rot.T).as_rotvec(),.0001*(x-prior)]
        fit=least_squares(residual,np.clip(prior,bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=(bounds[:,0],bounds[:,1]),max_nfev=300,gtol=1e-9,ftol=1e-9,xtol=1e-9)
        error=residual(fit.x);position=float(np.linalg.norm(error[:3])/10);angle=float(np.linalg.norm(error[3:6]));prior=fit.x.copy()
        results.append(dict(index=index,position_error_m=position,rotation_error_rad=angle,solver_success=bool(fit.success)));solutions.append(d.qpos[qa].copy().tolist())
        if position>.001 or angle>.01:break
    passed=len(results)==len(path) and all(r['position_error_m']<.001 and r['rotation_error_rad']<.01 for r in results)
    a.output.mkdir(parents=True)
    report=dict(passed=passed,scope=__doc__,physics_steps=0,samples=results,initial_root=d.qpos[:7].tolist(),input_sha256={str(x):sha(x) for x in (a.reference,a.stance_run/'independent-locomotion-audit.json',Path(cfg['robot']),Path(cfg['door'])/'door.xml',Path(__file__))})
    (a.output/'workspace-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    if passed:
        ref['initial_root']=d.qpos[:7].tolist();ref['acquisition']['path_qpos']=solutions;ref['scope']='Standing-height geometric variant; no physical task qualification'
        (a.output/'reference.json').write_text(json.dumps(ref,indent=2)+'\n');(a.output/'motor-contract.json').write_text(json.dumps(motors,indent=2)+'\n')
    sim.close();print(json.dumps(dict(passed=passed,samples=len(results),last=results[-1])));return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
