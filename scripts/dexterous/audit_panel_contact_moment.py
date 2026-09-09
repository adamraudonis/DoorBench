#!/usr/bin/env python3
"""Explain measured panel contact torque without rerunning contact dynamics."""
import argparse,hashlib,json
from pathlib import Path
import mujoco
import numpy as np
from doorbench.dexterous.contact_moment import contact_moment
from doorbench.dexterous.environment import DexterousDoorEnv


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def audit(trial,at):
    trial=Path(trial);manifest=json.loads((trial/'manifest.json').read_text())
    archive=json.loads((trial/'raw-transitions/manifest.json').read_text())
    if not archive['complete'] or not np.isfinite(at):raise ValueError('Require complete archived run and finite sample time')
    chunk=next((c for c in archive['chunks'] if c['interval_start_s']<=at<c['interval_end_s']),None)
    if chunk is None:raise ValueError('Requested time is outside archived intervals')
    path=trial/'raw-transitions'/chunk['file']
    if sha(path)!=chunk['sha256']:raise ValueError('Archived chunk changed')
    robot=Path(manifest['configuration']['robot']);door=Path(manifest['configuration']['door'])
    if sha(robot)!=manifest['inputs']['robot']['sha256'] or sha(door/'door.xml')!=manifest['inputs']['door']['door.xml']:raise ValueError('Recorded model inputs changed')
    with np.load(path) as z:
        i=int(np.searchsorted(z['interval_start_s'],at,side='right')-1)
        sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()))
        try:
            m,d=sim.m,sim.d;d.qpos[:]=z['qpos_before'][i]
            # Kinematics only, at the original contact epoch. Never mj_forward/step.
            mujoco.mj_kinematics(m,d)
            b,e=z['body_offsets'][i:i+2];ids=z['body_ids'][b:e]
            error=max(float(np.max(np.abs(d.xpos[ids]-z['body_positions_world_m'][b:e]),initial=0)),float(np.max(np.abs(d.xmat[ids].reshape(-1,3,3)-z['body_rotations_world'][b:e]),initial=0)))
            if error>1e-8:raise ValueError('Contact epoch FK does not match recorded transforms')
            joint=m.joint('leaf_hinge').id;leaf=m.body('leaf').id;anchor=d.xanchor[joint].copy();axis=d.xaxis[joint].copy()
            entries=[];b,e=z['contact_offsets'][i:i+2]
            for k in range(b,e):
                pair=z['contact_body'][k].tolist()
                if leaf not in pair:continue
                side=pair.index(leaf)
                item=contact_moment(z['contact_position_world_m'][k],z['contact_frame_world'][k],z['contact_wrench_contact_frame'][k],anchor,axis,body_index=side)
                item['other_body']=m.body(pair[1-side]).name;entries.append(item)
            fields=('moment_about_hinge_Nm','normal_force_moment_Nm','tangential_force_moment_Nm','contact_couple_moment_Nm')
            return dict(scope='Archived direct leaf contact moment, not full generalized torque balance or task qualification. Frictionloss is the original model limit, not a measured solver multiplier.',interval_s=[float(z['interval_start_s'][i]),float(z['interval_end_s'][i])],maximum_actual_fk_error=error,leaf_angle_rad=float(d.qpos[m.jnt_qposadr[joint]]),leaf_velocity_rad_s=float(z['qvel_before'][i,m.jnt_dofadr[joint]]),hinge_frictionloss_Nm=float(m.dof_frictionloss[m.jnt_dofadr[joint]]),totals={key:sum(v[key] for v in entries) for key in fields},contacts=entries,input_sha256={str(path):sha(path),str(robot):sha(robot),str(door/'door.xml'):sha(door/'door.xml')},auditor_sha256=sha(__file__))
        finally:sim.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);p.add_argument('--at',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh diagnostic output required')
    result=audit(a.trial,a.at);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('interval_s','leaf_angle_rad','leaf_velocity_rad_s','hinge_frictionloss_Nm','totals')}))


if __name__=='__main__':main()
