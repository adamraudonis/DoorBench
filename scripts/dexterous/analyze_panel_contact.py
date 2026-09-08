#!/usr/bin/env python3
"""Separate actual rigid-contact impulse timing from gross palm support loss."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv

spec=importlib.util.spec_from_file_location('_contact_archive',Path(__file__).resolve().parents[2]/'doorbench/dexterous/native_transition_archive.py')
archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)


metrics_spec=importlib.util.spec_from_file_location('_contact_metrics',Path(__file__).resolve().parents[2]/'doorbench/dexterous/contact_impulse_diagnostic.py')
metrics=importlib.util.module_from_spec(metrics_spec);metrics_spec.loader.exec_module(metrics)
longest_gap=metrics.longest_gap
impulse_summary=metrics.impulse_summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','robot','door'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();path=args.run
    rows=json.load(gzip.open(path/'physics-steps.json.gz','rt'))
    final=rows[-1]['sim_time_s'];dt=rows[1]['contact_interval_end_s']-rows[1]['contact_interval_start_s']
    # Integrate exactly the final0.5s, not the extra boundary sample retained by
    # the frozen251-sample pointwise gate. The original report is not modified.
    tail=[r for r in rows if r.get('contact_interval_end_s',-1)>final-.5+1e-8]
    if len(tail)!=round(.5/dt):raise ValueError('Missing exact final-half-second interval evidence')
    for row in tail:
        if row['contact_force_source']!='actual_mj_step_dynamics' or abs(row['contact_interval_end_s']-row['contact_interval_start_s']-dt)>1e-8:
            raise ValueError('Require complete actual force intervals')
    palm=np.array([r['left_surface_audit']['palm_normal_load_N'] for r in tail])
    total=np.array([r['left_surface_audit']['total_normal_load_N'] for r in tail])
    report=dict(schema='doorbench.panel-contact-diagnostic.v1',scope='Diagnostic only; frozen pass/fail remains unchanged',
        physics_dt_s=dt,window_start_s=final-.5,window_end_s=final,
        palm=impulse_summary(palm,dt),total_left_hand=impulse_summary(total,dt),
        low_palm_samples_with_total_hand_at_least_2N=int(np.sum((palm<2)&(total>=2))),
        competing_body_mean_load_N={name:float(np.mean([r['left_surface_audit']['body_normal_loads_N'].get(name,0.) for r in tail])) for name in sorted({name for r in tail for name in r['left_surface_audit']['body_normal_loads_N'] if name!='robot/lh_palm'})})
    sim=DexterousDoorEnv(args.door,args.robot,json.loads(args.robot.with_suffix('.audit.json').read_text()))
    m,d=sim.m,sim.d;pbody=m.body('robot/lh_palm').id;lbody=m.body('leaf').id
    pg=[g for g in range(m.ngeom) if m.geom_bodyid[g]==pbody and m.geom_contype[g]]
    lg=[g for g in range(m.ngeom) if m.geom_bodyid[g]==lbody and m.geom_contype[g]]
    rawpath=path/'raw-transitions';manifest=json.loads((rawpath/'manifest.json').read_text())
    if not manifest['complete']:raise ValueError('Incomplete raw transition archive')
    geometry=[];jp=np.zeros((3,m.nv));jl=jp.copy()
    for chunk in manifest['chunks']:
        if chunk['interval_end_s']<=final-.5+1e-8:continue
        file=rawpath/chunk['file']
        if hashlib.file_digest(file.open('rb'),'sha256').hexdigest()!=chunk['sha256']:raise ValueError('Raw chunk hash mismatch')
        with np.load(file,allow_pickle=False) as arrays:raw=list(archive.unpacked({k:arrays[k] for k in arrays.files}))
        for row in raw:
            if row['interval_end_s']<=final-.5+1e-8:continue
            d.qpos[:]=row['qpos_before'];d.qvel[:]=row['qvel_before'];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            ids=row['body_ids']
            if not np.allclose(d.xpos[ids],row['body_positions_world_m'],atol=1e-9,rtol=0) or not np.allclose(d.xmat[ids].reshape((-1,3,3)),row['body_rotations_world'],atol=1e-9,rtol=0):raise ValueError('Saved pre-state contact geometry does not match analytic FK')
            best=(1.,None)
            for p in pg:
                for l in lg:
                    points=np.zeros(6);gap=float(mujoco.mj_geomDistance(m,d,p,l,1.,points))
                    if gap<best[0]:best=(gap,points.copy())
            point=best[1];normal=d.xmat[lbody].reshape(3,3)[:,1]
            mujoco.mj_jac(m,d,jp,None,point[:3],pbody);mujoco.mj_jac(m,d,jl,None,point[3:],lbody)
            velocity=float(normal@(jp-jl)@d.qvel)
            geometry.append((row['interval_end_s'],best[0],velocity))
    sim.close();geometry=np.asarray(geometry)
    if len(geometry)!=len(tail) or not np.allclose(geometry[:,0],[r['contact_interval_end_s'] for r in tail],atol=1e-9,rtol=0):raise ValueError('Geometry and force intervals differ')
    low=geometry[palm<2,1]
    report['matching_pre_state_geometry']=dict(checked_frames=len(geometry),max_positive_palm_gap_m=max(0.,float(geometry[:,1].max())),
        maximum_gap_during_below_2N_m=max(0.,float(low.max(initial=0.))),minimum_gap_during_below_2N_m=float(low.min()) if len(low) else None,
        max_normal_relative_speed_m_s=float(np.abs(geometry[:,2]).max()),analytic_model_steps=0)
    np.savez_compressed(path/'contact-diagnostic.npz',interval_end_s=geometry[:,0],palm_load_N=palm,total_hand_load_N=total,palm_gap_m=geometry[:,1],normal_relative_speed_m_s=geometry[:,2])
    (path/'contact-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
