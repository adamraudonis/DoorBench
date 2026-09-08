#!/usr/bin/env python3
"""Replay exact held controls at finer steps; never reclassify an opening run."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv

ROOT=Path(__file__).resolve().parents[2]
def local_module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
recording=local_module('doorbench.dexterous.native_transition_audit','doorbench/dexterous/native_transition_audit.py')
storage=local_module('_convergence_storage','doorbench/dexterous/native_transition_archive.py')
analysis=local_module('_convergence_analysis','scripts/dexterous/analyze_panel_contact.py')


def source_chunks(path):
    receipt=json.loads((path/'manifest.json').read_text())
    if not receipt['complete']:raise ValueError('Source force stream is incomplete')
    for chunk in receipt['chunks']:
        file=path/chunk['file']
        if hashlib.file_digest(file.open('rb'),'sha256').hexdigest()!=chunk['sha256']:raise ValueError('Source chunk changed')
        with np.load(file,allow_pickle=False) as arrays:
            yield {k:arrays[k] for k in storage.STATE_FIELDS}


def physical_checks(rows):
    return dict(finite=all(r['finite'] for r in rows),upright=all(r['root_height_m']>.7 and r['torso_tilt_deg']<12 for r in rows),
        joint_limits=all(r['max_joint_limit_violation_rad']<=.02 for r in rows),
        passive_loopbacks=all(r['max_shadow_loopback_violation_rad']<=.02 for r in rows),
        scene_penetration=all(r['max_nonfoot_penetration_m']<=.003 for r in rows),
        original_motor_limits=all(r['native_motor_limits'] for r in rows),
        no_helper_forces=all(r['external_wrench_max']==0 and r['applied_generalized_force_max']==0 for r in rows),
        no_wrong_right_pad_patches=all(all(c['pad_qualified'] for c in r['pad_grasp']['contacts']) for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source_run','robot','door','motors','output'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--seconds',type=float,default=.5)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output directory')
    a.output.mkdir(parents=True)
    (a.output/'configuration.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2)+'\n')
    for file in (Path(__file__),Path(recording.__file__),Path(storage.__file__),Path(analysis.__file__),Path(analysis.metrics.__file__)):
        shutil.copy2(file,a.output/file.name)
    motors=json.loads(a.motors.read_text());sim=DexterousDoorEnv(a.door,a.robot,json.loads(a.robot.with_suffix('.audit.json').read_text()))
    sim.reset(randomize=False,images=False);m,d=sim.m,sim.d
    if m.na:raise ValueError('This reset replay requires the source fixture without actuator activation states')
    aids=np.array([m.actuator('robot/'+x['name']).id for x in motors['actuators']]);caps=np.array([x['force_range'] for x in motors['actuators']])
    m.actuator_gainprm[aids,0]=1.;m.actuator_biasprm[aids,:3]=0.;m.actuator_ctrlrange[aids]=caps
    srcpath=a.source_run/'raw-transitions';manifest=json.loads((srcpath/'manifest.json').read_text())
    source_dt=round(float(manifest['chunks'][0]['interval_end_s']/manifest['chunks'][0]['rows']),12)
    final=float(manifest['chunks'][-1]['interval_end_s']);target_start=final-a.seconds
    full_max_q=full_max_v=0.;prefix_steps=0;tail=[];snapshot=None;first=True
    state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
    # Recover the complete integration state, including warmstarts, by replaying
    # the source's actual controls from its exact reset. Every endpoint checks
    # against the original archive before accepting a held snapshot.
    for arrays in source_chunks(srcpath):
        for i,t in enumerate(arrays['interval_start_s']):
            if first:
                d.qpos[:]=arrays['qpos_before'][i];d.qvel[:]=arrays['qvel_before'][i];d.time=float(t)
                mujoco.mj_forward(m,d);d.qacc_warmstart[:]=0.;first=False
            if t>=target_start-1e-8:
                if snapshot is None:
                    snapshot=np.zeros(mujoco.mj_stateSize(m,state_spec));mujoco.mj_getState(m,d,snapshot,state_spec)
                    exact_start=float(t)
                tail.append({key:arrays[key][i].copy() for key in storage.STATE_FIELDS})
                continue
            d.ctrl[:]=arrays['controls'][i];sim.plant.step();mujoco.mj_kinematics(m,d)
            q_error=float(np.max(np.abs(d.qpos-arrays['qpos_after'][i])));v_error=float(np.max(np.abs(d.qvel-arrays['qvel_after'][i])))
            full_max_q=max(full_max_q,q_error);full_max_v=max(full_max_v,v_error);prefix_steps+=1
            if q_error>1e-9 or v_error>1e-8:raise ValueError(f'Source prefix replay diverged at {d.time}: q={q_error}, v={v_error}')
    if snapshot is None or len(tail)!=round(a.seconds/source_dt):raise ValueError('Missing exact held control interval')
    np.savez_compressed(a.output/'held-input.npz',integration_state=snapshot,integration_state_spec=int(state_spec),
        interval_start_s=[r['interval_start_s'] for r in tail],controls=[r['controls'] for r in tail],
        source_qpos_after=[r['qpos_after'] for r in tail],source_qvel_after=[r['qvel_after'] for r in tail])
    palm=m.body('robot/lh_palm').id;leaf=m.body('leaf').id
    pg=[g for g in range(m.ngeom) if m.geom_bodyid[g]==palm and m.geom_contype[g]];lg=[g for g in range(m.ngeom) if m.geom_bodyid[g]==leaf and m.geom_contype[g]]
    config=dict(plugin_count=int(m.nplugin),plugin_state_size=int(d.plugin_state.size),integrator=int(m.opt.integrator),solver=int(m.opt.solver),iterations=int(m.opt.iterations),tolerance=float(m.opt.tolerance),disableflags=int(m.opt.disableflags),
        palm_solref=m.geom_solref[pg].tolist(),leaf_solref=m.geom_solref[lg].tolist(),passive_tendon_solref=m.tendon_solref_lim.tolist(),
        source_dt_s=source_dt,original_motor_caps=caps.tolist())
    results=[]
    for dt in (.002,.001,.0005):
        out=a.output/f'dt-{dt:.4f}';out.mkdir();m.opt.timestep=dt
        mujoco.mj_setState(m,d,snapshot,state_spec);mujoco.mj_kinematics(m,d)
        recorder=recording.NativeTransitionRecorder(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge')
        archive=storage.NativeTransitionArchive(out/'raw-transitions')
        rows=[];gaps=[];errors=[];states=[]
        for step in range(round(a.seconds/dt)):
            action_index=min(int((step*dt+1e-10)/source_dt),len(tail)-1)
            d.ctrl[:]=tail[action_index]['controls']
            gap=min(float(mujoco.mj_geomDistance(m,d,g,h,1.,None)) for g in pg for h in lg)
            recorder.before_step();sim.plant.step();row,raw=recorder.after_step();archive.write(raw)
            rows.append(row);gaps.append(gap);states.append(d.qpos.copy())
            if dt==source_dt:
                errors.append([float(np.max(np.abs(d.qpos-tail[action_index]['qpos_after']))),float(np.max(np.abs(d.qvel-tail[action_index]['qvel_after'])))])
        archive.close();force=[r['left_surface_audit']['palm_normal_load_N'] for r in rows];total=[r['left_surface_audit']['total_normal_load_N'] for r in rows]
        result=dict(physics_dt_s=dt,palm=analysis.impulse_summary(force,dt),total_hand=analysis.impulse_summary(total,dt),
            maximum_positive_palm_gap_m=max(0.,max(gaps)),final_leaf_rad=rows[-1]['door_q'],checks=physical_checks(rows),
            source_tail_replay_max_error=np.max(errors,axis=0).tolist() if errors else None,
            initialized_diagnostic=True,source_opening_report_unchanged=True)
        with gzip.open(out/'physics-steps.json.gz','wt') as stream:json.dump(rows,stream)
        np.savez_compressed(out/'trace.npz',time_s=[r['sim_time_s'] for r in rows],palm_load_N=force,total_hand_load_N=total,palm_gap_m=gaps,qpos=states)
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result);print(json.dumps(result),flush=True)
    sim.close()
    report=dict(schema='doorbench.panel-contact-convergence.v1',scope='Initialized same-state same-control numerical diagnostic only; no new full-opening qualification',
        exact_held_start_s=exact_start,duration_s=a.seconds,prefix_replay_steps=prefix_steps,
        prefix_max_q_error=full_max_q,prefix_max_velocity_error=full_max_v,integration_state_includes_warmstart=True,
        unchanged_physical_settings=config,source_run=str(a.source_run),source_manifest_sha256=hashlib.file_digest((srcpath/'manifest.json').open('rb'),'sha256').hexdigest(),results=results)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
