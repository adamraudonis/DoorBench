#!/usr/bin/env python3
"""Audit archived Isaac acquisition contact accounting; never upgrade task success.

New acquisitions retain synchronized raw contact patches and body transforms.
Older archives permit formula/load accounting only, explicitly incomplete for
independent world-to-hand and opposition reconstruction.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from doorbench.dexterous.grasp_verification import shadow_surface_qualified
from doorbench.dexterous.isaac_pad_audit import shadow_physx_pad_grasp


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(trial):
    trial=Path(trial)
    name='operation-report.json' if (trial/'operation-report.json').exists() else 'acquisition-report.json'
    report=json.loads((trial/name).read_text())
    with gzip.open(trial/'acquisition-pad-steps.json.gz','rt') as f:rows=json.load(f)
    dt=report['physics_dt_s'];duration=report['duration_s'];expected=round(duration/dt)
    checks=dict(complete_clock=len(rows)==expected+1 and np.allclose([r['sim_time_s'] for r in rows],np.arange(expected+1)*dt,atol=1e-10,rtol=0))
    mismatch=0;force_error=0.;invalid=0;raw_count=0;raw_mismatch=0;raw_force_error=0.;best=current=0;bad_tail=[]
    raw_clocks=True;pair_error=0.
    checks['one_original_distal_profile']=all(r.get('grasp_profile')=='distal-pad-v1' for r in rows)
    digits=('ff','mf','rf','lf','th');minimum={d:float('inf') for d in digits}
    for row in rows[1:]:
        totals={d:0. for d in digits}
        for c in row['contacts']:
            body=c['body'].rsplit('/',1)[-1];digit=c['digit'];segment=body[len('rh_'+digit):]
            good=bool(shadow_surface_qualified(digit,segment,c['body_position_m'],c['hand_outward_normal_body']) and c['axial_clearance_m']>=.001 and c['on_lever_cylindrical_side'] and c['inward_radial_normal_alignment']>.8)
            mismatch+=good!=c['pad_qualified']
            invalid+=not good and c['normal_force_N']>1e-6
            if good:totals[digit]+=c['normal_force_N']
        force_error=max(force_error,max(abs(totals[d]-row['qualified_pad_forces_N'][d]) for d in digits))
        raw=row.get('raw_evidence')
        if raw is not None:
            raw_count+=1;lever=raw['lever']
            raw_clocks=raw_clocks and raw['schema']=='doorbench.shadow-raw-pad-evidence.v1' and all(abs(raw[k]-value)<1e-9 for k,value in (('interval_start_s',row['sim_time_s']-dt),('interval_end_s',row['sim_time_s']),('geometry_time_s',row['sim_time_s'])))
            for body,wanted in raw['handle_pair_forces_world_N'].items():
                force=sum((np.asarray(c['normal'])*c['normal_force_N'] for c in raw['contacts'] if c['body']==body),start=np.zeros(3))
                pair_error=max(pair_error,float(np.linalg.norm(force-wanted)))
            recomputed=shadow_physx_pad_grasp(raw['contacts'],raw['body_transforms_xyzw'],lever['center'],lever['axis'],half_length=lever['half_length'],radius=lever['radius'])
            raw_mismatch+=recomputed['valid_pad_grasp']!=row['valid_pad_grasp']
            raw_force_error=max(raw_force_error,max(abs(recomputed['qualified_pad_forces_N'][d]-row['qualified_pad_forces_N'][d]) for d in digits))
        current=current+1 if row['valid_pad_grasp'] else 0;best=max(best,current)
        if row['sim_time_s']>=duration-.5-1e-9:
            for d in digits:minimum[d]=min(minimum[d],totals[d])
            if not row['valid_pad_grasp']:bad_tail.append(dict(time_s=row['sim_time_s'],reason=row['reason'],pad_loads_N=totals))
    checks.update(anatomical_formula_matches=mismatch==0,qualified_load_accounting_matches=force_error<1e-6,
        final_hold_report_matches=report['checks']['sustained_pad_grasp']==(not bad_tail),raw_reductions_match=raw_mismatch==0 and raw_force_error<1e-6,
        raw_contact_clocks_match=raw_clocks,raw_patch_loads_match_pair_forces=pair_error<1e-3)
    return dict(schema='doorbench.isaac-acquisition-contact-audit.v1',accounting_passed=all(checks.values()),task_passed=report['passed'],
        independent_raw_contact_audit_complete=raw_count==expected and all(checks.values()),checks=checks,
        physical_intervals=expected,raw_intervals=raw_count,invalid_loaded_patches=invalid,
        longest_recorded_opposed_hold_s=best*dt,final_half_second_minimum_pad_force_N=minimum,
        final_half_second_failed_samples=bad_tail,maximum_load_accounting_error_N=force_error,
        maximum_raw_reduction_load_error_N=raw_force_error,raw_classification_mismatches=raw_mismatch,
        maximum_raw_pair_force_error_N=pair_error,
        limitation='Contact audit only; controller replay, collision geometry and whole-task qualification are separate. Without raw frames, world-to-hand and opposition cannot be independently reconstructed.',
        input_sha256={n:sha(trial/n) for n in (name,'acquisition-pad-steps.json.gz','configuration.json','provenance.json')},auditor_sha256=sha(__file__))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    r=audit(a.trial);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({k:r[k] for k in ('accounting_passed','task_passed','independent_raw_contact_audit_complete','physical_intervals','raw_intervals','invalid_loaded_patches','longest_recorded_opposed_hold_s')}))


if __name__=='__main__':main()
