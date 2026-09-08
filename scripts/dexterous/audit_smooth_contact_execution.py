#!/usr/bin/env python3
"""Reconstruct mode history from numeric actor touch and audit actual recovery.

Mode/algebra pass is separate from the unchanged physical task gates.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from doorbench.dexterous.tactile_contact_mode import TactileContactMode


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):
    with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('trial','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh audit required')
    rows=lines(a.trial/'physics.jsonl.gz');infos=lines(a.trial/'controller.jsonl.gz');layout=json.loads((a.trial/'sensor-layout.json').read_text())
    mode=TactileContactMode(json.loads((a.trial/'contact-mode-protocol.json').read_text()));slices={};offset=0
    for r in layout['sensors']:
        slices[r['name']]=slice(offset,offset+r['dimension']);offset+=r['dimension']
    with np.load(a.trial/'actor-inputs.npz') as z:packets={k:z[k].copy() for k in ('tactile','sensor_valid','sensor_time_s')}
    with np.load(a.trial/'trajectory.npz') as z:forces=z['force'].copy();actions=list(z['action_names'])
    if not len(rows)==len(infos)==len(forces)==len(packets['tactile']):raise ValueError('Actual decisions/intervals disagree')
    maxraw=maxweight=maxfrozen=maxstep=0.;progress_errors=0;prior=None;firstzero=None;events=[]
    for i in range(9500,len(rows)):
        t=i*.002;r=infos[i]
        if not packets['sensor_valid'][i,4] or not 0<=t-packets['sensor_time_s'][i,4]<=.006+1e-9:raise ValueError('Recorded mode consumed stale touch')
        raw=np.array([max(0.,float(packets['tactile'][i,slices['rh_'+d+'distal_touch']].reshape(3,2,4)[0,:,1:3].sum())) for d in ('ff','mf','rf','lf','th')])
        weight,info=mode.update(raw,now_s=t);maxraw=max(maxraw,float(abs(raw-r['distal_projected_force_N']).max()))
        maxweight=max(maxweight,float(abs(weight-r['contact_mode_weight']).max()),float(abs(weight-r['local_contact_weight']).max()))
        if prior is not None:
            maxstep=max(maxstep,float(abs(weight-prior).max()))
            lost=raw==0
            for field in ('digit_force_integral_N','virtual_digit_force_state_N','finger_motor_closure_offsets_rad'):
                if np.any(lost):maxfrozen=max(maxfrozen,float(abs(np.array(r[field])-infos[i-1][field])[lost].max()))
            if lost[0]:maxfrozen=max(maxfrozen,abs(r['index_proximal_offset_rad']-infos[i-1]['index_proximal_offset_rad']))
            if np.any(raw<.2):progress_errors+=int(r['local_touch_progression_ready'] or r['virtual_press_clock_s']!=infos[i-1]['virtual_press_clock_s'])
        if firstzero is None and raw[0]==0:firstzero=i
        if firstzero is not None and firstzero-1<=i<=firstzero+5:
            j=actions.index('rh_A_FFJ3')
            events.append(dict(decision_s=t,available_FF_local_N=float(raw[0]),mode_weight=float(weight[0]),
                actual_interval_start_s=rows[i]['contact_interval_start_s'],actual_opposed_grasp=rows[i]['pad_grasp']['valid_pad_grasp'],
                actual_FF_qualified_normal_N=float(sum(c['normal_force_N'] for c in rows[i]['pad_grasp']['contacts'] if c['digit']=='ff' and c['pad_qualified'])),
                actual_FFJ3_force_Nm=float(forces[i,j]),command_step_Nm=float(forces[i,j]-forces[i-1,j])))
        prior=weight.copy()
    bad=next((r for r in rows if r['invalid_loaded_distal_patches']),None)
    firstloss=next((i for i,r in enumerate(rows) if i>=9500 and not r['pad_grasp']['valid_pad_grasp']),None)
    recovered=next((i for i in range(firstloss+1,len(rows)) if rows[i]['pad_grasp']['valid_pad_grasp']),None) if firstloss is not None else None
    checks=dict(local_numeric_projection_exact=maxraw==0.,recorded_mode_reconstructed=maxweight<1e-10,weight_slew=maxstep<=.01+1e-10,
        all_zero_touch_integrators_frozen=maxfrozen==0.,press_frozen_on_missing_contact=progress_errors==0)
    result=dict(scope=__doc__,passed=all(checks.values()),checks=checks,actual_task_passed=json.loads((a.trial/'report.json').read_text())['passed'],
        maximum_local_projection_error_N=maxraw,maximum_mode_weight_error=maxweight,maximum_weight_step=maxstep,
        maximum_frozen_integrator_change=maxfrozen,progression_errors=progress_errors,
        first_loss_interval_s=None if firstloss is None else firstloss*.002,first_recovered_interval_s=None if recovered is None else recovered*.002,
        first_zero_feedback_window=events,first_invalid_patch=None if bad is None else dict(interval_start_s=bad['contact_interval_start_s'],
            contacts=[c for c in bad['pad_grasp']['contacts'] if c['normal_force_N']>1e-6 and not c['pad_qualified']]),
        source_sha256=sha(__file__),inputs_sha256={n:sha(a.trial/n) for n in ['provenance.json','controller.jsonl.gz','physics.jsonl.gz','actor-inputs.npz','trajectory.npz','sensor-layout.json','contact-mode-protocol.json']})
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
