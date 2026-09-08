#!/usr/bin/env python3
"""Reduce existing LH-support/RH-grip overlap without changing physical gates."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


DIGITS=('ff','mf','rf','lf','th')
DT=.002


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spans(mask,times):
    mask=np.asarray(mask,bool);edges=np.diff(np.r_[False,mask,False].astype(int))
    intervals=[]
    for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        intervals.append(dict(start_s=float(times[start]),end_s=float(times[end-1]),samples=int(end-start),
                              span_s=float(times[end-1]-times[start])))
    return sorted(intervals,key=lambda r:(-r['span_s'],r['start_s']))


def reduce_run(path):
    path=Path(path)
    with gzip.open(path/'acquisition-pad-steps.json.gz','rt') as f:pads=json.load(f)
    with gzip.open(path/'full-opening-steps.json.gz','rt') as f:steps=json.load(f)
    report=json.loads((path/'full-opening-report.json').read_text());protocol=json.loads((path/'full-opening-protocol.json').read_text())
    times=np.array([r['time_s'] for r in steps]);pt=np.array([r['sim_time_s'] for r in pads])
    if len(pads)!=len(steps)+1 or not np.allclose(times,np.arange(1,len(times)+1)*DT,atol=1e-8,rtol=0) or not np.allclose(pt,np.arange(len(pt))*DT,atol=1e-8,rtol=0):raise ValueError('Actual step/pad clocks are incomplete or misaligned')
    p=pads[1:];valid=np.array([r['valid_pad_grasp'] for r in p],bool)
    raw=np.array([[r['digit_forces_N'][d] for d in DIGITS] for r in p]);qualified=np.array([[r['qualified_pad_forces_N'][d] for d in DIGITS] for r in p])
    patch=np.array([any(not c['pad_qualified'] for c in r['contacts']) for r in p])
    finger=np.array([r.get('minimum_pairwise_finger_alignment',np.nan) for r in p]);thumb=np.array([r.get('maximum_thumb_finger_dot',np.nan) for r in p])
    misplaced=np.array([any(r.get('misplaced_force_N',{}).get(d,0)>.05*r['digit_forces_N'][d]+1e-12 for d in DIGITS) for r in p])
    support=np.array([r['surface']['total_normal_load_N'] for r in steps]);palm=np.array([r['surface']['palm_normal_load_N'] for r in steps])
    first_bad=int(np.flatnonzero(patch)[0]) if patch.any() else len(patch)
    first_time=float(times[first_bad]) if first_bad<len(times) else None
    start=report['handoffs']['left_approach'];before=(times>=start)&(np.arange(len(times))<first_bad)
    after30=times>=30.-1e-8
    masks={'right_grip':valid,'left_support_ge2N':support>=2.,'both_same_interval':valid&(support>=2.),
           'all_digits_raw_loaded':(raw>=.2).all(axis=1),'all_digits_qualified_loaded':(qualified>=.2).all(axis=1),
           'finger_alignment_failed':finger<=.5,'thumb_opposition_failed':thumb>=-.5,
           'misplaced_fraction_failed':misplaced,'invalid_patch':patch}
    result=dict(path=str(path),input_sha256={name:digest(path/name) for name in ['acquisition-pad-steps.json.gz','full-opening-steps.json.gz','full-opening-report.json','full-opening-protocol.json']},
        original_task_passed=report['passed'],seconds=float(times[-1]),transfer_load_target_N=protocol.get('transfer_load_target',4.),
        handoffs=report['handoffs'],original_failed_checks=[k for k,v in report['checks'].items() if not v],
        first_invalid_pad_time_s=first_time,first_invalid_pad_record=p[first_bad] if first_bad<len(p) else None,
        before_invalid_after_left_begin=dict(start_s=start,end_exclusive_s=first_time,samples=int(before.sum()),
            counts={name:int((mask&before).sum()) for name,mask in masks.items()},
            evaluated_opposition_samples=int((np.isfinite(finger)&before).sum()),
            per_digit_raw_unloaded={d:int(((raw[:,i]<.2)&before).sum()) for i,d in enumerate(DIGITS)},
            per_digit_qualified_unloaded={d:int(((qualified[:,i]<.2)&before).sum()) for i,d in enumerate(DIGITS)},
            first_raw_unload_s={d:float(times[np.flatnonzero((raw[:,i]<.2)&before)[0]]) if ((raw[:,i]<.2)&before).any() else None for i,d in enumerate(DIGITS)},
            first_finger_alignment_failure_s=float(times[np.flatnonzero((finger<=.5)&before)[0]]) if ((finger<=.5)&before).any() else None,
            first_thumb_opposition_failure_s=float(times[np.flatnonzero((thumb>=-.5)&before)[0]]) if ((thumb>=-.5)&before).any() else None,
            first_invalid_grip_s=float(times[np.flatnonzero(~valid&before)[0]]) if (~valid&before).any() else None),
        strongest_intervals_before_first_bad_patch={name:spans(mask&before,times)[:5] for name,mask in masks.items() if name in ['right_grip','left_support_ge2N','both_same_interval']},
        strongest_intervals_at_or_after_release_time={name:spans(mask&before&after30,times)[:5] for name,mask in masks.items() if name in ['right_grip','left_support_ge2N','both_same_interval']})
    # At each actual possible release time >=30, examine its exact251 sampled
    # endpoints. Do not substitute phase row i's preceding teacher evidence.
    eligible=[]
    for end in np.flatnonzero(before&after30):
        begin=end-250
        if begin<0 or not before[begin]:continue
        selection=slice(begin,end+1)
        eligible.append(dict(start_s=float(times[begin]),end_s=float(times[end]),span_s=float(times[end]-times[begin]),
            joint_good_samples=int((valid[selection]&(support[selection]>=2.)).sum()),
            right_good_samples=int(valid[selection].sum()),left_good_samples=int((support[selection]>=2.).sum()),
            raw_digit_unloaded_samples={d:int((raw[selection,i]<.2).sum()) for i,d in enumerate(DIGITS)},
            finger_alignment_failure_samples=int((finger[selection]<=.5).sum()),thumb_opposition_failure_samples=int((thumb[selection]>=-.5).sum()),
            min_total_support_N=float(support[selection].min()),min_palm_support_N=float(palm[selection].min()),
            invalid_patch_samples=int(patch[selection].sum())))
    eligible.sort(key=lambda r:(-r['joint_good_samples'],r['end_s']))
    result['best_half_second_release_window']=eligible[0] if eligible else None
    if eligible:
        best=eligible[0];selection=(times>=best['start_s']-1e-8)&(times<=best['end_s']+1e-8)
        best['bad_samples']=[dict(time_s=float(times[i]),left_support_N=float(support[i]),left_palm_N=float(palm[i]),right_grip=bool(valid[i]),raw_digit_loads_N=dict(zip(DIGITS,raw[i].tolist())),finger_alignment=float(finger[i]) if np.isfinite(finger[i]) else None,thumb_opposition=float(thumb[i]) if np.isfinite(thumb[i]) else None) for i in np.flatnonzero(selection&~(valid&(support>=2.)))]
    result['actual_qualified_release_windows']=sum(r['joint_good_samples']==251 for r in eligible)
    result['pre_bad_support_distribution_N']={key:float(value) for key,value in zip(['min','p05','median','p95','max'],np.quantile(support[before],[0,.05,.5,.95,1]))}
    result['first_invalid_patch_contacts']=[c for c in p[first_bad]['contacts'] if not c['pad_qualified']] if first_bad<len(p) else []
    # Verify that saved controller inputs are exactly the preceding actual
    # interval, preserving the distinction from each row's integrated state.
    discrepancies=[]
    for i,r in enumerate(steps):
        ev=r['teacher']['measured_evidence'];actual_index=i-1
        if abs(r['teacher']['evidence_time_s']-(times[i]-DT))>1e-7:discrepancies.append(i);continue
        if actual_index>=0 and (ev['grasp_qualified']!=p[actual_index]['valid_pad_grasp'] or abs(ev['left_panel_load_N']-support[actual_index])>1e-7):discrepancies.append(i)
    result['teacher_input_clock_discrepancies']=discrepancies
    result['qualification_semantics']='Longest spans use endpoint-time difference;0.5s requires251 consecutive2ms endpoints. Opposition is evaluated only when every digit has>=0.2N qualified load. Overlapping failure categories are not summed. Support is the stored actual projected normal-plus-friction surface load.'
    return result,dict(times=times,raw=raw,qualified=qualified,valid=valid,finger=finger,thumb=thumb,support=support,palm=palm,patch=patch)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='append',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    results=[];series=[]
    for path in a.run:
        result,arrays=reduce_run(path);results.append(result);series.append(arrays)
    result=dict(scope=__doc__,auditor_sha256=digest(__file__),runs=results,changed_gates=False,physics_mutations=False)
    (a.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(5,len(series),figsize=(15,12),sharex='col',squeeze=False)
    for col,(r,s) in enumerate(zip(results,series)):
        keep=s['times']>=22;t=s['times'][keep]
        axs[0,col].plot(t,s['support'][keep],lw=.5,label='LH total');axs[0,col].plot(t,s['palm'][keep],lw=.5,label='LH palm');axs[0,col].axhline(2,color='black',ls='--',lw=.8);axs[0,col].legend()
        for i,d in enumerate(DIGITS):axs[1,col].plot(t,s['raw'][keep,i],lw=.5,label=d)
        axs[1,col].axhline(.2,color='black',ls='--',lw=.8);axs[1,col].legend(ncols=5)
        axs[2,col].plot(t,s['finger'][keep],lw=.5,label='min finger pair');axs[2,col].axhline(.5,color='black',ls='--',lw=.8);axs[2,col].legend()
        axs[3,col].plot(t,s['thumb'][keep],lw=.5,label='max thumb/finger');axs[3,col].axhline(-.5,color='black',ls='--',lw=.8);axs[3,col].legend()
        axs[4,col].fill_between(t,0,s['valid'][keep].astype(int),step='post',alpha=.5,label='qualified RH');axs[4,col].plot(t,((s['support']>=2)&s['valid'])[keep].astype(int),lw=.7,label='LH+RH overlap');axs[4,col].plot(t,s['patch'][keep].astype(int),lw=.7,label='invalid RH patch');axs[4,col].legend()
        axs[0,col].set_title(f"{r['transfer_load_target_N']:g} N transfer target; run{'004' if col==0 else '005'}")
        for row,label in enumerate(['LH support (N)','RH digit loads (N)','Finger alignment','Thumb/finger dot','Boolean state']):
            axs[row,col].set_ylabel(label);axs[row,col].grid(alpha=.2)
            if r['first_invalid_pad_time_s']:axs[row,col].axvline(r['first_invalid_pad_time_s'],color='red',ls=':',lw=.8)
        axs[-1,col].set_xlabel('Actual endpoint time (s)')
    fig.tight_layout();fig.savefig(a.output/'overlap.png',dpi=160);plt.close(fig)
    print(json.dumps([dict(target=r['transfer_load_target_N'],first_bad=r['first_invalid_pad_time_s'],best_window=r['best_half_second_release_window'],strongest_overlap=r['strongest_intervals_before_first_bad_patch']['both_same_interval'][:1]) for r in results],indent=2))


if __name__=='__main__':main()
