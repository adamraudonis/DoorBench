"""Reduce recorded pad scoring into load and opposition failures; no simulation."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

DIGITS=('ff','mf','rf','lf','th')


def summarize(rows):
    failed=[row for row in rows if not row['valid_pad_grasp']]
    unloaded=[row for row in failed if any(row['qualified_pad_forces_N'].get(d,0)<.2 for d in DIGITS)]
    opposition=[row for row in failed if row not in unloaded]
    windows=[];start=None
    for row in rows:
        t=row['sim_time_s']
        if row['valid_pad_grasp']:
            if start is None:start=t
            end=t
        elif start is not None:
            windows.append(dict(start_s=start,end_s=end,span_s=end-start));start=None
    if start is not None:windows.append(dict(start_s=start,end_s=end,span_s=end-start))
    return dict(samples=len(rows),failed_grasp_samples=len(failed),low_load_samples=len(unloaded),
        opposed_geometry_failure_with_all_digits_loaded=len(opposition),
        per_digit_low_load={d:[dict(time_s=r['sim_time_s'],force_N=r['qualified_pad_forces_N'].get(d,0))
            for r in unloaded if r['qualified_pad_forces_N'].get(d,0)<.2] for d in DIGITS},
        finger_alignment_failure_samples=sum(r.get('minimum_pairwise_finger_alignment',1)<=.5 for r in failed),
        thumb_opposition_failure_samples=sum(r.get('maximum_thumb_finger_dot',-1)>=-.5 for r in failed),
        invalid_patch_samples=sum(any(not c['pad_qualified'] and c['normal_force_N']>1e-6 for c in r['contacts']) for r in rows),
        valid_windows=sorted(windows,key=lambda r:-r['span_s']))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args();run=args.run
    report=json.loads((run/'operation-report.json').read_text())
    with gzip.open(run/'acquisition-pad-steps.json.gz','rt') as f:rows=json.load(f)
    start=report['operation_reference']['operation_start_s'];end=report['duration_s']
    result=dict(scope='Recorded selected-profile diagnostic; original failed task is unchanged',
        run=str(run),task_passed=report['passed'],grasp_profile=report.get('grasp_profile','distal-pad-v1'),
        original_report_sha256=hashlib.sha256((run/'operation-report.json').read_bytes()).hexdigest(),
        original_pad_steps_sha256=hashlib.sha256((run/'acquisition-pad-steps.json.gz').read_bytes()).hexdigest(),
        legacy_counter_note='operation_digit_unload_samples counts every failed valid_pad_grasp, including opposition failures with all five digits loaded.',
        operation=summarize([r for r in rows if r['sim_time_s']>=start]),
        final_half_second=summarize([r for r in rows if r['sim_time_s']>=end-.5-1e-8]),
        representative_rows=[min(rows,key=lambda r:abs(r['sim_time_s']-t)) for t in (18.8,20.104,21.5,21.702,21.704,22.)])
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('task_passed','grasp_profile','legacy_counter_note')}))


if __name__=='__main__':main()
