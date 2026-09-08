#!/usr/bin/env python3
"""Source-bound actual trajectory comparison for the smooth-gain experiment."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_rows(path):
    with gzip.open(path,'rt') as stream:return [json.loads(line) for line in stream]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','trial','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh comparison receipt required')
    baseline=np.load(a.baseline/'trajectory.npz');trial=np.load(a.trial/'trajectory.npz')
    report=json.loads((a.trial/'report.json').read_text())
    rows=load_rows(a.trial/'physics.jsonl.gz');infos=load_rows(a.trial/'controller.jsonl.gz')
    baseline_infos=load_rows(a.baseline/'controller.jsonl.gz')
    if len(rows)!=len(infos) or len(rows)!=len(trial['force']):raise ValueError('Unmatched physical/control epochs')
    prefix={key:bool(np.array_equal(baseline[key][:9501 if key!='force' else 9500],trial[key][:9501 if key!='force' else 9500])) for key in ('qpos','qvel','force','time')}
    prefix['joint_goals']=len(infos)>=9500 and all(infos[i]['goal_joint_position_rad']==baseline_infos[i]['goal_joint_position_rad'] for i in range(9500))
    names=list(trial['action_names']);finger=np.array([n.startswith('rh_') and 'WRJ' not in n for n in names])
    forces=np.asarray(trial['force']);delta=np.diff(forces,axis=0)
    times=np.arange(len(delta))*.002+.002
    def first(predicate):
        return next(({'time_s':r['sim_time_s'],'handle_rad':r['handle_angle_rad'],'detail':predicate(r)} for r in rows if predicate(r)),None)
    phases={}
    for label,lo,hi in [('acquisition',0.,19.),('gain_ramp',19.,21.),('after_ramp',21.,36.)]:
        selected=(times>=lo)&(times<hi)
        i=np.flatnonzero(selected)
        phases[label]=dict(interval_count=int(len(i)),maximum_finger_command_step_Nm=float(abs(delta[selected][:,finger]).max()) if len(i) else None,
            maximum_all_motor_command_step_Nm=float(abs(delta[selected]).max()) if len(i) else None)
    start=None;best=(0.,None,None)
    for r in rows:
        good=bool(r['pad_grasp']['valid_pad_grasp'])
        if good and start is None:start=r['contact_interval_start_s']
        if good and r['sim_time_s']-start>best[0]:best=(r['sim_time_s']-start,start,r['sim_time_s'])
        if not good:start=None
    result=dict(scope=__doc__,trial_passed=report['passed'],first19_seconds_bitwise=prefix,
        initial_ramp_coefficient=infos[9500].get('effective_finger_position_gain_multiplier') if len(infos)>9500 else None,
        final_ramp_coefficient=infos[-1].get('effective_finger_position_gain_multiplier'),
        command_continuity=phases,
        first_invalid_loaded_distal=first(lambda r:r['invalid_loaded_distal_patches']),
        first_unintended_hand_contact=first(lambda r:r['unintended_hand_contacts']),
        first_post_acquisition_opposed_failure=first(lambda r:r['pad_grasp']['reason'] if r['sim_time_s']>19. and not r['pad_grasp']['valid_pad_grasp'] else None),
        longest_actual_opposed_hold_s=best[0],longest_hold_interval_s=list(best[1:]),
        failed_checks=[k for k,v in report['checks'].items() if not v],
        last_virtual_clock_s=infos[-1]['virtual_press_clock_s'],last_closing_offsets_rad=infos[-1]['finger_motor_closure_offsets_rad'],
        source_sha256=sha(__file__),inputs_sha256={str(path):sha(path) for path in [a.baseline/'trajectory.npz',a.baseline/'controller.jsonl.gz',a.trial/'trajectory.npz',a.trial/'controller.jsonl.gz',a.trial/'physics.jsonl.gz',a.trial/'report.json']})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
