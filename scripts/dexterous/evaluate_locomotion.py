#!/usr/bin/env python3
"""Execute a frozen, bounded CPU locomotion development protocol and retain failures."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('robot','checkpoint','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=Path('configs/dexterous/h1-locomotion-development.json'))
    p.add_argument('--reference',type=Path,default=Path('configs/dexterous/isaac-door55-reference.json'))
    args=p.parse_args()
    if args.output.exists():raise SystemExit('Use a fresh output directory')
    protocol=json.loads(args.protocol.read_text());args.output.mkdir(parents=True);(args.output/'protocol.json').write_bytes(args.protocol.read_bytes());(args.output/'reference.json').write_bytes(args.reference.read_bytes());(args.output/'run.pid').write_text(str(os.getpid()))
    total=len(protocol['seeds'])*len(protocol['groups']);rows=[]
    pipeline=dict(stage='CPU H1 locomotion seeded evaluation',started_at_unix=time.time(),completion_marker='LOCOMOTION_EVALUATION_FINISHED',total_trials=total,completed_trials=0)
    (args.output/'pipeline.json').write_text(json.dumps(pipeline)+'\n')
    for group in protocol['groups']:
        for seed in protocol['seeds']:
            trial=args.output/f'{group["name"]}-seed-{seed:02d}'
            command=[sys.executable,str(Path(__file__).with_name('probe_locomotion.py')),'--robot',str(args.robot),'--checkpoint',str(args.checkpoint),'--output',str(trial),'--mode',group['mode'],'--initial-pose',group['initial_pose'],'--speed',str(group['speed']),'--seconds',str(group['seconds']),'--seed',str(seed),'--joint-noise',str(protocol['joint_noise_rad']),'--reference',str(args.output/'reference.json')]
            if protocol['phase_stop']:command.append('--phase-stop')
            result=subprocess.run(command,capture_output=True,text=True,timeout=120)
            (args.output/f'{trial.name}.process.log').write_text(result.stdout+result.stderr)
            if (trial/'report.json').exists():
                report=json.loads((trial/'report.json').read_text());trace=json.loads((trial/'trace.json').read_text());tail=[r for r in trace if r['time_s']>group['seconds']-1]
                limits=protocol['extra_acceptance_checks'];positions=np.array([r['root'][:2] for r in tail]);excursion=float(np.max(np.linalg.norm(positions-positions[0],axis=1))) if len(tail) else float('inf')
                minload=min((min(r['foot_loads_N']) for r in tail),default=0.)
                extra=dict(quiet_root=report['final_second_max_speed_m_s'] is not None and report['final_second_max_speed_m_s']<limits['final_second_max_root_speed_m_s'],quiet_position=excursion<limits['final_second_max_root_excursion_m'],both_feet_support=minload>limits['final_second_min_each_foot_load_N'])
                row=dict(group=group['name'],seed=seed,passed=result.returncode==0 and report['passed'] and all(extra.values()),checks=report['checks'],extra_checks=extra,distance_m=report['net_distance_m'],max_tilt_deg=report['max_tilt_deg'],max_joint_violation_rad=report['max_joint_violation_rad'],final_second_max_speed_m_s=report['final_second_max_speed_m_s'],final_second_excursion_m=excursion,final_second_min_foot_load_N=minload,report=str(trial/'report.json'))
            else:row=dict(group=group['name'],seed=seed,passed=False,error='Trial did not produce a report',exit_code=result.returncode)
            rows.append(row);pipeline['completed_trials']=len(rows);pipeline['passed_trials']=sum(r['passed'] for r in rows);(args.output/'pipeline.json').write_text(json.dumps(pipeline)+'\n');(args.output/'summary.json').write_text(json.dumps(dict(scope=protocol['scope'],completed=len(rows),total=total,passed=sum(r['passed'] for r in rows),rows=rows),indent=2)+'\n')
            line=json.dumps(row);print(line,flush=True)
            with (args.output/'run.log').open('a') as stream:stream.write(line+'\n')
    pipeline['stage']='Completed CPU H1 locomotion development evaluation';(args.output/'pipeline.json').write_text(json.dumps(pipeline)+'\n')
    with (args.output/'run.log').open('a') as stream:stream.write('LOCOMOTION_EVALUATION_FINISHED\n')
    print(f'{sum(r["passed"] for r in rows)}/{total} complete checks passed',flush=True)
    raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
if __name__=='__main__':main()
