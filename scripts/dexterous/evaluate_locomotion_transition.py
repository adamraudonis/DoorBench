#!/usr/bin/env python3
"""Run a frozen seed/heading matrix of continuous native H1 body transitions."""
import argparse,json,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=Path('configs/dexterous/h1-locomotion-transition-development.json'));args=p.parse_args()
    if args.output.exists():raise SystemExit('Use a fresh output directory')
    config=json.loads(args.protocol.read_text());args.output.mkdir(parents=True);(args.output/'protocol.json').write_bytes(args.protocol.read_bytes());rows=[];started=time.time();total=len(config['seeds'])*len(config['initial_yaw_rad'])
    for index,yaw in enumerate(config['initial_yaw_rad']):
        for seed in config['seeds']:
            trial=args.output/f'yaw-{index}-seed-{seed}'
            command=[sys.executable,str(Path(__file__).with_name('probe_locomotion_transition.py')),'--robot',str(args.robot),'--checkpoint',str(args.checkpoint),'--output',str(trial),'--controller',config['controller'],'--seconds',str(config['seconds']),'--seed',str(seed),'--joint-noise',str(config['joint_noise_rad']),'--yaw',str(yaw)]
            command+=['--landing-hip-spread',str(config.get('landing_hip_spread_rad',0.))]
            result=subprocess.run(command,capture_output=True,text=True,timeout=180);(args.output/f'{trial.name}.process.log').write_text(result.stdout+result.stderr)
            if (trial/'report.json').exists():
                report=json.loads((trial/'report.json').read_text());report.pop('arguments');row=dict(trial=trial.name,seed=seed,yaw_rad=yaw,exit_code=result.returncode,**report)
            else:row=dict(trial=trial.name,seed=seed,yaw_rad=yaw,passed=False,exit_code=result.returncode,error='Missing trial report')
            rows.append(row);summary=dict(scope=config['scope'],started_at_unix=started,total=total,completed=len(rows),passed=sum(r['passed'] for r in rows),rows=rows);(args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            line=json.dumps(row);print(line,flush=True)
            with (args.output/'run.log').open('a') as stream:stream.write(line+'\n')
    print(f'{sum(r["passed"] for r in rows)}/{total} complete transition checks passed',flush=True)
    raise SystemExit(0 if all(r['passed'] and r['exit_code']==0 for r in rows) else 1)
if __name__=='__main__':main()
