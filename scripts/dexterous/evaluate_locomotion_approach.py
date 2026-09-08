#!/usr/bin/env python3
"""Run every frozen Door55 waypoint approach development case, retaining failures."""
import argparse,json,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','reference','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=Path('configs/dexterous/h1-door55-approach-development.json'));args=p.parse_args()
    if args.output.exists():raise SystemExit('Use a fresh output directory')
    config=json.loads(args.protocol.read_text());args.output.mkdir(parents=True);(args.output/'protocol.json').write_bytes(args.protocol.read_bytes());rows=[];started=time.time()
    for index,case in enumerate(config['cases']):
        for seed in config['seeds']:
            trial=args.output/f'case-{index}-seed-{seed}'
            command=[sys.executable,str(Path(__file__).with_name('probe_locomotion_approach.py'))]
            for name in ('robot','door','reference','checkpoint'):command+=['--'+name,str(getattr(args,name))]
            command+=['--output',str(trial),'--seed',str(seed)]
            values={'seconds':config['seconds'],'joint-noise':config['joint_noise_rad'],'gain':config['gain'],'brake-prediction':config['brake_prediction_s'],'brake-radius':config['brake_radius_m'],'max-speed':config['max_forward_speed_m_s'],'brake-velocity-window':config.get('brake_velocity_window_s',0.),'distance':case['distance_m'],'lateral':case['lateral_m'],'yaw-offset':case['yaw_offset_rad']}
            for name,value in values.items():command+=['--'+name,str(value)]
            if config.get('precision_stance'):command+=['--precision-stance']
            process=subprocess.run(command,capture_output=True,text=True,timeout=180);(args.output/f'{trial.name}.process.log').write_text(process.stdout+process.stderr)
            if (trial/'report.json').exists():
                report=json.loads((trial/'report.json').read_text());report.pop('arguments');row={'trial':trial.name,'seed':seed,'case':case,'exit_code':process.returncode,**report}
            else:row={'trial':trial.name,'seed':seed,'case':case,'exit_code':process.returncode,'passed':False,'error':'Missing report'}
            rows.append(row);summary={'scope':config['scope'],'started_at_unix':started,'total':len(config['cases'])*len(config['seeds']),'completed':len(rows),'passed':sum(r['passed'] for r in rows),'rows':rows};(args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(row),flush=True)
    print(f'{sum(r["passed"] for r in rows)}/{len(rows)} complete approach checks passed',flush=True)
    raise SystemExit(0 if all(r['passed'] and r['exit_code']==0 for r in rows) else 1)
if __name__=='__main__':main()
