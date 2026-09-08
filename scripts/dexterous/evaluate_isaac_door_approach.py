#!/usr/bin/env python3
"""Run the frozen native-matched reset matrix as sequential live Isaac processes.

Use only while this work package owns the GPU slot. Every failed process and
report stays in the aggregate; initialization failure never counts as success.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot-usd','door-usd','motors','checkpoint','reset-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stop-on-failure',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--record',action='store_true');p.add_argument('--timeout',type=float,default=900.);a=p.parse_args()
    if a.output.exists():raise SystemExit('Use a fresh output directory')
    a.output.mkdir(parents=True);protocol=json.loads((a.reset_dir/'protocol.json').read_text());resets=json.loads((a.reset_dir/'resets.json').read_text());(a.output/'protocol.json').write_bytes((a.reset_dir/'protocol.json').read_bytes());rows=[];started=time.time()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for reset in resets:
        trial=a.output/Path(reset).stem
        cmd=[sys.executable,str(Path(__file__).with_name('isaac_door_approach.py')),'--headless','--device',a.device]
        for name in ('robot_usd','door_usd','motors','checkpoint'):cmd+=['--'+name.replace('_','-'),str(getattr(a,name))]
        cmd+=['--reset',str(a.reset_dir/reset),'--output',str(trial),'--seconds',str(protocol['seconds']),'--gain',str(protocol['gain']),'--brake-prediction',str(protocol['brake_prediction_s']),'--brake-radius',str(protocol['brake_radius_m']),'--max-speed',str(protocol['max_forward_speed_m_s']),'--brake-velocity-window',str(protocol.get('brake_velocity_window_s',0.))]
        if a.record:cmd+=['--record']
        process_log=a.output/f'{trial.name}.process.log'
        try:
            with process_log.open('w') as log:process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout,env=env)
            code=process.returncode
        except subprocess.TimeoutExpired:code=124
        row={'trial':trial.name,'reset':reset,'exit_code':code,'passed':False}
        if (trial/'report.json').exists():row.update(json.loads((trial/'report.json').read_text()))
        row['passed']=bool(row['passed'] and code==0);rows.append(row)
        summary={'scope':__doc__,'started_at_unix':started,'total':len(resets),'completed':len(rows),'passed':sum(r['passed'] for r in rows),'rows':rows};(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'trial':trial.name,'exit_code':code,'passed':row['passed'],'completed':len(rows),'total':len(resets)}),flush=True)
        if a.stop_on_failure and not row['passed']:break
    raise SystemExit(0 if len(rows)==len(resets) and all(row['passed'] for row in rows) else 1)
if __name__=='__main__':main()
