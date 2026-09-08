#!/usr/bin/env python3
"""Run the native-matched H1 Door55 approach after the one-click Isaac setup."""
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--matrix',action='store_true');p.add_argument('--record',action='store_true',help='Record live RTX video; slower than numerical qualification')
    a=p.parse_args();ready=ROOT/'out/isaac-ready';receipt=json.loads((ready/'ready.json').read_text())
    if not receipt.get('ready'):raise RuntimeError('Complete scripts/isaac/prepare.sh first')
    config=json.loads((Path(receipt['trial'])/'configuration.json').read_text())
    if not config.get('contact_material_audit',{}).get('backend_offsets_verified'):raise RuntimeError('Readiness requires verified native friction/collision offsets')
    work=Path(os.environ.get('DOORBENCH_WORK','/workspace'));asset_python=work/'asset-venv/bin/python'
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y')
    checkpoint=a.checkpoint or ROOT/'out/h1-walking-upstream/deploy/pre_train/h1/motion.pt'
    if not checkpoint.exists():
        if a.checkpoint:raise FileNotFoundError(checkpoint)
        subprocess.run([str(asset_python),'scripts/dexterous/setup_h1_walking.py','--output',str(checkpoint.parents[3])],cwd=ROOT,env=env,check=True)
    output=(a.output or ROOT/'out/isaac-approach'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    if output.exists():raise FileExistsError('Use a fresh output directory')
    output.parent.mkdir(parents=True,exist_ok=True);resets=output.with_suffix('.resets')
    subprocess.run([str(asset_python),'scripts/dexterous/export_h1_door_approach.py','--robot',str(ready/'h1-shadow.xml'),'--door',str(ROOT/'assets/doors/db0055_swing_single'),'--reference',str(ROOT/'configs/dexterous/h1-door55-approach-target.json'),'--output',str(resets)],cwd=ROOT,env=env,check=True)
    script='evaluate_isaac_door_approach.py' if a.matrix else 'isaac_door_approach.py'
    cmd=[sys.executable,'scripts/dexterous/'+script,'--robot-usd',config['args']['robot_usd'],'--door-usd',config['args']['door_usd'],'--motors',str(ready/'h1-import.motors.json'),'--checkpoint',str(checkpoint.resolve()),'--output',str(output)]
    cmd+=['--reset-dir',str(resets)] if a.matrix else ['--reset',str(resets/'case-1-seed-0.json'),'--headless']
    if a.record:cmd+=['--record']
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    report=json.loads((output/('summary.json' if a.matrix else 'report.json')).read_text())
    passed=(report['completed']==report['total']==report['passed']) if a.matrix else report['passed']
    if not passed:raise SystemExit('Approach did not pass; inspect '+str(output))
    print(f'Approach evidence: {output}\nPrivileged waypoint approach only; no acquisition, opening or traversal.')
if __name__=='__main__':main()
