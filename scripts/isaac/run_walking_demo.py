#!/usr/bin/env python3
"""Run the pinned H1 plane walking/quiet-stop fixture in a ready Isaac environment."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--checkpoint',type=Path,help='Existing exact pinned official H1 checkpoint')
    p.add_argument('--no-video',action='store_true')
    args=p.parse_args()
    ready=ROOT/'out/isaac-ready'
    receipt=json.loads((ready/'ready.json').read_text())
    if not receipt.get('ready'):raise RuntimeError('Complete scripts/isaac/prepare.sh first')
    configuration=json.loads((Path(receipt['trial'])/'configuration.json').read_text())
    if not configuration.get('contact_material_audit',{}).get('backend_offsets_verified'):
        raise RuntimeError('Rerun readiness with verified native friction and collision offsets')
    work=Path(os.environ.get('DOORBENCH_WORK','/workspace'))
    asset_python=work/'asset-venv/bin/python'
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y')
    checkpoint=args.checkpoint or ROOT/'out/h1-walking-upstream/deploy/pre_train/h1/motion.pt'
    if not checkpoint.exists():
        if args.checkpoint:raise FileNotFoundError(checkpoint)
        subprocess.run([str(asset_python),'scripts/dexterous/setup_h1_walking.py',
                        '--output',str(ROOT/'out/h1-walking-upstream')],cwd=ROOT,env=env,check=True)
    output=(args.output or ROOT/'out/isaac-walking'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    if output.exists():raise FileExistsError('Use a fresh output directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    reset=output.with_suffix('.reset.json')
    subprocess.run([str(asset_python),'scripts/dexterous/export_h1_walking_reset.py',
                    '--robot',str(ready/'h1-shadow.xml'),'--output',str(reset)],cwd=ROOT,env=env,check=True)
    subprocess.run([sys.executable,'scripts/dexterous/isaac_h1_walking.py',
        '--robot-usd',configuration['args']['robot_usd'],'--motors',str(ready/'h1-import.motors.json'),
        '--reset',str(reset),'--checkpoint',str(checkpoint.resolve()),'--output',str(output),
        '--headless','--device','cuda:0',*([] if args.no_video else ['--record','--enable_cameras'])],cwd=ROOT,env=env,check=True)
    print(f'Walking evidence: {output}\nThis is a plane motor-skill fixture, not a door task.')


if __name__=='__main__':main()
