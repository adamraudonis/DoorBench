#!/usr/bin/env python3
"""Run the initialized, privileged one-door H1/Shadow teacher in a ready environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view',choices=['wide','hand'],default='wide')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    ready=ROOT/'out/isaac-ready'
    receipt=json.loads((ready/'ready.json').read_text())
    if not receipt.get('ready'):
        raise RuntimeError('Run scripts/isaac/prepare.sh and resolve its readiness checks first')
    configuration=json.loads((Path(receipt['trial'])/'configuration.json').read_text())
    if not configuration.get('contact_material_audit',{}).get('backend_offsets_verified'):
        raise RuntimeError('This receipt predates verified collider offsets; rerun preparation')
    robot_usd=Path(configuration['args']['robot_usd'])
    output=args.output or ROOT/'out/isaac-demos'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    reference=ROOT/'configs/dexterous/isaac-door55-reference.json'
    work=Path(os.environ.get('DOORBENCH_WORK','/workspace'))
    asset_python=work/'asset-venv/bin/python'
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y')
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([str(asset_python),'scripts/dexterous/audit_isaac_reference.py',
        '--robot',str(ready/'h1-shadow.xml'),'--door',str(ROOT/'assets/doors/db0055_swing_single'),
        '--reference',str(reference),'--output',str(output.with_suffix('.reset-audit.json'))],cwd=ROOT,env=env,check=True)
    subprocess.run([sys.executable,'scripts/dexterous/isaac_opening.py',
        '--robot-usd',str(robot_usd),'--door-usd',str(ROOT/'assets/doors/db0055_swing_single/door.usda'),
        '--motors',str(ready/'h1-import.motors.json'),'--reference',str(reference),
        '--output',str(output),'--seconds','10','--record','--view',args.view,
        '--native-robot',str(ready/'h1-shadow.xml'),'--arm-impedance','10','--grip-reset-targets',
        '--grip-force','6','--torso-damping','20','--stance-qp','--press-feedforward',
        '--enable_cameras','--headless','--device','cuda:0'],cwd=ROOT,env=env,check=True)
    subprocess.run([str(asset_python),'scripts/dexterous/audit_isaac_opening.py','--trial',str(output),
        '--motors',str(ready/'h1-import.motors.json')],cwd=ROOT,env=env,check=True)
    print(f'Opening evidence: {output}\nVideo: {output / "live-isaac.mp4"}')


if __name__=='__main__':
    main()
