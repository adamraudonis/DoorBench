#!/usr/bin/env python3
"""Run the frozen privileged contact-free acquisition in a verified v2 environment.

The reference's absolute source XML hash is historical provenance; a new cluster
has different mesh paths. The ready native XML must match its own motor contract
and the corrected pinned robot profile. Frozen reference numbers are unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from doorbench.dexterous.isaac_readiness import load_ready_receipt,ready_directory,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--view',choices=('wide','hand'),default='wide');p.add_argument('--no-record',action='store_true')
    p.add_argument('--check-only',action='store_true',help='Verify inputs and native initial geometry without launching Isaac')
    a=p.parse_args()
    ready=Path(os.environ.get('DOORBENCH_READY_DIR',ready_directory(ROOT,'shadow-loopback-v2')))
    receipt_path=a.receipt or ready/'ready.json';receipt=load_ready_receipt(receipt_path,expected_profile='shadow-loopback-v2')
    frozen=ROOT/'configs/dexterous/door55-precurl-v2';reference=frozen/'reference.json';protocol=json.loads((frozen/'protocol.json').read_text())
    if file_sha256(reference)!=protocol['reference_sha256']:raise ValueError('Frozen acquisition reference changed')
    ref=json.loads(reference.read_text());motors=json.loads(Path(receipt['motor_contract']).read_text())
    initial=dict(zip(ref['acquisition']['joint_names'],ref['acquisition']['path_qpos'][0]))
    for tendon in motors['passive_tendons']:
        length=sum(initial[name]*coefficient for name,coefficient in tendon['terms'].items())
        if not tendon['range_rad'][0]-1e-6<=length<=tendon['range_rad'][1]+1e-6:raise ValueError('Acquisition reset violates passive loopback')
    output=(a.output or ROOT/'out/isaac-acquisition'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    if output.exists():raise FileExistsError('Use a fresh output directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    # The legacy workspace_fit includes opening poses. Screen the acquisition's
    # actual closed, contact-free initial pose instead of that different task.
    reset=dict(ref);reset['initial_joints']=initial;reset.pop('workspace_fit',None)
    reset_path=output.with_suffix('.initial-reference.json');reset_path.write_text(json.dumps(reset)+'\n')
    work=Path(os.environ.get('DOORBENCH_WORK','/workspace'));asset_python=work/'asset-venv/bin/python'
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y')
    subprocess.run([str(asset_python),'scripts/dexterous/audit_isaac_reference.py','--robot',receipt['native_robot'],
        '--door',str(Path(receipt['door_usd']).parent),'--reference',str(reset_path),'--output',str(output.with_suffix('.initial-audit.json'))],cwd=ROOT,env=env,check=True)
    if a.check_only:
        print('V2 inputs and native contact-free initialization verified; no Isaac trial launched.');return
    cmd=[sys.executable,'scripts/dexterous/isaac_opening.py','--robot-usd',receipt['robot_usd'],'--door-usd',receipt['door_usd'],
         '--motors',receipt['motor_contract'],'--native-robot',receipt['native_robot'],'--reference',str(reference),
         '--acquisition','--seconds',str(protocol['strict_audit']['expected_duration_s']),'--output',str(output),
         '--view',a.view,'--headless','--device','cuda:0']
    if not a.no_record:cmd+=['--record','--enable_cameras']
    invocation={'scope':__doc__,'argv':cmd,'readiness_receipt':str(receipt_path.resolve()),
        'reference_sha256':file_sha256(reference),'protocol_sha256':file_sha256(frozen/'protocol.json'),
        'ready_native_xml_sha256':receipt['native_robot_sha256'],'reference_source_xml_sha256':protocol['model_xml_sha256']}
    output.with_suffix('.invocation.json').write_text(json.dumps(invocation,indent=2)+'\n')
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    report=json.loads((output/'acquisition-report.json').read_text())
    if report.get('passed') is not True:raise SystemExit('Acquisition checks failed; retained evidence: '+str(output))
    print('Acquisition checks passed: '+str(output)+'\nPrivileged grasp acquisition only; no approach, opening, traversal or sensor-only claim.')


if __name__=='__main__':main()
