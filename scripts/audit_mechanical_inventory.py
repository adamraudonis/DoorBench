#!/usr/bin/env python3
"""Inventory current IR and optional published JSON, without exporting assets."""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from doorbench.build import build_model
from doorbench.mechanical_audit import audit_model
from doorbench.physics import derive
from doorbench.spec import generate_all


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--published',type=Path,default=ROOT/'assets')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    source_paths=[p for p in (ROOT/'doorbench').rglob('*.py') if '__pycache__' not in p.parts]+[Path(__file__)]
    digest=lambda p:sha256(p.read_bytes()).hexdigest()
    before={str(p.relative_to(ROOT)):digest(p) for p in source_paths}
    started=time.monotonic();rows=[]
    for spec in generate_all():
        model=build_model(spec);report=audit_model(spec,model.to_dict('full'))
        report['physics_mass']=derive(spec)['mass']
        folder=args.published/'doors'/spec['id']
        if (folder/'model.json').exists() and (folder/'spec.json').exists():
            old_spec=json.loads((folder/'spec.json').read_text())
            old_model=json.loads((folder/'model.json').read_text())
            report['published']=audit_model(old_spec,old_model)
            report['published']['physics_mass']=old_spec.get('physics',{}).get('mass')
            report['published']['sha256']={name:digest(folder/name) for name in ('spec.json','model.json')}
        rows.append(report)
    after={str(p.relative_to(ROOT)):digest(p) for p in source_paths}
    result={'schema':'doorbench.mechanical-inventory.v1','scope':'Independent material area and operator inventory; not collision/dynamic certification or a replacement bill of materials.','python':platform.python_version(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':before,'source_files_changed_during_audit':[p for p in before if before[p]!=after[p]],'published_root':str(args.published),'elapsed_s':time.monotonic()-started,'count':len(rows),'family_counts':dict(Counter(r['family'] for r in rows)),'current_issue_counts':dict(Counter(i['code'] for r in rows for i in r['issues'])),'published_issue_counts':dict(Counter(i['code'] for r in rows if 'published' in r for i in r['published']['issues'])),'rows':rows}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('count','elapsed_s','current_issue_counts','published_issue_counts','source_files_changed_during_audit')},indent=2))
    print('report_sha256',digest(args.out))
    return 0 if result['count']==1000 else 1

if __name__=='__main__':raise SystemExit(main())
