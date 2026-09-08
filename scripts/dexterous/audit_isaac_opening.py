#!/usr/bin/env python3
"""Audit saved live opening evidence without replaying or altering the plant."""
import argparse,json
from pathlib import Path
from doorbench.dexterous.opening_audit import audit_opening

p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--motors',type=Path,required=True)
a=p.parse_args()
report=audit_opening(json.loads((a.trial/'trace.json').read_text()),json.loads((a.trial/'configuration.json').read_text()),json.loads(a.motors.read_text()))
report['checks']['no_runtime_error']=not (a.trial/'error.txt').exists() and not (a.trial/'early-stop.json').exists()
report['passed']=all(report['checks'].values())
(a.trial/'opening-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
raise SystemExit(int(not report['passed']))
