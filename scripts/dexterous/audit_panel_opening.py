#!/usr/bin/env python3
"""Audit panel opening from saved evidence without rerunning the physics."""
import argparse
import json
from pathlib import Path
from doorbench.dexterous.panel_opening_audit import audit_panel_opening

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trial',type=Path,required=True)
a=p.parse_args()
report=audit_panel_opening(json.loads((a.trial/'trace.json').read_text()),
    json.loads((a.trial/'opening-audit.json').read_text()),json.loads((a.trial/'mechanical-audit.json').read_text()))
(a.trial/'panel-opening-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
raise SystemExit(int(not report['passed']))
