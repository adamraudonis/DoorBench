#!/usr/bin/env python3
"""Prepare an opt-in LH route from a frozen actual runtime packet; no simulation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from doorbench.dexterous.runtime_left_planner import plan_attained_left_contact,RuntimeLeftPlanFailure,PROFILES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','targets','measured','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--at-time',type=float,required=True)
    p.add_argument('--clearance-profile',choices=PROFILES,default='strict-v1')
    p.add_argument('--subdivisions',type=int,default=20)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    try:
        config,receipt=plan_attained_left_contact(a.robot,a.door,a.targets,json.loads(a.measured.read_text()),
            at_time_s=a.at_time,clearance_profile=a.clearance_profile,subdivisions=a.subdivisions)
    except RuntimeLeftPlanFailure as error:
        (a.output/'report.json').write_text(json.dumps(error.receipt,indent=2)+'\n')
        print(str(error));return 1
    (a.output/'target-config.json').write_text(json.dumps(config,indent=2)+'\n')
    (a.output/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({key:receipt[key] for key in ('passed','selected_profile','wall_seconds','attained_state_sha256','target_content_sha256')}));return 0


if __name__=='__main__':raise SystemExit(main())
