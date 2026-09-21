#!/usr/bin/env python3
"""Audit an actual-Isaac release candidate without running or promoting it."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from doorbench.dexterous.isaac_release_geometry_audit import audit_isaac_release_candidate
from doorbench.dexterous.qualified_isaac_grasp import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','isaac-source','robot','door-xml','door-usd','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--duration',type=float,default=16.)
    parser.add_argument('--grasp-profile',choices=('distal-pad-v1','volar-phalange-v1'),default='volar-phalange-v1')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous independent audit evidence')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=audit_isaac_release_candidate(args.candidate,source=args.isaac_source,
            robot=args.robot,door_xml=args.door_xml,door_usd=args.door_usd,
            duration_s=args.duration,profile=args.grasp_profile)
        result['input_sha256'][str(Path(__file__).resolve())]=digest(__file__)
    except Exception as error:
        result=dict(schema='doorbench.isaac-release-dense-geometry-audit.v1',passed=False,
            admission_failed=True,error=repr(error),samples=0,physics_steps=0,
            runtime_route_exported=False,physical_contact_qualification=False,
            scope='Failed explicit source/candidate admission; no geometric or physical promotion')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('input_sha256','source_admission','failures')}),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
