#!/usr/bin/env python3
"""Bind a qualified actual Isaac endpoint and independently screened RH route."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from doorbench.dexterous.isaac_coupled_release_planning import make_isaac_withdrawal_source_config
from doorbench.dexterous.isaac_release_source_dispatch import PAUSED_SOURCE_KIND


def run(args):
    output=args.output.resolve()
    failure=output.with_name(output.stem+'-failure.json')
    if output.exists() or failure.exists():raise FileExistsError('Fresh detached source-configuration evidence required')
    output.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=make_isaac_withdrawal_source_config(source=args.isaac_source,candidate=args.candidate,
            dense_audit=args.dense_audit,robot=args.robot,door_xml=args.door_xml,door_usd=args.door_usd,
            profile=args.grasp_profile,source_kind=getattr(args,'source_kind',None),
            phase_audit_path=getattr(args,'phase_audit',None))
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(source_config=str(output),initial_episode_time_s=result['start_time_s'],
            duration_s=result['duration_s'],authorized_stages=0,runtime_route_exported=False)),flush=True)
        return 0
    except Exception as error:
        failure.write_text(json.dumps(dict(passed=False,error=repr(error),physics_steps=0,
            authorized_stages=0,runtime_route_exported=False),indent=2)+'\n')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('isaac-source','candidate','dense-audit','robot','door-xml','door-usd','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--grasp-profile',choices=('distal-pad-v1','volar-phalange-v1'),default='volar-phalange-v1')
    parser.add_argument('--source-kind',choices=(PAUSED_SOURCE_KIND,))
    parser.add_argument('--phase-audit',type=Path)
    return run(parser.parse_args())


if __name__=='__main__':raise SystemExit(main())
