#!/usr/bin/env python3
"""Audit a detached actual-Isaac coupled map without physical/runtime promotion."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from doorbench.dexterous.isaac_coupled_release_audit import audit_isaac_coupled_envelope,SCHEMA
from doorbench.dexterous.qualified_isaac_grasp import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('envelope','source-config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--coarse',action='store_true',help='Diagnostic only; never passes')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve prior independent evidence')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=audit_isaac_coupled_envelope(args.envelope,source_config=args.source_config,
            coarse=args.coarse,progress=lambda row:print(json.dumps(row),flush=True))
        result['input_sha256'][str(Path(__file__).resolve())]=digest(__file__)
    except Exception as error:
        result=dict(schema=SCHEMA,source_engine='isaac-physx',passed=False,
            admission_failed=True,error=repr(error),samples=0,physics_steps=0,
            authorized_stages=0,runtime_route_exported=False,physical_admission=False)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:value for key,value in result.items()
        if key not in ('input_sha256','source_admission','initial_qpos','failures')}),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
