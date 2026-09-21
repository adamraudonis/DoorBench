#!/usr/bin/env python3
"""Detached finite panel/mechanism screen; never launches a plant/controller."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from doorbench.dexterous.isaac_panel_domain_audit import audit_panel_domain, _read
from doorbench.dexterous.qualified_isaac_grasp import digest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--static-audit',type=Path,required=True)
    parser.add_argument('--sampling',type=Path,required=True,help='Explicit versioned JSON sampling specification')
    parser.add_argument('--output',type=Path,required=True,help='Fresh receipt JSON path; failed samples are retained')
    args=parser.parse_args(argv)
    if args.output.exists():parser.error('Fresh output path required; preserve prior audit')
    sampling=args.sampling.resolve();before=digest(sampling);source_before=digest(__file__)
    report=audit_panel_domain(args.candidate,args.plan,args.static_audit,_read(sampling),
        progress=lambda row:print(json.dumps(row),flush=True))
    if digest(sampling)!=before:raise ValueError('Sampling document changed during screen')
    if digest(__file__)!=source_before:raise ValueError('Domain CLI changed during screen')
    report['input_sha256'][str(sampling)]=before
    report['input_sha256'][str(Path(__file__).resolve())]=source_before
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create avoids silently overwriting another audit even if a
    # concurrent process created the requested name while sampling ran.
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('passed','attempted_samples','failed_samples','rejected_samples','authorized_stages')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
