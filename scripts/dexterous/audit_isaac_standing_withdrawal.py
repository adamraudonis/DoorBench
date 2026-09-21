#!/usr/bin/env python3
"""Independently audit the phase-specific actual Isaac withdrawal evidence."""
import argparse
import json
from pathlib import Path
from doorbench.dexterous.isaac_withdrawal_audit import audit_withdrawal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--contact-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh audit output required')
    try:result=audit_withdrawal(args.trial,args.contact_audit)
    except Exception as error:
        result=dict(schema='doorbench.isaac-standing-withdrawal-audit.v1',passed=False,
            error_type=type(error).__name__,error=str(error),physics_steps=0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','checks')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
