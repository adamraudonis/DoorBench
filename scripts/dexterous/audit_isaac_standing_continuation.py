#!/usr/bin/env python3
"""Audit saved actual continuation observations without launching a simulator."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from doorbench.dexterous.isaac_standing_continuation_audit import audit_standing_continuation, SCHEMA


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True, help='Completed run or its trial directory')
    parser.add_argument('--output', type=Path, required=True, help='New JSON receipt; existing evidence is never overwritten')
    args = parser.parse_args(argv)
    if args.output.exists(): raise ValueError('Preserve the existing audit; choose a new output path')
    try:
        receipt = audit_standing_continuation(args.trial)
    except (ValueError, KeyError, TypeError, OSError, EOFError, zipfile.BadZipFile, zlib.error) as exc:
        receipt = dict(schema=SCHEMA, passed=False, accounting_passed=False,
            source_trial=str(args.trial.resolve()), error=type(exc).__name__+': '+str(exc),
            physical_task_qualification=False, authorized_stages=0,
            scope='Incomplete or corrupt observation evidence; no observation or stage admission')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({key:receipt[key] for key in ('schema', 'passed', 'accounting_passed', 'authorized_stages')}))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
