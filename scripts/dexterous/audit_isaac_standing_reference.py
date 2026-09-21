#!/usr/bin/env python3
"""Analyze the recorded final three references without evaluating a controller."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from doorbench.dexterous.isaac_standing_reference_audit import admit_standing_reference_tail, SCHEMA


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--continuation-audit', type=Path, required=True)
    parser.add_argument('--expected-epoch-s', type=float, required=True)
    parser.add_argument('--expected-physics-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New receipt; existing evidence is preserved')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError('Preserve the existing analysis; choose a new output path')
    try:
        receipt = admit_standing_reference_tail(args.trial, args.continuation_audit,
            args.expected_epoch_s, args.expected_physics_sha256)
    except (ValueError, KeyError, TypeError, OSError, EOFError, zipfile.BadZipFile, zlib.error) as exc:
        receipt = dict(schema=SCHEMA, passed=False, reference_tail_accounting_passed=False,
            source_trial=str(args.trial.resolve()), error=type(exc).__name__+': '+str(exc),
            requires_separate_physical_source_qualification=True,
            controller_state_restoration_supported=False, bridge_feasibility_qualified=False,
            physical_task_qualification=False, authorized_stages=0,
            scope='Incomplete or corrupt reference/observation evidence; no bridge or stage admission')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: receipt[key] for key in (
        'schema', 'passed', 'reference_tail_accounting_passed', 'authorized_stages')}))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
