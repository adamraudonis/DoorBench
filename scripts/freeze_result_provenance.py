#!/usr/bin/env python3
"""Freeze audit identities of unchanged historical runs from available Git objects."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from doorbench.result_provenance import freeze_historical_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh receipt; inspect changes before replacing an existing registry')
    registry = freeze_historical_results(args.files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(registry, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(f'Frozen {len(registry["results"])} unchanged results against {len(registry["manifests"])} source revisions')


if __name__ == '__main__':
    main()
