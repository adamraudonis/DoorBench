#!/usr/bin/env python3
"""Export one exact recorded state for planning, without claiming qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from doorbench.dexterous.isaac_attained_state import extract_attained_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--time-s', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = {name: args.run / name for name in (
        'configuration.json', 'motor-contract.json', 'provenance.json', 'acquisition-physics.npz')}
    if args.output.exists():
        raise FileExistsError('Choose a new output; existing evidence is preserved')
    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    hashes = {name: digest(path) for name, path in files.items()}
    with np.load(files['acquisition-physics.npz'], allow_pickle=False) as physics:
        binding = extract_attained_state(
            configuration=json.loads(files['configuration.json'].read_text()),
            motor_contract=json.loads(files['motor-contract.json'].read_text()),
            provenance=json.loads(files['provenance.json'].read_text()),
            physics=physics, time_s=args.time_s)
    if hashes != {name: digest(path) for name, path in files.items()}:
        raise ValueError('Archive changed during extraction; wait for verified collection')
    result = dict(binding=binding, input_sha256=hashes,
                  scope='Recorded state identity only; no contact, geometry or task qualification')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(time_s=binding['time_s'], state_sha256=binding['sha256'], output=str(args.output))))


if __name__ == '__main__':
    main()
