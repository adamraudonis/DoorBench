#!/usr/bin/env python3
"""Reproduce the frozen native Door55 pre-curl grasp acquisition.

This privileged reference controller has demonstrated grasp acquisition only.
Opening, traversal, Isaac parity and sensor-only actor control are separate gates.
"""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', type=Path, required=True)
    parser.add_argument('--door', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    command = [sys.executable, str(root/'scripts/dexterous/probe_acquisition.py'),
               '--robot', str(args.robot.resolve()), '--door', str(args.door.resolve()),
               '--reference', str(root/'configs/dexterous/door55-precurl-v2/reference.json'),
               '--output', str(args.output.resolve()),
               '--reach-seconds', '6.6', '--hold-seconds', '3',
               '--grip-force', '6', '--grip-start', '.995',
               '--finger-grip-scale', '.333333333', '--grip-reaction',
               '--palm-integral', '1', '--torso-impedance', '10',
               '--cartesian-tracking', '--tracking-gate', '--explicit-motors']
    raise SystemExit(subprocess.call(command, cwd=root))


if __name__ == '__main__':
    main()
