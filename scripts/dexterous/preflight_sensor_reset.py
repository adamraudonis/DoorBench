#!/usr/bin/env python3
"""Compile the frozen actor reset on native CPU before any Isaac actor job."""
import argparse
import json
from pathlib import Path

from doorbench.dexterous.sensor_reset_preflight import build_sensor_reset_preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'motors', 'native-robot', 'native-door', 'robot-usd', 'door-usd', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    options = vars(parser.parse_args())
    output = options.pop('output')
    if output.exists():
        parser.error('Use a fresh output receipt; previous failures must be retained')
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = build_sensor_reset_preflight(**options)
    except Exception as error:
        report = dict(passed=False, scope='Failed sensor reset preflight', error=str(error))
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(receipt=str(output), passed=report['passed'], checks=report.get('checks'), error=report.get('error'))))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
