#!/usr/bin/env python3
"""Print or execute a portable standing-transfer experiment command.

Provision and verify the environment and teardown guards first. This wrapper
only supplies named experiment settings to the existing audited coordinator.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ARITY = {'--operation-leaf-target-rad': 1, '--operation-handle-hub-avoidance': 0,
         '--operation-grasp-offset-in-handle-m': 3, '--isaac-timeout-seconds': 1,
         '--actual-material-pads': 0, '--material-pad-profile': 1,
         '--operation-operator-lead-limit-rad': 1,
         '--operation-index-proximal-offset-rad': 1,
         '--operation-index-tendon-offset-rad': 1,
         '--standing-transfer-hybrid-support': 0}
OPTIONAL = {'--standing-transfer-hybrid-support'}


def build_command(profile, *, source, ready, reference, route, output, work,
                  deadline, now=None, python=sys.executable):
    if profile.get('schema') != 'doorbench.standing-transfer-recipe.v1':
        raise ValueError('Versioned standing-transfer recipe required')
    args = profile.get('arguments')
    if not isinstance(args, list) or not all(isinstance(v, str) for v in args):
        raise ValueError('Recipe arguments must be a string array')
    i = 0
    seen = set()
    while i < len(args):
        flag = args[i]
        if flag not in ARITY or flag in seen or i + ARITY[flag] >= len(args):
            raise ValueError('Unknown, duplicate or incomplete recipe option: ' + flag)
        seen.add(flag)
        i += ARITY[flag] + 1
    if seen - OPTIONAL != set(ARITY) - OPTIONAL:
        raise ValueError('All declared experiment options must be explicit')
    camera = Path(profile['camera_profile'])
    if camera.is_absolute() or '..' in camera.parts:
        raise ValueError('Camera profile must be inside the source snapshot')
    paths = dict(source=source, ready=ready, reference=reference, output=output, work=work)
    if not all(Path(p).is_absolute() for p in [*paths.values(), route]):
        raise ValueError('Explicit absolute environment and artifact paths required')
    if not math.isfinite(deadline) or deadline <= (time.time() if now is None else now):
        raise ValueError('A future externally guarded deadline is required')
    command = [str(python), str(Path(source)/'scripts/isaac/run_standing_operation.py')]
    for flag, value in paths.items():
        command += ['--' + flag, str(value)]
    return command + ['--deadline-unix', str(deadline), *args,
                      '--standing-transfer-route', str(route),
                      '--camera-profile', str(Path(source)/camera)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=ROOT/'configs/isaac/standing-transfer-h1-shadow-v1.json')
    for name in ('source', 'ready', 'reference', 'route', 'output', 'work'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--execute', action='store_true', help='Run the printed command; default only prints it')
    args = parser.parse_args()
    command = build_command(json.loads(args.profile.read_text()),
                            **{k: getattr(args, k) for k in ('source','ready','reference','route','output','work')},
                            deadline=args.deadline_unix)
    print(json.dumps(dict(argv=command, execution_requested=args.execute,
                         scope='Recipe only; environment readiness and physical success are not inferred')), flush=True)
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
