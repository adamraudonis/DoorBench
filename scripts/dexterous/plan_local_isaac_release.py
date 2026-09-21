#!/usr/bin/env python3
"""Generate an unadmitted release candidate from a qualified actual Isaac rest.

An old report supplies numeric design preferences only. No native trajectory or
synthetic source metadata is accepted. This entrypoint runs unstepped geometry,
never a physical episode; it deliberately does not export a runtime route.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from doorbench.dexterous.isaac_release_planning import (
    admit_isaac_release_context, generate_release_candidate, numeric_release_preferences,
)
from doorbench.dexterous.qualified_isaac_grasp import digest


def run(args):
    if args.output.exists():
        raise FileExistsError('Fresh planning evidence directory required')
    output = args.output.resolve()
    output.mkdir(parents=True)
    try:
        preferences = numeric_release_preferences(
            json.loads(args.preferences.read_text()) if args.preferences else {})
        names = ('isaac_release_planning.py', 'isaac_release_source.py', 'isaac_prefix_witness.py',
            'qualified_isaac_grasp.py', 'isaac_attained_state.py', 'standing_body_record.py',
            'motor_contract_identity.py', 'destination_state_binding.py',
            'destination_planner_admission.py', 'destination_planning_coordinates.py',
            'destination_return_kinematics.py', 'landed_left_planner.py',
            'release_material_targets.py', 'operation_teacher.py')
        sources = [Path(__file__).resolve(), ROOT/'scripts/dexterous/plan_local_isaac_transfer.py']
        sources += [ROOT/'doorbench/dexterous'/name for name in names]
        inputs = sources + ([args.preferences.resolve()] if args.preferences else [])
        hashes = {str(path): digest(path) for path in inputs}
        for path in sources:
            shutil.copy2(path, output/('source-'+path.name))
        context = admit_isaac_release_context(args.isaac_source, robot=args.robot,
            door_xml=args.door_xml, door_usd=args.door_usd, profile=args.grasp_profile)
        (output/'source-admission.json').write_text(json.dumps(context.admission, indent=2)+'\n')
        report = generate_release_candidate(context, preferences)
        report['input_sha256'].update(hashes)
        report['preferences_source'] = str(args.preferences.resolve()) if args.preferences else None
        report['preferences_are_numeric_only'] = True
        for name, expected in report['input_sha256'].items():
            if digest(name) != expected:
                raise ValueError('Input changed during source-bound release planning: '+name)
        path = output/'candidate.json'
        path.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(candidate=str(path), physics_steps=0,
            source_time_s=context.terminal_time_s, nodes=len(report['trials'][0]['rows']),
            runtime_route_exported=False, physical_contact_qualification=False)), flush=True)
        return 0
    except Exception as error:
        (output/'failure.json').write_text(json.dumps(dict(
            passed=False, physics_steps=0, runtime_route_exported=False,
            error=repr(error), scope='Retained admission/planning failure; no source or route promoted'), indent=2)+'\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('isaac-source', 'robot', 'door-xml', 'door-usd', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--preferences', type=Path,
                        help='Old report contributes bounded numeric design choices only')
    parser.add_argument('--grasp-profile', choices=('distal-pad-v1', 'volar-phalange-v1'),
                        default='volar-phalange-v1')
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
