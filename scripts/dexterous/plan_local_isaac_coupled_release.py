#!/usr/bin/env python3
"""Generate an unadmitted moving-door map from a qualified actual Isaac source."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from doorbench.dexterous.isaac_coupled_release_planning import (
    generate_isaac_coupled_envelope,numeric_coupled_preferences)
from doorbench.dexterous.qualified_isaac_grasp import digest


def run(args):
    output=args.output.resolve()
    if output.exists():raise FileExistsError('Fresh coupled planning evidence directory required')
    output.mkdir(parents=True)
    try:
        preference_hash=digest(args.preferences) if args.preferences else None
        preferences=numeric_coupled_preferences(json.loads(args.preferences.read_text()) if args.preferences else {})
        paths=[Path(__file__).resolve()]+[ROOT/'doorbench/dexterous'/name for name in
            ('isaac_coupled_release_planning.py','isaac_coupled_release_geometry.py',
             'coupled_release_geometry.py','isaac_release_geometry_audit.py','withdrawal_source_context.py',
             'isaac_release_planning.py','isaac_release_source.py','landed_left_planner.py','operation_teacher.py')]
        hashes={str(path):digest(path) for path in paths}
        for path in paths:
            frozen=output/('source-'+path.name)
            shutil.copy2(path,frozen);hashes[str(frozen)]=digest(frozen)
            if hashes[str(frozen)]!=hashes[str(path)]:raise ValueError('Planner changed during source capture')
        if args.preferences:hashes[str(args.preferences.resolve())]=preference_hash
        plan=generate_isaac_coupled_envelope(args.source_config,preferences,
            progress=lambda row:print(json.dumps(row),flush=True))
        plan['input_sha256'].update(hashes)
        plan['preferences_source']=str(args.preferences.resolve()) if args.preferences else None
        for name,expected in plan['input_sha256'].items():
            if digest(name)!=expected:raise ValueError('Bound coupled planning input changed: '+name)
        path=output/'envelope.json'
        path.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(envelope=str(path),initial_episode_time_s=plan['initial_episode_time_s'],
            grid_points=plan['solver_grid_points'],unconverged_solves=plan['unconverged_solves'],
            physics_steps=0,authorized_stages=0,runtime_route_exported=False,geometric_admission=False)),flush=True)
        return 0
    except Exception as error:
        (output/'failure.json').write_text(json.dumps(dict(passed=False,error=repr(error),physics_steps=0,
            source_sample_playback=0,authorized_stages=0,runtime_route_exported=False,
            scope='Preserved actual-source admission or numerical planning failure; no map promoted'),indent=2)+'\n')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-config','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preferences',type=Path,help='Only allowlisted bounded numerical preferences are used')
    return run(parser.parse_args())


if __name__=='__main__':raise SystemExit(main())
