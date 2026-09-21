#!/usr/bin/env python3
"""Fit a detached panel segment from a qualified actual Isaac withdrawal."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from doorbench.dexterous.isaac_panel_planning import admit_isaac_panel_context,generate_panel_candidate
from doorbench.dexterous.qualified_isaac_grasp import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','robot','door-xml','door-usd','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--numeric-preferences',type=Path)
    parser.add_argument('--target-aperture-rad',type=float,default=1.62)
    parser.add_argument('--solver-method',choices=('constrained','least-squares'),default='constrained')
    parser.add_argument('--flatten-palm',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a new output directory; prior evidence is preserved')
    context=admit_isaac_panel_context(args.source,robot=args.robot,door_xml=args.door_xml,door_usd=args.door_usd)
    preference_bytes=None if args.numeric_preferences is None else args.numeric_preferences.read_bytes()
    preferences={} if preference_bytes is None else json.loads(preference_bytes)
    preference_hash=None if preference_bytes is None else hashlib.sha256(preference_bytes).hexdigest()
    candidate=generate_panel_candidate(context,preferences,target_aperture_rad=args.target_aperture_rad,
        solver_method=args.solver_method,flatten_palm=args.flatten_palm,
        progress=lambda row:print(json.dumps(row,allow_nan=False),flush=True))
    if args.numeric_preferences is not None:
        if digest(args.numeric_preferences)!=preference_hash:raise ValueError('Numeric preference input changed during solve')
        candidate['input_sha256'][str(args.numeric_preferences.resolve())]=preference_hash
        candidate['numeric_preference_input']={'path':str(args.numeric_preferences.resolve()),'sha256':preference_hash,
            'scope':'Bounded numeric preferences only; historical state and qualification discarded'}
    candidate['input_sha256'][str(Path(__file__).resolve())]=digest(__file__)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'candidate.json').write_text(json.dumps(candidate,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(candidate=str(args.output/'candidate.json'),authorized_stages=0,
        physical_admission=False,target_aperture_rad=args.target_aperture_rad)))


if __name__=='__main__':main()
