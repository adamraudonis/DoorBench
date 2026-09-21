#!/usr/bin/env python3
"""Independently screen actual-Isaac panel geometry; never authorize physics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from doorbench.dexterous.isaac_panel_geometry_audit import audit_panel_candidate
from doorbench.dexterous.qualified_isaac_grasp import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--duration-s',type=float,default=40.)
    parser.add_argument('--samples',type=int,default=2001)
    parser.add_argument('--aperture-speed-limit-rad-s',type=float,default=.149)
    parser.add_argument('--aperture-acceleration-limit-rad-s2',type=float,default=.08)
    parser.add_argument('--actual-leaf-lag-rad',type=float,default=0.)
    parser.add_argument('--lag-start-angle-rad',type=float)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a new audit directory; failed screens remain evidence')
    receipt,plan,traces=audit_panel_candidate(args.candidate,duration_s=args.duration_s,samples=args.samples,
        aperture_speed_limit_rad_s=args.aperture_speed_limit_rad_s,
        aperture_acceleration_limit_rad_s2=args.aperture_acceleration_limit_rad_s2,
        actual_leaf_lag_rad=args.actual_leaf_lag_rad,lag_start_angle_rad=args.lag_start_angle_rad,
        progress=lambda row:print(json.dumps(row),flush=True))
    receipt['input_sha256'][str(Path(__file__).resolve())]=digest(__file__)
    plan['input_sha256']=receipt['input_sha256'].copy()
    args.output.mkdir(parents=True,exist_ok=False)
    for name,value in [('report.json',receipt),('geometry-plan.json',plan)]:
        (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.output/'target-traces.npz',values=traces)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('failures','source_admission','input_sha256')},indent=2))
    return 0 if receipt['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
