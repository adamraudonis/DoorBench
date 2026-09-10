#!/usr/bin/env python3
"""Reproduce an exact failed arm solve without stepping a physical plant."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from doorbench.dexterous.handle_relative_arm import ARM_NAMES,HandleRelativeArmTarget


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new diagnostic output')
    manifest=json.loads((args.trial/'manifest.json').read_text())
    robot=Path(manifest['configuration']['robot'])
    robot_sha=hashlib.sha256(robot.read_bytes()).hexdigest()
    if robot_sha!=manifest['inputs']['robot']['sha256']:raise ValueError('Original robot XML bytes required')
    source=args.trial/'relative-arm-failure.json';snapshot=json.loads(source.read_text())
    if snapshot['schema']!='doorbench.relative-arm-failure.v1':raise ValueError('Exact failure snapshot required')
    m=mujoco.MjModel.from_xml_path(str(robot))
    c=HandleRelativeArmTarget(m,snapshot['names'],snapshot['root'],snapshot['joints'],snapshot['reference_pose'],reference_frame=snapshot['reference_frame'])
    c.relative_position=np.array(snapshot['relative_position']);c.relative_rotation=np.array(snapshot['relative_rotation'])
    c.goal=np.array(snapshot['previous_goal']);c.correction=np.array(snapshot['previous_correction'])
    error=None
    try:c.target(snapshot['time_s'],snapshot['root'],snapshot['joints'],snapshot['reference_pose'],snapshot['nominal'])
    except ValueError as exc:error=str(exc)
    replay=c.failure_snapshot
    comparisons={} if replay is None else {k:float(np.max(np.abs(np.asarray(replay[k])-np.asarray(snapshot[k])))) for k in ['lower','upper','fit','residual']}
    result=dict(reproduced=bool(replay is not None and all(v<1e-10 for v in comparisons.values())),error=error,maximum_absolute_differences=comparisons,
        replay_solve=c.solve_info,reference_frame=snapshot['reference_frame'],time_s=snapshot['time_s'],physics_steps=0,
        robot_sha256=robot_sha,snapshot_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Reproduction of a failed kinematic solve only; not physical success or feasibility proof')
    if replay is not None:
        result['active_bounds']={n:dict(value=replay['fit'][i],lower=replay['lower'][i],upper=replay['upper'][i]) for i,n in enumerate(ARM_NAMES) if min(replay['fit'][i]-replay['lower'][i],replay['upper'][i]-replay['fit'][i])<1e-5}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))
    return 0 if result['reproduced'] else 1


if __name__=='__main__':raise SystemExit(main())
