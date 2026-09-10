#!/usr/bin/env python3
"""Check original-source planner coordinates against recorded Isaac body poses.

Run where original MJCF asset paths resolve. This does not prove collision
cooking parity, contact qualification, path feasibility or physical execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from doorbench.dexterous.landed_left_planner import LandedLeftScene
from doorbench.dexterous.destination_planner_admission import admit_destination_planner
from doorbench.dexterous.destination_return_kinematics import DestinationKinematicsFailure
from doorbench.dexterous.robot_design_identity import robot_design_identity


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','door-usd','extracted','motors','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new evidence output')
    inputs=[a.robot,a.door,a.door_usd,a.extracted,a.motors,Path(__file__)]
    hashes={str(path):digest(path) for path in inputs}
    extracted=json.loads(a.extracted.read_text());binding=extracted['binding']
    if digest(a.robot)!=binding['robot_source_sha256']:
        raise ValueError('Original bound robot XML required; relocation needs a separate identity proof')
    if digest(a.door_usd)!=binding['door_source_sha256']:
        raise ValueError('Original bound door USD required')
    designs={str(path):robot_design_identity(path) for path in (a.robot,a.door)}
    scene=LandedLeftScene(a.robot,a.door)
    try:
        data,receipt=admit_destination_planner(scene.m,extracted,
            motor_contract=json.loads(a.motors.read_text()),door_source_sha256=digest(a.door_usd))
        receipt.update(qpos=data.qpos.tolist(),qvel=data.qvel.tolist())
    except DestinationKinematicsFailure as error:
        receipt=dict(passed=False,measured_body_admission=error.receipt)
    if hashes!={str(path):digest(path) for path in inputs}:
        raise ValueError('Input changed during admission')
    if designs!={str(path):robot_design_identity(path) for path in (a.robot,a.door)}:
        raise ValueError('Referenced model assets changed during admission')
    receipt.update(input_sha256=hashes,source_designs=designs,
        limitation=__doc__.strip())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=receipt['passed'],output=str(a.output))))
    return 0 if receipt['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
