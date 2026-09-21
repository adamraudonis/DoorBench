#!/usr/bin/env python3
"""Bind the existing volar anatomy contract to a prepared local robot.

This declares the next trial's scoring contract. It performs no physics and
does not qualify any prior recording or change any numerical requirement.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from doorbench.dexterous.robot_design_identity import robot_design_identity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','motors','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a fresh declaration output')
    audit_path=args.robot.with_suffix('.audit.json')
    audit=json.loads(audit_path.read_text());motors=json.loads(args.motors.read_text())
    digest=sha(args.robot)
    if audit.get('mechanics_profile')!='shadow-loopback-v2' or audit.get('robot_xml_sha256')!=digest or motors.get('source_xml_sha256')!=digest:
        raise ValueError('Prepared corrected robot, audit and motor contract must match')
    parent=ROOT/'configs/dexterous/grasp-profiles/volar-phalange-v1.json'
    declaration=json.loads(parent.read_text())
    old_sha=declaration['robot_xml_sha256'];declaration['robot_xml_sha256']=digest
    declaration['declared_before']='Next fresh physical trial explicitly selecting this declaration; historical reports remain unchanged.'
    declaration['local_provenance']=dict(declared_at_utc=datetime.now(timezone.utc).isoformat(),
        parent_profile_definition=str(parent),parent_profile_sha256=sha(parent),parent_robot_xml_sha256=old_sha,
        local_robot=str(args.robot.absolute()),local_robot_audit_sha256=sha(audit_path),
        local_motor_contract_sha256=sha(args.motors),robot_source_design_identity=robot_design_identity(args.robot))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(declaration,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(profile=declaration['profile'],robot_sha256=digest,
        output=str(args.output),declaration_sha256=sha(args.output),physics_started=False)))


if __name__=='__main__':main()
