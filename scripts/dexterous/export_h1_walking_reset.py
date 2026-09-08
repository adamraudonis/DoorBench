#!/usr/bin/env python3
"""Export the existing native standing reset using the actual collision soles."""
import argparse
import hashlib
import json
from pathlib import Path
from probe_locomotion import make_plant


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sim=make_plant(a.robot,initial_pose='standing')
    m,d=sim.m,sim.d
    result=dict(robot_xml_sha256=hashlib.sha256(a.robot.read_bytes()).hexdigest(),mass_kg=float(m.body_mass.sum()),
        initial_root=d.qpos[:7].tolist(),
        joints={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in sim.joints},
        motor_targets={m.actuator(i).name:float(d.ctrl[i]) for i in range(m.nu)},
        feet={m.body(i).name:d.xpos[i].tolist() for i in sim.feet},
        scope='Reset only; root is freely simulated after initialization')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'root':result['initial_root'],'mass_kg':result['mass_kg']}))


if __name__=='__main__':main()
