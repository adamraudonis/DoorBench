#!/usr/bin/env python3
"""Standing initialization for importer readiness; not a grasp/opening policy."""
import argparse
import json
from pathlib import Path
import mujoco

p=argparse.ArgumentParser();p.add_argument('--robot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();m=mujoco.MjModel.from_xml_path(str(a.robot))
audit=json.loads(a.robot.with_suffix('.audit.json').read_text())
q=audit['nominal_joint_positions']
motors=json.loads(a.output.parent.joinpath('h1-import.motors.json').read_text())
controls=[sum(q[n]*c for n,c in motor['terms'].items()) for motor in motors['actuators']]
probe=next(i for i,motor in enumerate(motors['actuators']) if motor['name']=='right_wrist_yaw')
sequence=[]
for frame in range(50):
    row=controls.copy();blend=max(0.,min(1.,(frame/50-.2)/.3))
    row[probe]+=.15*blend*blend*(3-2*blend);sequence.append(row)
r=dict(scope='Importer readiness only; nominal standing, no grasp or opening claim',
       initial_root=[-.6,-.65,1.01,1.,0.,0.,0.],initial_joints=q,controls=sequence)
a.output.write_text(json.dumps(r)+'\n')
