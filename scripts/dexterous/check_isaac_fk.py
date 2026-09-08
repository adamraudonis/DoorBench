#!/usr/bin/env python3
"""Compare live PhysX link poses with independent native kinematics at identical q."""
import argparse
import json
import re
from pathlib import Path
import mujoco
import numpy as np


def check(robot, run):
    m=mujoco.MjModel.from_xml_path(str(robot));d=mujoco.MjData(m)
    run=Path(run)
    config=json.loads((run/'configuration.json').read_text())
    poses=json.loads((run/'initial-body-poses.json').read_text())['robot']
    row=json.loads((run/'trace.json').read_text())[0]
    d.qpos[:7]=row['root'][:7]
    for name,value in zip(config['robot_joint_names'],row['joints']):
        d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
    mujoco.mj_forward(m,d)
    errors=[]
    for b in range(1,m.nbody):
        name=m.body(b).name
        usd_name=re.sub(r'[^A-Za-z0-9_]','_',name)
        if usd_name not in poses:raise ValueError('Missing live link: '+name)
        actual=np.array(poses[usd_name])
        distance=float(np.linalg.norm(actual[:3]-d.xpos[b]))
        dot=float(abs(np.dot(actual[3:]/np.linalg.norm(actual[3:]),d.xquat[b])))
        angle=float(np.degrees(2*np.arccos(np.clip(dot,-1,1))))
        errors.append(dict(body=name,position_m=distance,rotation_deg=angle))
    mass_error=abs(config['robot_mass_kg']-float(m.body_mass.sum()))
    result=dict(passed=all(x['position_m']<.002 and x['rotation_deg']<.5 for x in errors) and mass_error<1e-4,
                position_tolerance_m=.002,rotation_tolerance_deg=.5,mass_error_kg=mass_error,
                max_position_m=max(x['position_m'] for x in errors),max_rotation_deg=max(x['rotation_deg'] for x in errors),
                bodies=errors)
    (run/'kinematics-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--robot',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();r=check(a.robot,a.run)
    print(json.dumps({k:v for k,v in r.items() if k!='bodies'},indent=2))
    raise SystemExit(0 if r['passed'] else 1)

if __name__=='__main__':main()
