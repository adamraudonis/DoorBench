#!/usr/bin/env python3
"""Bind a nominal acquisition collision screen to this exact robot and door.

This never executes physics or establishes physical grasp success.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from scripts.dexterous.probe_sensor_acquisition_balance import screen_motion,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh screen output required')
    ref=json.loads(a.reference.read_text());names=ref['acquisition']['joint_names'];path=np.asarray(ref['acquisition']['path_qpos'])
    sim=DexterousDoorEnv(a.door,a.robot,json.loads(a.robot.with_suffix('.audit.json').read_text()));sim.reset(randomize=False,images=False)
    m,d=sim.m,sim.d;qa=np.array([m.joint('robot/'+n).qposadr[0] for n in names]);d.qpos[sim.root_qadr:sim.root_qadr+7]=ref['initial_root'];d.qpos[qa]=path[0];d.qvel[:]=0;mujoco.mj_forward(m,d)
    class Route:
        def values_at_fraction(self,u):
            x=u*(len(path)-1);i=min(int(x),len(path)-2);return (1-(x-i))*path[i]+(x-i)*path[i+1]
    result=screen_motion(sim,SimpleNamespace(goal_names=names),Route());result['input_sha256']={str(x):sha(x) for x in (a.robot,a.robot.with_suffix('.audit.json'),a.door/'door.xml',a.reference,Path(__file__))}
    a.output.mkdir(parents=True);(a.output/'geometry-audit.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'reference.json').write_bytes(a.reference.read_bytes());sim.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ('bad_samples','moving_joints')}));return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
