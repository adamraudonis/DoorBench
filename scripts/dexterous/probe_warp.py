#!/usr/bin/env python3
"""Fail-closed compatibility check of the unchanged tactile humanoid plant."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import mujoco
import mujoco_warp as mjw
import warp as wp
from doorbench.dexterous.environment import DexterousDoorEnv

p=argparse.ArgumentParser()
p.add_argument('--robot',default='out/dexterous/robot/h1-shadow.xml')
p.add_argument('--door',default='out/dexterous/assets/doors/db0055_swing_single')
p.add_argument('--output',default='out/dexterous/warp-compatibility.json')
a=p.parse_args();robot=Path(a.robot)
env=DexterousDoorEnv(a.door,robot,json.loads(robot.with_suffix('.audit.json').read_text()))
env.reset(images=False,randomize=False)
report={'checked_at_unix':time.time(),'mujoco':mujoco.__version__,
        'mujoco_warp':importlib.metadata.version('mujoco-warp'),'model_modified':False,
        'door_opening_claim':False,'plugins':env.m.nplugin,'actuators':env.m.nu}
try:
    wp.init()
    model=mjw.put_model(env.m)
    data=mjw.put_data(env.m,env.d,nworld=1)
    mjw.step(model,data);wp.synchronize()
    report.update(compiles_and_steps=True,verified_tactile_parity=False,
        production_ready=False,reason='A single step does not establish callback or tactile parity')
except Exception as exc:
    report.update(compiles_and_steps=False,production_ready=False,error=type(exc).__name__+': '+str(exc))
finally:
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));env.close()
