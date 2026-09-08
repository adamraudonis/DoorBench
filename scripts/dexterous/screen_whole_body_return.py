#!/usr/bin/env python3
"""Screen a bounded H1 whole-body lever return from an exact archived state.

This creates and modifies only an unstepped geometry model. Root and foot poses
are not imposed on an active plant. Use the source package frozen with the run,
then independently validate interpolated geometry and physical motor execution.
"""
import argparse
from pathlib import Path
import sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-run',type=Path,required=True)
p.add_argument('--state-run',type=Path,required=True)
p.add_argument('--source-package',type=Path,required=True)
p.add_argument('--at',type=float,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
if args.output.exists():raise ValueError('Use a new evidence directory')
args.output.mkdir(parents=True)
sys.path.insert(0,str(args.source_package.resolve()))
import json,hashlib,importlib.util,numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.native_transition_archive import unpacked
baseline=args.source_run.resolve();conf=json.loads((baseline/'manifest.json').read_text())['configuration'];r=Path(conf['robot']);s=DexterousDoorEnv(conf['door'],r,json.loads(r.with_suffix('.audit.json').read_text()));m,d=s.m,s.d
manifest=json.loads((args.state_run/'raw-transitions/manifest.json').read_text());time=args.at
chunk=next(c for c in manifest['chunks'] if c['interval_start_s']-1e-8<=time<c['interval_end_s']-1e-8);file=args.state_run/'raw-transitions'/chunk['file'];assert hashlib.file_digest(file.open('rb'),'sha256').hexdigest()==chunk['sha256']
with np.load(file) as a:rows=list(unpacked({k:a[k] for k in a.files}))
row=min(rows,key=lambda r:abs(r['interval_start_s']-time))
if abs(row['interval_start_s']-time)>1e-8:raise ValueError('Require the exact requested attained interval')
from doorbench.dexterous.whole_body_return_planner import copy_attained_return_state, iter_whole_body_return
state=dict(qpos=row['qpos_before'],qvel=row['qvel_before'],time_s=row['interval_start_s'],
           geometry_time_s=row['geometry_time_s'],body_ids=row['body_ids'],
           body_positions_world_m=row['body_positions_world_m'],body_rotations_world=row['body_rotations_world'],observation_scope='archived-native-contact-bodies')
_,state_sha=copy_attained_return_state(m,**state)
results=[]
for result in iter_whole_body_return(m,**state):
 results.append(result)
 print(json.dumps({k:v for k,v in result.items() if k not in ('qpos','joints','joint_margins_rad','forbidden_collisions')}|{'forbidden_collision_count':len(result['forbidden_collisions'])}),flush=True)
out=args.output;(out/'report.json').write_text(json.dumps(dict(scope='Unstepped whole-body IK; actual landed feet fixed; no dynamics qualification; original limits plus0.01rad planning margin',input_files={str(Path(p)/'door.xml' if Path(p).is_dir() else Path(p)):hashlib.file_digest((Path(p)/'door.xml' if Path(p).is_dir() else Path(p)).open('rb'),'sha256').hexdigest() for p in (conf['robot'],conf['door'])},source_time_s=row['interval_start_s'],source_raw_chunk_sha256=chunk['sha256'],source_state_sha256=state_sha,observation_scope=state['observation_scope'],mujoco_version=mujoco.__version__,results=results),indent=2));s.close()
