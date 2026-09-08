#!/usr/bin/env python3
"""A receipt is emitted only after independent live simulation checks pass."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();audit=json.loads((a.trial/'kinematics-audit.json').read_text());assert audit['passed']
rows=json.loads((a.trial/'trace.json').read_text());config=json.loads((a.trial/'configuration.json').read_text())
assert len(rows)>=45 and rows[-1]['time_s']>=.9
assert config['runtime_pose_writes']==0 and not config['direct_door_commands']
assert not (a.trial/'error.txt').exists()
assert min(r['root'][2] for r in rows)>.7 and max(r['torso_tilt_deg'] for r in rows)<20
assert max(abs(r['sim_time_s']-r['time_s']) for r in rows)<.003
assert len(config['simulator_effort_limits'])==69 and min(config['simulator_effort_limits'])>0
wrist=config['robot_joint_names'].index('right_wrist_yaw')
assert max(r['joints'][wrist] for r in rows)-rows[0]['joints'][wrist]>.05,'Commanded wrist motion did not occur'
assert all(max(abs(x-y) for x,y in zip(r['joint_torque_command'],r['joint_torque_sent']))<1e-4 for r in rows),'Motor commands were dropped before reaching PhysX'
receipt=dict(ready=True,verified_at_utc=datetime.now(timezone.utc).isoformat(),
    scope='Runtime, generated door QA, imported free robot, bounded motors, live standing physics and rendering. Not an opening-policy success.',
    trial=str(a.trial.resolve()),kinematics=audit,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True),
    checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.trial.iterdir() if p.suffix in ('.json','.mp4')})
a.output.write_text(json.dumps(receipt,indent=2)+'\n')
