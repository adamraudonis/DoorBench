import importlib.util,json
from pathlib import Path
import numpy as np,mujoco
from doorbench.dexterous.environment import DexterousDoorEnv
spec=importlib.util.spec_from_file_location('doorbench.dexterous.grasp_verification_review','/tmp/doorbench-route-review/doorbench/dexterous/grasp_verification.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
root=Path('/tmp/doorbench-dexterous-humanoid/out/dexterous');s=DexterousDoorEnv(root/'assets/doors/db0055_swing_single',root/'robot/h1-shadow.xml',json.loads((root/'robot/h1-shadow.audit.json').read_text()));s.reset(images=False,randomize=False);m,d=s.m,s.d
m.actuator_gainprm[s.actuators,0]=1;m.actuator_biasprm[s.actuators,:3]=0;m.actuator_ctrlrange[s.actuators]=m.actuator_forcerange[s.actuators]
report=[]
for trial in ['/tmp/doorbench-acquisition-audit/initialized-six-force','/tmp/doorbench-isaac-integration/out/dexterous/acquisition-physics-013','/tmp/doorbench-standing-grasp/initialized-hold-001','/tmp/doorbench-ppo-transfer/thumb-first-001','/tmp/doorbench-ppo-transfer/thumb-first-002']:
 z=np.load(Path(trial)/'trajectory.npz');rows=[]
 for index in [0,len(z['qpos'])//2,len(z['qpos'])-1]:
  d.qpos[:]=z['qpos'][index];d.qvel[:]=z['qvel'][index];d.ctrl[:]=z['ctrl'][index];mujoco.mj_forward(m,d)
  result=v.shadow_lever_pad_grasp(m,d,'leaf_handle_lever_col_n');result['source_frame']=index
  rows.append(result)
  print(Path(trial).name,index,result['valid_pad_grasp'],result['qualified_pad_forces_N'],[(c['digit'],round(c['axial_clearance_m'],4),c['pad_qualified']) for c in result['contacts'] if c['normal_force_N']>.2])
 report.append(dict(source=trial,scope='Sampled stored states re-evaluated with native FK/contact solver; not a replacement for per-step rollout evidence',samples=rows))
mat=v.scalar_transmission_matrix(m,s.actuators,s.joints);print('NATIVE_MATRIX_ERROR',np.max(np.abs(mat@d.qpos[s.qadr]-d.actuator_length[s.actuators])))
Path('/tmp/doorbench-standing-grasp/strict-pad-review.json').write_text(json.dumps(report,indent=2)+'\n')
