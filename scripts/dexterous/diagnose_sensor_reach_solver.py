"""Counterfactual solver replay diagnostics; no runtime policy or physics changes."""
from pathlib import Path
import hashlib,json,gzip,argparse,inspect
import osqp,mujoco
import numpy as np
from doorbench.dexterous.sensor_reach_runtime import SensorReachBalanceRuntime
from doorbench.dexterous.sensor_actor import ActorDimensions
from replay_isaac_sensor_balance import arrays,decision_packet
p=argparse.ArgumentParser(description='Diagnose strict Isaac reach replay with original and explicit adaptive-rho intervals; no physical simulation')
for name in ('run','robot','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();root=a.run;out=a.output;robot=a.robot
if out.exists():raise FileExistsError('Preserve earlier numerical diagnostics')
motors=json.loads((root/'motor-contract.json').read_text());layout=json.loads((root/'sensors/layout.json').read_text())
initial=arrays(root/'sensors/actor-initial-decision.npz');numeric=arrays(root/'sensors/actor-sensors.npz');rgb=arrays(root/'sensors/actor-rgb.npz');physical=arrays(root/'acquisition-physics.npz')
with gzip.open(root/'balance-steps.json.gz','rt') as f:steps=json.load(f)
assert len(steps)==5500 and len(numeric['time_s'])==5500 and physical['motor_forces'].shape==(5500,61)
results=[]
for interval in (None,25):
 controller=SensorReachBalanceRuntime(robot,motors,layout,root/'sensor-balance-calibration.json',root/'balance-reach-protocol.json',root/'balance-reach-route.json')
 controller.reset_episode()
 if interval is not None:controller._controller.balance.sim.stance_solver_settings['adaptive_rho_interval']=interval
 errors=[];failure=None;first=None;last=None
 for i in range(len(steps)):
  try:
   packet,t=decision_packet(i,initial,numeric,rgb,ActorDimensions(tactile=layout['tactile_dimension']))
   force=controller.force(packet,t);error=float(np.max(abs(force-physical['motor_forces'][i])))
   info=controller.last_info;actual=steps[i]['controller_info'];errors.append(error)
   last=dict(decision=i,time_s=t,error_Nm=error,replay_qp=info['qp_solver'],actual_qp=actual['qp_solver'],estimated_root_error_m=float(np.max(abs(np.array(info['estimated_root_local'])-actual['estimated_root_local']))),estimated_velocity_error=float(np.max(abs(np.array(info['estimated_velocity_local'])-actual['estimated_velocity_local']))))
   if first is None:first=last
   if i%1000==0:print(json.dumps(dict(interval=interval,decision=i,error_Nm=error)),flush=True)
  except Exception as exc:
   failure=dict(decision=i,time_s=i*.002,error=type(exc).__name__+': '+str(exc));break
 results.append(dict(adaptive_rho_interval=interval,replayed_decisions=len(errors),maximum_error_Nm=max(errors,default=None),first=first,last=last,failure=failure,calculator_time_s=controller.last_info.get('calculator_time_s')))
receipt=dict(scope='Detached numerical diagnostic on frozen actual Isaac reach packets. Explicit interval is a counterfactual solver setting, not the original qualified controller or a new physical trial. Strict actual previous-command ownership stays enforced; no command/history overwrites.',physical_steps=0,results=results,run=str(root),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),versions=dict(numpy=np.__version__,osqp=osqp.__version__,mujoco=mujoco.__version__),input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [robot,root/'balance-steps.json.gz',root/'acquisition-physics.npz',root/'sensors/actor-sensors.npz',root/'sensors/actor-rgb.npz',root/'sensors/actor-initial-decision.npz',root/'sensor-balance-calibration.json',root/'balance-reach-protocol.json',root/'balance-reach-route.json',Path(inspect.getfile(SensorReachBalanceRuntime))]})
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(output=str(out),summary=[{k:r[k] for k in ('adaptive_rho_interval','replayed_decisions','maximum_error_Nm','failure')} for r in results])),flush=True)
