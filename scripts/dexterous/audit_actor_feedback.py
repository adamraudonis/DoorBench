"""Audit a failed actor's valid correction prefix and local input sensitivities.

All sensor trajectories remain recorded. Jacobians describe one inference at a
frozen actual-history state, not a plant stability margin or a physical recovery.
No weights are changed and no simulator is stepped.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from doorbench.dexterous.correction_demonstrations import CorrectionDemonstration
from doorbench.dexterous.sensor_actor import prepare_actor_packet
from doorbench.dexterous.sensor_policy_controller import SensorPolicyController
from doorbench.dexterous.offline_teacher_queries import sha
from doorbench.dexterous.sensor_contract import SENSOR_KEYS


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('corrections','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(1)
 if a.output.exists():raise FileExistsError('Retain previous feedback evidence')
 data=CorrectionDemonstration(a.corrections);run=data.path
 motors=json.loads((run/'motor-contract.json').read_text());names=data.layout['action_order'];joints=data.layout['joint_order']
 caps=np.array([x['force_range'] for x in motors['actuators']]);half=(caps[:,1]-caps[:,0])/2
 body=np.array([i for i,n in enumerate(names) if not n.startswith(('rh_','lh_'))]);leg=np.array([i for i,n in enumerate(names) if any(s in n for s in ('hip','knee','ankle'))])
 actor=SensorPolicyController(a.checkpoint,motor_contract=motors,sensor_layout=data.layout,physics_dt_s=.002);actor.reset_episode()
 physical=np.load(run/'acquisition-physics.npz',allow_pickle=False)
 labels=np.load(a.corrections/'counterfactual-teacher-actions.npz',allow_pickle=False)['motor_force'][:len(data)]
 step_queries=json.loads((a.corrections/'queries.json').read_text())
 forces=[];sensitivities=[];hidden=None
 selected={int(round(t/.002)) for t in (.0,.02,.05,.1,.2,.28)}
 for i,t in enumerate(data.times):
  packet=data.packet(i)
  # Exact recorded prior command, with continuous hidden state, reproduces the live actor.
  values=prepare_actor_packet(packet,float(t),actor.dimensions)
  values={k:torch.as_tensor(v)[None,None] for k,v in values.items()}
  m=actor._actor
  if i in selected:
   fixed_h=None if hidden is None else hidden.clone().detach()
   with torch.no_grad():
    vision=m.vision(values['images'].reshape(2,3,128,128)).reshape(1,1,128)
    touch=m.touch(values['tactile'])
   def output(z):
    features=torch.cat((vision,touch,m.proprio(z.reshape(1,1,-1))),dim=-1)
    sequence,_=m.memory(features,fixed_h)
    return m.action(sequence)[0,0]*torch.tensor(half,dtype=torch.float32)
   z=values['proprio'].flatten().detach().requires_grad_(True)
   jac=torch.autograd.functional.jacobian(output,z).detach().numpy()
   start=2*len(joints)+6
   for key,sl,scale in [('joint_position',slice(0,69),np.pi),('joint_velocity',slice(69,138),10.),('imu_gyro',slice(138,141),10.),('imu_accelerometer',slice(141,144),20.)]:
    mask=(abs(packet[key]/scale)<5.) & packet['sensor_valid'][SENSOR_KEYS.index(key)]
    jac[:,sl]*=mask[None,:]
   command_jac=jac[:,start:start+61]/half[None,:]
   rr,cc=np.unravel_index(np.argmax(abs(command_jac[np.ix_(body,body)])),(len(body),len(body)))
   rr,cc=body[rr],body[cc];epsilon=.02
   plus=z.detach().clone();minus=plus.clone();plus[start+cc]+=epsilon/half[cc];minus[start+cc]-=epsilon/half[cc]
   with torch.no_grad():fd=float((output(plus)[rr]-output(minus)[rr])/(2*epsilon))
   fd_error=abs(fd-command_jac[rr,cc])
   if fd_error>.01:raise AssertionError('Previous-command Jacobian disagrees with finite difference')
   def top(matrix,row_names,col_names,k=8):
    pairs=np.dstack(np.unravel_index(np.argsort(-abs(matrix).ravel())[:k],matrix.shape))[0]
    return [dict(output=row_names[r],input=col_names[c],gain=float(matrix[r,c])) for r,c in pairs]
   groups=dict(joint_position=(slice(0,69),np.pi,joints),joint_velocity=(slice(69,138),10.,joints),imu_gyro=(slice(138,141),10.,['x','y','z']),imu_accelerometer=(slice(141,144),20.,['x','y','z']))
   direct_joint_feedback={n:dict(position_gain_Nm_per_rad=float(jac[names.index(n),joints.index(n)]/np.pi),velocity_gain_Nm_per_rad_s=float(jac[names.index(n),69+joints.index(n)]/10.)) for n in [names[k] for k in body]}
   sensitivities.append(dict(time_s=float(t),direct_same_joint_feedback=direct_joint_feedback,finite_difference_previous_action_max_gain_error=fd_error,previous_motor_force_one_step_singular_value=float(np.linalg.svd(command_jac[np.ix_(body,body)],compute_uv=False)[0]),previous_motor_force_feedback=top(command_jac[np.ix_(body,body)],[names[n] for n in body],[names[n] for n in body]),sensor_derivative={name:dict(units='Nm per '+{'joint_position':'rad','joint_velocity':'rad/s','imu_gyro':'rad/s','imu_accelerometer':'m/s2'}[name],largest_body_gains=top(jac[body,s]/scale,[names[n] for n in body],cols)) for name,(s,scale,cols) in groups.items()},sensor_valid=packet['sensor_valid'].tolist(),gyro=packet['imu_gyro'].tolist(),accelerometer=packet['imu_accelerometer'].tolist()))
  with torch.no_grad():action,hidden=m(**values,hidden=hidden)
  forces.append((caps[:,0]+(action[0,0].numpy()+1)*half))
 forces=np.asarray(forces);actual=physical['motor_forces'][:len(data)];error=actual-labels
 bands=[]
 for lo,hi in [(0,.062),(.064,.1),(.102,.2),(.202,float(data.times[-1]))]:
  ids=np.flatnonzero((data.times>=lo-1e-9)&(data.times<=hi+1e-9));e=error[ids];order=np.argsort(-np.mean(e[:,body]**2,axis=0))[:8]
  bands.append(dict(first_time_s=float(data.times[ids[0]]),last_time_s=float(data.times[ids[-1]]),examples=len(ids),body_force_rmse_Nm=float(np.sqrt(np.mean(e[:,body]**2))),leg_force_rmse_Nm=float(np.sqrt(np.mean(e[:,leg]**2))),dominant_motors=[dict(motor=names[body[k]],signed_error_Nm=float(np.mean(e[:,body[k]])),rmse_Nm=float(np.sqrt(np.mean(e[:,body[k]]**2))),actual_mean_Nm=float(np.mean(actual[ids,body[k]])),teacher_mean_Nm=float(np.mean(labels[ids,body[k]]))) for k in order]))
 report=dict(scope=__doc__,checkpoint_sha256=sha(a.checkpoint),actual_actor_report_sha256=sha(run/'actor-report.json'),source_correction_report_sha256=sha(a.corrections/'report.json'),script_sha256=sha(Path(__file__)),examples=len(data),last_time_s=float(data.times[-1]),recorded_actor_replay_max_absolute_error_Nm=float(abs(forces-actual).max()),corrective_error_bands=bands,local_input_sensitivities=sensitivities,teacher_query_first_invalid=next(r for r in step_queries if not r['candidate_label_valid']),physics_steps=0,optimizer_updates=0,limitations=['Counterfactual corrective forces are analytic, not executed recovery.','One-step derivatives hold all other inputs and GRU history fixed; they do not establish closed-loop stability or causal importance.','Derivative input scales follow physical units; compare gains only within matching units.','Recorded-command replay may differ from CUDA by floating point roundoff.'])
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ('examples','last_time_s','recorded_actor_replay_max_absolute_error_Nm','corrective_error_bands')}),flush=True)
if __name__=='__main__':main()
