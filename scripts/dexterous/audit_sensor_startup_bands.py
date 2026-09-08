"""Localize startup force errors using the actual sensor-only runtime boundary.

Predicted commands own their complete history from the actual reset. Teacher
forces remain labels; no simulator or robot state is advanced by this audit.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from doorbench.dexterous.sensor_actor import native_motor_forces
from doorbench.dexterous.sensor_policy_controller import SensorPolicyController
from doorbench.dexterous.sensor_training_bundle import load_bundle,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(1)
    episodes,_=load_bundle(args.dataset_manifest);e=episodes[0]
    if e.times[0]!=0 or e.metadata['physics_dt_s']!=.002 or len(e)<251:raise ValueError('Actual nominal500ms cold prefix required')
    motors=json.loads((e.path/'motor-contract.json').read_text());caps=np.array([a['force_range'] for a in motors['actuators']])
    names=e.layout['action_order'];body=[i for i,n in enumerate(names) if not n.startswith(('rh_','lh_'))]
    rows=[]
    bands=[('cold_prefix_0_to_before64ms',0,32),('64_through100ms',32,51),('after100_through500ms',51,251)]
    for checkpoint in args.checkpoint:
        actor=SensorPolicyController(checkpoint,motor_contract=motors,sensor_layout=e.layout,physics_dt_s=.002);actor.reset_episode();errors=[]
        for i in range(251):
            packet=e.packet(i);packet['previous_action']=actor.previous_action
            actual=actor.force(packet,float(e.times[i]));target=native_motor_forces(e.numeric['previous_action'][i+1],caps)
            errors.append(actual-target)
        err=np.asarray(errors);out=[]
        for name,start,end in bands:
            delta=err[start:end];squared=np.sum(delta[:,body]**2)
            out.append(dict(name=name,first_label=start,last_label=end-1,examples=end-start,
                first_time_s=float(e.times[start]),last_time_s=float(e.times[end-1]),
                body_force_rmse_Nm=float(np.sqrt(np.mean(delta[:,body]**2))),body_squared_error_Nm2=float(squared),
                body_motors={names[i]:dict(signed_mean_error_Nm=float(delta[:,i].mean()),rmse_Nm=float(np.sqrt(np.mean(delta[:,i]**2))),
                    max_absolute_error_Nm=float(np.max(np.abs(delta[:,i])))) for i in body}))
        row=dict(checkpoint=str(checkpoint),checkpoint_sha256=actor.checkpoint_sha256,bands=out,
            first100ms_body_force_rmse_Nm=float(np.sqrt(np.mean(err[:51,body]**2))),
            first500ms_body_force_rmse_Nm=float(np.sqrt(np.mean(err[:,body]**2))),
            fraction_first100ms_squared_error_in64_to100ms=out[1]['body_squared_error_Nm2']/(out[0]['body_squared_error_Nm2']+out[1]['body_squared_error_Nm2']))
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('checkpoint','bands')}|{'band_rmse_Nm':{b['name']:b['body_force_rmse_Nm'] for b in out}}),flush=True)
    result=dict(scope=__doc__,dataset_manifest_sha256=digest(args.dataset_manifest),source_sha256=digest(Path(__file__)),rows=rows,
        physical_state_steps=0,closed_loop_evaluated=False,teacher_commands_delivered=False,
        interval_contract='Cold-prefix objective supervises labels0..31 at0..62ms. The original100ms check uses labels0..50 inclusive;64..100ms therefore means labels32..50. Later band uses51..250, without duplicate boundary samples.',
        limitation='Frozen recorded sensors cannot react to actor predictions. This locates fitting errors, not physical recovery or causal proof that the loss weights cause instability.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
