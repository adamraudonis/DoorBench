#!/usr/bin/env python3
"""Compare causal local touch, qualified loads and motor mechanics at a stall."""
import argparse,gzip,hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from doorbench.dexterous.grasp_verification import scalar_transmission_matrix
from doorbench.dexterous.native_transition_archive import NativeTransitionArchive


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh diagnostic receipt required')
    prov=json.loads((a.trial/'provenance.json').read_text());layout=json.loads((a.trial/'sensor-layout.json').read_text())
    m=mujoco.MjModel.from_xml_path(prov['parameters']['robot']);names=[m.joint(i).name for i in range(1,m.njnt)];actions=[m.actuator(i).name for i in range(m.nu)]
    matrix=scalar_transmission_matrix(m,np.arange(m.nu),np.arange(1,m.njnt));state=np.load(a.trial/'trajectory.npz');packets=np.load(a.trial/'actor-inputs.npz')
    rows=[json.loads(s) for s in gzip.open(a.trial/'physics.jsonl.gz','rt')];info=[json.loads(s) for s in gzip.open(a.trial/'controller.jsonl.gz','rt')]
    reset=json.loads((a.trial/'reset.json').read_text());matches=[i for i in range(state['qpos'].shape[1]-6) if np.array_equal(state['qpos'][0,i:i+7],reset['root'])]
    if len(matches)!=1:raise ValueError('Ambiguous original root-state block')
    offset=matches[0];qa=m.jnt_qposadr[1:]+offset
    q=state['qpos'][1:,qa];coords=q@matrix.T;desired=[]
    for row in info:
        goal=np.array([reset['joints'][n] for n in names])
        for name,value in zip(row['goal_joint_names'],row['goal_joint_position_rad']):goal[names.index(name)]=value
        desired.append(matrix@goal)
    desired=np.asarray(desired);start=max(1,len(info)-500)
    if list(state['action_names'])!=actions:raise ValueError('Original motor order mismatch')
    delivered_forces=np.asarray([r['actuator_force'] for r in NativeTransitionArchive.read(a.trial/'actual-transitions')])
    if delivered_forces.shape!=state['force'].shape:raise ValueError('Actual motor-force array order or length mismatch')
    delivery_error=float(np.max(abs(delivered_forces-state['force'])))
    if delivery_error>1e-5:raise ValueError('Actual motor delivery differs from the issued force')
    causal=all(abs(float(packets['sensor_time_s'][i,4])-rows[i-1]['contact_interval_start_s'])<1e-5 for i in range(start,len(info)))
    digits={}
    for k,digit in enumerate(('ff','mf','rf','lf','th')):
        local=np.array([r['distal_projected_force_N'][k] for r in info[start:]])
        force=np.array([r['pad_grasp']['qualified_pad_forces_N'][digit] for r in rows[start-1:-1]])
        digits[digit]=dict(local_projection_mean_N=float(local.mean()),matching_interval_qualified_load_mean_N=float(force.mean()),projection_minus_normal_mean_N=float((local-force).mean()),projection_range_N=[float(local.min()),float(local.max())])
    motors={}
    for name in actions:
        if not name.startswith('rh_') or 'WRJ' in name:continue
        i=actions.index(name);error=desired[start:,i]-coords[start:,i];delivered=delivered_forces[start:,i]
        motors[name]=dict(requested_coordinate_mean_rad=float(desired[start:,i].mean()),actual_coordinate_mean_rad=float(coords[start:,i].mean()),mean_tracking_error_rad=float(error.mean()),maximum_tracking_error_rad=float(abs(error).max()),delivered_force_mean_Nm=float(delivered.mean()),maximum_abs_delivered_force_Nm=float(abs(delivered).max()),original_force_range_Nm=m.actuator_forcerange[i].tolist())
    result=dict(scope=__doc__,maximum_actual_motor_delivery_error_Nm=delivery_error,same_interval_sensor_load_alignment=causal,final_second_start_s=start*.002,digits=digits,motors=motors,
        virtual_clock_s=info[-1]['virtual_press_clock_s'],closing_offsets_rad=info[-1]['finger_motor_closure_offsets_rad'],
        encoder_tracking_ready_fraction=float(np.mean([r['encoder_press_tracking_ready'] for r in info[9500:]])),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_sha256={n:hashlib.sha256((a.trial/n).read_bytes()).hexdigest() for n in ('provenance.json','physics.jsonl.gz','controller.jsonl.gz','trajectory.npz','actor-inputs.npz','actual-transitions/manifest.json')})
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
