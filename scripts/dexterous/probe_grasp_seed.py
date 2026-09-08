#!/usr/bin/env python3
"""Short free-body physics test from an offline seed, not a task rollout."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.provenance import capture
from doorbench.dexterous.contact_audit import lever_contacts


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--robot',default='out/dexterous/robot/h1-shadow.xml')
    p.add_argument('--door',default='out/dexterous/assets/doors/db0055_swing_single')
    p.add_argument('--seed-pose',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bias-feedforward',action='store_true')
    p.add_argument('--preload-json',type=Path);p.add_argument('--seconds',type=float,default=1.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    capture(Path(__file__).resolve().parents[2],a.output,vars(a))
    robot=Path(a.robot);env=DexterousDoorEnv(a.door,robot,json.loads(robot.with_suffix('.audit.json').read_text()))
    seed=np.load(a.seed_pose)['qpos'];trials=[]
    try:
        delta=json.loads(a.preload_json.read_text())['thumb_delta_and_curl_rad'] if a.preload_json else None
        for preload in ([delta[5]] if delta else (0.,.05,.10,.20)):
            env.reset(randomize=False,images=False)
            env.d.qpos[:]=seed;env.d.qvel[:]=0;mujoco.mj_forward(env.m,env.d)
            controls=env.d.actuator_length[env.actuators].copy()
            for i,aid in enumerate(env.actuators):
                name=env.m.actuator(aid).name
                if name.startswith('robot/rh_') and any(k in name for k in ('FFJ0','MFJ0','RFJ0','LFJ0')):
                    controls[i]+=preload
                if delta and name.startswith("robot/rh_A_THJ"):
                    controls[i]+=delta[(5,4,3,2,1).index(int(name[-1]))]
            action=env.normalize(np.clip(controls,env.low,env.high));trace=[];poses=[];velocities=[];ctrl=[]
            for step in range(round(a.seconds/.02)):
                if a.bias_feedforward:
                    adjusted=controls.copy()
                    for local,aid in enumerate(env.actuators):
                        if int(env.m.actuator_trntype[aid])==int(mujoco.mjtTrn.mjTRN_JOINT):
                            joint=env.m.actuator_trnid[aid,0];gain=env.m.actuator_gainprm[aid,0]
                            if gain>0:adjusted[local]+=env.d.qfrc_bias[env.m.jnt_dofadr[joint]]/gain
                    action=env.normalize(np.clip(adjusted,env.low,env.high))
                env.step(action,images=False)
                contact=lever_contacts(env.m,env.d,'leaf_handle_lever_col_n')
                diag=env.diagnostics()
                trace.append({'time':float(env.d.time),**contact,**diag})
                poses.append(env.d.qpos.copy());velocities.append(env.d.qvel.copy());ctrl.append(env.d.ctrl.copy())
                if diag['root_height_m']<.5 or not diag['finite'] or diag['numerical_warnings']:break
            row={'preload_rad':preload,'thumb_delta_rad':delta[:5] if delta else None,'opposed_steps':sum(x['opposed'] for x in trace),
                 'steps':len(trace),'max_loaded_digits':max(sum(v>=.2 for v in x['digit_forces_N'].values()) for x in trace),
                 'max_operator_angle_rad':float(max(q[env.m.jnt_qposadr[env.m.joint('leaf_handle_hinge').id]] for q in poses)),
                 'final_root_height_m':trace[-1]['root_height_m']}
            trials.append(row);print(json.dumps(row),flush=True)
            name=f'preload-{preload:.2f}'
            (a.output/(name+'.json')).write_text(json.dumps(trace)+'\n')
            np.savez_compressed(a.output/(name+'.npz'),qpos=np.array(poses),qvel=np.array(velocities),ctrl=np.array(ctrl))
        report={'stage':'free-body contact probe from initialized grasp pose','door_opening_claim':False,
                'qualifying_start':False,'external_force_assistance':False,'bias_feedforward_through_bounded_motors':a.bias_feedforward,'trials':trials,
                'limitations':['Starts near the handle in an offline fitted pose','Constant joint targets, not a learned opening policy',f'{a.seconds:g}-second probe cannot establish general grasp or balance reliability']}
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:env.close()

if __name__=='__main__':main()
