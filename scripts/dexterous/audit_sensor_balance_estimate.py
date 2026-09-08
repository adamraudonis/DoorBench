#!/usr/bin/env python3
"""Withheld-root audit and exact sensor-only replay of a native balance episode.

True robot poses are evaluator inputs only. No true state enters the controller
replay, whose only runtime input is the frozen numeric sensor packet and clock.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.spatial.transform import Rotation
from doorbench.dexterous.sensor_balance import SensorBalanceController


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_rows(path):return [json.loads(s) for s in gzip.open(path,'rt')]


def audit(trial):
    trial=Path(trial);prov=json.loads((trial/'provenance.json').read_text());report=json.loads((trial/'report.json').read_text())
    config=json.loads((trial/'calibration.json').read_text());reset=json.loads((trial/'reset.json').read_text())
    source=Path(__file__).resolve().parents[2]/'doorbench/dexterous/sensor_balance.py'
    if sha(source)!=prov['sources']['doorbench/dexterous/sensor_balance.py']:raise ValueError('Replay controller source differs from the frozen physical run')
    state=np.load(trial/'trajectory.npz');packets=np.load(trial/'actor-inputs.npz');info=read_rows(trial/'controller.jsonl.gz');physics=read_rows(trial/'physics.jsonl.gz')
    qpos=state['qpos'];times=state['time'];n=len(info)
    if len(physics)!=n or len(qpos)!=n+1 or len(state['force'])!=n or len(times)!=n+1 or any(len(v)!=n for v in packets.values()):raise ValueError('Sensor/physics decision archive lengths differ')
    if np.max(abs(np.diff(times)-.002))>1e-10 or abs(times[-1]-report['duration_s'])>1e-9:raise ValueError('Unaligned physical clock')
    # Locate the reset's unique seven-number free-root block in the complete
    # state archive; never pass this block to the controller below.
    roots=[i for i in range(qpos.shape[1]-6) if np.array_equal(qpos[0,i:i+7],reset['root'])]
    if len(roots)!=1:raise ValueError('Root reset block is not uniquely bound to the physical archive')
    adr=roots[0];actual=qpos[:-1,adr:adr+7]
    true_rot=Rotation.from_quat(actual[:,[4,5,6,3]])
    yaw=Rotation.from_euler('z',true_rot[0].as_euler('ZYX')[0])
    local_true=yaw.inv()*true_rot
    estimated=np.array([v['estimated_root_local'] for v in info])
    estimate_rot=Rotation.from_quat(estimated[:,[4,5,6,3]])
    attitude=np.rad2deg((local_true.inv()*estimate_rot).magnitude())
    # XY is a relative local gauge; Z is inferred using the calibrated sole plane.
    position=yaw.inv().apply(actual[:,:3]-actual[0,:3]);position[:,2]=actual[:,2]
    position_error=np.linalg.norm(position-estimated[:,:3],axis=1)
    c=SensorBalanceController(Path(prov['parameters']['robot']),json.loads((trial/'motors.json').read_text()),json.loads((trial/'sensor-layout.json').read_text()),config['desired_posture'],gravity_correction=prov['parameters']['gravity_correction'])
    force_error=0.;replayed_estimate_error=0.
    for i in range(n):
        packet={key:packets[key][i].copy() for key in packets.files}
        if packet['sensor_valid'][5:].any():raise ValueError('RGB values were omitted; cannot replay valid RGB')
        packet['rgb_left']=np.zeros(c.shapes['rgb_left'],np.uint8);packet['rgb_right']=np.zeros(c.shapes['rgb_right'],np.uint8)
        force,entry=c.force(packet,now_s=float(times[i]))
        force_error=max(force_error,float(np.max(abs(force-state['force'][i]))))
        replayed_estimate_error=max(replayed_estimate_error,float(np.max(abs(np.asarray(entry['estimated_root_local'])-estimated[i]))))
    checks=dict(physical_trial_passed=report['passed'],all_decisions_replayed=c.ticks==n,
        no_calculator_time_advance=c.d.time==0.,same_packet_forces=force_error<1e-8,
        same_sensor_estimates=replayed_estimate_error<1e-10,
        actual_timestamps_match=all(abs(r['measurement_pose_time_s']-times[i+1])<1e-9 and abs(r['contact_geometry_time_s']-times[i])<1e-9 for i,r in enumerate(physics)))
    return dict(scope=__doc__,passed=all(checks.values()),checks=checks,decisions=n,
        root_state_block_index=adr,max_orientation_error_deg=float(attitude.max()),max_position_error_m=float(position_error.max()),
        final_orientation_error_deg=float(attitude[-1]),final_position_error_m=float(position_error[-1]),
        maximum_replayed_motor_difference_Nm=force_error,maximum_replayed_estimate_difference=replayed_estimate_error,
        final_second_horizontal_speed_max_mps=max(r['actual_root_horizontal_speed_mps'] for r in physics[-500:]),
        final_second_per_foot_support_min_N=min(min(r['actual_floor_support_normal_N']) for r in physics[-500:]),
        input_sha256={f:sha(trial/f) for f in ('report.json','provenance.json','trajectory.npz','actor-inputs.npz','physics.jsonl.gz','controller.jsonl.gz','reset.json','calibration.json','motors.json','sensor-layout.json')},
        auditor_source_sha256=sha(__file__))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh output receipt required')
    result=audit(a.trial);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
