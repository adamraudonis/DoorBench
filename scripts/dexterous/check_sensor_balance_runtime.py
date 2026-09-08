"""Replay actual stationary-run packets through the adapter without physics.

The original run's physical evidence is scored separately. Replay neither
changes those states nor qualifies a new executed balance trajectory.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from doorbench.dexterous.sensor_balance_runtime import SensorBalanceRuntime,evaluate_sensor_balance


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):
    with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','robot','calibration','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Keep earlier adapter evidence')
    run=a.run;motors=json.loads((run/'motors.json').read_text());layout=json.loads((run/'sensor-layout.json').read_text())
    controller=SensorBalanceRuntime(a.robot,motors,layout,a.calibration);controller.reset_episode()
    data=np.load(run/'actor-inputs.npz',allow_pickle=False);trajectory=np.load(run/'trajectory.npz',allow_pickle=False)
    physical=lines(run/'physics.jsonl.gz');infos=lines(run/'controller.jsonl.gz');original=json.loads((run/'report.json').read_text())
    errors=[]
    for i,row in enumerate(physical):
        packet={k:data[k][i].copy() for k in data.files}
        packet.update(rgb_left=np.zeros((128,128,3),np.uint8),rgb_right=np.zeros((128,128,3),np.uint8))
        if np.any(packet['sensor_valid'][-2:]):raise ValueError('This replay requires the actual disabled RGB contract')
        t=0. if i==0 else physical[i-1]['sim_time_s']
        force=controller.force(packet,t)
        errors.append(float(abs(force-trajectory['force'][i]).max()))
    reset=json.loads((run/'reset.json').read_text());q=np.array(reset['qpos']);root=np.array(reset['root'])
    candidates=[i for i in range(len(q)-6) if np.array_equal(q[i:i+7],root)]
    if len(candidates)!=1 or len(reset['qvel'])!=len(q)-1:raise ValueError('Require the single native free-root mapping')
    adr=candidates[0];rows=[]
    for i,(row,info) in enumerate(zip(physical,infos,strict=True)):
        qr=trajectory['qpos'][i+1,adr:adr+7];vr=trajectory['qvel'][i+1,adr:adr+6]
        rotation=Rotation.from_quat(qr[[4,5,6,3]]).as_matrix()
        root13=np.r_[qr,vr[:3],rotation@vr[3:6]]
        if (abs(root13[2]-row['root_height_m'])>1e-8 or abs(np.linalg.norm(root13[7:9])-row['actual_root_horizontal_speed_mps'])>1e-8 or
                abs(np.linalg.norm(root13[10:13])-row['actual_root_angular_speed_radps'])>1e-8):raise ValueError('Native actual root mapping differs from recorded diagnostics')
        rows.append(dict(time_s=row['sim_time_s'],root13_actororigin=root13.tolist(),torso_tilt_deg=row['torso_tilt_deg'],
            foot_floor_loads=row['actual_floor_support_normal_N'],hand_contact_count=row['left_and_right_hand_contacts'],controller_info=info))
    result=evaluate_sensor_balance(rows,original['checks'])
    report=dict(scope=__doc__,source_run=str(run),calibration_sha256=sha(a.calibration),runtime_sha256=sha(Path(__file__).resolve().parents[2]/'doorbench/dexterous/sensor_balance_runtime.py'),
        controller_sha256=sha(Path(__file__).resolve().parents[2]/'doorbench/dexterous/sensor_balance.py'),examples=len(physical),
        maximum_replayed_force_difference_Nm=max(errors),replay_passed=max(errors)<1e-7,calculator_time_s=controller.last_info['calculator_time_s'],
        independent_original_run_evaluation=result,physics_steps_in_replay=0,
        source_sha256={name:sha(run/name) for name in ('report.json','provenance.json','actor-inputs.npz','trajectory.npz','physics.jsonl.gz','controller.jsonl.gz')})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(replay_passed=report['replay_passed'],max_force_error_Nm=max(errors),original_run_checks=result['checks'])),flush=True)
if __name__=='__main__':main()
