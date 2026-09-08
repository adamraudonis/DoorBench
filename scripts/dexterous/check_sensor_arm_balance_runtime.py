"""Replay a native scripted-arm capture and independently score its real states.

No physics is stepped. Original states are used only by the evaluator; runtime
inference receives the original numeric sensor packet and frozen schedule.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from doorbench.dexterous.sensor_arm_balance_runtime import SensorArmBalanceRuntime,evaluate_sensor_arm_balance


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):
    with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','robot','calibration','schedule','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous replay evidence')
    run=a.run;motors=json.loads((run/'motors.json').read_text());layout=json.loads((run/'sensor-layout.json').read_text())
    runtime=SensorArmBalanceRuntime(a.robot,motors,layout,a.calibration,a.schedule);runtime.reset_episode()
    data=np.load(run/'actor-inputs.npz',allow_pickle=False);trajectory=np.load(run/'trajectory.npz',allow_pickle=False)
    physical=lines(run/'physics.jsonl.gz');infos=lines(run/'controller.jsonl.gz');original=json.loads((run/'report.json').read_text())
    errors=[];failure=None
    for i,row in enumerate(physical):
        try:
            packet={k:data[k][i].copy() for k in data.files}
            packet.update(rgb_left=np.zeros((128,128,3),np.uint8),rgb_right=np.zeros((128,128,3),np.uint8))
            if np.any(packet['sensor_valid'][-2:]):raise ValueError('This replay requires actual disabled RGB')
            force=runtime.force(packet,i*.002)
            errors.append(float(abs(force-trajectory['force'][i]).max()))
        except Exception as exc:failure=dict(decision=i,error=type(exc).__name__+': '+str(exc));break
    reset=json.loads((run/'reset.json').read_text());q=np.array(reset['qpos']);root=np.array(reset['root'])
    candidates=[i for i in range(len(q)-6) if np.array_equal(q[i:i+7],root)]
    if len(candidates)!=1 or len(reset['qvel'])!=len(q)-1:raise ValueError('Require the single native free-root mapping')
    adr=candidates[0];calculator=runtime._controller.balance
    arm_adr={n:adr+int(calculator.m.jnt_qposadr[calculator.m.joint(n).id]) for n in runtime.goal_names}
    initial={n:float(trajectory['qpos'][0,k]) for n,k in arm_adr.items()};rows=[]
    for i,(row,info) in enumerate(zip(physical,infos,strict=True)):
        qr=trajectory['qpos'][i+1,adr:adr+7];vr=trajectory['qvel'][i+1,adr:adr+6]
        rotation=Rotation.from_quat(qr[[4,5,6,3]]).as_matrix();root13=np.r_[qr,vr[:3],rotation@vr[3:6]]
        if (abs(root13[2]-row['root_height_m'])>1e-8 or abs(np.linalg.norm(root13[7:9])-row['actual_root_horizontal_speed_mps'])>1e-8 or
                abs(np.linalg.norm(root13[10:13])-row['actual_root_angular_speed_radps'])>1e-8):raise ValueError('Recorded native root convention mismatch')
        rows.append(dict(time_s=row['sim_time_s'],root13_actororigin=root13.tolist(),torso_tilt_deg=row['torso_tilt_deg'],
            foot_floor_loads=row['actual_floor_support_normal_N'],hand_contact_count=row['left_and_right_hand_contacts'],controller_info=info,
            actual_arm_joint_position={n:float(trajectory['qpos'][i+1,k]) for n,k in arm_adr.items()}))
    scored=evaluate_sensor_arm_balance(rows,original['checks'],initial_arm_joint_position=initial,schedule=a.schedule)
    report=dict(scope=__doc__,source_run=str(run),replay_passed=failure is None and len(errors)==len(physical) and max(errors)<1e-7,
        actual_original_run_passed=scored['passed'],maximum_force_difference_Nm=max(errors,default=None),replayed_decisions=len(errors),failure=failure,
        independent_original_evaluation=scored,physics_steps_in_replay=0,calculator_time_s=runtime.last_info.get('calculator_time_s'),
        robot_sha256=sha(a.robot),calibration_sha256=sha(a.calibration),schedule_sha256=sha(a.schedule),script_sha256=sha(__file__),
        source_sha256={name:sha(run/name) for name in ('report.json','provenance.json','reset.json','static-screen.json','actor-inputs.npz','trajectory.npz','physics.jsonl.gz','controller.jsonl.gz')})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('replay_passed','actual_original_run_passed','maximum_force_difference_Nm','failure')}),flush=True)
if __name__=='__main__':main()
