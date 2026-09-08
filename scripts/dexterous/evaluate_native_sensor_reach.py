"""Independently score recorded native reach states with robot-only FK.

Actual state is evaluator-only; no policy runs and no physics is stepped.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from doorbench.dexterous.sensor_reach_evaluation import ReachEvaluationRobot,evaluate_sensor_reach_balance


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):
    with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','robot','calibration','protocol','joint-route','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve earlier independent evidence')
    run=a.run;motors=json.loads((run/'motors.json').read_text())
    model=ReachEvaluationRobot(a.robot,motors,a.calibration,a.protocol,a.joint_route)
    with np.load(run/'trajectory.npz',allow_pickle=False) as saved:
        trajectory={name:saved[name] for name in ('qpos','qvel')}
    physical=lines(run/'physics.jsonl.gz');infos=lines(run/'controller.jsonl.gz');original=json.loads((run/'report.json').read_text())
    reset=json.loads((run/'reset.json').read_text());q=np.array(reset['qpos']);root=np.array(reset['root'])
    candidates=[i for i in range(len(q)-6) if np.array_equal(q[i:i+7],root)]
    if len(candidates)!=1 or len(reset['qvel'])!=len(q)-1:raise ValueError('Require single actual free-root mapping')
    adr=candidates[0];addresses={n:adr+int(model.m.jnt_qposadr[model.m.joint(n).id]) for n in model.names}
    def root13(q,v):
        qr=q[adr:adr+7];vr=v[adr:adr+6];r=Rotation.from_quat(qr[[4,5,6,3]]).as_matrix()
        return np.r_[qr,vr[:3],r@vr[3:6]]
    if not np.array_equal(trajectory['qpos'][0],q) or not np.array_equal(trajectory['qvel'][0],reset['qvel']):raise ValueError('Recorded initial state differs from reset')
    initial={n:float(q[k]) for n,k in addresses.items()};initial_root=root13(q,np.array(reset['qvel']));rows=[]
    for i,(row,info) in enumerate(zip(physical,infos,strict=True)):
        r=root13(trajectory['qpos'][i+1],trajectory['qvel'][i+1])
        if (abs(r[2]-row['root_height_m'])>1e-8 or abs(np.linalg.norm(r[7:9])-row['actual_root_horizontal_speed_mps'])>1e-8 or
                abs(np.linalg.norm(r[10:13])-row['actual_root_angular_speed_radps'])>1e-8):raise ValueError('Native root frame mismatch')
        rows.append(dict(time_s=row['sim_time_s'],root13_actororigin=r.tolist(),torso_tilt_deg=row['torso_tilt_deg'],
            foot_floor_loads=row['actual_floor_support_normal_N'],hand_contact_count=row['left_and_right_hand_contacts'],controller_info=info,
            actual_joint_position={n:float(trajectory['qpos'][i+1,k]) for n,k in addresses.items()}))
    scored=evaluate_sensor_reach_balance(rows,original['checks'],robot_xml=a.robot,motors=motors,initial_root13_actororigin=initial_root,
        initial_joint_position=initial,calibration=a.calibration,protocol=a.protocol,joint_route=a.joint_route)
    report=dict(scope=__doc__,source_run=str(run),independent_evaluation=scored,physics_steps_in_evaluator=0,
        robot_sha256=sha(a.robot),calibration_sha256=sha(a.calibration),protocol_sha256=sha(a.protocol),joint_route_sha256=sha(a.joint_route),script_sha256=sha(__file__),
        evaluator_sha256=sha(Path(__file__).parents[2]/'doorbench/dexterous/sensor_reach_evaluation.py'),
        source_sha256={name:sha(run/name) for name in ('report.json','provenance.json','reset.json','static-screen.json','trajectory.npz','physics.jsonl.gz','controller.jsonl.gz')})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:scored[k] for k in ('passed','checks','errors','maximum_motor_coordinate_tracking_error_rad','maximum_nominal_joint_tracking_error_rad','palm_endpoint_error_m')}),flush=True)
if __name__=='__main__':main()
