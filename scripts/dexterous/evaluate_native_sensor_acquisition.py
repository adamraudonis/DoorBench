"""Independently score recorded native acquisition states with robot-only FK.

Actual state is evaluator-only; no policy runs and no physics is stepped.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco,re
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.sensor_acquisition_evidence import RAW_SCHEMA
from doorbench.dexterous.sensor_acquisition_evaluation import AcquisitionEvaluationRobot,evaluate_sensor_acquisition_balance


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):
    with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','robot','calibration','protocol','joint-route','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve earlier independent evidence')
    run=a.run;motors=json.loads((run/'motors.json').read_text())
    model=AcquisitionEvaluationRobot(a.robot,motors,a.calibration,a.protocol,a.joint_route)
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
    prov=json.loads((run/'provenance.json').read_text());door=Path(prov['parameters']['door']);door=door if door.is_dir() else door.parent
    if sha(a.robot)!=prov['robot_xml_sha256'] or sha(door/'door.xml')!=prov['door_xml_sha256']:raise ValueError('Native scene source bytes changed')
    scene=DexterousDoorEnv(door,a.robot,json.loads(a.robot.with_suffix('.audit.json').read_text()),frame_skip=1)
    m=scene.m;detached=mujoco.MjData(m);lever=m.geom('leaf_handle_lever_col_n').id
    raw_stream=gzip.open(run/'actual-transitions.jsonl.gz','rt')
    initial_door={name:float(q[m.joint(name).qposadr[0]]) for name in ('leaf_hinge','leaf_handle_hinge')}
    detached.qpos[:]=q;mujoco.mj_kinematics(m,detached);mujoco.mj_collision(m,detached)
    initial_hand_count=sum(c.dist<=0 and any(m.body(m.geom_bodyid[g]).name.startswith(('robot/rh_','robot/lh_')) for g in c.geom) for c in detached.contact[:detached.ncon])
    for i,(row,info) in enumerate(zip(physical,infos,strict=True)):
        r=root13(trajectory['qpos'][i+1],trajectory['qvel'][i+1])
        if (abs(r[2]-row['root_height_m'])>1e-8 or abs(np.linalg.norm(r[7:9])-row['actual_root_horizontal_speed_mps'])>1e-8 or
                abs(np.linalg.norm(r[10:13])-row['actual_root_angular_speed_radps'])>1e-8):raise ValueError('Native root frame mismatch')
        raw=json.loads(next(raw_stream));detached.qpos[:]=raw['qpos_before'];mujoco.mj_kinematics(m,detached)
        poses={};contact_patches=[];hand_count=bad_count=0
        for bid,pos,rotation in zip(raw['body_ids'],raw['body_positions_world_m'],raw['body_rotations_world'],strict=True):
            name=m.body(bid).name
            if not np.allclose(detached.xpos[bid],pos,atol=1e-9,rtol=0) or not np.allclose(detached.xmat[bid].reshape(3,3),rotation,atol=1e-9,rtol=0):raise ValueError('Raw native geometry differs from actual pre-step pose')
            if name.startswith('robot/rh_'):poses[name]=np.r_[pos,Rotation.from_matrix(rotation).as_quat()].tolist()
        for contact in raw['contacts']:
            bodies=[m.body(b).name for b in contact['body']];geoms=contact['geom'];force=max(0.,contact['wrench_contact_frame'][0])
            touching=contact['distance_m']<=0 or force>1e-6
            if any(n.startswith(('robot/rh_','robot/lh_')) for n in bodies) and touching:
                hand_count+=1
                if not (lever in geoms and any(re.fullmatch(r'robot/rh_(ff|mf|rf|lf|th).+',n) for n in bodies)):bad_count+=1
            if lever not in geoms:continue
            other=geoms[1] if geoms[0]==lever else geoms[0];name=m.body(m.geom_bodyid[other]).name
            if not name.startswith('robot/rh_'):continue
            normal=(-1 if geoms[0]==other else 1)*np.array(contact['frame_world'])[0]
            contact_patches.append(dict(body=name,position=contact['position_world_m'],normal=normal.tolist(),normal_force_N=force))
        if hand_count!=row['left_and_right_hand_contacts'] or bad_count!=row['unintended_hand_contacts']:raise ValueError('Raw all-hand contact counts disagree')
        evidence=dict(schema=RAW_SCHEMA,interval_start_s=raw['interval_start_s'],interval_end_s=raw['interval_end_s'],geometry_time_s=raw['geometry_time_s'],
            clock='native-interval-start',scope='native-lever-collider',contacts=contact_patches,body_transforms_xyzw=poses,
            handle_pair_forces_world_N=None,contact_capacity=None,active_contact_count=len(raw['contacts']),normal_pair_force_consistency_error_N=None,
            lever=dict(center=detached.geom_xpos[lever].tolist(),axis=detached.geom_xmat[lever].reshape(3,3)[:,2].tolist(),half_length=float(m.geom_size[lever,1]),radius=float(m.geom_size[lever,0])))
        rows.append(dict(time_s=row['sim_time_s'],root13_actororigin=r.tolist(),torso_tilt_deg=row['torso_tilt_deg'],
            foot_floor_loads=row['actual_floor_support_normal_N'],hand_contact_count=row['left_and_right_hand_contacts'],controller_info=info,unintended_hand_contact_count=bad_count,pad_evidence=evidence,
            actual_joint_position={n:float(trajectory['qpos'][i+1,k]) for n,k in addresses.items()}))
    if next(raw_stream,None) is not None:raise ValueError('Additional unmatched raw contact intervals')
    raw_stream.close();scene.close()
    scored=evaluate_sensor_acquisition_balance(rows,original['checks'],robot_xml=a.robot,motors=motors,initial_root13_actororigin=initial_root,
        initial_joint_position=initial,initial_hand_contact_count=int(initial_hand_count),initial_door_position=initial_door,calibration=a.calibration,protocol=a.protocol,joint_route=a.joint_route)
    report=dict(scope=__doc__,source_run=str(run),independent_evaluation=scored,physics_steps_in_evaluator=0,
        robot_sha256=sha(a.robot),calibration_sha256=sha(a.calibration),protocol_sha256=sha(a.protocol),joint_route_sha256=sha(a.joint_route),script_sha256=sha(__file__),
        evaluator_sha256=sha(Path(__file__).parents[2]/'doorbench/dexterous/sensor_acquisition_evaluation.py'),
        source_sha256={name:sha(run/name) for name in ('report.json','provenance.json','reset.json','static-screen.json','trajectory.npz','physics.jsonl.gz','controller.jsonl.gz','actual-transitions.jsonl.gz')})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:scored[k] for k in ('passed','checks','errors','maximum_motor_coordinate_tracking_error_rad','maximum_nominal_joint_tracking_error_rad','palm_endpoint_error_m')}),flush=True)
if __name__=='__main__':main()
