"""Detached all-decision palm correction screen on native and Isaac recordings.

Actor inputs are recorded encoders and the existing previous-decision estimator
state. Actual robot/door states enter only the separate collision calculator.
No physics steps, model changes, hand forces or live plant writes occur.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
import mujoco,numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.palm_ground_reference import ARM_NAMES,GroundPalmReference,RobotPalmReferenceIK,project_palm_reference
from doorbench.dexterous.sensor_palm_reference import corrected_goals,PROTOCOL
from scripts.dexterous.probe_sensor_acquisition_balance import intended_contact_geometry


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_lines(path):
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','reference','native','isaac','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior candidate/screen evidence')
    a.output.mkdir(parents=True)
    for name in ('mj_step','mj_step1','mj_step2'):
        setattr(mujoco,name,lambda *args,**kw:(_ for _ in ()).throw(RuntimeError('No physics in a recorded-state screen')))
    reference=project_palm_reference(a.robot,a.reference)
    (a.output/'palm-reference.json').write_text(json.dumps(reference,indent=2)+'\n')
    (a.output/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2)+'\n')
    m=mujoco.MjModel.from_xml_path(str(a.robot));names=[m.joint(i).name for i in range(1,m.njnt)]
    ref=GroundPalmReference(reference,sha(a.robot));solver=RobotPalmReferenceIK(m,names)
    scene=DexterousDoorEnv(a.door,a.robot,json.loads(a.robot.with_suffix('.audit.json').read_text()),frame_skip=1)
    sm=scene.m;w=mujoco.MjData(sm);qa=np.array([sm.joint('robot/'+n).qposadr[0] for n in names])
    armids=[names.index(n) for n in ARM_NAMES];lever=sm.geom('leaf_handle_lever_col_n').id
    report=dict(schema='doorbench.recorded-palm-reference-screen.v1',scope=__doc__,passed=False,
        physics_steps=0,model_parameters_modified=False,protocol=PROTOCOL,results={},
        source_sha256={str(p):sha(p) for p in (a.robot,a.door/'door.xml',a.reference,Path(__file__))})
    for label,run in [('native',a.native),('isaac',a.isaac)]:
        if label=='native':
            infos=read_lines(run/'controller.jsonl.gz');saved=np.load(run/'trajectory.npz',allow_pickle=False)
            packets=np.load(run/'actor-inputs.npz',allow_pickle=False)
            states=saved['qpos'];encoders=packets['joint_position']
            record_paths=[run/n for n in ('controller.jsonl.gz','trajectory.npz','actor-inputs.npz')]
        else:
            with gzip.open(run/'balance-steps.json.gz','rt') as f:rows=json.load(f)
            infos=[r['controller_info'] for r in rows];packets=np.load(run/'sensors/actor-sensors.npz',allow_pickle=False)
            actual=np.load(run/'acquisition-physics.npz',allow_pickle=False);reset=json.loads((run/'balance-acquisition-reset.json').read_text())
            dnames=json.loads((run/'configuration.json').read_text())['door_joint_names']
            states=np.zeros((9501,sm.nq));states[0,scene.root_qadr:scene.root_qadr+7]=reset['root13_actororigin'][:7]
            states[0,qa]=[reset['joint_position'][n] for n in names]
            for i,r in enumerate(rows):
                states[i+1,scene.root_qadr:scene.root_qadr+7]=r['root13_actororigin'][:7]
                states[i+1,qa]=[r['actual_joint_position'][n] for n in names]
                for name,value in zip(dnames,actual['door'][i]):states[i+1,sm.joint(name).qposadr[0]]=value
            encoders=np.r_[np.zeros((1,69),np.float32),packets['joint_position'][:-1]]
            record_paths=[run/n for n in ('balance-steps.json.gz','acquisition-physics.npz','balance-acquisition-reset.json','sensors/actor-sensors.npz')]
        if len(infos)!=9500 or states.shape!=(9501,sm.nq) or encoders.shape!=(9500,69):raise ValueError('Complete original nineteen-second records required')
        report['source_sha256'].update({str(p):sha(p) for p in record_paths})
        delta=np.zeros(7);errors=[];bad=[];samples=[];max_joint=0.;max_pos=0.;max_rot=0.;max_pen=0.;maxrate=0.;rate_limited=0;collisions=0
        previous_goals=None
        for i in range(9500):
            t=i*.002;info=infos[i];nominal=dict(zip(info['goal_joint_names'],info['goal_joint_position_rad']))
            previous_root=None if i==0 else infos[i-1]['estimated_root_local']
            try:goals,new_delta,detail=corrected_goals(solver,ref,nominal,encoders[i],previous_root,t,delta)
            except Exception as exc:
                errors.append(dict(decision=i,time_s=t,error=type(exc).__name__+': '+str(exc)));break
            delta=new_delta;max_joint=max(max_joint,float(abs(delta).max()));rate_limited+=int(detail['correction_rate_limited'])
            if 'position_correction_m' in detail:max_pos=max(max_pos,float(np.linalg.norm(detail['position_correction_m'])))
            if 'rotation_correction_rad' in detail:max_rot=max(max_rot,float(np.linalg.norm(detail['rotation_correction_rad'])))
            current=np.array([goals[n] for n in info['goal_joint_names']])
            if previous_goals is not None:maxrate=max(maxrate,float(np.max(abs(current-previous_goals))/.002))
            previous_goals=current
            # Every2ms through contact; every20ms while hand-clear. This remains
            # a static screen, followed by mandatory actual500Hz qualification.
            if i%10==0 or t>=14.:
                for mode in ('measured_plus_correction','commanded_arm_pose'):
                    w.qpos[:]=states[i]
                    if mode=='measured_plus_correction':w.qpos[qa[armids]]+=delta
                    else:w.qpos[qa[armids]]=[goals[n] for n in ARM_NAMES]
                    mujoco.mj_kinematics(sm,w);mujoco.mj_collision(sm,w);collisions+=1
                    failures=[]
                    for c in w.contact[:w.ncon]:
                        bodies=[sm.body(sm.geom_bodyid[g]).name for g in c.geom];geoms=[sm.geom(g).name for g in c.geom]
                        hand=any(n.startswith(('robot/rh_','robot/lh_')) for n in bodies)
                        if any(n.startswith('robot/') for n in bodies) and not ('floor' in geoms and any(n.endswith('_ankle_link') for n in bodies)):
                            max_pen=max(max_pen,-float(c.dist))
                            if c.dist<-.003:failures.append(dict(kind='nonfoot_penetration',bodies=bodies,gap_m=float(c.dist)))
                        if hand and c.dist<=0 and not intended_contact_geometry(sm,w,c,lever):
                            failures.append(dict(kind='unqualified_hand_surface',bodies=bodies,gap_m=float(c.dist)))
                    if failures:bad.append(dict(decision=i,time_s=t,mode=mode,failures=failures))
            if i in (0,500,4000,7500,8500,9499):samples.append(dict(decision=i,time_s=t,correction=delta.tolist(),diagnostic=detail))
            if i%1000==0:print(json.dumps(dict(run=label,decision=i,max_joint=max_joint,bad_geometry=len(bad))),flush=True)
        result=dict(passed=not errors and not bad and maxrate<=1.5+1e-8,decisions=i+1,errors=errors,
            collision_pose_samples=collisions,maximum_joint_correction_rad=max_joint,maximum_requested_position_correction_m=max_pos,
            maximum_requested_orientation_correction_rad=max_rot,maximum_total_goal_speed_radps=maxrate,
            maximum_nonfoot_penetration_m=max_pen,rate_limited_decisions=rate_limited,
            bad_geometry_samples=len(bad),first_bad_geometry=bad[:30],diagnostic_samples=samples,
            original_source_pelvis_z=reference['source_pelvis_height_m'],estimator_initial_pelvis_z=infos[0]['estimated_root_local'][2])
        report['results'][label]=result
        (a.output/(label+'-bad-geometry.json')).write_text(json.dumps(bad,indent=2)+'\n')
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['passed']=all(v['passed'] for v in report['results'].values())
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{n:v[n] for n in ('passed','decisions','errors','bad_geometry_samples','maximum_nonfoot_penetration_m','maximum_total_goal_speed_radps')} for k,v in report['results'].items()},indent=2))


if __name__=='__main__':main()
