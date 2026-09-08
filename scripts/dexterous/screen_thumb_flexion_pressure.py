#!/usr/bin/env python3
"""Independent authored-axis/virtual-work and attained-pose thumb envelope screen.

No active physics, no reset/replay success, no inference of loaded contact from
geometry. The grid includes all rejected candidates and cannot qualify a grasp.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.grasp_verification import scalar_transmission_matrix
from doorbench.dexterous.robot_digit_force import RobotDigitForce
from doorbench.dexterous.robot_thumb_flexion_force import RobotThumbFlexionForce
from scripts.dexterous.probe_sensor_touch_operation import intended_contact_geometry


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('trial','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh screen required')
    prov=json.loads((a.trial/'provenance.json').read_text());robot=Path(prov['parameters']['robot']);door=Path(prov['parameters']['door'])
    if sha(robot)!=prov['robot_xml_sha256'] or sha(door/'door.xml')!=prov['door_xml_sha256']:raise ValueError('Frozen source scene changed')
    m=mujoco.MjModel.from_xml_path(str(robot));names=[m.joint(i).name for i in range(1,m.njnt)];actions=[m.actuator(i).name for i in range(m.nu)]
    A=scalar_transmission_matrix(m,np.arange(61),np.arange(1,m.njnt));full=RobotDigitForce(m,names,actions,A);small=RobotThumbFlexionForce(m,names,actions,A)
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()));scene=sim.m;d=mujoco.MjData(scene)
    with np.load(a.trial/'trajectory.npz') as z:states=z['qpos'].copy();jn=list(z['joint_names'])
    if jn!=names:raise ValueError('Exact archived joint ordering required')
    with gzip.open(a.trial/'controller.jsonl.gz','rt') as f:infos=[json.loads(line) for line in f]
    qa=np.array([scene.joint('robot/'+n).qposadr[0] for n in names]);lever=scene.geom('leaf_handle_lever_col_n').id
    thumb_names=['rh_THJ'+str(i) for i in (5,4,3,2,1)];axis=[];mapping=[];grid=[]
    for t in [19.,21.,23.,25.008]:
        i=round(t/.002);q=states[i,qa]
        unit,whole=full.motor_bias(q,np.ones(5));limited,part=small.motor_bias(q,np.ones(5))
        mujoco.mj_kinematics(m,full.d)
        axes={n:full.d.xmat[m.jnt_bodyid[m.joint(n).id]].reshape(3,3)@m.jnt_axis[m.joint(n).id] for n in thumb_names}
        axis.append(dict(time_s=t,world_gauge_axes={n:v.tolist() for n,v in axes.items()},
            th1_th2_axis_dot=float(axes['rh_THJ1']@axes['rh_THJ2']),
            authored_axis={n:m.jnt_axis[m.joint(n).id].tolist() for n in thumb_names},
            authored_range={n:m.jnt_range[m.joint(n).id].tolist() for n in thumb_names}))
        rows=full.groups['th'][1];kept=small.groups['th'][1]
        mapping.append(dict(time_s=t,whole_thumb=whole['th'],flexion_only=part['th'],
            whole_normal_moment_norm_Nm_per_N=float(np.linalg.norm(unit[rows])),
            flexion_normal_moment_norm_Nm_per_N=float(np.linalg.norm(limited[kept])),
            retained_squared_moment_fraction=float(limited[kept]@limited[kept]/(unit[rows]@unit[rows])),
            opposition_pressure_moments_exact_zero=bool(np.all(limited[[actions.index('rh_A_THJ'+str(k)) for k in (3,4,5)]]==0))))
        goals=dict(zip(infos[i]['goal_joint_names'],infos[i]['goal_joint_position_rad']))
        for style in ('actual_opposition','nominal_opposition'):
            for th2 in np.linspace(-.035,.035,9):
                for th1 in np.linspace(-.035,.035,9):
                    d.qpos[:]=states[i]
                    if style=='nominal_opposition':
                        for k in (3,4,5):d.qpos[scene.joint('robot/rh_THJ'+str(k)).qposadr[0]]=goals['rh_THJ'+str(k)]
                    for k,delta in ((2,th2),(1,th1)):d.qpos[scene.joint('robot/rh_THJ'+str(k)).qposadr[0]]+=delta
                    mujoco.mj_kinematics(scene,d);mujoco.mj_collision(scene,d)
                    failures=[];thumb_count=0;maxpen=0.
                    for n in thumb_names:
                        j=scene.joint('robot/'+n).id;v=float(d.qpos[scene.jnt_qposadr[j]])
                        if not scene.jnt_range[j,0]<=v<=scene.jnt_range[j,1]:failures.append('authored thumb bound:'+n)
                    for c in d.contact[:d.ncon]:
                        bn=[scene.body(scene.geom_bodyid[g]).name for g in c.geom]
                        if not any(n.startswith(('robot/rh_','robot/lh_')) for n in bn) or c.dist>0:continue
                        maxpen=max(maxpen,-float(c.dist))
                        if not intended_contact_geometry(scene,d,c,lever):failures.append('unintended hand geometry:'+','.join(bn))
                        if c.dist<-.003:failures.append('hand penetration above3mm')
                        if lever in c.geom and any(n=='robot/rh_thdistal' for n in bn):thumb_count+=1
                    grid.append(dict(time_s=t,opposition=style,th2_offset_rad=float(th2),th1_offset_rad=float(th1),
                        passed=not failures,failures=sorted(set(failures)),thumb_distal_lever_overlap_count=thumb_count,
                        maximum_hand_penetration_m=maxpen))
    handoff=json.loads((a.trial/'acquisition-handoff-audit.json').read_text())
    result=dict(scope=__doc__,axis_audit=axis,mapping=mapping,grid_samples=len(grid),grid_passed=sum(r['passed'] for r in grid),
        initial19s_grasp_envelope_passed=all(r['passed'] for r in grid if r['time_s']==19.),
        source_initial19s_acquisition_physically_qualified=handoff.get('passed') is True,
        whole_envelope_passed=all(r['passed'] for r in grid),grid=grid,
        declared_scope='THJ1/THJ2 pressure only; THJ3/4/5 keep original posture controller; other digits unchanged',
        geometry_interpretation='Overlap is static only and does not establish load, continued opposed grasp or passage',
        calculators_never_stepped=full.d.time==0 and small.d.time==0 and d.time==0,
        initial_reference_sha256=prov['reference_sha256'],initial_calibration_sha256=prov['calibration_sha256'],
        motor_contract_sha256=prov['motor_contract_sha256'],initial_schedule_sha256=prov['arm_schedule_sha256'],
        robot_xml_sha256=sha(robot),door_xml_sha256=sha(door/'door.xml'),source_sha256=sha(__file__),
        mapper_sha256=sha(Path(__file__).resolve().parents[2]/'doorbench/dexterous/robot_thumb_flexion_force.py'),
        inputs_sha256={n:sha(a.trial/n) for n in ['provenance.json','trajectory.npz','controller.jsonl.gz','acquisition-handoff-audit.json']})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');sim.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ('grid','axis_audit','mapping')},indent=2))

if __name__=='__main__':main()
