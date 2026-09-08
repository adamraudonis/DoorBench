#!/usr/bin/env python3
"""Separate first actual pad loss from the following sensor/motor mode reaction."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.native_transition_archive import unpacked
from doorbench.dexterous.grasp_verification import scalar_transmission_matrix


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['trial','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh evidence required')
    infos=lines(a.trial/'controller.jsonl.gz');rows=lines(a.trial/'physics.jsonl.gz')
    first=next(i for i,r in enumerate(rows) if r['contact_interval_start_s']>=19 and not r['pad_grasp']['valid_pad_grasp'])
    prov=json.loads((a.trial/'provenance.json').read_text());robot=Path(prov['parameters']['robot']);door=Path(prov['parameters']['door'])
    if sha(robot)!=prov['robot_xml_sha256'] or sha(door/'door.xml')!=prov['door_xml_sha256']:raise ValueError('Actual scene identity changed')
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()));m=sim.m;d=mujoco.MjData(m)
    lever=m.geom('leaf_handle_lever_col_n').id;body=m.body('robot/rh_ffdistal').id
    def compatible(g):return bool((m.geom_contype[g]&m.geom_conaffinity[lever]) or (m.geom_contype[lever]&m.geom_conaffinity[g]))
    geometries=[g for g in range(m.ngeom) if m.geom_bodyid[g]==body and compatible(g)]
    manifest=json.loads((a.trial/'actual-transitions/manifest.json').read_text());selected={}
    for chunk in manifest['chunks']:
        if chunk['interval_end_s']<(first-5)*.002 or chunk['interval_start_s']>(first+5)*.002:continue
        path=a.trial/'actual-transitions'/chunk['file']
        if sha(path)!=chunk['sha256']:raise ValueError('Actual transition chunk changed')
        with np.load(path) as z:raw=list(unpacked({k:z[k] for k in z.files}))
        for r in raw:
            i=round(r['interval_start_s']/.002)
            if first-5<=i<=first+5:selected[i]=r
    state=np.load(a.trial/'trajectory.npz');actions=list(state['action_names']);motor=actions.index('rh_A_FFJ3');out=[]
    if [m.actuator(j).name.removeprefix('robot/') for j in range(m.nu)]!=actions:raise ValueError('Actual motor evidence ordering differs')
    joints=np.array([m.joint('robot/'+str(n)).id for n in state['joint_names']]);M=scalar_transmission_matrix(m,np.arange(m.nu),joints)
    first_tracking=None
    for i,row in enumerate(rows):
        if row['command_coordinate_tracking_error_max_rad']>=.04:
            current=M@state['qpos'][i+1,m.jnt_qposadr[joints]];targets=dict(zip(infos[i]['goal_motor_names'],infos[i]['goal_motor_coordinates']))
            name=max(targets,key=lambda n:abs(current[actions.index(n)]-targets[n]));actual=float(current[actions.index(name)])
            first_tracking=dict(interval_start_s=row['contact_interval_start_s'],endpoint_time_s=row['sim_time_s'],motor=name,actual_rad=actual,target_rad=targets[name],error_rad=abs(actual-targets[name]));break
    for i,raw in sorted(selected.items()):
        d.qpos[:]=raw['qpos_before'];mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
        distances=[dict(geom_id=g,geom_name=m.geom(g).name,geometry_type=int(m.geom_type[g]),distance_m=float(mujoco.mj_geomDistance(m,d,g,lever,.05,None))) for g in geometries]
        computed=[float(c.dist) for c in d.contact[:d.ncon] if lever in c.geom and any(g in c.geom for g in geometries)]
        actual=[c for c in raw['contacts'] if lever in c['geom'] and any(g in c['geom'] for g in geometries)]
        qj={n:float(d.qpos[m.joint('robot/'+n).qposadr[0]]) for n in ['rh_FFJ4','rh_FFJ3','rh_FFJ2','rh_FFJ1']}
        out.append(dict(interval_start_s=raw['interval_start_s'],actual_raw_FF_contact_count=len(actual),
            actual_FF_contact_normal_load_N=sum(max(0.,c['wrench_contact_frame'][0]) for c in actual),
            actual_contact_distances_m=[c['distance_m'] for c in actual],detached_collision_distances_m=computed,
            detached_geom_distance_m=distances,actual_joint_angles=qj,
            actual_FFJ3_motor_force_Nm=float(raw['actuator_force'][motor]),
            available_local_FF_touch_N=infos[i]['distal_projected_force_N'][0],projection_alpha=infos[i]['normal_posture_transfer']['ff']['alpha'],
            current_position_effort_equivalent_N=infos[i]['normal_posture_transfer']['ff']['current_position_effort_equivalent_N']))
    result=dict(scope=__doc__,first_opposed_loss_interval_s=first*.002,first_motor_tracking_violation=first_tracking,
        geometry_scope='Detached exact pre-integration pose collision and signed distance; no physical step or force recomputation',
        entries=out,robot_xml_sha256=sha(robot),door_xml_sha256=sha(door/'door.xml'),calculator_time_s=d.time,
        source_sha256=sha(__file__),inputs_sha256={n:sha(a.trial/n) for n in ['provenance.json','physics.jsonl.gz','controller.jsonl.gz','trajectory.npz','actual-transitions/manifest.json']})
    a.output.write_text(json.dumps(result,indent=2)+'\n');sim.close();print(json.dumps(result,indent=2))

if __name__=='__main__':main()
