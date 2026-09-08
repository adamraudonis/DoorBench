#!/usr/bin/env python3
"""Export exact native Door55 separated resets for the matching Isaac trials."""
import argparse,json,hashlib
import mujoco
from pathlib import Path
from doorbench.dexterous.locomotion_approach import make_door_approach


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=Path('configs/dexterous/h1-door55-approach-development.json'));a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);protocol=json.loads(a.protocol.read_text());rows=[]
    for case_index,case in enumerate(protocol['cases']):
        for seed in protocol['seeds']:
            sim,goal,yaw=make_door_approach(a.door,a.robot,a.reference,distance=case['distance_m'],lateral=case['lateral_m'],yaw_offset=case['yaw_offset_rad'],seed=seed,joint_noise=protocol['joint_noise_rad']);m,d=sim.m,sim.d
            bodies=[b for b in range(m.nbody) if m.body(b).name.startswith('robot/')]
            result={'scope':'Exact native reset only; no physics rollout or runtime pose tracking','case':case,'seed':seed,'goal_xy':goal.tolist(),'goal_yaw_rad':yaw,'robot_xml_sha256':hashlib.sha256(a.robot.read_bytes()).hexdigest(),'mass_kg':float(m.body_mass[bodies].sum()),'initial_root':d.qpos[sim.root_qadr:sim.root_qadr+7].tolist(),'joints':{m.joint(j).name.removeprefix('robot/'):float(d.qpos[m.jnt_qposadr[j]]) for j in sim.joints},'motor_targets':{m.actuator(i).name.removeprefix('robot/'):float(d.ctrl[i]) for i in sim.actuators},'feet':{m.body(i).name.removeprefix('robot/'):d.xpos[i].tolist() for i in sim.feet}}
            result['passive_tendons']=[]
            for tendon in range(m.ntendon):
                if not m.tendon(tendon).name.startswith('robot/') or not m.tendon_limited[tendon]:continue
                terms={}
                for k in range(m.tendon_adr[tendon],m.tendon_adr[tendon]+m.tendon_num[tendon]):
                    if m.wrap_type[k]!=mujoco.mjtWrap.mjWRAP_JOINT:raise ValueError('Unsupported native passive tendon')
                    terms[m.joint(int(m.wrap_objid[k])).name.removeprefix('robot/')]=float(m.wrap_prm[k])
                result['passive_tendons'].append(dict(name=m.tendon(tendon).name.removeprefix('robot/'),terms=terms,range_rad=m.tendon_range[tendon].tolist()))
            name=f'case-{case_index}-seed-{seed}.json';(a.output/name).write_text(json.dumps(result,indent=2)+'\n');rows.append(name);sim.close()
    (a.output/'protocol.json').write_bytes(a.protocol.read_bytes());(a.output/'resets.json').write_text(json.dumps(rows,indent=2)+'\n');print(a.output)
if __name__=='__main__':main()
