#!/usr/bin/env python3
"""Offline screen of bounded finger offsets around a fixed press hypothesis."""
import argparse,hashlib,itertools,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from scripts.dexterous.probe_sensor_acquisition_balance import intended_contact_geometry


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--acquisition',type=Path,required=True);p.add_argument('--press-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh screen receipt required')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source=a.acquisition;prov=json.loads((source/'provenance.json').read_text());plan=json.loads(a.press_plan.read_text())
    if not plan['passed'] or plan['source_trajectory_sha256']!=sha(source/'trajectory.npz'):raise ValueError('Matching attained-state press plan required')
    robot=Path(prov['parameters']['robot']);door=Path(prov['parameters']['door']);door=door if door.is_dir() else door.parent
    if sha(robot)!=plan['robot_xml_sha256'] or sha(door/'door.xml')!=plan['door_xml_sha256']:raise ValueError('Changed authored plant')
    q=np.load(source/'trajectory.npz')['qpos'][-1];sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()),frame_skip=1);m=sim.m;d=mujoco.MjData(m)
    qa=[m.jnt_qposadr[m.joint('robot/'+n).id] for n in plan['joint_names']];lever=m.geom('leaf_handle_lever_col_n').id
    hj=m.jnt_qposadr[m.joint('leaf_handle_hinge').id];bj=m.jnt_qposadr[m.joint('leaf_latch_bolt_slide').id]
    bad=[];maxdepth=0.;count=0
    for i,arm in enumerate(plan['path_qpos']):
        for choice in itertools.product((0,1),repeat=5):
            d.qpos[:]=q;d.qpos[qa]=np.asarray(arm)-plan['constant_tracking_offset'];d.qpos[hj]=plan['screen'][i]['handle_angle_rad'];d.qpos[bj]=.014598*d.qpos[hj]
            for digit,flag in zip(('FF','MF','RF','LF','TH'),choice):
                for j in (('1',) if digit=='TH' else ('1','2')):d.qpos[m.jnt_qposadr[m.joint('robot/rh_'+digit+'J'+j).id]]+=flag*(.08 if digit=='TH' else .03)
            mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d);problems=[]
            for c in d.contact[:d.ncon]:
                bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom];geoms=[m.geom(g).name for g in c.geom]
                if c.dist<=0 and any(n.startswith(('robot/rh_','robot/lh_')) for n in bodies) and not intended_contact_geometry(m,d,c,lever):problems.append(dict(reason='unintended hand surface',bodies=bodies,distance_m=float(c.dist)))
                if any(n.startswith('robot/') for n in bodies) and not ('floor' in geoms and any(n.endswith('_ankle_link') for n in bodies)):
                    maxdepth=max(maxdepth,-float(c.dist))
                    if c.dist<-.003:problems.append(dict(reason='nonfoot penetration',bodies=bodies,distance_m=float(c.dist)))
            count+=1
            if problems:bad.append(dict(press_index=i,offset_corner=choice,problems=problems))
    result=dict(passed=not bad,samples=count,max_nonfoot_penetration_m=maxdepth,bad_samples=bad,
        scope='Offline offset-corner geometry at121 rigid-grasp press poses; no force or continuous-volume qualification',
        press_plan_sha256=sha(a.press_plan),source_trajectory_sha256=sha(source/'trajectory.npz'),screen_source_sha256=sha(__file__))
    a.output.write_text(json.dumps(result,indent=2)+'\n');sim.close();print(json.dumps({k:v for k,v in result.items() if k!='bad_samples'},indent=2));print('First failures',bad[:2])
if __name__=='__main__':main()
