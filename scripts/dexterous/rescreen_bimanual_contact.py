#!/usr/bin/env python3
"""Destination-runtime static rescreen of an archived actual contact trajectory.

The model is never stepped. This qualifies sampled FK/collision compatibility,
not physical replay or Isaac dynamics. Exact destination compiled identity and
source-design identity are both retained; neither replaces the other.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.robot_identity import robot_file_identity
from doorbench.dexterous.robot_design_identity import verify_robot_design_identity


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','trajectory','target_config','expected_design','output'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Use a fresh output file')
    config=json.loads(a.target_config.read_text())
    expected=json.loads(a.expected_design.read_text())
    if config.get('source_design_identity')!=expected:raise ValueError('Expected design is not bound to the frozen targets')
    if config.get('runtime_rescreen_trajectory_sha256')!=sha(a.trajectory):raise ValueError('Recorded source path is not bound to the frozen targets')
    design=verify_robot_design_identity(a.robot,expected)
    compiled=robot_file_identity(a.robot)
    sim=DexterousDoorEnv(a.door,a.robot,json.loads(a.robot.with_suffix('.audit.json').read_text()))
    m,d=sim.m,sim.d
    try:
        path=np.load(a.trajectory,allow_pickle=False)
        states=np.asarray(path['qpos'])
        if states.ndim!=2 or states.shape[1]!=m.nq or len(states)<2 or not np.isfinite(states).all():
            raise ValueError('Expected a complete finite recorded state matrix for the exact model')
        joints=np.asarray(sim.joints);qa=m.jnt_qposadr[joints]
        limited=m.jnt_limited[joints].astype(bool)
        loop_pairs=[]
        for side in ('lh','rh'):
            for digit in ('FF','MF','RF','LF'):
                loop_pairs.append([m.jnt_qposadr[m.joint(f'robot/{side}_{digit}J{k}').id] for k in (1,2)])
        maximum_joint=maximum_loopback=maximum_penetration=0.;worst_contact=None
        for i,q in enumerate(states):
            d.qpos[:]=q;d.qvel[:]=0.;mujoco.mj_forward(m,d)
            v=np.maximum(m.jnt_range[joints,0]-q[qa],q[qa]-m.jnt_range[joints,1])
            maximum_joint=max(maximum_joint,float(v[limited].max(initial=0.)))
            maximum_loopback=max(maximum_loopback,max(float(q[x]-q[y]) for x,y in loop_pairs))
            for c in d.contact[:d.ncon]:
                bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom]
                geoms=[m.geom(int(g)).name for g in c.geom]
                if any(n.startswith('robot/') for n in bodies) and not ('floor' in geoms and any(n.endswith('_ankle_link') for n in bodies)):
                    if -float(c.dist)>maximum_penetration:
                        maximum_penetration=-float(c.dist);worst_contact=dict(sample=i,geoms=geoms,depth_m=maximum_penetration)
        checks=dict(expected_source_design=True,finite_complete_state_matrix=True,
                    original_robot_joint_limits=maximum_joint<=.02,
                    documented_loopback_limits=maximum_loopback<=.02,
                    all_nonfoot_collision_depth=maximum_penetration<=.003)
        receipt=dict(schema='doorbench.bimanual-runtime-contact-rescreen.v1',passed=all(checks.values()),checks=checks,
            scope='Recorded 50 Hz native pose FK/collision rescreen only; no destination physical rollout or interpolation guarantee',
            target_config_sha256=sha(a.target_config),trajectory_sha256=sha(a.trajectory),
            source_design_sha256=design['sha256'],source_compiled_robot_sha256=config['compiled_robot_identity']['sha256'],destination_compiled_robot_sha256=compiled['sha256'],
            destination_mujoco_version=mujoco.__version__,door_xml_sha256=sha(a.door/'door.xml' if a.door.is_dir() else a.door),
            recorded_samples=len(states),maximum_joint_limit_violation_rad=maximum_joint,
            maximum_loopback_violation_rad=maximum_loopback,maximum_nonfoot_penetration_m=maximum_penetration,
            worst_contact=worst_contact,physics_steps=0,source_script_sha256=sha(__file__))
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt));return 0 if receipt['passed'] else 1
    finally:sim.close()


if __name__=='__main__':raise SystemExit(main())
