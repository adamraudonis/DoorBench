#!/usr/bin/env python3
"""Export the existing native standing reset using the actual collision soles."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from doorbench.dexterous.locomotion import JOINT_NAMES, DEFAULT_ANGLES
from doorbench.dexterous.reset import check_joint_reset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    # This export needs only native kinematics, not training/stance packages.
    m=mujoco.MjModel.from_xml_path(str(a.robot));d=mujoco.MjData(m)
    audit=json.loads(a.robot.with_suffix('.audit.json').read_text())
    if hashlib.sha256(a.robot.read_bytes()).hexdigest()!=audit['robot_xml_sha256']:
        raise ValueError('Robot source differs from its audit')
    d.qpos[2]=audit['free_root_height']
    for name,value in audit['nominal_joint_positions'].items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
    for name,value in zip(JOINT_NAMES,DEFAULT_ANGLES):
        d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
    for side in ('left','right'):
        d.qpos[m.jnt_qposadr[m.joint(side+'_elbow').id]]=.7
        d.qpos[m.jnt_qposadr[m.joint(side+'_shoulder_roll').id]]=.1 if side=='left' else -.1
    joints=[j for j in range(m.njnt) if m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE]
    check_joint_reset([m.joint(j).name for j in joints],d.qpos[m.jnt_qposadr[joints]],m.jnt_range[joints])
    mujoco.mj_forward(m,d)
    feet=[m.body(side+'_ankle_link').id for side in ('left','right')]
    bottoms=[]
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] not in feet or not m.geom_contype[gid]:continue
        if m.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Audit non-mesh sole before use')
        mid=m.geom_dataid[gid]
        vertices=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
        world=vertices@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
        bottoms.append(float(world[:,2].min()))
    if not bottoms:raise ValueError('No collision sole geometry found')
    d.qpos[2]+=.0001-min(bottoms);mujoco.mj_forward(m,d)
    d.ctrl[:]=d.actuator_length
    result=dict(robot_xml_sha256=hashlib.sha256(a.robot.read_bytes()).hexdigest(),mass_kg=float(m.body_mass.sum()),
        initial_root=d.qpos[:7].tolist(),
        joints={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in joints},
        motor_targets={m.actuator(i).name:float(d.ctrl[i]) for i in range(m.nu)},
        feet={m.body(i).name:d.xpos[i].tolist() for i in feet},
        scope='Reset only; root is freely simulated after initialization')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'root':result['initial_root'],'mass_kg':result['mass_kg']}))


if __name__=='__main__':main()
