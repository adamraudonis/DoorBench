#!/usr/bin/env python3
"""Record official H1 deployment-model differences without changing our robot."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import mujoco,numpy as np
from doorbench.dexterous.locomotion import JOINT_NAMES,UPSTREAM_REVISION,POLICY_RELATIVE_PATH,POLICY_SHA256,NativeH1MotorAdapter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('robot','upstream','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();revision=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()
    if revision!=UPSTREAM_REVISION:raise ValueError('Unexpected Unitree revision')
    policy=args.upstream/POLICY_RELATIVE_PATH
    if hashlib.sha256(policy.read_bytes()).hexdigest()!=POLICY_SHA256:raise ValueError('Unexpected H1 weights')
    m=mujoco.MjModel.from_xml_path(str(args.robot));u=mujoco.MjModel.from_xml_path(str(args.upstream/'resources/robots/h1/scene.xml'));a=NativeH1MotorAdapter(m,prefix='');joints=[]
    for name,j,aid in zip(JOINT_NAMES,a.joints,a.actuators):
        k=u.joint(name+'_joint').id;v=m.jnt_dofadr[j];uv=u.jnt_dofadr[k]
        joints.append(dict(name=name,policy_order=len(joints),native_qpos_index=int(m.jnt_qposadr[j]),native_qvel_index=int(v),native_actuator_index=int(aid),axis_native=m.jnt_axis[j].tolist(),axis_official=u.jnt_axis[k].tolist(),axes_match=bool(np.allclose(m.jnt_axis[j],u.jnt_axis[k])),range_native=m.jnt_range[j].tolist(),range_official=u.jnt_range[k].tolist(),damping_native=float(m.dof_damping[v]),damping_official=float(u.dof_damping[uv]),armature_native=float(m.dof_armature[v]),armature_official=float(u.dof_armature[uv]),frictionloss_native=float(m.dof_frictionloss[v]),frictionloss_official=float(u.dof_frictionloss[uv]),force_range_native=m.actuator_forcerange[aid].tolist(),force_range_official=u.jnt_actfrcrange[k].tolist()))
    report=dict(scope=__doc__,upstream_revision=revision,policy_sha256=POLICY_SHA256,native=dict(mass_kg=float(m.body_mass.sum()),nq=m.nq,nv=m.nv,motors=m.nu,free_base=True,dual_shadow_hands=True),official_deployment=dict(mass_kg=float(u.body_mass.sum()),nq=u.nq,nv=u.nv,motors=u.nu,upper_body='Rigidly folded into pelvis; not copied to DoorBench'),joints=joints,physics_changes_made=False,compatibility_claim='Same named H1 leg axes/order, empirically tested on the unchanged full robot. Dynamics and hip-pitch limits differ; this is not exact model parity.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(args.output)
if __name__=='__main__':main()
