#!/usr/bin/env python3
"""Detached virtual-work/transmission audit on archived attained robot angles."""
import argparse,hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from doorbench.dexterous.grasp_verification import scalar_transmission_matrix
from doorbench.dexterous.robot_digit_force import RobotDigitForce


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh audit receipt required')
    prov=json.loads((a.trial/'provenance.json').read_text());robot=Path(prov['parameters']['robot'])
    if sha(robot)!=prov['robot_xml_sha256']:raise ValueError('Archived robot design changed')
    m=mujoco.MjModel.from_xml_path(str(robot));names=[m.joint(i).name for i in range(1,m.njnt)];actions=[m.actuator(i).name for i in range(m.nu)]
    M=scalar_transmission_matrix(m,np.arange(m.nu),np.arange(1,m.njnt));calc=RobotDigitForce(m,names,actions,M)
    state=np.load(a.trial/'trajectory.npz');reset=json.loads((a.trial/'reset.json').read_text())
    matches=[i for i in range(state['qpos'].shape[1]-6) if np.array_equal(state['qpos'][0,i:i+7],reset['root'])]
    if len(matches)!=1 or list(state['joint_names'])!=names or list(state['action_names'])!=actions:raise ValueError('Archived original root/joint/motor order required')
    qa=m.jnt_qposadr[1:]+matches[0];max_bias=0.;max_orthogonal_error=0.;last=None;count=0
    for i in np.unique(np.linspace(0,len(state['qpos'])-1,501).astype(int)):
        bias,details=calc.motor_bias(state['qpos'][i,qa],np.ones(5)*2.)
        max_bias=max(max_bias,float(abs(bias).max()));count+=1
        for digit,entry in details.items():
            columns,rows,A,_=calc.groups[digit]
            residual=np.asarray(entry['unavailable_passive_split_moment_Nm'])
            max_orthogonal_error=max(max_orthogonal_error,float(abs(A@residual).max()))
        last=details
    result=dict(scope='Detached original motor projection over archived encoder poses; no actual force, grasp or dynamic qualification',
        passed=max_orthogonal_error<1e-12 and calc.d.time==0.,samples=count,
        maximum_requested_motor_bias_Nm=max_bias,maximum_projection_orthogonality_error_Nm=max_orthogonal_error,
        correction_force_per_digit_N=2.,final_pose_projection=last,calculator_never_stepped=calc.d.time==0.,
        robot_xml_sha256=sha(robot),source_sha256=sha(__file__),mapper_source_sha256=sha(Path(__file__).resolve().parents[2]/'doorbench/dexterous/robot_digit_force.py'),
        inputs_sha256={n:sha(a.trial/n) for n in ['provenance.json','trajectory.npz','reset.json']})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
