#!/usr/bin/env python3
"""Counterfactual effort algebra on preserved actual poses, without physics steps.

This does not predict the new compliant trajectory, contact loads or collisions.
The original static nominal path receipts remain separate admission evidence.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from doorbench.dexterous.grasp_verification import scalar_transmission_matrix
from doorbench.dexterous.robot_digit_force import RobotDigitForce
from doorbench.dexterous.sensor_hierarchical_digit_force import normal_posture_transfer,HIERARCHICAL_PROTOCOL


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['trial','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh screen required')
    prov=json.loads((a.trial/'provenance.json').read_text());robot=Path(prov['parameters']['robot'])
    if sha(robot)!=prov['robot_xml_sha256']:raise ValueError('Archived robot identity changed')
    audit=json.loads((a.trial/'regulation-decomposition.json').read_text())
    if not audit['passed'] or audit['trial']['motor_saturated_samples']!=0:raise ValueError('Original command reconstruction and unsaturated effort required')
    m=mujoco.MjModel.from_xml_path(str(robot));names=[m.joint(i).name for i in range(1,m.njnt)];actions=[m.actuator(i).name for i in range(m.nu)]
    M=scalar_transmission_matrix(m,np.arange(61),np.arange(1,m.njnt));calc=RobotDigitForce(m,names,actions,M)
    info=rows(a.trial/'controller.jsonl.gz')
    with np.load(a.trial/'actor-inputs.npz') as z:q=z['joint_position'].copy()
    with np.load(a.trial/'trajectory.npz') as z:forces=z['force'].copy()
    initial=None;max_tangent=max_normal=max_adjust=0.;would_saturate=0;modified=forces.copy();capture=None
    for i in range(9500,len(info)):
        r=info[i];goal=q[i].astype(float).copy()
        for n,x in zip(r['goal_joint_names'],r['goal_joint_position_rad']):goal[names.index(n)]=x
        position=r['effective_finger_position_gain_multiplier']*(m.actuator_gainprm[:,0]*(M@goal)+m.actuator_biasprm[:,1]*(M@q[i]))
        unit,_=calc.motor_bias(q[i],np.ones(5));groups=calc.groups
        if initial is None:
            initial=np.array([unit[rows]@position[rows]/(unit[rows]@unit[rows]) for _,rows,_,_ in groups.values()]);capture=initial.tolist()
        u=float(np.clip((i*.002-19)/2,0,1));alpha=np.array(r['local_contact_weight'])*u**3*(10+u*(-15+6*u))
        adjustment,details=normal_posture_transfer(position,unit,groups,initial,r['virtual_digit_force_state_N'],alpha)
        for k,(digit,(_,motor,_,_)) in enumerate(groups.items()):
            p=unit[motor];P=np.outer(p,p)/(p@p);added=unit[motor]*r['applied_virtual_digit_force_N'][k]
            new=position[motor]+added+adjustment[motor]
            max_tangent=max(max_tangent,float(abs((np.eye(len(motor))-P)@(new-position[motor])).max()))
            expected=(1-alpha[k])*(p@position[motor]/(p@p))+alpha[k]*details[digit]['total_virtual_normal_effort_N']
            max_normal=max(max_normal,abs(float(p@new/(p@p))-expected))
        modified[i]+=adjustment;max_adjust=max(max_adjust,float(abs(adjustment).max()))
        would_saturate+=int(np.count_nonzero((modified[i]<m.actuator_forcerange[:,0])|(modified[i]>m.actuator_forcerange[:,1])))
    gates=dict(capture_admitted=bool(np.all(initial>=0) and np.all(initial<=2)),tangential_motor_posture_preserved=max_tangent<1e-12,
        declared_normal_transfer=max_normal<1e-12,recorded_pose_efforts_within_original_caps=would_saturate==0,
        calculator_never_stepped=calc.d.time==0.,first19s_commands_unchanged=bool(np.array_equal(modified[:9500],forces[:9500])))
    gates={k:bool(v) for k,v in gates.items()}
    out=dict(scope=__doc__,passed=all(gates.values()),checks=gates,recorded_decisions=len(info),initial_position_effort_equivalent_N=capture,
        maximum_tangent_error_Nm=max_tangent,maximum_normal_transfer_error_N=max_normal,maximum_added_projection_adjustment_Nm=max_adjust,
        maximum_projected_command_step_after19_Nm=float(abs(np.diff(modified,axis=0)[9500:]).max()),
        predicted_saturated_motor_samples=would_saturate,protocol=HIERARCHICAL_PROTOCOL,
        robot_xml_sha256=sha(robot),motor_contract_sha256=sha(a.trial/'motors.json'),
        source_sha256=sha(__file__),controller_sha256=sha(Path(__file__).resolve().parents[2]/'doorbench/dexterous/sensor_hierarchical_digit_force.py'),
        input_sha256={n:sha(a.trial/n) for n in ['provenance.json','controller.jsonl.gz','actor-inputs.npz','trajectory.npz','regulation-decomposition.json','index-screen.json','preload-screen.json','static-screen.json']})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
