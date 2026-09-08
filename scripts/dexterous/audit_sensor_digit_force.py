#!/usr/bin/env python3
"""Reconstruct every recorded tactile virtual-work bias without stepping physics."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from doorbench.dexterous.grasp_verification import scalar_transmission_matrix
from doorbench.dexterous.robot_digit_force import RobotDigitForce
from doorbench.dexterous.sensor_digit_force_control import FORCE_PROTOCOL


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('trial','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh audit receipt required')
    prov=json.loads((a.trial/'provenance.json').read_text());robot=Path(prov['parameters']['robot'])
    if sha(robot)!=prov['robot_xml_sha256']:raise ValueError('Archived robot changed')
    if json.loads((a.trial/'digit-force-protocol.json').read_text())!=FORCE_PROTOCOL:raise ValueError('Exact frozen force profile required')
    infos=rows(a.trial/'controller.jsonl.gz');physics=rows(a.trial/'physics.jsonl.gz')
    with np.load(a.trial/'actor-inputs.npz') as saved:encoders=saved['joint_position'].copy()
    if len(infos)!=len(physics) or len(infos)!=len(encoders):raise ValueError('Unmatched decision/evidence count')
    m=mujoco.MjModel.from_xml_path(str(robot));names=[m.joint(i).name for i in range(1,m.njnt)];actions=[m.actuator(i).name for i in range(m.nu)]
    M=scalar_transmission_matrix(m,np.arange(m.nu),np.arange(1,m.njnt));calc=RobotDigitForce(m,names,actions,M)
    rows_finger=np.concatenate([g[1] for g in calc.groups.values()]);other=np.setdiff1d(np.arange(61),rows_finger)
    hierarchy=(a.trial/'hierarchical-force-protocol.json').exists()
    initial_normal=None;max_hierarchy=max_tangent=0.
    max_map=max_orthogonal=max_passive=0.;first_loss=None;prefix_zero=True;max_nonfinger=0.
    states=[];applied=[];weights=[];biases=[]
    for i,(info,row,q) in enumerate(zip(infos,physics,encoders,strict=True)):
        if abs(row['contact_interval_start_s']-i*.002)>1e-8 or abs(row['contact_interval_end_s']-(i+1)*.002)>1e-8:raise ValueError('Actual contact epoch mismatch')
        f=np.asarray(info['applied_virtual_digit_force_N']);bias=np.asarray(info['requested_additional_motor_bias_Nm'])
        if i<9500:prefix_zero &= bool(np.all(f==0.) and np.all(bias==0.))
        else:
            expected,details=calc.motor_bias(q,f);max_map=max(max_map,float(abs(expected-bias).max()))
            for digit,detail in details.items():
                residual=np.asarray(detail['unavailable_passive_split_moment_Nm']);A=calc.groups[digit][2]
                max_orthogonal=max(max_orthogonal,float(abs(A@residual).max()));max_passive=max(max_passive,float(abs(residual).max()))
            if hierarchy:
                q=q.astype(float);goal=q.copy()
                for name,value in zip(info['goal_joint_names'],info['goal_joint_position_rad'],strict=True):goal[names.index(name)]=value
                position=info['effective_finger_position_gain_multiplier']*(m.actuator_gainprm[:,0]*(M@goal)+m.actuator_biasprm[:,1]*(M@q))
                unit,_=calc.motor_bias(q,np.ones(5))
                if initial_normal is None:initial_normal=np.array([unit[r]@position[r]/(unit[r]@unit[r]) for _,r,_,_ in calc.groups.values()])
                adjustment=np.zeros(61);t=float(np.clip((row['contact_interval_start_s']-19.)/2.,0.,1.))
                alpha=np.array(info['local_contact_weight'])*t**3*(10+t*(-15+6*t))
                for k,(_,motor,_,_) in enumerate(calc.groups.values()):
                    p=unit[motor];P=np.outer(p,p)/(p@p);correction=info['virtual_digit_force_state_N'][k]
                    total=np.clip(initial_normal[k]+correction,0.,4.)
                    adjustment[motor]=alpha[k]*(p*(total-correction)-P@position[motor])
                    max_tangent=max(max_tangent,float(abs((np.eye(len(motor))-P)@adjustment[motor]).max()))
                max_hierarchy=max(max_hierarchy,float(abs(adjustment-info['hierarchical_motor_bias_adjustment_Nm']).max()),float(abs(expected+adjustment-info['additional_bias_after_projection_Nm']).max()))
            if not row['pad_grasp']['valid_pad_grasp'] and first_loss is None:first_loss=dict(interval_start_s=i*.002,reason=row['pad_grasp']['reason'])
        max_nonfinger=max(max_nonfinger,float(abs(bias[other]).max()))
        states.append(info['virtual_digit_force_state_N']);applied.append(f);weights.append(info['local_contact_weight']);biases.append(bias)
    states=np.asarray(states);applied=np.asarray(applied);weights=np.asarray(weights);biases=np.asarray(biases)
    final=np.arange(len(infos))*.002>=len(infos)*.002-1.
    internal_step=float(abs(np.diff(states,axis=0)).max());applied_step=float(abs(np.diff(applied,axis=0)).max())
    checks=dict(all_recorded_biases_reconstructed=max_map<1e-12,passive_residual_unactuated=max_orthogonal<1e-12,
        no_nonfinger_bias=max_nonfinger==0.,first19s_no_force_bias=bool(prefix_zero),virtual_force_bound=bool(np.max(abs(states))<=2.+1e-12),
        internal_force_slew_bound=internal_step<=.004+1e-12,zero_force_without_local_contact=bool(np.all(applied[weights==0.] == 0.)),
        calculator_never_stepped=calc.d.time==0.)
    if hierarchy:checks.update(hierarchical_bias_reconstructed=max_hierarchy<1e-12,tangential_motor_posture_preserved=max_tangent<1e-12)
    report=dict(hierarchical_profile=hierarchy,maximum_hierarchical_bias_error_Nm=max_hierarchy,maximum_tangential_projection_error_Nm=max_tangent,initial_position_effort_equivalent_N=None if initial_normal is None else initial_normal.tolist(),scope=__doc__+'; this verifies controller algebra, not physical task success',passed=all(checks.values()),checks=checks,
        decisions=len(infos),maximum_bias_reconstruction_error_Nm=max_map,maximum_projection_orthogonality_error_Nm=max_orthogonal,
        maximum_unavailable_passive_moment_Nm=max_passive,maximum_additional_motor_bias_Nm=float(abs(biases).max()),
        maximum_internal_virtual_force_step_N=internal_step,maximum_contact_weighted_virtual_force_step_N=applied_step,
        maximum_requested_bias_step_Nm=float(abs(np.diff(biases,axis=0)).max()),first_post19_opposed_loss=first_loss,
        last_second=dict(mean_virtual_correction_N=applied[final].mean(axis=0).tolist(),
            mean_local_projection_N=np.array([r['distal_projected_force_N'] for r in infos])[final].mean(axis=0).tolist() if 'distal_projected_force_N' in infos[0] else None,
            maximum_bias_Nm=float(abs(biases[final]).max())),
        source_sha256=sha(__file__),robot_sha256=sha(robot),input_sha256={n:sha(a.trial/n) for n in ['provenance.json','controller.jsonl.gz','physics.jsonl.gz','actor-inputs.npz','digit-force-protocol.json']})
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
