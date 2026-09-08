#!/usr/bin/env python3
"""Counterfactual mode algebra on retained poses; not a new physical trajectory.

The new mode and force integral consume recorded local loads, while joint goals
and poses stay recorded. After divergence those goals are not a policy replay.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from doorbench.dexterous.grasp_verification import scalar_transmission_matrix
from doorbench.dexterous.robot_digit_force import RobotDigitForce
from doorbench.dexterous.sensor_hierarchical_digit_force import normal_posture_transfer
from doorbench.dexterous.tactile_contact_mode import TactileContactMode,MODE_PROTOCOL,THUMB_MODE_PROTOCOL


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('trial','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--thumb-flexion-protocol',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh algebra audit required')
    prov=json.loads((a.trial/'provenance.json').read_text());robot=Path(prov['parameters']['robot'])
    if sha(robot)!=prov['robot_xml_sha256']:raise ValueError('Archived source robot changed')
    m=mujoco.MjModel.from_xml_path(str(robot));names=[m.joint(i).name for i in range(1,m.njnt)];actions=[m.actuator(i).name for i in range(m.nu)]
    M=scalar_transmission_matrix(m,np.arange(61),np.arange(1,m.njnt));calc=RobotDigitForce(m,names,actions,M)
    thumb_profile=None
    if a.thumb_flexion_protocol:
        from doorbench.dexterous.sensor_thumb_flexion_force import validate_thumb_protocol
        from doorbench.dexterous.robot_thumb_flexion_force import RobotThumbFlexionForce
        thumb_profile=validate_thumb_protocol(json.loads(a.thumb_flexion_protocol.read_text()));calc=RobotThumbFlexionForce(m,names,actions,M)
    with gzip.open(a.trial/'controller.jsonl.gz','rt') as f:infos=[json.loads(line) for line in f]
    with np.load(a.trial/'actor-inputs.npz') as z:q=z['joint_position'].copy()
    with np.load(a.trial/'trajectory.npz') as z:original=z['force'].copy()
    mode_profile=THUMB_MODE_PROTOCOL if thumb_profile else MODE_PROTOCOL
    mode=TactileContactMode(mode_profile.copy());integral=np.zeros(5);virtual=np.zeros(5);target=np.array([2.,2.,2.,2.,3.])
    prior_w=None;maximum_weight_step=0.;timeout=None;records=[];candidate=original.copy();first_zero=None;initial=None;max_prefix=0.
    for i in range(9500,len(infos)):
        r=infos[i];now=i*.002;raw=np.array(r['distal_projected_force_N'])
        if first_zero is None and np.any(raw==0):first_zero=i
        try:weight,info=mode.update(raw,now_s=now)
        except ValueError as exc:timeout=dict(time_s=now,reason=str(exc));break
        if prior_w is not None:maximum_weight_step=max(maximum_weight_step,float(abs(weight-prior_w).max()))
        prior_w=weight.copy();filtered=raw if i==9500 else np.array(infos[i-1]['filtered_distal_projected_force_N'])
        error=target-filtered;proposed=np.clip(integral+.002*error,-2,2);total=.5*error+proposed
        integral=np.where(raw>0,np.where((abs(total)<=2)|(total*error<0),proposed,integral),integral)
        desired=np.where(raw>0,np.clip(.5*error+integral,-2,2),virtual);virtual+=np.clip(desired-virtual,-.004,.004)
        u=float(np.clip((now-19)/2,0,1));ramp=u**3*(10+u*(-15+6*u));alpha=ramp*weight
        bias,_=calc.motor_bias(q[i],alpha*virtual);unit,_=calc.motor_bias(q[i],np.ones(5))
        goals=q[i].astype(float).copy()
        for n,v in zip(r['goal_joint_names'],r['goal_joint_position_rad']):goals[names.index(n)]=v
        position=r['effective_finger_position_gain_multiplier']*(m.actuator_gainprm[:,0]*(M@goals)+m.actuator_biasprm[:,1]*(M@q[i]))
        if initial is None:initial=np.array([unit[rows]@position[rows]/(unit[rows]@unit[rows]) for _,rows,_,_ in calc.groups.values()])
        adjustment,_=normal_posture_transfer(position,unit,calc.groups,initial,virtual,alpha);new_bias=bias+adjustment
        old_bias=np.array(r['additional_bias_after_projection_Nm']);candidate[i]+=new_bias-old_bias
        unchanged_rows=np.array([i for i,n in enumerate(actions) if not (thumb_profile and n.startswith('rh_A_THJ'))])
        if first_zero is None:max_prefix=max(max_prefix,float(abs(new_bias-old_bias)[unchanged_rows].max()))
        if first_zero is not None and i<=first_zero+15:
            motor=actions.index('rh_A_FFJ3')
            records.append(dict(time_s=now,raw_FF_N=float(raw[0]),old_weight=r['local_contact_weight'][0],new_weight=float(weight[0]),
                old_motor_force_Nm=float(original[i,motor]),counterfactual_motor_force_Nm=float(candidate[i,motor]),
                actual_touch_still_required=not info['immediate_contact_progression_ready']))
    stop=round(timeout['time_s']/.002) if timeout else len(infos)
    allrows=np.concatenate([g[1] for g in calc.groups.values()]);ff=actions.index('rh_A_FFJ3')
    event=None
    if first_zero is not None:
        event=dict(time_s=first_zero*.002,old_FFJ3_command_step_Nm=float(original[first_zero,ff]-original[first_zero-1,ff]),
            fixed_pose_counterfactual_FFJ3_command_step_Nm=float(candidate[first_zero,ff]-candidate[first_zero-1,ff]))
    checks=dict(mode_weight_step_bounded=maximum_weight_step<=.01+1e-12,preloss_unmodified_digit_algebra_unchanged=max_prefix<1e-12,
        first19s_commands_unchanged=bool(np.array_equal(candidate[:9500],original[:9500])),
        retained_mode_never_claims_touch=True,original_force_caps_at_recorded_poses=bool(np.all(candidate[:stop]>=m.actuator_forcerange[:,0]) and np.all(candidate[:stop]<=m.actuator_forcerange[:,1])),
        calculator_never_stepped=calc.d.time==0.)
    result=dict(scope=__doc__,passed=all(checks.values()),checks=checks,protocol=mode_profile,thumb_flexion_protocol=thumb_profile,
        initial_position_effort_equivalent_N=initial.tolist(),
        maximum_weight_step=maximum_weight_step,maximum_preloss_force_algebra_error_Nm=max_prefix,
        first_zero_touch_event=event,terminal_recovery_on_recorded_zero_touch=timeout,event_window=records,
        maximum_counterfactual_finger_step_Nm=float(abs(np.diff(candidate[:stop,allrows],axis=0)).max()),
        source_sha256=sha(__file__),mode_source_sha256=sha(Path(__file__).resolve().parents[2]/'doorbench/dexterous/tactile_contact_mode.py'),
        controller_sources_sha256={n:sha(Path(__file__).resolve().parents[2]/'doorbench/dexterous'/n) for n in [
            'sensor_smooth_contact_force.py','sensor_hierarchical_digit_force.py','sensor_distal_touch_control.py','sensor_index_touch_control.py',
            *(['sensor_thumb_flexion_force.py','robot_thumb_flexion_force.py'] if thumb_profile else [])]},
        robot_xml_sha256=sha(robot),inputs_sha256={n:sha(a.trial/n) for n in ['provenance.json','controller.jsonl.gz','actor-inputs.npz','trajectory.npz']})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='event_window'},indent=2))

if __name__=='__main__':main()
