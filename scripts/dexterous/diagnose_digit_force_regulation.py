#!/usr/bin/env python3
"""Separate actual finger commands into force bias, position, damping and gravity.

Recomputes the policy's robot-only bias dynamics at its saved sensor-derived
estimate. It does not step a plant or treat this calculation as contact force.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from doorbench.dexterous.grasp_verification import scalar_transmission_matrix


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]


def reduce(run):
    prov=json.loads((run/'provenance.json').read_text());robot=Path(prov['parameters']['robot'])
    if sha(robot)!=prov['robot_xml_sha256']:raise ValueError('Robot bytes differ from run')
    m=mujoco.MjModel.from_xml_path(str(robot));d=mujoco.MjData(m)
    names=[m.joint(i).name for i in range(1,m.njnt)];actions=[m.actuator(i).name for i in range(m.nu)]
    M=scalar_transmission_matrix(m,np.arange(m.nu),np.arange(1,m.njnt));qa=m.jnt_qposadr[1:];va=m.jnt_dofadr[1:]
    infos=rows(run/'controller.jsonl.gz');physical=rows(run/'physics.jsonl.gz')
    reset=json.loads((run/'reset.json').read_text());desired=np.array([reset['joints'][n] for n in names])
    with np.load(run/'actor-inputs.npz') as z:packets={k:z[k].copy() for k in ['joint_position','joint_velocity']}
    with np.load(run/'trajectory.npz') as z:actual=z['force'].copy()
    motor=np.array([i for i,n in enumerate(actions) if n.startswith('rh_') and 'WRJ' not in n])
    if len(infos)!=len(actual) or len(physical)!=len(actual):raise ValueError('Unmatched command/evidence count')
    parts={k:[] for k in ['position','velocity','original_constant','virtual_bias','robot_gravity_coriolis','uncapped','capped','actual','target','measured','q']}
    maxerror=0.;saturations=0
    for i,info in enumerate(infos):
        q=desired.copy() if i==0 else packets['joint_position'][i].astype(float)
        v=np.zeros(69) if i==0 else packets['joint_velocity'][i].astype(float)
        d.qpos[:7]=info['estimated_root_local'];d.qpos[qa]=q;d.qvel[:6]=info['estimated_velocity_local'];d.qvel[va]=v
        mujoco.mj_forward(m,d)
        goal=desired.copy()
        for n,x in zip(info['goal_joint_names'],info['goal_joint_position_rad'],strict=True):goal[names.index(n)]=x
        target=M@goal;length=M@q;speed=M@v
        vtarget=np.zeros(61)
        for n,x in zip(info['goal_motor_names'],info['goal_motor_velocity_radps'],strict=True):vtarget[actions.index(n)]=x
        gain=info['effective_finger_position_gain_multiplier']
        position=gain*(m.actuator_gainprm[:,0]*target+m.actuator_biasprm[:,1]*length)
        velocity=(m.actuator_biasprm[:,2]-.05)*speed+.05*vtarget
        constant=m.actuator_biasprm[:,0].copy();bias=np.array(info.get('requested_additional_motor_bias_Nm',np.zeros(61)))
        gravity=np.zeros(61)
        for aid in motor:
            if m.actuator_trntype[aid]==mujoco.mjtTrn.mjTRN_JOINT:gravity[aid]=d.qfrc_bias[m.jnt_dofadr[m.actuator_trnid[aid,0]]]
        uncapped=position+velocity+constant+bias+gravity;capped=np.clip(uncapped,m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])
        maxerror=max(maxerror,float(abs(capped[motor]-actual[i,motor]).max()));saturations+=int(np.count_nonzero(capped[motor]!=uncapped[motor]))
        values=[position,velocity,constant,bias,gravity,uncapped,capped,actual[i],target,length]
        for key,value in zip(list(parts)[:-1],values):parts[key].append(value[motor])
        parts['q'].append(q)
    parts={k:np.asarray(v) for k,v in parts.items()};last=slice(max(0,len(actual)-500),None);summary={}
    for j,aid in enumerate(motor):
        summary[actions[aid]]={k+'_mean':float(parts[k][last,j].mean()) for k in parts if k!='q'}
        summary[actions[aid]]['original_cap_Nm']=m.actuator_forcerange[aid].tolist()
    digits={}
    for digit in ['ff','mf','rf','lf','th']:
        columns=[names.index(n) for n in names if n.startswith('rh_'+digit.upper()+'J')]
        digits[digit]=dict(actual_joint_angles_mean=dict(zip([names[k] for k in columns],parts['q'][last][:,columns].mean(axis=0).tolist())),
            qualified_normal_load_mean_N=float(np.mean([r['pad_grasp']['qualified_pad_forces_N'][digit] for r in physical[last]])))
    return dict(passed=maxerror<1e-9,decisions=len(actual),maximum_policy_reconstruction_error_Nm=maxerror,
        motor_saturated_samples=saturations,robot_calculator_time_s=d.time,final_second_motors=summary,final_second_digits=digits,
        source_robot_sha256=sha(robot),input_sha256={n:sha(run/n) for n in ['provenance.json','controller.jsonl.gz','physics.jsonl.gz','actor-inputs.npz','trajectory.npz','reset.json']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('trial','baseline','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh reduction receipt required')
    actual=reduce(a.trial);baseline=reduce(a.baseline)
    result=dict(scope=__doc__,passed=actual['passed'] and baseline['passed'],trial=actual,baseline=baseline,
        comparison_scope='Different closed-loop attained equilibria and press phases; not a matched-state intervention or isolated causal gain estimate',
        source_sha256=sha(__file__))
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=result['passed'],trial_max_error=actual['maximum_policy_reconstruction_error_Nm'],baseline_max_error=baseline['maximum_policy_reconstruction_error_Nm']),indent=2))

if __name__=='__main__':main()
