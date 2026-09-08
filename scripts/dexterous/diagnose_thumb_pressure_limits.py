#!/usr/bin/env python3
"""Decompose pressure/posture error and actual thumb-contact moments at soft stops.

Actual contact forces come only from saved mj_step buffers. Jacobians, signed
distances and small pose perturbations are detached calculations; they neither
step physics nor establish the force response of a perturbed pose.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.grasp_verification import scalar_transmission_matrix
from doorbench.dexterous.robot_thumb_flexion_force import RobotThumbFlexionForce
from doorbench.dexterous.native_transition_archive import unpacked


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):
    with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('trial','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh diagnosis required')
    prov=json.loads((a.trial/'provenance.json').read_text());robot=Path(prov['parameters']['robot']);door=Path(prov['parameters']['door'])
    if sha(robot)!=prov['robot_xml_sha256'] or sha(door/'door.xml')!=prov['door_xml_sha256']:raise ValueError('Actual source design changed')
    m=mujoco.MjModel.from_xml_path(str(robot));names=[m.joint(i).name for i in range(1,m.njnt)];actions=[m.actuator(i).name for i in range(m.nu)]
    M=scalar_transmission_matrix(m,np.arange(61),np.arange(1,m.njnt));calc=RobotThumbFlexionForce(m,names,actions,M)
    info=lines(a.trial/'controller.jsonl.gz');physics=lines(a.trial/'physics.jsonl.gz')
    with np.load(a.trial/'actor-inputs.npz') as z:q=z['joint_position'].copy()
    with np.load(a.trial/'trajectory.npz') as z:actual_q=z['qpos'].copy();forces=z['force'].copy()
    motor=calc.groups['th'][1];thumb_all=[actions.index('rh_A_THJ'+str(k)) for k in (5,4,3,2,1)]
    no_pressure=[actions.index('rh_A_THJ'+str(k)) for k in (5,4,3)]
    out=[];first=None;first_stop=None;worst=None
    for i in range(9500,len(info)):
        r=info[i];goals=q[i].astype(float).copy()
        for n,v in zip(r['goal_joint_names'],r['goal_joint_position_rad']):goals[names.index(n)]=v
        K=r['effective_finger_position_gain_multiplier']*m.actuator_gainprm[:,0]
        error=M@(q[i]-goals);unit,_=calc.motor_bias(q[i],np.ones(5));punit=unit[motor];P=np.outer(punit,punit)/(punit@punit)
        # This is the actual effort-space projector: converting its components
        # through K^-1 differs from naively projecting angle errors by P.
        posture=-K[motor]*error[motor];normal=P@posture;tangent=posture-normal
        endpoint_error=M@(actual_q[i+1,10:]-goals)
        j2=names.index('rh_THJ2');stop=max(0.,float(m.jnt_range[m.joint('rh_THJ2').id,0]-actual_q[i+1,10+j2]))
        entry=dict(time_s=(i+1)*.002,pressure_motor_names=[actions[j] for j in motor],
            measured_motor_error_rad=error[motor].tolist(),actual_endpoint_motor_error_rad=endpoint_error[motor].tolist(),
            normal_removed_position_effort_Nm=normal.tolist(),retained_tangent_position_effort_Nm=tangent.tolist(),
            normal_effort_equivalent_error_rad=(-normal/K[motor]).tolist(),retained_tangent_equivalent_error_rad=(-tangent/K[motor]).tolist(),
            opposition_error_rad=error[no_pressure].tolist(),th2_lower_stop_penetration_rad=stop,
            actual_total_virtual_normal_N=r['normal_posture_transfer']['th']['total_virtual_normal_effort_N'],
            local_thumb_load_N=r['distal_projected_force_N'][4],qualified_thumb_normal_N=physics[i]['pad_grasp']['qualified_pad_forces_N']['th'])
        out.append(entry)
        if first is None and np.max(abs(endpoint_error[thumb_all]))>=.04:first=entry
        if first_stop is None and stop>0:first_stop=entry
        if worst is None or np.max(abs(endpoint_error[thumb_all]))>worst[0]:worst=(float(np.max(abs(endpoint_error[thumb_all]))),entry)
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()));scene=sim.m;d=mujoco.MjData(scene)
    lever=scene.geom('leaf_handle_lever_col_n').id;thbody=scene.body('robot/rh_thdistal').id
    geoms=[g for g in range(scene.ngeom) if scene.geom_bodyid[g]==thbody and ((scene.geom_contype[g]&scene.geom_conaffinity[lever]) or (scene.geom_contype[lever]&scene.geom_conaffinity[g]))]
    qa=[scene.joint('robot/'+n).qposadr[0] for n in names];va=[scene.joint('robot/'+n).dofadr[0] for n in names]
    thumbq=[qa[names.index('rh_THJ'+str(k))] for k in (2,1)];thumbv=[va[names.index('rh_THJ'+str(k))] for k in (2,1)]
    jp=np.zeros((3,scene.nv));jr=np.zeros_like(jp);moments=[];normal_moments=[];raw_last=None
    manifest=json.loads((a.trial/'actual-transitions/manifest.json').read_text())
    for chunk in manifest['chunks']:
        if chunk['interval_end_s']<=35.-1e-9:continue
        path=a.trial/'actual-transitions'/chunk['file']
        if sha(path)!=chunk['sha256']:raise ValueError('Actual force archive changed')
        with np.load(path) as z:raw=list(unpacked({k:z[k] for k in z.files}))
        for r in raw:
            if r['interval_start_s']<35.-1e-9:continue
            d.qpos[:]=r['qpos_before'];mujoco.mj_kinematics(scene,d);mujoco.mj_comPos(scene,d)
            tau=np.zeros(2);normal_tau=np.zeros(2)
            for c in r['contacts']:
                frame=np.array(c['frame_world']);w=np.array(c['wrench_contact_frame']);world=frame.T@w[:3];twist=frame.T@w[3:]
                for sign,body in zip((-1,1),c['body']):
                    if not scene.body(body).name.startswith('robot/rh_th'):continue
                    mujoco.mj_jac(scene,d,jp,jr,np.array(c['position_world_m']),body)
                    tau+=sign*(jp[:,thumbv].T@world+jr[:,thumbv].T@twist)
                    normal_tau+=sign*(jp[:,thumbv].T@(frame[0]*w[0]))
            moments.append(tau);normal_moments.append(normal_tau);raw_last=r
    if raw_last is None:raise ValueError('Final actual force interval required')
    base=np.array(raw_last['qpos_before']);perturb=[]
    for delta in ([0.,0.],[.002,0.],[-.002,0.],[0.,.002],[0.,-.002],[.03,0.],[.03,-.02]):
        d.qpos[:]=base;d.qpos[thumbq]+=delta;mujoco.mj_kinematics(scene,d);mujoco.mj_collision(scene,d)
        signed=[float(mujoco.mj_geomDistance(scene,d,g,lever,.05,None)) for g in geoms]
        perturb.append(dict(th2_th1_delta_rad=delta,minimum_distal_lever_signed_distance_m=min(signed),
            actual_joint_angles_rad=d.qpos[thumbq].tolist(),scope='Static derivative/interior-recovery candidate only; no force prediction'))
    last=out[-500:]
    result=dict(scope=__doc__,first_tracking_failure=first,first_authored_th2_stop_penetration=first_stop,worst_tracking=worst[1],
        maximum_retained_opposition_error_rad=float(max(np.max(np.abs(r['opposition_error_rad'])) for r in out)),
        maximum_retained_tangent_equivalent_error_rad=float(max(np.max(np.abs(r['retained_tangent_equivalent_error_rad'])) for r in out)),
        final_second={key:np.mean([r[key] for r in last],axis=0).tolist() for key in last[0] if key not in ('time_s','pressure_motor_names')},
        actual_contact_moment_thumb_th2_th1_mean_Nm=np.mean(moments,axis=0).tolist(),
        actual_normal_only_contact_moment_thumb_th2_th1_mean_Nm=np.mean(normal_moments,axis=0).tolist(),
        actual_contact_moment_intervals=len(moments),actual_contact_force_scope='Original saved mj_step wrenches with matching pre-integration FK Jacobians',
        joint_limit_force_scope='No actual constraint-torque buffer was archived; limit-force magnitude is not reconstructed from a fresh dynamics solve',
        final_pose_signed_distance_perturbations=perturb,calculators_never_stepped=calc.d.time==0 and d.time==0,
        source_sha256=sha(__file__),robot_xml_sha256=sha(robot),door_xml_sha256=sha(door/'door.xml'),
        inputs_sha256={n:sha(a.trial/n) for n in ['provenance.json','controller.jsonl.gz','physics.jsonl.gz','actor-inputs.npz','trajectory.npz','actual-transitions/manifest.json']})
    a.output.write_text(json.dumps(result,indent=2)+'\n');sim.close();print(json.dumps(result,indent=2))

if __name__=='__main__':main()
