#!/usr/bin/env python3
"""Compare actual attained hand/lever transforms and commanded palm correction."""
import argparse,gzip,hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.robot_palm_translation import ARM_NAMES
from doorbench.dexterous.sensor_acquisition_schedule import ScriptedAcquisitionSchedule
from doorbench.dexterous.sensor_handle_operation_schedule import ScriptedHandleOperationSchedule


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','trial','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh response receipt required')
    prov=json.loads((a.trial/'provenance.json').read_text());oldprov=json.loads((a.baseline/'provenance.json').read_text())
    robot=Path(prov['parameters']['robot']);door=Path(prov['parameters']['door']);door=door if door.is_dir() else door.parent
    for key,file in [('robot_xml_sha256',robot),('door_xml_sha256',door/'door.xml')]:
        if sha(file)!=prov[key] or prov[key]!=oldprov[key]:raise ValueError('Compared actual plants must match exactly')
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()),frame_skip=1);m=sim.m;d=mujoco.MjData(m)
    palm=m.site('robot/rh_palm_touch').id;handle=m.body('leaf_handle').id
    armqa=np.array([m.jnt_qposadr[m.joint('robot/'+n).id] for n in ARM_NAMES])
    outputs={};means={}
    for label,path in [('index_only',a.baseline),('palm_feedback',a.trial)]:
        state=np.load(path/'trajectory.npz');q=state['qpos'];times=state['time']
        infos=[json.loads(line) for line in gzip.open(path/'controller.jsonl.gz','rt')]
        ref=json.loads((path/'reference.json').read_text())['acquisition'];plan=json.loads((path/'press-plan.json').read_text())
        base=ScriptedAcquisitionSchedule(infos[0]['goal_joint_names'],ref['joint_names'],ref['path_qpos'])
        route=ScriptedHandleOperationSchedule(base,plan['joint_names'],plan['path_qpos'],press_seconds=plan['press_seconds'],settle_seconds=plan['settle_seconds'])
        relative=[];normals=[];arm_errors=[];corrections=[];counterfactual=[];loads=[];nominal_residual=[];shifted_tracking=[]
        for i in range(len(infos)-500,len(infos)):
            # Geometry at the actual force interval's beginning, never a
            # counterfactual constraint solve. Comparisons are between two
            # different attained plateaus, not a matched-state causal trial.
            d.qpos[:]=q[i];mujoco.mj_kinematics(m,d)
            Rh=d.xmat[handle].reshape(3,3).copy();ph=d.xpos[handle].copy()
            actual_p=d.site_xpos[palm].copy();relative.append(Rh.T@(actual_p-ph))
            n=np.mean([d.xmat[m.body('robot/rh_'+f+'distal').id].reshape(3,3)@np.array([0.,-1.,0.]) for f in ('ff','mf','rf','lf')],axis=0);n/=np.linalg.norm(n);normals.append(Rh.T@n)
            info=infos[i];nominal=route.goals(info['virtual_press_clock_s']);goal=dict(zip(info['goal_joint_names'],info['goal_joint_position_rad']))
            requested=np.array([goal[n] for n in ARM_NAMES]);unshifted=np.array([nominal[n] for n in ARM_NAMES]);delta=requested-unshifted
            arm_errors.append(requested-q[i,armqa]);corrections.append(delta);loads.append(info['distal_projected_force_N'])
            d.qpos[armqa]=unshifted;mujoco.mj_kinematics(m,d)
            nominal_residual.append((actual_p-d.site_xpos[palm])@n)
            d.qpos[armqa]=requested;mujoco.mj_kinematics(m,d)
            shifted_tracking.append((d.site_xpos[palm]-actual_p)@n)
            # Applying just the commanded delta to attained arm angles is an
            # independent local geometry sensitivity, not a physical replay.
            d.qpos[armqa]=q[i,armqa]+delta;mujoco.mj_kinematics(m,d)
            counterfactual.append((d.site_xpos[palm]-actual_p)@n)
        means[label]=np.mean(relative,axis=0)
        outputs[label]=dict(actual_palm_position_in_handle_mean_m=means[label].tolist(),mean_four_finger_normal_in_handle=np.mean(normals,axis=0).tolist(),
            actual_plateau_virtual_clock_s=infos[-1]['virtual_press_clock_s'],
            requested_palm_translation_m=infos[-1].get('palm_translation_m',0.),
            mean_static_command_shift_sensitivity_m=float(np.mean(counterfactual)),
            actual_palm_minus_unshifted_arm_fk_along_normal_mean_m=float(np.mean(nominal_residual)),
            shifted_arm_goal_fk_minus_actual_palm_along_normal_mean_m=float(np.mean(shifted_tracking)),
            mean_actual_arm_target_error_rad=dict(zip(ARM_NAMES,np.mean(arm_errors,axis=0).tolist())),
            mean_palm_command_joint_correction_rad=dict(zip(ARM_NAMES,np.mean(corrections,axis=0).tolist())),
            mean_local_distal_projection_N=np.mean(loads,axis=0).tolist(),
            actual_interval_start_range_s=[float(times[len(infos)-500]),float(times[len(infos)-1])])
    direction=np.array(outputs['index_only']['mean_four_finger_normal_in_handle']);direction/=np.linalg.norm(direction)
    delta=means['palm_feedback']-means['index_only']
    result=dict(scope=__doc__,between_run_actual_relative_palm_shift_handle_m=delta.tolist(),
        between_run_actual_relative_palm_shift_along_four_finger_normal_m=float(delta@direction),runs=outputs,
        between_run_difference_actual_palm_minus_unshifted_arm_fk_m=outputs['palm_feedback']['actual_palm_minus_unshifted_arm_fk_along_normal_mean_m']-outputs['index_only']['actual_palm_minus_unshifted_arm_fk_along_normal_mean_m'],
        interpretation='Actual plateau geometry is measured but plateaus differ in press phase and load. Static delta-on-attained-state is a counterfactual geometry sensitivity, not realized displacement or force. No isolated matched-state displacement claim.',
        source_sha256=sha(__file__),inputs_sha256={str(f):sha(f) for path in [a.baseline,a.trial] for f in [path/'provenance.json',path/'trajectory.npz',path/'controller.jsonl.gz']})
    a.output.write_text(json.dumps(result,indent=2)+'\n');sim.close();print(json.dumps(result,indent=2))


if __name__=='__main__':main()
