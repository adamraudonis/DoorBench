#!/usr/bin/env python3
"""Join actual contact moments to the exact same-step panel reference trace.

This is evaluator-only privileged analysis. It never steps a plant or solves
new contact forces. The native frictionloss value is reported as a model limit,
not as a measured friction-constraint multiplier.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np



def validate_panel_reference_row(row, previous=None):
    """Reject corrupt target evidence and check the actually consumed LH path."""
    sizes={'previous_contact_interval_s':2,'body_coordinates':6,'planned_coordinates':31,
           'planned_velocity':31,'planned_acceleration':31,'latched_motor_targets':61,
           'left_arm_targets':7,'left_arm_target_velocity':7}
    for key,size in sizes.items():
        value=np.asarray(row[key],float)
        if value.shape!=(size,) or not np.isfinite(value).all():
            raise ValueError('Require complete finite panel target evidence: '+key)
    scalars=['episode_time_s','local_time_s','opening_clock_offset_s','pose_time_s',
             'reference_aperture_rad','actual_aperture_rad','normal_feedforward_N','tracking_lead_rad']
    if not np.isfinite([row[k] for k in scalars]).all():
        raise ValueError('Require finite target clocks and commands')
    velocity=np.asarray(row['left_arm_target_velocity']);speed=float(max(abs(velocity)));acceleration=0.
    if speed>1.2+1e-9:raise ValueError('Actual consumed LH target exceeds original speed bound')
    correction=row.get('actual_base_correction')
    if correction is not None:
        if not np.isfinite(list(correction.values())).all():raise ValueError('Nonfinite correction evidence')
        if abs(correction['time_s']-row['local_time_s'])>1e-9:raise ValueError('Correction and target clocks differ')
        if previous is not None:
            dt=row['episode_time_s']-previous['episode_time_s']
            if not abs(dt-.002)<1e-9:raise ValueError('Missing corrected target interval')
            previous_velocity=np.asarray(previous['left_arm_target_velocity'])
            acceleration=float(max(abs((velocity-previous_velocity)/dt)))
            change=np.asarray(row['left_arm_targets'])-np.asarray(previous['left_arm_targets'])
            if np.max(abs(change-.5*dt*(velocity+previous_velocity)))>1e-12:
                raise ValueError('Consumed LH targets differ from bounded target integration')
            if acceleration>3.+1e-8:raise ValueError('Consumed LH target exceeds original acceleration bound')
    chain=row.get('corrected_chain_targets')
    if chain is not None:
        names=row['corrected_chain_joint_names'];chain=np.asarray(chain,float);chainv=np.asarray(row['corrected_chain_velocity'],float)
        expected=['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1']
        if names not in (expected,['torso']+expected) or chain.shape!=(len(names),) or chainv.shape!=chain.shape or not np.isfinite(np.r_[chain,chainv]).all():
            raise ValueError('Require the declared finite corrected actuator chain')
        if not np.array_equal(chain[-7:],row['left_arm_targets']) or not np.array_equal(chainv[-7:],velocity):
            raise ValueError('Corrected chain and consumed LH targets differ')
        if np.max(abs(chainv))>1.2+1e-9:raise ValueError('Corrected chain exceeds original speed bound')
        if previous is not None and previous.get('corrected_chain_targets') is not None:
            prior=np.asarray(previous['corrected_chain_targets']);priorv=np.asarray(previous['corrected_chain_velocity']);dt=row['episode_time_s']-previous['episode_time_s']
            if prior.shape!=chain.shape or np.max(abs(chain-prior-.5*dt*(chainv+priorv)))>1e-12 or np.max(abs((chainv-priorv)/dt))>3.+1e-8:
                raise ValueError('Corrected whole chain differs from bounded target integration')
    normal=row.get('normal_admittance')
    if normal is not None:
        if not np.isfinite(list(normal.values())).all():raise ValueError('Nonfinite palm-normal evidence')
        if abs(normal['time_s']-row['local_time_s'])>1e-9:raise ValueError('Palm-normal target and pose clocks differ')
        if (not 0.-1e-12<=normal['normal_offset_m']<=.002+1e-12
                or abs(normal['normal_offset_velocity_m_s'])>.00025+1e-12
                or abs(normal['normal_offset_acceleration_m_s2'])>.0005+1e-12):
            raise ValueError('Normal target exceeds its frozen geometry/rate envelope')
        if previous is not None and previous.get('normal_admittance') is not None:
            prior=previous['normal_admittance'];dt=row['episode_time_s']-previous['episode_time_s']
            change=normal['normal_offset_m']-prior['normal_offset_m']
            expected=.5*dt*(normal['normal_offset_velocity_m_s']+prior['normal_offset_velocity_m_s'])
            if abs(change-expected)>1e-12:raise ValueError('Normal target differs from bounded integration')
    return speed,acceleration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run=args.run.resolve();source=run.with_name(run.name+'-source')
    sys.path.insert(0,str(source))
    from doorbench.dexterous.environment import DexterousDoorEnv
    from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
    helper_path=Path(__file__).with_name('audit_leaf_contact_moment.py')
    spec=importlib.util.spec_from_file_location('_actual_hinge_moment',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    config=json.loads((run/'manifest.json').read_text())['configuration']
    robot=Path(config['robot'])
    digest=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
    if digest(robot)!=digest(run/'robot-input.xml'):
        raise ValueError('The archived actual robot source changed')
    with gzip.open(run/'panel-phase.jsonl.gz','rt') as file:
        phase=[json.loads(line) for line in file]
    if not phase:raise ValueError('Require the complete panel reference trace')
    times=np.array([row['episode_time_s'] for row in phase])
    if not np.isfinite(times).all() or not np.allclose(np.diff(times),.002,rtol=0,atol=1e-9):
        raise ValueError('Require consecutive 2ms panel reference rows')
    run_report=json.loads((run/'report.json').read_text())
    if run_report.get('opening_handoff') is not None:
        if abs(times[-1]+.002-run_report['opening_handoff']['time_s'])>1e-8:
            raise ValueError('Panel target trace does not reach the selected continuation handoff')
    simulation=DexterousDoorEnv(config['door'],robot,json.loads(robot.with_suffix('.audit.json').read_text()))
    model,data=simulation.m,simulation.d
    joint=model.joint('leaf_hinge').id;leaf=model.body('leaf').id
    qa=model.jnt_qposadr[joint];va=model.jnt_dofadr[joint]
    def belongs_to_leaf(body):
        while body:
            if body==leaf:return True
            body=model.body_parentid[body]
        return False
    members={b for b in range(model.nbody) if belongs_to_leaf(b)}
    index=0;values=[];max_frame_error=0.;max_phase_error=0.;max_force_excess=0.;previous_row=None;max_torso_delivery_error=0.
    for raw in NativeTransitionArchive.read(run/'raw-transitions'):
        time=float(raw['interval_start_s'])
        if time<times[0]-1e-9:continue
        if time>times[-1]+1e-9:
            # A continuous episode may release the panel and traverse later.
            # The complete selected panel interval remains audited below.
            continue
        if index>=len(phase) or abs(time-times[index])>1e-9:
            raise ValueError('Actual intervals and panel targets do not align')
        row=phase[index];index+=1
        consumed_speed,consumed_acceleration=validate_panel_reference_row(row,previous_row);previous_row=row
        if abs(row['pose_time_s']-time)>1e-9 or not np.allclose(row['previous_contact_interval_s'],[time-.002,time],rtol=0,atol=1e-9):
            raise ValueError('Current pose and preceding force clocks were mixed')
        data.qpos[:]=raw['qpos_before'];mujoco.mj_kinematics(model,data)
        ids=raw['body_ids']
        frame_error=max(float(np.max(abs(data.xpos[ids]-raw['body_positions_world_m']))),float(np.max(abs(data.xmat[ids].reshape(-1,3,3)-raw['body_rotations_world']))))
        max_frame_error=max(max_frame_error,frame_error)
        max_phase_error=max(max_phase_error,abs(float(data.qpos[qa])-row['actual_aperture_rad']))
        if frame_error>1e-9 or max_phase_error>1e-12:
            raise ValueError('Actual body geometry or current leaf observation differs')
        forces=np.asarray(raw['actuator_force'])
        max_force_excess=max(max_force_excess,float(np.max(forces-model.actuator_forcerange[:,1])),float(np.max(model.actuator_forcerange[:,0]-forces)))
        torso=row.get('actual_torso_force')
        if torso is not None:
            if not np.isfinite(list(torso.values())).all():raise ValueError('Nonfinite actual torso delivery evidence')
            aid=model.actuator('robot/torso').id
            error=abs(float(forces[aid])-torso['consumed_torso_effort_Nm']);max_torso_delivery_error=max(max_torso_delivery_error,error)
            if error>1e-9:raise ValueError('Corrected torso effort was overwritten before the actual plant step')
        normal_moment=0.;tangential_moment=0.;couple_moment=0.;palm_load=0.;finger_load=0.
        for contact in raw['contacts']:
            membership=[int(b) in members for b in contact['body']]
            if sum(membership)!=1:continue
            side=membership.index(True);other=model.body(int(contact['body'][1-side])).name
            frame=np.asarray(contact['frame_world']);wrench=np.asarray(contact['wrench_contact_frame'])
            full,normal,_=helper.contact_moment_about_axis(data.xanchor[joint],data.xaxis[joint],contact['position_world_m'],frame,wrench,side)
            couple=float(data.xaxis[joint]@((1 if side==1 else -1)*(frame.T@wrench[3:])))
            normal_moment+=normal;couple_moment+=couple;tangential_moment+=full-normal-couple
            if other=='robot/lh_palm':palm_load+=max(0.,float(wrench[0]))
            elif other.startswith('robot/lh_'):finger_load+=max(0.,float(wrench[0]))
        velocity=np.asarray(row['planned_velocity']);acceleration=np.asarray(row['planned_acceleration'])
        joint_speed=float(np.max(abs(velocity[6:])));joint_acceleration=float(np.max(abs(acceleration[6:])))
        root_speed=float(np.linalg.norm(velocity[:3]));root_rotation_speed=float(np.linalg.norm(velocity[3:6]))
        if joint_speed>1.2 or joint_acceleration>3. or root_speed>.02 or root_rotation_speed>.03:
            raise ValueError('The archived consumed reference exceeds the unchanged motion bounds')
        leaf_speed=float(raw['qvel_before'][va])
        values.append([time,row['reference_aperture_rad'],float(data.qpos[qa]),leaf_speed,
                       row['tracking_lead_rad'],row['reference_aperture_rad']-float(data.qpos[qa]),
                       normal_moment,tangential_moment,couple_moment,normal_moment+tangential_moment+couple_moment,
                       -float(model.dof_damping[va])*leaf_speed,palm_load,finger_load,
                       joint_speed,joint_acceleration,root_speed,root_rotation_speed,consumed_speed,consumed_acceleration])
    if index!=len(phase):raise ValueError('The panel target trace extends beyond the actual dynamics')
    columns=['episode_time_s','reference_leaf_rad','actual_leaf_rad','actual_leaf_velocity_rad_s',
             'desired_lead_rad','actual_reference_lead_rad','normal_contact_moment_Nm','tangential_contact_moment_Nm',
             'contact_couple_moment_Nm','total_contact_moment_Nm','original_viscous_hinge_moment_Nm',
             'actual_palm_normal_load_N','actual_other_lh_normal_load_N','target_joint_speed_rad_s',
             'target_joint_acceleration_rad_s2','target_root_speed_m_s','target_root_rotvec_speed_rad_s',
             'consumed_lh_target_speed_rad_s','consumed_lh_target_acceleration_rad_s2']
    array=np.array(values)
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'phase-moment.npz',values=array)
    result=dict(scope='Actual current-interval wrenches joined to same-time reference inputs; evaluator-only. Original frictionloss is a model limit, not a measured multiplier.',
                samples=len(values),first_interval_start_s=float(array[0,0]),last_interval_start_s=float(array[-1,0]),
                columns=columns,maximum_actual_body_frame_error=max_frame_error,maximum_current_leaf_observation_error=max_phase_error,
                maximum_original_motor_cap_excess=max_force_excess,maximum_corrected_torso_delivery_error_Nm=max_torso_delivery_error,original_hinge_frictionloss_limit_Nm=float(model.dof_frictionloss[va]),
                minimums=dict(zip(columns,array.min(axis=0).tolist())),maximums=dict(zip(columns,array.max(axis=0).tolist())),
                final=dict(zip(columns,array[-1].tolist())),raw_manifest_sha256=digest(run/'raw-transitions/manifest.json'),
                phase_trace_sha256=digest(run/'panel-phase.jsonl.gz'),moment_helper_sha256=digest(helper_path),
                original_trial_passed=json.loads((run/'report.json').read_text())['passed'])
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output/'audit-source.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'moment-helper-source.py').write_bytes(helper_path.read_bytes())
    simulation.close();print(json.dumps(result,indent=2))


if __name__=='__main__':main()
