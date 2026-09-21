#!/usr/bin/env python3
"""Replay the actual controller on recorded observations, without advancing physics.

Prefix commands must reproduce the recorded motor commands. The proposed suffix
is counterfactual: recorded observations do not become outcomes of new commands.
"""
import argparse
import gzip
import json
from pathlib import Path

import mujoco
import numpy as np

from doorbench.dexterous.acquisition_teacher import AcquisitionTeacher
from doorbench.dexterous.coupled_release_geometry import sha
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.json_record_stream import iter_json_object_array
from doorbench.dexterous.operation_teacher import DoorOperationTeacher
from doorbench.dexterous.provenance import capture
from doorbench.dexterous.resting_transfer import RestingTransferBridge
from doorbench.dexterous.standing_transfer import StandingTransferTeacher
from doorbench.dexterous.standing_withdrawal import StandingWithdrawalTeacher


def cached_chunk(path,expected_sha256):
    """Decompress each required array once, never once per physical sample."""
    if sha(path)!=expected_sha256:raise ValueError('Recorded raw chunk changed')
    names=('interval_start_s','qpos_before','qvel_before','controls','contact_offsets',
           'contact_frame_world','contact_wrench_contact_frame','contact_body')
    with np.load(path,allow_pickle=False) as archive:
        arrays={name:archive[name] for name in names}
    if sha(path)!=expected_sha256:raise ValueError('Recorded raw chunk changed while reading')
    for array in arrays.values():array.setflags(write=False)
    return arrays


def check_bindings(bindings):
    for name,expected in bindings.items():
        if sha(name)!=expected:raise ValueError('Counterfactual replay input changed: '+name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recorded-run',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh counterfactual command replay required')
    bound_paths=[a.recorded_run/name for name in ('manifest.json','physics-steps.json.gz','controller-steps.json.gz','raw-transitions/manifest.json')]+[a.config]
    bindings={str(path.resolve()):sha(path) for path in bound_paths}
    manifest=json.loads((a.recorded_run/'manifest.json').read_text());cfg=manifest['configuration']
    if (not cfg['portable_wrapper'] or not cfg['standing_measured_rest'] or cfg.get('standing_return_path')
            or cfg.get('jev_progress_plan') or cfg.get('standing_transfer_hybrid_support')):
        raise ValueError('Explicit deterministic native measured-rest source without unsupported transfer hybrid feedback required')
    robot=Path(cfg['robot']);door=Path(cfg['door'])
    proposed=json.loads(a.config.read_text())
    bound_paths=[robot,robot.with_suffix('.audit.json'),door/'door.xml',Path(cfg['motors']),Path(cfg['reference']),Path(cfg['standing_transfer_path'])]
    bound_paths += [Path(proposed[key]) for key in ('coupled_envelope_path','coupled_audit_path') if proposed.get(key)]
    bindings.update({str(path.resolve()):sha(path) for path in bound_paths})
    motors=json.loads(Path(cfg['motors']).read_text());ref=json.loads(Path(cfg['reference']).read_text())
    captured=capture(Path(__file__).resolve().parents[2],a.output,dict(robot=str(robot),door=str(door),recorded_run=str(a.recorded_run.resolve()),proposed_config=str(a.config.resolve()),physics_steps=0,scope=__doc__))
    from doorbench.dexterous.controller_input_snapshot import snapshot_controller_inputs
    snapshot_controller_inputs([Path(cfg['reference']),Path(cfg['motors']),Path(cfg['standing_transfer_path']),a.config,
        *[Path(proposed[key]) for key in ('coupled_envelope_path','coupled_audit_path') if proposed.get(key)]],a.output/'controller-inputs')
    scene=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()));m,d=scene.m,scene.d
    hub=m.geom('leaf_handle_hub_col_n').id
    geometry=dict(size=m.geom_size[hub].tolist(),position=m.geom_pos[hub].tolist(),quaternion_wxyz=m.geom_quat[hub].tolist()) if cfg['operation_handle_hub_avoidance'] else None
    teacher=AcquisitionTeacher(robot,motors,ref,handle_hub_geometry=geometry,landed_foot_max_iterations=cfg['landed_foot_max_iterations'],stance_profile=cfg['stance_profile'],pressure_segment=cfg['pressure_segment'],middle_finger_force=cfg['middle_finger_force'],index_finger_force=cfg['index_finger_force'])
    hj,lj,bj=[m.joint(n).id for n in ('leaf_handle_hinge','leaf_hinge','leaf_latch_bolt_slide')]
    operation=DoorOperationTeacher(teacher,dict(operator_origin=m.jnt_pos[hj],operator_axis=m.jnt_axis[hj],leaf_origin=m.jnt_pos[lj],leaf_axis=m.jnt_axis[lj]),release_operator_threshold=cfg['operation_opening_trigger_rad'],handle_hub_avoidance=cfg['operation_handle_hub_avoidance'],hub_clearance_m=cfg['operation_hub_clearance_m'],leaf_target=cfg['operation_leaf_target_rad'],leaf_lead_limit_rad=cfg['operation_leaf_lead_limit_rad'],operator_lead_limit_rad=cfg['operation_operator_lead_limit_rad'],operator_follow_after_leaf_rad=cfg['operation_operator_follow_after_leaf_rad'],min_acquisition_seconds=cfg['min_acquisition_seconds'],press_seconds=cfg['press_seconds'],wait_for_press_completion=not cfg['open_on_latch_clear'],operator_compliance_gain=cfg['operator_compliance_gain'],grasp_offset_in_handle_m=cfg['grasp_offset_in_handle_m'],index_proximal_offset_rad=cfg['index_proximal_offset_rad'],index_tendon_offset_rad=cfg['index_tendon_offset_rad'],fixed_pad_control=cfg['operation_fixed_pad_control'],pad_control_profile=cfg['operation_pad_control_profile'],hold_attained_grasp=cfg['hold_attained_grasp'],attained_hold_stage=cfg['attained_hold_stage'])
    transfer=StandingTransferTeacher(operation,motors,cfg['standing_transfer_path'],start_seconds=cfg['standing_transfer_start_seconds'],fixed_pad_tracking=not cfg['standing_transfer_no_fixed_pads'],attained_arm_tracking=cfg['standing_transfer_attained_arm'],preload_profile=cfg['standing_transfer_preload_profile'],grasp_shift=cfg['standing_transfer_grasp_shift'],hold_route=cfg['standing_transfer_hold_route'],handoff_seconds=cfg['standing_transfer_handoff_seconds'],handle_relative_arm=cfg['standing_transfer_handle_relative_arm'],leaf_relative_arm=cfg['standing_transfer_leaf_relative_arm'],support_load_target=4. if cfg['standing_transfer_support_load'] is None else cfg['standing_transfer_support_load'])
    controller=StandingWithdrawalTeacher(RestingTransferBridge(transfer),motors,a.config)
    ids=np.array([m.joint('robot/'+n).id for n in teacher.names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids]
    aids=np.array([m.actuator('robot/'+v['name']).id for v in motors['actuators']]);hb=m.body('leaf_handle').id;lb=m.body('leaf').id
    hand_names={b:m.body(b).name.removeprefix('robot/') for b in range(m.nbody) if m.body(b).name.startswith(('robot/rh_','robot/lh_'))}
    loads={n:np.zeros(3) for n in hand_names.values()}
    archive=a.recorded_run/'raw-transitions';raw_manifest=json.loads((archive/'manifest.json').read_text())
    if not raw_manifest['complete']:raise ValueError('Complete actual recording required')
    count=prefix=0;maximum_prefix_error=0.;maximum_clock_error=0.;failure=None;started=None;last=None;last_info={};first_capture_error=None
    clocks={}
    with gzip.open(a.recorded_run/'controller-steps.json.gz','rt') as f:
        for row in iter_json_object_array(f):
            if 'withdrawal_clock_s' in row:clocks[round(row['time_s'],9)]=row['withdrawal_clock_s']
    with gzip.open(a.recorded_run/'physics-steps.json.gz','rt') as stream:
        observations=iter_json_object_array(stream);observation=next(observations)
        for chunk in raw_manifest['chunks']:
            file=(archive/chunk['file']).resolve()
            if file.parent!=archive.resolve():raise ValueError('Raw chunk must stay within the recorded archive')
            rows=cached_chunk(file,chunk['sha256']);bindings[str(file)]=chunk['sha256']
            for i,t in enumerate(rows['interval_start_s']):
                t=float(t)
                try:
                    if abs(observation['sim_time_s']-t)>1e-8:raise ValueError('Actual observations and motor interval must share a clock')
                    d.qpos[:]=rows['qpos_before'][i];d.qvel[:]=rows['qvel_before'][i];mujoco.mj_kinematics(m,d)
                    root=np.r_[d.qpos[scene.root_qadr:scene.root_qadr+7],d.qvel[scene.root_vadr:scene.root_vadr+3],d.xmat[scene.pelvis].reshape(3,3)@d.qvel[scene.root_vadr+3:scene.root_vadr+6]]
                    joints=dict(zip(teacher.names,d.qpos[qa]));velocities=dict(zip(teacher.names,d.qvel[va]));left=observation.get('left_surface',{})
                    force,info=controller.force(t,root,joints,velocities,np.r_[d.xpos[hb],d.xquat[hb]],np.r_[d.xpos[lb],d.xquat[lb]],dict(operator=d.qpos[m.jnt_qposadr[hj]],leaf=d.qpos[m.jnt_qposadr[lj]],latch=d.qpos[m.jnt_qposadr[bj]]),loads,grasp_qualified=observation['pad_grasp']['valid_pad_grasp'],left_panel_load=left.get('total_normal_load_N',0.),left_palm_load=left.get('palm_normal_load_N',0.))
                    if not np.isfinite(force).all() or np.any(force<teacher.caps[:,0]-1e-10) or np.any(force>teacher.caps[:,1]+1e-10):raise ValueError('Original finite motor caps required')
                    expected=rows['controls'][i,aids]
                    if t<controller.start_time-1e-8:
                        error=float(np.max(abs(force-expected)));maximum_prefix_error=max(maximum_prefix_error,error);prefix+=1
                        if error>1e-7:raise ValueError('Counterfactual teacher prefix diverged from actual controls by '+str(error)+' Nm')
                    else:
                        if started is None:started=t;first_capture_error=float(np.max(abs(force-expected)))
                        error=abs(info['withdrawal_clock_s']-clocks[round(t,9)]);maximum_clock_error=max(maximum_clock_error,error)
                        if error>1e-10:raise ValueError('Actual teacher clock differs from recorded once-warped clock')
                    last_info=info;last=t;count+=1
                    loads={n:np.zeros(3) for n in hand_names.values()}
                    begin,end=rows['contact_offsets'][i:i+2]
                    for k in range(begin,end):
                        force_world=rows['contact_frame_world'][k].reshape(3,3).T@rows['contact_wrench_contact_frame'][k,:3]
                        for sign,body in zip((-1,1),rows['contact_body'][k]):
                            if int(body) in hand_names:loads[hand_names[int(body)]]+=sign*force_world
                    observation=next(observations)
                except Exception as exc:
                    failure=dict(time_s=t,error=type(exc).__name__+': '+str(exc),coupled_failure=getattr(controller.coupled,'failure_snapshot',None));break
            print(json.dumps(dict(recorded_time_s=last,controller_evaluations=count,maximum_prefix_error_Nm=maximum_prefix_error)),flush=True)
            if failure:break
    unchanged=True
    try:check_bindings(bindings)
    except ValueError as exc:
        unchanged=False
        failure=dict(error=str(exc),prior_controller_failure=failure)
    result=dict(schema='doorbench.counterfactual-coupled-command-replay.v1',passed=failure is None and count==sum(v['rows'] for v in raw_manifest['chunks']),scope=__doc__,physics_steps=0,physical_admission=False,controller_evaluations=count,prefix_evaluations=prefix,maximum_prefix_motor_error_Nm=maximum_prefix_error,first_capture_s=started,first_capture_motor_error_Nm=first_capture_error,last_accepted_s=last,maximum_recorded_clock_error_s=maximum_clock_error,failure=failure,final_info=last_info,recorded_raw_manifest_sha256=bindings[str((archive/'manifest.json').resolve())],input_sha256=bindings,inputs_unchanged=unchanged,runtime_source_archive_sha256=captured['source_archive_sha256'])
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');scene.close()
    print(json.dumps({k:v for k,v in result.items() if k!='final_info'},indent=2))


if __name__=='__main__':main()
