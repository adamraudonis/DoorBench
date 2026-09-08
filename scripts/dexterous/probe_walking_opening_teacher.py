#!/usr/bin/env python3
"""Native development of continuous walking, preparation and full opening."""
import argparse
import gzip
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.grasp_verification import native_grasp_sample, audited_native_step, audit_grasp_steps
from doorbench.dexterous.hand_surface_audit import native_hand_surface_loads
from doorbench.dexterous.provenance import capture
from doorbench.dexterous.native_post_opening_measurements import measured_contacts, measured_state, outward_release_normal

transition_spec = importlib.util.spec_from_file_location('doorbench.dexterous.native_transition_audit', Path(__file__).resolve().parents[2]/'doorbench/dexterous/native_transition_audit.py')
transition_module = importlib.util.module_from_spec(transition_spec)
sys.modules[transition_spec.name] = transition_module
transition_spec.loader.exec_module(transition_module)
NativeTransitionRecorder=transition_module.NativeTransitionRecorder
archive_spec=importlib.util.spec_from_file_location('doorbench.dexterous.native_transition_archive',Path(__file__).resolve().parents[2]/'doorbench/dexterous/native_transition_archive.py')
archive_module=importlib.util.module_from_spec(archive_spec);sys.modules[archive_spec.name]=archive_module;archive_spec.loader.exec_module(archive_module)
NativeTransitionArchive=archive_module.NativeTransitionArchive

left_spec = importlib.util.spec_from_file_location('doorbench.dexterous.bimanual_transfer', Path(__file__).resolve().parents[2]/'doorbench/dexterous/bimanual_transfer.py')
left_module = importlib.util.module_from_spec(left_spec)
sys.modules[left_spec.name] = left_module
left_spec.loader.exec_module(left_module)
panel_spec = importlib.util.spec_from_file_location('doorbench.dexterous.panel_continuation', Path(__file__).resolve().parents[2]/'doorbench/dexterous/panel_continuation.py')
panel_module = importlib.util.module_from_spec(panel_spec)
sys.modules[panel_spec.name] = panel_module
panel_spec.loader.exec_module(panel_module)
spec = importlib.util.spec_from_file_location('_full_opening_teacher', Path(__file__).resolve().parents[2]/'doorbench/dexterous/full_opening_teacher.py')
module = importlib.util.module_from_spec(spec)
sys.modules['_full_opening_teacher'] = module
spec.loader.exec_module(module)
FullOpeningTeacher = module.FullOpeningTeacher


def physical_sample_passed(row):
    return bool(row['finite'] and row['numerical_warnings']==0 and
                row.get('mujoco_warning_interval',{}).get('passed',True) and
                row['root_height_m']>.7 and row['torso_tilt_deg']<12 and
                row['max_joint_limit_violation_rad']<=.02 and
                row['max_nonfoot_penetration_m']<=.003 and
                row['max_shadow_loopback_violation_rad']<=.02 and
                row['native_motor_limits'] and row['external_wrench_max']==0 and
                row['applied_generalized_force_max']==0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','reference','motors','plan','release_path','output','body_reset','preparation','checkpoint'):
        p.add_argument('--'+name.replace('_','-'), type=Path, required=True)
    p.add_argument('--runtime-screen', type=Path)
    p.add_argument('--seconds', type=float, default=85.)
    p.add_argument('--target-aperture', type=float, default=1.2)
    p.add_argument('--open-on-latch-clear',action='store_true')
    p.add_argument('--operator-compliance-gain',type=float,default=0.)
    p.add_argument('--follow-leaf-during-transfer',action='store_true')
    p.add_argument('--panel-profile',choices=('plain-v1','hybrid-surface-v2'),default='hybrid-surface-v2')
    p.add_argument('--palm-load-target',type=float)
    p.add_argument('--transfer-load-target',type=float,default=4.,help='Pre-release total left-panel load target; original motor caps unchanged')
    p.add_argument('--traverse',action='store_true',help='Require one uninterrupted qualified opening, stow, rise, passage and quiet finish')
    p.add_argument('--whole-body-return-path',type=Path,help='Opt-in exact-attained-state screened body path; returns lever while retaining grip, not a completed withdrawal')
    a = p.parse_args()
    if a.output.exists(): raise SystemExit('Use a new output directory')
    ref=json.loads(a.reference.read_text());motors=json.loads(a.motors.read_text())
    reset=json.loads(a.body_reset.read_text());preparation=json.loads(a.preparation.read_text())
    if not json.loads((a.reference.parent/'geometry-audit.json').read_text())['passed']:
        raise ValueError('Unscreened acquisition route')
    root_source = Path(inspect.getfile(DexterousDoorEnv)).resolve().parents[2]
    capture(root_source, a.output, {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    for key in ('body_reset','preparation','checkpoint'):
        source=getattr(a,key);shutil.copy2(source,a.output/(key+source.suffix))
    if a.whole_body_return_path is not None:
        shutil.copy2(a.whole_body_return_path,a.output/'whole-body-return-path.json')
    for source,name in ((Path(__file__),'diagnostic-source.py'),(Path(inspect.getfile(FullOpeningTeacher)),'full-opening-teacher-source.py'),(a.robot,'robot-input.xml'),(a.robot.with_suffix('.audit.json'),'robot-input.audit.json'),(a.motors,'motors-input.json'),(a.door/'door.xml','door-input.xml'),(a.reference,'reference.json'),(a.plan,'static-plan.json'),(a.release_path,'release-path.json')):
        shutil.copy2(source,a.output/name)
    shutil.copy2(Path(panel_spec.origin),a.output/'panel-continuation-source.py')
    shutil.copy2(Path(left_spec.origin),a.output/'bimanual-transfer-source.py')
    shutil.copy2(Path(transition_spec.origin),a.output/'native-transition-audit-source.py')
    shutil.copy2(Path(archive_spec.origin),a.output/'native-transition-archive-source.py')
    sim = DexterousDoorEnv(a.door,a.robot,json.loads(a.robot.with_suffix('.audit.json').read_text()))
    m,d=sim.m,sim.d;sim.reset(randomize=False,images=False)
    hj=m.joint('leaf_handle_hinge').id;lj=m.joint('leaf_hinge').id;bj=m.joint('leaf_latch_bolt_slide').id
    hb=m.body('leaf_handle').id;leaf=m.body('leaf').id
    from doorbench.dexterous.walking_opening_teacher import WalkingOpeningTeacher
    from doorbench.dexterous.continuous_door_teacher import ContinuousDoorTeacher, ContinuousDoorFailure
    controller_type=ContinuousDoorTeacher if a.traverse else WalkingOpeningTeacher
    controller=controller_type(a.robot,motors,ref,preparation,reset,a.checkpoint,
        dict(operator_origin=m.jnt_pos[hj],operator_axis=m.jnt_axis[hj],leaf_origin=m.jnt_pos[lj],leaf_axis=m.jnt_axis[lj]),
        door_xml=a.door,left_targets=a.plan,release_screen=a.release_path,runtime_screen=a.runtime_screen,
        opening_options=dict(target_aperture=a.target_aperture,open_on_latch_clear=a.open_on_latch_clear,
            operator_compliance_gain=a.operator_compliance_gain,follow_leaf_during_transfer=a.follow_leaf_during_transfer,panel_profile=a.panel_profile,palm_load_target=a.palm_load_target,transfer_load_target=a.transfer_load_target,whole_body_return_path=a.whole_body_return_path),
        **({'maximum_seconds':a.seconds} if a.traverse else {}))
    sequence=controller.walking if a.traverse else controller
    opening=sequence.opening
    teacher=opening.acquisition
    d.qpos[sim.root_qadr:sim.root_qadr+7]=reset['initial_root']
    ids=np.array([m.joint('robot/'+n).id for n in teacher.names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids]
    d.qpos[qa]=[reset['joints'][n] for n in teacher.names];d.qvel[:]=0.;mujoco.mj_forward(m,d)
    aids=np.array([m.actuator('robot/'+v['name']).id for v in motors['actuators']])
    m.actuator_gainprm[aids,0]=1.;m.actuator_biasprm[aids,:3]=0.;m.actuator_ctrlrange[aids]=teacher.caps;d.ctrl[aids]=0.
    hand_names={b:m.body(b).name.removeprefix('robot/') for b in range(m.nbody) if m.body(b).name.startswith(('robot/rh_','robot/lh_'))}
    geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
    lever=m.geom('leaf_handle_lever_col_n').id
    feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')];foot_loads=np.zeros(2)
    initial_feet=d.xpos[feet].copy();maximum_foot_lift=np.zeros(2)
    immutable=('body_mass','body_inertia','body_gravcomp','jnt_range','tendon_range','geom_friction','geom_contype','geom_conaffinity','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange')
    originals={k:getattr(m,k).copy() for k in immutable}
    physics=[native_grasp_sample(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge')]
    recorder=NativeTransitionRecorder(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge')
    _,previous_raw=transition_module._contact_solution(sim)
    previous_raw.update(interval_start_s=0.,interval_end_s=0.)
    raw_stream=NativeTransitionArchive(a.output/'raw-transitions')
    completed=False;controller_error=None
    states={key:[] for key in ('qpos','qvel','ctrl')};traces=[]
    try:
        for step in range(round(a.seconds/m.opt.timestep)):
            previous=physics[-1]
            loads=recorder.hand_forces
            rotation=d.xmat[sim.pelvis].reshape(3,3)
            root=np.r_[d.qpos[sim.root_qadr:sim.root_qadr+7],d.qvel[sim.root_vadr:sim.root_vadr+3],rotation@d.qvel[sim.root_vadr+3:sim.root_vadr+6]]
            angles=dict(operator=float(d.qpos[m.jnt_qposadr[hj]]),leaf=float(d.qpos[m.jnt_qposadr[lj]]),latch=float(d.qpos[m.jnt_qposadr[bj]]))
            surface=recorder.left_surface
            gap=min(float(mujoco.mj_geomDistance(m,d,g,lever,1.,None)) for g in geoms)
            evidence=dict(grasp_qualified=previous['pad_grasp']['valid_pad_grasp'],physics_qualified=physical_sample_passed(previous),right_pad_patches_valid=all(c['pad_qualified'] for c in previous['pad_grasp']['contacts']),hand_contact_count=previous['hand_contact_count'],left_panel_load_N=surface['total_normal_load_N'],left_palm_load_N=surface['palm_normal_load_N'],right_lever_clearance_m=gap)
            palm=m.site('robot/rh_palm_touch').id;palm_quat=np.empty(4);mujoco.mju_mat2Quat(palm_quat,d.site_xmat[palm])
            measured=dict(evidence=evidence,right_palm_pose=np.r_[d.site_xpos[palm],palm_quat],pose_time_s=float(d.time),contact_interval_s=(recorder.contact_time_s,recorder.contact_interval_end_s))
            force_args=(float(d.time),root,dict(zip(teacher.names,d.qpos[qa])),dict(zip(teacher.names,d.qvel[va])),foot_loads,np.r_[d.xpos[hb],d.xquat[hb]],np.r_[d.xpos[leaf],d.xquat[leaf]],angles,loads)
            if a.traverse:
                _,_,_,post_state=measured_state(sim,controller.names,controller.post.door_names,controller.post.pose_names)
                _,_,post_evidence=measured_contacts(m,previous_raw,physics_qualified=physical_sample_passed(previous))
                normal=None
                if angles['leaf']>=a.target_aperture and controller.handoff is None:
                    normal=outward_release_normal(m,previous_raw)
                try:
                    force,continuous_info=controller.force(*force_args,**measured,
                        body_poses=post_state['body_poses'],door_velocities=post_state['door_velocities'],
                        continuation_evidence=post_evidence,
                        applied_motor_forces=np.asarray(previous_raw['actuator_force'])[aids] if step else np.zeros(61),
                        release_normal_world=normal)
                except ContinuousDoorFailure as exc:
                    controller_error=dict(time_s=float(d.time),reason=str(exc));break
                info=continuous_info['component']
                if controller.opening_audit is not None and not (a.output/'opening-handoff-audit.json').exists():
                    (a.output/'opening-handoff-audit.json').write_text(json.dumps(controller.opening_audit,indent=2)+'\n')
                    (a.output/'opening-handoff-state.json').write_text(json.dumps(controller.handoff,indent=2)+'\n')
                    (a.output/'post-opening-geometry-screen.json').write_text(json.dumps(controller.post.plan,indent=2)+'\n')
            else:
                force,info=sequence.force(*force_args,**measured)
            d.ctrl[aids]=force
            recorder.before_step()
            sim.plant.step()
            row,raw=recorder.after_step()
            raw_stream.write(raw)
            previous_raw=raw
            foot_loads[:]=0.
            for c in raw['contacts']:
                if not any(m.geom(g).name=='floor' for g in c['geom']):continue
                for body in c['body']:
                    if body in feet:foot_loads[feet.index(body)]+=max(0.,c['wrench_contact_frame'][0])
            maximum_foot_lift=np.maximum(maximum_foot_lift,d.xpos[feet,2]-initial_feet[:,2])
            if sequence.readiness_screen is not None and not (a.output/'actual-preparation-screen.json').exists():
                (a.output/'actual-preparation-screen.json').write_text(json.dumps(sequence.readiness_screen,indent=2)+'\n')
                (a.output/'actual-preparation-reference.json').write_text(json.dumps(sequence.actual_preparation)+'\n')
            row.update(bolt_slide_m=float(d.qpos[m.jnt_qposadr[bj]]),teacher=info)
            if a.traverse:row['continuous_controller']=continuous_info
            physics.append(row)
            if step%10==0:
                trace=dict(**sim.diagnostics(),teacher=info);traces.append(trace)
                for key in states:states[key].append(getattr(d,key).copy())
                (a.output/'latest.json').write_text(json.dumps(trace)+'\n')
                if step%500==0:print(json.dumps(trace),flush=True)
            if (controller.done if a.traverse else row['door_q']>=a.target_aperture) or sequence.blocked_reason or not row['finite'] or row['torso_tilt_deg']>35:break
        end=float(d.time)
        opening_end=controller.opening_audit['time_s'] if a.traverse and controller.opening_audit is not None else end
        opening_rows=[r for r in physics if r['sim_time_s']<=opening_end+1e-8]
        opening_final=opening_rows[-1]
        tail=[r for r in opening_rows if r['sim_time_s']>=opening_end-.5-1e-8]
        if a.traverse:
            for name,value in (('opening-handoff-audit',controller.opening_audit),
                               ('opening-handoff-state',controller.handoff),
                               ('post-opening-geometry-screen',controller.post.plan)):
                if value is not None:(a.output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
        report=audit_grasp_steps(physics,physics_dt=m.opt.timestep,expected_duration=end)
        report['checks'].pop('sustained_pad_grasp')
        report['checks'].update(acquisition_precedes_operation=opening.operation_started is not None,operator_driven_to_release=max(r['handle_angle_rad'] for r in physics)>=.8 and max(r.get('bolt_slide_m',0) for r in physics)>=.011,left_contact_reached=opening.left.started is not None and opening.left.progress>=.999,sustained_left_panel_load=bool(tail) and all(r.get('left_surface_audit',{}).get('total_normal_load_N',0)>=2. for r in tail),usable_aperture_under_palm_load=opening_final['door_q']>=a.target_aperture and opening_final['left_surface_audit']['palm_normal_load_N']>=2.,no_invalid_right_pad_patch=all(all(c['pad_qualified'] for c in r['pad_grasp']['contacts']) for r in physics),sustained_left_palm_load=bool(tail) and all(r.get('left_surface_audit',{}).get('palm_normal_load_N',0)>=2. for r in tail),qualified_grasp_before_intentional_release=opening.release.started is not None,right_release_completed=opening.release.info.get('release_fraction',0)>=.999 and min(float(mujoco.mj_geomDistance(m,d,g,lever,1.,None)) for g in geoms)>=.02)
        report['checks']={key:bool(value) for key,value in report['checks'].items()}
        report['checks']['actual_transition_geometry_matches_pre_state']=all(r.get('pre_integration_body_poses_match',True) for r in physics)
        report['checks']['complete_runtime_warning_audit']=all(r.get('mujoco_warning_interval',{}).get('passed',False) for r in physics[1:])
        report['checks']['pre_integration_state_limits']=all(r['pre_integration_state']['max_joint_limit_violation_rad']<=.02 and r['pre_integration_state']['max_shadow_loopback_violation_rad']<=.02 and r['pre_integration_state']['root_height_m']>.7 and r['pre_integration_state']['torso_tilt_deg']<12 and r['pre_integration_state']['finite'] for r in physics[1:])
        preparation_rows=[r for r in physics[1:] if r.get('teacher',{}).get('phase')=='arm preparation']
        report['checks'].update(separated_start=float(np.linalg.norm(np.array(reset['initial_root'])[:2]-reset['goal_xy']))>=.5,
            both_feet_swung=bool(np.all(maximum_foot_lift>.015)),actual_preparation_screen=bool(sequence.readiness_screen and sequence.readiness_screen['passed']),
            contact_free_preparation=bool(preparation_rows) and all(r['hand_contact_count']==0 for r in preparation_rows),
            continuous_walk_to_acquisition=sequence.acquisition_started is not None,landed_stance_solver=sequence.body.controller.solver_failures==0,
            plant_parameters_unchanged=all(np.array_equal(v,getattr(m,k)) for k,v in originals.items()))
        report['sequence_handoffs']=sequence.handoffs
        report['opening_clock_offset_s']=sequence.acquisition_started
        report['maximum_foot_lift_m']=maximum_foot_lift.tolist()
        report['blocked_reason']=sequence.blocked_reason
        report.update(passed=all(report['checks'].values()),handoffs=opening.handoffs,declared_target_aperture_rad=a.target_aperture,final_leaf_rad=physics[-1]['door_q'],final_palm_load_N=physics[-1]['left_surface_audit']['palm_normal_load_N'],runtime_pose_writes=0,native_mirror_steps=0,scope='Continuous native approach, preparation and full-opening development with actual transition contacts; no traversal/Isaac/actor claim',measurement_refresh='mj_kinematics only after archived actual mj_step solution',pose_joint_clock_aligned=True,contact_force_source='actual_mj_step_dynamics',final_contact_interval_s=[physics[-1]['contact_interval_start_s'],physics[-1]['contact_interval_end_s']])
        if a.traverse:
            report['checks'].update(qualified_opening_before_traversal=bool(controller.opening_audit and controller.opening_audit['passed'] and controller.handoff),
                continuous_controller_contract=controller_error is None,
                qualified_post_opening_geometry=bool(controller.post.plan and controller.post.plan['passed']),
                whole_body_passage_and_quiet_finish=controller.done,
                complete_full_episode=bool(controller.done and end<=a.seconds+1e-7))
            report.update(passed=all(report['checks'].values()),controller_error=controller_error,
                opening_end_time_s=opening_end,opening_audit=controller.opening_audit,
                continuous_handoffs=controller.handoffs,completed_time_s=controller.completed_time_s,
                scope='Continuous actual native approach, grasp, mechanism opening, release, stow, rise, whole-body passage and quiet finish; privileged teacher, no Isaac or sensor-only claim')
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
        completed=True
    finally:
        raw_stream.close(complete=completed)
        (a.output/'trace.json').write_text(json.dumps(traces)+'\n')
        with gzip.open(a.output/'physics-steps.json.gz','wt') as stream:json.dump(physics,stream)
        np.savez_compressed(a.output/'trajectory.npz',**states,terminal_qpos=d.qpos.copy(),terminal_qvel=d.qvel.copy(),terminal_ctrl=d.ctrl.copy(),terminal_time_s=float(d.time))
        sim.close()
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
