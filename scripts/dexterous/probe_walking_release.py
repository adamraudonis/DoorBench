#!/usr/bin/env python3
"""Repeat a frozen walking-opening run with one opt-in pressed-frame release.

The immutable source run supplies the complete baseline code and configuration.
The original active robot/door is stepped continuously from reset. Declared
release and optional existing panel-profile interventions are recorded.
Byte-identical raw prefix chunks are hardlinked
after verification to avoid storing another large copy of the walking prefix.
"""
import argparse
import ast
import subprocess
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile


def digest(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=66.)
    p.add_argument("--retain-grip-until-clear", action="store_true")
    p.add_argument("--release-mode", choices=("pressed-frame", "controlled-return", "whole-body-return", "whole-body-ungrip"), default="pressed-frame")
    p.add_argument("--whole-body-path", type=Path)
    p.add_argument("--ungrip-path", type=Path)
    p.add_argument("--ungrip-goal-frame", choices=("attained-resting-world", "measured-handle"), default="attained-resting-world")
    p.add_argument("--verified-prefix-run", type=Path)
    p.add_argument("--require-identical-prefix-until-s",type=float)
    p.add_argument("--hold-full-left-orientation", action="store_true")
    p.add_argument("--panel-profile", choices=("plain-v1", "hybrid-surface-v2"),
                   help="Explicit existing profile comparison; omitted inherits frozen baseline")
    p.add_argument("--handoff-posture-targets", action="store_true")
    p.add_argument("--hybrid-include-waist", action="store_true")
    p.add_argument("--record-panel-targets", action="store_true")
    p.add_argument("--whole-body-panel-plan",type=Path)
    p.add_argument("--screened-panel-feedforward-n",type=float,default=3.5)
    p.add_argument("--screened-panel-lead-rad",type=float,default=.005)
    p.add_argument("--screened-panel-lead-start-rad",type=float)
    p.add_argument("--screened-panel-lead-ramp-rad",type=float,default=.1)
    p.add_argument("--record-screened-panel-phase",action="store_true")
    p.add_argument("--screened-panel-lead-audit",type=Path)
    p.add_argument("--screened-panel-actual-base-correction",action="store_true")
    p.add_argument("--screened-panel-normal-admittance",action="store_true")
    p.add_argument("--screened-panel-include-waist",action="store_true")
    p.add_argument("--screened-palm-recontact",action="store_true")
    p.add_argument("--moving-body-recontact",action="store_true")
    p.add_argument("--wait-for-loaded-aperture",action="store_true",
                   help="Keep the original half-second load requirement before early aperture termination")
    p.add_argument("--continuous-traversal",action="store_true")
    a = p.parse_args()
    if a.moving_body_recontact and not a.screened_palm_recontact:
        raise ValueError('Moving-body targets require the declared timed recontact experiment')
    if a.wait_for_loaded_aperture and not a.moving_body_recontact:
        raise ValueError('Loaded-aperture termination is declared only for this moving-body continuation')
    if a.continuous_traversal and not (a.moving_body_recontact and a.wait_for_loaded_aperture):
        raise ValueError('Continuous traversal requires the declared loaded moving-body opening')
    if a.screened_palm_recontact and not (a.whole_body_panel_plan and a.screened_panel_actual_base_correction
            and a.screened_panel_normal_admittance and a.screened_panel_include_waist and a.screened_panel_lead_rad==0
            and a.screened_panel_lead_start_rad is None):
        raise ValueError("Timed palm recontact requires its explicit actual-base/tactile/waist profile and zero opening lead")
    if a.screened_panel_include_waist and not a.screened_panel_actual_base_correction:
        raise ValueError("Waist correction requires the actual-base correction")
    if a.screened_panel_normal_admittance and not a.screened_panel_actual_base_correction:
        raise ValueError("Normal admittance requires the actual-base correction")
    if a.screened_panel_actual_base_correction and not a.whole_body_panel_plan:
        raise ValueError("Actual-base correction requires the screened panel")
    if a.screened_panel_lead_rad>.005 and a.screened_panel_lead_audit is None:
        raise ValueError("Increased panel lead requires the passed shifted-geometry audit")
    if not math.isfinite(a.screened_panel_lead_rad) or not 0 <= a.screened_panel_lead_rad <= .02:
        raise ValueError("Require a finite declared screened-panel tracking lead in [0,.02] rad")
    if a.record_screened_panel_phase and not a.whole_body_panel_plan:
        raise ValueError("Phase recording requires the screened whole-body panel")
    if a.screened_panel_lead_rad != .005 and not a.whole_body_panel_plan:
        raise ValueError("Tracking lead applies only to the screened whole-body panel")
    if not math.isfinite(a.screened_panel_feedforward_n) or not 0 < a.screened_panel_feedforward_n <= 8.:
        raise ValueError("Require a finite declared screened-panel feedforward in (0,8] N")
    if a.screened_panel_feedforward_n != 3.5 and not a.whole_body_panel_plan:
        raise ValueError("The declared feedforward applies only to the screened whole-body panel")
    if (a.release_mode in ('whole-body-return', 'whole-body-ungrip')) != (a.whole_body_path is not None):
        raise ValueError('Whole-body return requires the exact screened path')
    if (a.release_mode == 'whole-body-ungrip') != (a.ungrip_path is not None):
        raise ValueError('Whole-body ungrip requires its screened actual-state path')
    if a.handoff_posture_targets and a.release_mode != 'whole-body-ungrip':
        raise ValueError('Posture handoff requires the measured whole-body ungrip helper')
    if a.whole_body_panel_plan and (a.hybrid_include_waist or a.record_panel_targets):
        raise ValueError('New screened panel has its own declared position-force profile/telemetry')
    if a.hybrid_include_waist and a.panel_profile!='hybrid-surface-v2':
        raise ValueError('Waist projection requires the explicit hybrid profile')
    if a.release_mode == 'controlled-return' and a.retain_grip_until_clear:
        raise ValueError('Controlled return already retains grip; do not combine release options')
    if a.require_identical_prefix_until_s is not None and (not math.isfinite(a.require_identical_prefix_until_s) or a.require_identical_prefix_until_s<0 or not a.verified_prefix_run):
        raise ValueError("Require a finite nonnegative prefix horizon and explicit complete source run")
    source = a.source_run.resolve()
    output = a.output.resolve()
    stage = output.with_name(output.name + "-source")
    if output.exists() or stage.exists():
        raise ValueError("Use a new output and source staging directory")
    baseline = json.loads((source / "manifest.json").read_text())
    if digest(source / "source.tar.gz") != baseline["source_archive_sha256"]:
        raise ValueError("Frozen baseline source archive changed")
    stage.mkdir(parents=True)
    with tarfile.open(source / "source.tar.gz") as archive:
        for member in archive.getmembers():
            path = stage / member.name
            if not member.isfile() or not path.resolve().is_relative_to(stage):
                raise ValueError("Require plain files within the frozen source archive")
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as incoming, path.open("wb") as target:
                shutil.copyfileobj(incoming, target)
    # Explicitly saved local modules can differ from a repository capture.
    for stored, module in (("full-opening-teacher-source.py", "full_opening_teacher.py"),
                           ("bimanual-transfer-source.py", "bimanual_transfer.py"),
                           ("panel-continuation-source.py", "panel_continuation.py"),
                           ("native-transition-audit-source.py", "native_transition_audit.py"),
                           ("native-transition-archive-source.py", "native_transition_archive.py")):
        shutil.copy2(source / stored, stage / "doorbench/dexterous" / module)
    own = Path(__file__).resolve().parents[2]
    if a.continuous_traversal:
        for module in ('continuous_door_teacher.py','post_opening_teacher.py','post_opening.py',
                       'post_opening_route.py','passage.py','native_post_opening_measurements.py',
                       'native_continuous_bridge.py','locomotion_manipulation.py'):
            shutil.copy2(own/'doorbench/dexterous'/module,stage/'doorbench/dexterous'/module)
    if a.whole_body_panel_plan:
        for module in ('screened_panel_path.py','screened_panel_teacher.py','actual_base_palm.py','palm_normal_admittance.py','panel_torso_target.py'):
            shutil.copy2(own/'doorbench/dexterous'/module,stage/'doorbench/dexterous'/module)
        shutil.copy2(a.whole_body_panel_plan,stage/'whole-body-panel-plan.json')
        if a.screened_palm_recontact:
            shutil.copy2(own/'doorbench/dexterous/palm_recontact_teacher.py',stage/'doorbench/dexterous/palm_recontact_teacher.py')
        if a.moving_body_recontact:
            for module in ('moving_body_recontact.py','whole_body_contact_targets.py'):
                shutil.copy2(own/'doorbench/dexterous'/module,stage/'doorbench/dexterous'/module)
        if a.screened_panel_lead_audit:shutil.copy2(a.screened_panel_lead_audit,stage/'panel-lead-audit.json')
    if a.hybrid_include_waist or a.record_panel_targets:
        shutil.copy2(own/'doorbench/dexterous/panel_chain_projection.py',stage/'doorbench/dexterous/panel_chain_projection.py')
    shutil.copy2(own / "doorbench/dexterous/right_hand_release.py",
                 stage / "doorbench/dexterous/right_hand_release.py")
    if a.release_mode in ("controlled-return", "whole-body-return", "whole-body-ungrip"):
        shutil.copy2(own / "doorbench/dexterous/right_release_return.py",
                     stage / "doorbench/dexterous/right_release_return.py")
    if a.release_mode in ('whole-body-return', 'whole-body-ungrip'):
        shutil.copy2(own/'doorbench/dexterous/whole_body_return.py',stage/'doorbench/dexterous/whole_body_return.py')
        shutil.copy2(a.whole_body_path,stage/'whole-body-path.json')
    if a.release_mode == 'whole-body-ungrip':
        shutil.copy2(own/'doorbench/dexterous/whole_body_ungrip.py',stage/'doorbench/dexterous/whole_body_ungrip.py')
        shutil.copy2(a.ungrip_path,stage/'ungrip-path.json')
        # Apply only the parent's reviewed readiness method to the frozen core.
        # All other baseline behavior is retained and the actual prefix is checked.
        approved = subprocess.check_output(['git','show','15ad820a3:doorbench/dexterous/full_opening_teacher.py'], cwd=own, text=True)
        tree=ast.parse(approved);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FullOpeningTeacher')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_freeze_release_if_ready')
        lines=approved.splitlines(keepends=True);method_source=''.join(lines[method.lineno-1:method.end_lineno])+'\n\n'
        core_path=stage/'doorbench/dexterous/full_opening_teacher.py';core=core_path.read_text()
        start=core.index('            if self.release.frozen is None and t-self.release.started > 1.')
        end=core.index('        self.release.update(float(t))', start)
        core=core[:start]+"        release_ready = self._freeze_release_if_ready(t, root, joints, right_palm_pose, evidence)\n"+core[end:]
        old='        if self.release.frozen is not None:\n            if self.push.started is None:'
        assert core.count(old)==1
        core=core.replace(old,'        if self.release.frozen is not None and release_ready:\n            if self.push.started is None:')
        core=core.replace('    def _begin_operation(',method_source+'    def _begin_operation(',1)
        core_path.write_text(core)
        (stage/'approved-readiness-source.py').write_text(approved)
    if a.hold_full_left_orientation:
        if a.release_mode != 'whole-body-ungrip':
            raise ValueError('Full left orientation comparison requires the attained ungrip stage')
        shutil.copy2(own/'doorbench/dexterous/left_palm_orientation.py',stage/'doorbench/dexterous/left_palm_orientation.py')
        left_path=stage/'doorbench/dexterous/bimanual_transfer.py';left_source=left_path.read_text()
        old='10*(d.site_xmat[self.palm].reshape(3,3)[:,2]-desired_z)'
        new='10*palm_orientation_error(d.site_xmat[self.palm].reshape(3,3),desired_z,rotation@self.attained_palm_rotation_leaf if hasattr(self,"attained_palm_rotation_leaf") else None)'
        if left_source.count(old)!=1:raise ValueError('Frozen left orientation residual changed')
        left_source=left_source.replace(old,new)
        left_source=left_source.replace('import json\n','from .left_palm_orientation import palm_orientation_error\nimport json\n',1)
        left_path.write_text(left_source)
    driver = stage / "scripts/dexterous/frozen_walking_opening_driver.py"
    shutil.copy2(source / "diagnostic-source.py", driver)
    if a.wait_for_loaded_aperture:
        text=driver.read_text()
        old="row['door_q']>=a.target_aperture or not row['finite']"
        new="(row['door_q']>=a.target_aperture and len(physics)>=251 and all(r.get('left_surface_audit',{}).get('palm_normal_load_N',0)>=2. for r in physics[-251:])) or not row['finite']"
        if text.count(old)!=1:raise ValueError('Frozen aperture stop condition changed')
        driver.write_text(text.replace(old,new))
    if a.continuous_traversal:
        text=driver.read_text()
        def replace_once(old,new):
            nonlocal text
            if text.count(old)!=1:raise ValueError('Frozen continuous driver integration point changed: '+old[:60])
            text=text.replace(old,new)
        replace_once('    completed=False',
            "    from doorbench.dexterous import native_continuous_bridge as bridge\n"
            "    from doorbench.dexterous.native_transition_audit import _contact_solution\n"
            "    _,previous_raw=_contact_solution(sim)\n"
            "    previous_raw.update(interval_start_s=0.,interval_end_s=0.)\n"
            "    body_bounds=bridge.RobotBounds(m)\n    bridge.annotate(sim,physics[0],body_bounds)\n    completed=False")
        line=next(line for line in text.splitlines() if line.strip().startswith('force,info=sequence.force('))
        newcall=line.rstrip()[:-1]+',**continuation_packet)'
        replace_once(line,
            "            continuation_packet=bridge.packet(sim,previous_raw,physical_sample_passed(previous),teacher.names,aids)\n"
            "            try:\n    "+newcall+"\n"
            "            except ValueError as error:\n"
            "                (a.output/'error.txt').write_text(str(error)+'\\n')\n                break")
        replace_once('            row,raw=recorder.after_step()',
            '            row,raw=recorder.after_step()\n            previous_raw=raw\n            bridge.annotate(sim,row,body_bounds)')
        stop=next(line for line in text.splitlines() if line.strip().startswith('if sequence.blocked_reason or'))
        replace_once(stop,"            if sequence.blocked_reason or sequence.continuous.done or not row['finite'] or row['torso_tilt_deg']>35:break")
        marker="        (a.output/'report.json').write_text"
        replace_once(marker,"        report=bridge.report(sequence.continuous,report,physics,sim)\n"+marker)
        driver.write_text(text)
    shutil.copy2(Path(__file__), stage / "scripts/dexterous/probe_walking_release.py")
    sys.path.insert(0, str(stage))

    # The baseline constructor is untouched; this isolated experiment replaces
    # only its release class and supplies the explicitly measured leaf channel.
    import doorbench.dexterous.right_hand_release as releases
    if a.release_mode in ("controlled-return", "whole-body-return", "whole-body-ungrip"):
        from doorbench.dexterous.right_release_return import ControlledLeverReturn
    if a.release_mode in ('whole-body-return', 'whole-body-ungrip'):
        from doorbench.dexterous.whole_body_return import WholeBodyLeverReturn, apply_stance_goal
    if a.release_mode == 'whole-body-ungrip':
        from doorbench.dexterous.whole_body_ungrip import WholeBodyMeasuredUngrip
    def selected_release(teacher, path):
        if a.release_mode == 'whole-body-ungrip':
            return WholeBodyMeasuredUngrip(teacher,path,stage/'whole-body-path.json',stage/'ungrip-path.json',goal_frame=a.ungrip_goal_frame,handoff_posture_targets=a.handoff_posture_targets)
        if a.release_mode in ('whole-body-return', 'whole-body-ungrip'):
            return WholeBodyLeverReturn(teacher,path,stage/'whole-body-path.json')
        if a.release_mode in ("controlled-return", "whole-body-return", "whole-body-ungrip"):
            return ControlledLeverReturn(teacher, path)
        return releases.PressedLeafFrameRightRelease(teacher, path,
            retain_grip_until_clear=a.retain_grip_until_clear)
    releases.AxialRightRelease = selected_release
    import doorbench.dexterous.full_opening_teacher as full
    if a.whole_body_panel_plan:
        from doorbench.dexterous.screened_panel_teacher import ScreenedWholeBodyPanel
        if a.screened_palm_recontact:
            from doorbench.dexterous.palm_recontact_teacher import TimedPalmRecontact
            ScreenedWholeBodyPanel=TimedPalmRecontact
        if a.moving_body_recontact:
            from doorbench.dexterous.moving_body_recontact import MovingBodyPalmRecontact
            ScreenedWholeBodyPanel=MovingBodyPalmRecontact
        full.CoordinatedPanelPush=lambda left,**options:ScreenedWholeBodyPanel(left,stage/'whole-body-panel-plan.json',normal_feedforward_N=a.screened_panel_feedforward_n,actual_base_correction=a.screened_panel_actual_base_correction,palm_normal_admittance=a.screened_panel_normal_admittance,correction_include_waist=a.screened_panel_include_waist,tracking_lead_rad=a.screened_panel_lead_rad,lead_start_angle=a.screened_panel_lead_start_rad,lead_ramp_rad=a.screened_panel_lead_ramp_rad,lead_receipt=stage/'panel-lead-audit.json' if a.screened_panel_lead_audit else None,**options)
    original_force = full.FullOpeningTeacher.force
    panel_trace=None
    panel_phase_trace=None

    def measured_force(self, t, root, joints, velocities, handle_pose, leaf_pose,
                       angles, hand_forces, **kwargs):
        nonlocal panel_trace,panel_phase_trace
        if a.release_mode in ("controlled-return", "whole-body-return", "whole-body-ungrip"):
            self.release.observe_operation(t, handle_pose, leaf_pose, angles, self.geometry)
            if a.release_mode == 'whole-body-ungrip':
                self.release.observe_state(t, root, joints)
                if a.hold_full_left_orientation and self.release.started is not None and t-self.release.started >= self.release.ungrip_delay-1e-8 and not hasattr(self.left,'attained_palm_rotation_leaf'):
                    from doorbench.dexterous.left_palm_orientation import bind_attained_palm_orientation
                    bind_attained_palm_orientation(self.left,t,root,joints,leaf_pose)
        else:
            self.release.observe_leaf(t, leaf_pose)
        force,info=original_force(self, t, root, joints, velocities, handle_pose,
                              leaf_pose, angles, hand_forces, **kwargs)
        if a.screened_panel_include_waist and self.push.started is not None:
            force,torso_info=self.push.apply_corrected_torso_force(force,joints,velocities)
            self._last_corrected_torso=(torso_info["motor_index"],torso_info["consumed_torso_effort_Nm"])
        projection=None
        if a.hybrid_include_waist:
            from doorbench.dexterous.panel_chain_projection import apply_waist_arm_projection
            force,projection=apply_waist_arm_projection(self,force)
            if projection is not None:
                self._last_projected_chain=(projection['chain_motor_indices'],force[projection['chain_motor_indices']].copy())
                info['eight_joint_projection']=projection
        if (a.record_panel_targets or a.hybrid_include_waist) and self.push.started is not None:
            import gzip
            from doorbench.dexterous.panel_chain_projection import measured_panel_chain
            chain=measured_panel_chain(self);teacher=self.acquisition
            if panel_trace is None:panel_trace=gzip.open(output/'panel-targets.jsonl.gz','wt')
            row=dict(time_s=float(t),pose_time_s=float(kwargs['pose_time_s']),
                     panel_update_time_s=self.push.last_update,motor_target_update_time_s=teacher.last_update,
                     nominal_ik_targets=self.push.previous.tolist(),nominal_joint_names=self.push.names,
                     consumed_motor_targets=teacher.target.tolist(),left_arm_targets=self.left.target.tolist(),
                     measured_chain_joint_positions=[joints[n] for n in self.left.names],
                     measured_chain_joint_velocities=[velocities[n] for n in self.left.names],
                     chain_motor_indices=chain['indices'].tolist(),chain_mass=chain['mass'].tolist(),
                     chain_normal_jacobian=chain['normal_jacobian'].tolist(),
                     weighted_task_jacobian_singular_values=chain['weighted_task_jacobian_singular_values'].tolist(),
                     assembled_motor_forces=force.tolist(),left_info=self.left.info,projection=projection)
            panel_trace.write(json.dumps(row)+'\n');panel_trace.flush()
        if a.record_screened_panel_phase and self.push.started is not None:
            import gzip
            if panel_phase_trace is None:panel_phase_trace=gzip.open(output/'panel-phase.jsonl.gz','wt')
            offset=self._diagnostic_episode_offset
            local_interval=info['contact_interval_s']
            row=dict(episode_time_s=float(t+offset),local_time_s=float(t),opening_clock_offset_s=float(offset),
                     pose_time_s=float(kwargs['pose_time_s']+offset),
                     previous_contact_interval_s=[float(x+offset) for x in local_interval],
                     reference_aperture_rad=float(self.push.latest['aperture']),actual_aperture_rad=float(angles['leaf']),
                     normal_feedforward_N=float(self.push.normal_feedforward_N),tracking_lead_rad=float(self.push.phase.lead_at(angles['leaf'])),
                     body_coordinates=self.push.latest['coordinate'][:6].tolist(),
                     planned_coordinates=self.push.latest['coordinate'].tolist(),
                     planned_velocity=self.push.latest['velocity'].tolist(),planned_acceleration=self.push.latest['acceleration'].tolist(),
                     latched_motor_targets=self.acquisition.target.tolist(),
                     left_arm_targets=self.left.target.tolist(),left_arm_target_velocity=self.left.target_velocity.tolist(),
                     actual_base_correction=self.push.latest.get('actual_base_correction'),
                     normal_admittance=self.push.latest.get('normal_admittance'),
                     whole_body_adaptation=self.push.latest.get('whole_body_adaptation'),
                     corrected_chain_joint_names=self.push.correction_names if self.push.correction is not None else None,
                     corrected_chain_targets=self.push.corrected_chain_target.tolist() if self.push.corrected_chain_target is not None else None,
                     corrected_chain_velocity=self.push.corrected_chain_velocity.tolist() if self.push.corrected_chain_velocity is not None else None,
                     actual_torso_force=self.push.latest.get('actual_torso_force'))
            if a.continuous_traversal:self._pending_panel_row=row
            else:panel_phase_trace.write(json.dumps(row,allow_nan=False)+'\n')
        if a.moving_body_recontact and self.push.started is None and self.release.started is not None:
            self.push.remember_output(t,self.release.body_goal(t))
        return force,info

    full.FullOpeningTeacher.force = measured_force
    if a.release_mode in ('whole-body-return', 'whole-body-ungrip'):
        from doorbench.dexterous.walking_opening_teacher import WalkingOpeningTeacher
        original_walking_force=WalkingOpeningTeacher.force
        def moving_stance_force(self,t,*args,**kwargs):
            release=self.opening.release
            self.opening._diagnostic_episode_offset=self.acquisition_started
            if a.whole_body_panel_plan and self.opening.push.started is not None:
                if a.moving_body_recontact:
                    if getattr(self.opening,'whole_body_return_path',None) is not None:
                        raise ValueError('Do not let a second legacy stance owner overwrite the moving-body target')
                    self.opening.push.update(t-self.acquisition_started,args[0],args[1],args[5],
                        kwargs['evidence']['left_palm_load_N'],args[6]['leaf'],True)
                else:
                    self.opening.push.advance(t-self.acquisition_started,args[6]['leaf'])
                apply_stance_goal(self.body.controller,self.opening.push.body_goal(t-self.acquisition_started))
            elif release.started is not None:
                goal=release.body_goal(t-self.acquisition_started)
                apply_stance_goal(self.body.controller,goal)
            result=original_walking_force(self,t,*args,**kwargs)
            if a.screened_panel_include_waist and hasattr(self.opening,"_last_corrected_torso"):
                index,expected=self.opening._last_corrected_torso
                if result[0][index]!=expected:raise AssertionError("Walking assembly overwrote the actual corrected torso effort")
            if a.hybrid_include_waist and hasattr(self.opening,'_last_projected_chain'):
                import numpy as np
                indices,expected=self.opening._last_projected_chain
                if not np.array_equal(result[0][indices],expected):
                    raise AssertionError('The walking force assembly overwrote the projected chain')
            return result
        WalkingOpeningTeacher.force=moving_stance_force
    if a.continuous_traversal:
        import doorbench.dexterous.walking_opening_teacher as walking_module
        from doorbench.dexterous.continuous_door_teacher import ContinuousDoorTeacher
        from doorbench.dexterous.post_opening_teacher import PostOpeningTeacher
        original_walking_class=walking_module.WalkingOpeningTeacher
        class NativeContinuousAdapter:
            def __init__(self,*args,**kwargs):
                self.walking=original_walking_class(*args,**kwargs)
                post=PostOpeningTeacher(args[0],args[1],args[4],args[5],door_xml=kwargs['door_xml'],
                    stow_profile='sequential-v2',phase_seconds=4.,inward_roll=.07,passage=True)
                self.continuous=ContinuousDoorTeacher.from_components(self.walking,post,args[1],args[4],
                    maximum_seconds=a.seconds,handoff_policy='loaded-hold-v2')
            def __getattr__(self,name):return getattr(self.walking,name)
            @property
            def blocked_reason(self):return self.continuous.blocked_reason
            def force(self,*args,**kwargs):
                force,info=self.continuous.force(*args,**kwargs)
                pending=getattr(self.walking.opening,'_pending_panel_row',None)
                if pending is not None:
                    # A crossing computes an opening proposal but returns the
                    # continuation force. Archive only actually selected targets.
                    if self.continuous.handoff is None:panel_phase_trace.write(json.dumps(pending,allow_nan=False)+'\n')
                    del self.walking.opening._pending_panel_row
                return force,dict(info['component'],continuous=info)
        walking_module.WalkingOpeningTeacher=NativeContinuousAdapter
    runner = load("_frozen_walking_release_probe", driver)
    base_archive = runner.NativeTransitionArchive
    prefix_source = a.verified_prefix_run.resolve() if a.verified_prefix_run else source
    old_raw = prefix_source / "raw-transitions"
    prior = json.loads((old_raw / "manifest.json").read_text())
    if not prior["complete"]:
        raise ValueError("A complete baseline contact archive is required")
    if a.require_identical_prefix_until_s is not None and a.require_identical_prefix_until_s>prior["chunks"][-1]["interval_end_s"]+1e-8:
        raise ValueError("Required prefix exceeds the complete original dynamics")
    old_chunks = {row["file"]: row for row in prior["chunks"]}
    baseline_report = json.loads((source / "report.json").read_text())
    release_at = baseline_report["opening_clock_offset_s"] + baseline_report["handoffs"]["right_release"]
    if a.verified_prefix_run:
        if a.release_mode != 'whole-body-ungrip':
            raise ValueError('Alternate prefix is reserved for the exact return continuation')
        release_at = json.loads(a.ungrip_path.read_text())['initial_episode_time_s']
    reuse = {"verified_linked_chunks": 0, "verified_linked_bytes": 0,
             "baseline_release_time_s": release_at, "matched_prefix_until_s": 0.,
             "required_identical_prefix_until_s":a.require_identical_prefix_until_s}

    class VerifiedPrefixArchive(base_archive):
        def flush(self):
            if not self.rows:
                return
            # Compare compressed bytes in memory first. An identical walking
            # prefix needs no temporary duplicate file, even during a disk
            # pressure event. New physical evidence is still written atomically.
            import numpy as np
            stream = io.BytesIO()
            np.savez_compressed(stream, **runner.archive_module.packed(self.rows))
            content = stream.getvalue()
            name = f'transitions-{len(self.chunks):05d}.npz'
            chunk = dict(file=name, rows=len(self.rows), bytes=len(content),
                         sha256=hashlib.sha256(content).hexdigest(),
                         interval_start_s=self.rows[0]['interval_start_s'],
                         interval_end_s=self.rows[-1]['interval_end_s'])
            previous = old_chunks.get(name)
            target = self.path / name
            if previous and chunk['sha256'] == previous['sha256']:
                old = old_raw / previous["file"]
                if digest(old) != previous["sha256"]:
                    raise ValueError("Original prefix chunk changed")
                os.link(old, target)
                reuse["verified_linked_chunks"] += 1
                reuse["verified_linked_bytes"] += chunk["bytes"]
                reuse["matched_prefix_until_s"] = chunk["interval_end_s"]
            elif (chunk["interval_end_s"] < release_at - 1e-8
                    or (a.require_identical_prefix_until_s is not None and chunk["interval_end_s"]<=a.require_identical_prefix_until_s+1e-8)):
                raise ValueError("Physical baseline prefix differs before the release intervention")
            else:
                temporary = self.path / (name + '.pending')
                temporary.write_bytes(content)
                temporary.replace(target)
            self.chunks.append(chunk)
            self.rows = []
            self._manifest(False)
            (output / "prefix-reuse.json").write_text(json.dumps(reuse, indent=2) + "\n")

    runner.NativeTransitionArchive = VerifiedPrefixArchive
    config = baseline["configuration"].copy()
    original_cwd = Path(baseline["working_directory"])
    for key in ("robot", "door", "reference", "motors", "plan", "release_path",
                "body_reset", "preparation", "checkpoint", "runtime_screen"):
        if config.get(key):
            path = Path(config[key])
            config[key] = str(path if path.is_absolute() else original_cwd / path)
    config["output"] = str(output)
    config["seconds"] = a.seconds
    if a.panel_profile is not None:
        config["panel_profile"] = a.panel_profile
    argv = [str(driver)]
    for key, value in config.items():
        if value is None or value is False:
            continue
        argv.append("--" + key.replace("_", "-"))
        if value is not True:
            argv.append(str(value))
    sys.argv = argv
    old_capture = runner.capture

    def experiment_capture(*args, **kwargs):
        result = old_capture(*args, **kwargs)
        metadata = dict(schema="doorbench.pressed-frame-release-experiment.v1",
            baseline_run=str(source), baseline_source_sha256=baseline["source_archive_sha256"],
            baseline_driver_sha256=digest(source / "diagnostic-source.py"),
            release_module_sha256=digest(stage / "doorbench/dexterous/right_hand_release.py"),
            intervention=("WholeBodyMeasuredUngrip with exact attained resting grasp and complete screened withdrawal" if a.release_mode=="whole-body-ungrip" else "WholeBodyLeverReturn with attained-foot stance motor targets" if a.release_mode=="whole-body-return" else "ControlledLeverReturn with actual joint geometry and measured operation" if a.release_mode == "controlled-return" else "PressedLeafFrameRightRelease with current same-clock measured leaf"),
            retain_grip_until_clear=a.retain_grip_until_clear,
            release_mode=a.release_mode,
            hold_full_left_orientation=a.hold_full_left_orientation,
            handoff_posture_targets=a.handoff_posture_targets,
            hybrid_include_waist=a.hybrid_include_waist,
            whole_body_panel_plan_sha256=digest(stage/'whole-body-panel-plan.json') if a.whole_body_panel_plan else None,
            whole_body_panel_profile='screened-position-v1' if a.whole_body_panel_plan else None,
            screened_panel_normal_feedforward_N=a.screened_panel_feedforward_n if a.whole_body_panel_plan else None,
            screened_panel_tracking_lead_rad=a.screened_panel_lead_rad if a.whole_body_panel_plan else None,
            record_screened_panel_phase=a.record_screened_panel_phase,
            screened_panel_actual_base_correction=a.screened_panel_actual_base_correction,
            screened_panel_normal_admittance=a.screened_panel_normal_admittance,
            screened_panel_include_waist=a.screened_panel_include_waist,
            screened_palm_recontact=a.screened_palm_recontact,
            moving_body_recontact=a.moving_body_recontact,
            wait_for_loaded_aperture=a.wait_for_loaded_aperture,
            continuous_traversal=a.continuous_traversal,
            screened_panel_lead_audit_sha256=digest(stage/"panel-lead-audit.json") if a.screened_panel_lead_audit else None,
            screened_panel_lead_start_rad=a.screened_panel_lead_start_rad,screened_panel_lead_ramp_rad=a.screened_panel_lead_ramp_rad,
            record_panel_targets=a.record_panel_targets or a.hybrid_include_waist,
            panel_profile=config.get('panel_profile'),
            panel_profile_changed_from_baseline=config.get('panel_profile')!=baseline['configuration'].get('panel_profile'),
            ungrip_goal_frame=a.ungrip_goal_frame if a.release_mode=="whole-body-ungrip" else None,
            prefix_source=str(prefix_source),
            required_identical_prefix_until_s=a.require_identical_prefix_until_s,
            ungrip_path_sha256=digest(stage/'ungrip-path.json') if a.release_mode=='whole-body-ungrip' else None,
            whole_body_path_sha256=digest(stage/'whole-body-path.json') if a.release_mode in ('whole-body-return','whole-body-ungrip') else None,
            default_release_unchanged=True, gates_unchanged=True,
            no_runtime_pose_reset=True, no_helper_forces=True,
            scope="Development continuous native single-release comparison; no actor or Isaac claim")
        (output / "release-experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return result

    runner.capture = experiment_capture
    try:
        return runner.main()
    finally:
        if panel_trace is not None:panel_trace.close()
        if panel_phase_trace is not None:panel_phase_trace.close()


if __name__ == "__main__":
    raise SystemExit(main())
