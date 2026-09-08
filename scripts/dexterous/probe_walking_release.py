#!/usr/bin/env python3
"""Repeat a frozen walking-opening run with one opt-in pressed-frame release.

The immutable source run supplies the complete baseline code and configuration.
The original active robot/door is stepped continuously from reset. Only the
release target frame differs. Byte-identical raw prefix chunks are hardlinked
after verification to avoid storing another large copy of the walking prefix.
"""
import argparse
import ast
import subprocess
import hashlib
import importlib.util
import io
import json
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
    p.add_argument("--hold-full-left-orientation", action="store_true")
    a = p.parse_args()
    if (a.release_mode in ('whole-body-return', 'whole-body-ungrip')) != (a.whole_body_path is not None):
        raise ValueError('Whole-body return requires the exact screened path')
    if (a.release_mode == 'whole-body-ungrip') != (a.ungrip_path is not None):
        raise ValueError('Whole-body ungrip requires its screened actual-state path')
    if a.release_mode == 'controlled-return' and a.retain_grip_until_clear:
        raise ValueError('Controlled return already retains grip; do not combine release options')
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
            return WholeBodyMeasuredUngrip(teacher,path,stage/'whole-body-path.json',stage/'ungrip-path.json',goal_frame=a.ungrip_goal_frame)
        if a.release_mode in ('whole-body-return', 'whole-body-ungrip'):
            return WholeBodyLeverReturn(teacher,path,stage/'whole-body-path.json')
        if a.release_mode in ("controlled-return", "whole-body-return", "whole-body-ungrip"):
            return ControlledLeverReturn(teacher, path)
        return releases.PressedLeafFrameRightRelease(teacher, path,
            retain_grip_until_clear=a.retain_grip_until_clear)
    releases.AxialRightRelease = selected_release
    import doorbench.dexterous.full_opening_teacher as full
    original_force = full.FullOpeningTeacher.force

    def measured_force(self, t, root, joints, velocities, handle_pose, leaf_pose,
                       angles, hand_forces, **kwargs):
        if a.release_mode in ("controlled-return", "whole-body-return", "whole-body-ungrip"):
            self.release.observe_operation(t, handle_pose, leaf_pose, angles, self.geometry)
            if a.release_mode == 'whole-body-ungrip':
                self.release.observe_state(t, root, joints)
                if a.hold_full_left_orientation and self.release.started is not None and t-self.release.started >= self.release.ungrip_delay-1e-8 and not hasattr(self.left,'attained_palm_rotation_leaf'):
                    from doorbench.dexterous.left_palm_orientation import bind_attained_palm_orientation
                    bind_attained_palm_orientation(self.left,t,root,joints,leaf_pose)
        else:
            self.release.observe_leaf(t, leaf_pose)
        return original_force(self, t, root, joints, velocities, handle_pose,
                              leaf_pose, angles, hand_forces, **kwargs)

    full.FullOpeningTeacher.force = measured_force
    if a.release_mode in ('whole-body-return', 'whole-body-ungrip'):
        from doorbench.dexterous.walking_opening_teacher import WalkingOpeningTeacher
        original_walking_force=WalkingOpeningTeacher.force
        def moving_stance_force(self,t,*args,**kwargs):
            release=self.opening.release
            if release.started is not None:
                goal=release.body_goal(t-self.acquisition_started)
                apply_stance_goal(self.body.controller,goal)
            return original_walking_force(self,t,*args,**kwargs)
        WalkingOpeningTeacher.force=moving_stance_force
    runner = load("_frozen_walking_release_probe", driver)
    base_archive = runner.NativeTransitionArchive
    prefix_source = a.verified_prefix_run.resolve() if a.verified_prefix_run else source
    old_raw = prefix_source / "raw-transitions"
    prior = json.loads((old_raw / "manifest.json").read_text())
    if not prior["complete"]:
        raise ValueError("A complete baseline contact archive is required")
    old_chunks = {row["file"]: row for row in prior["chunks"]}
    baseline_report = json.loads((source / "report.json").read_text())
    release_at = baseline_report["opening_clock_offset_s"] + baseline_report["handoffs"]["right_release"]
    if a.verified_prefix_run:
        if a.release_mode != 'whole-body-ungrip':
            raise ValueError('Alternate prefix is reserved for the exact return continuation')
        release_at = json.loads(a.ungrip_path.read_text())['initial_episode_time_s']
    reuse = {"verified_linked_chunks": 0, "verified_linked_bytes": 0,
             "baseline_release_time_s": release_at, "matched_prefix_until_s": 0.}

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
            elif chunk["interval_end_s"] < release_at - 1e-8:
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
            ungrip_goal_frame=a.ungrip_goal_frame if a.release_mode=="whole-body-ungrip" else None,
            prefix_source=str(prefix_source),
            ungrip_path_sha256=digest(stage/'ungrip-path.json') if a.release_mode=='whole-body-ungrip' else None,
            whole_body_path_sha256=digest(stage/'whole-body-path.json') if a.release_mode in ('whole-body-return','whole-body-ungrip') else None,
            default_release_unchanged=True, gates_unchanged=True,
            no_runtime_pose_reset=True, no_helper_forces=True,
            scope="Development continuous native single-release comparison; no actor or Isaac claim")
        (output / "release-experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return result

    runner.capture = experiment_capture
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
