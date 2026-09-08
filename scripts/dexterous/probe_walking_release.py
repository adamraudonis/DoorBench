#!/usr/bin/env python3
"""Repeat a frozen walking-opening run with one opt-in pressed-frame release.

The immutable source run supplies the complete baseline code and configuration.
The original active robot/door is stepped continuously from reset. Only the
release target frame differs. Byte-identical raw prefix chunks are hardlinked
after verification to avoid storing another large copy of the walking prefix.
"""
import argparse
import hashlib
import importlib.util
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
    a = p.parse_args()
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
    driver = stage / "scripts/dexterous/frozen_walking_opening_driver.py"
    shutil.copy2(source / "diagnostic-source.py", driver)
    shutil.copy2(Path(__file__), stage / "scripts/dexterous/probe_walking_release.py")
    sys.path.insert(0, str(stage))

    # The baseline constructor is untouched; this isolated experiment replaces
    # only its release class and supplies the explicitly measured leaf channel.
    import doorbench.dexterous.right_hand_release as releases
    def selected_release(teacher, path):
        return releases.PressedLeafFrameRightRelease(teacher, path,
            retain_grip_until_clear=a.retain_grip_until_clear)
    releases.AxialRightRelease = selected_release
    import doorbench.dexterous.full_opening_teacher as full
    original_force = full.FullOpeningTeacher.force

    def measured_force(self, t, root, joints, velocities, handle_pose, leaf_pose,
                       angles, hand_forces, **kwargs):
        self.release.observe_leaf(t, leaf_pose)
        return original_force(self, t, root, joints, velocities, handle_pose,
                              leaf_pose, angles, hand_forces, **kwargs)

    full.FullOpeningTeacher.force = measured_force
    runner = load("_frozen_walking_release_probe", driver)
    base_archive = runner.NativeTransitionArchive
    old_raw = source / "raw-transitions"
    prior = json.loads((old_raw / "manifest.json").read_text())
    if not prior["complete"]:
        raise ValueError("A complete baseline contact archive is required")
    old_chunks = {row["file"]: row for row in prior["chunks"]}
    baseline_report = json.loads((source / "report.json").read_text())
    release_at = baseline_report["opening_clock_offset_s"] + baseline_report["handoffs"]["right_release"]
    reuse = {"verified_linked_chunks": 0, "verified_linked_bytes": 0,
             "baseline_release_time_s": release_at, "matched_prefix_until_s": 0.}

    class VerifiedPrefixArchive(base_archive):
        def flush(self):
            before = len(self.chunks)
            super().flush()
            if len(self.chunks) == before:
                return
            chunk = self.chunks[-1]
            previous = old_chunks.get(chunk["file"])
            if previous and chunk["sha256"] == previous["sha256"]:
                old = old_raw / previous["file"]
                if digest(old) != previous["sha256"]:
                    raise ValueError("Original prefix chunk changed")
                target = self.path / chunk["file"]
                target.unlink()
                os.link(old, target)
                reuse["verified_linked_chunks"] += 1
                reuse["verified_linked_bytes"] += chunk["bytes"]
                reuse["matched_prefix_until_s"] = chunk["interval_end_s"]
            elif chunk["interval_end_s"] < release_at - 1e-8:
                raise ValueError("Physical baseline prefix differs before the release intervention")
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
            intervention="PressedLeafFrameRightRelease with current same-clock measured leaf",
            retain_grip_until_clear=a.retain_grip_until_clear,
            default_release_unchanged=True, gates_unchanged=True,
            no_runtime_pose_reset=True, no_helper_forces=True,
            scope="Development continuous native single-release comparison; no actor or Isaac claim")
        (output / "release-experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return result

    runner.capture = experiment_capture
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
