"""Read-only checks of an archived 500 Hz Isaac operation and actor recording.

This reduces existing evidence; it does not execute a simulator or upgrade a
failed task to a success. Raw contact transforms are not present in these old
archives, so local-pad checks below verify recorded derived geometry only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def audit(path, robot_xml):
    path = Path(path)
    config = read(path / "configuration.json")
    result = read(path / "operation-report.json")
    layout = read(path / "sensors/layout.json")
    motors = read(path / "motor-contract.json")
    trace = read(path / "trace.json")
    provenance = read(path / "provenance.json")
    manifest = read(path / "launch-source/manifest.json")
    checks = {}
    checks["source_tar_sha256"] = sha(path / "launch-source/source.tar.gz") == manifest["source_archive_sha256"]
    with tarfile.open(path / "launch-source/source.tar.gz") as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        hashes = {name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                  for name, m in members.items()}
    checks["all_packaged_source_hashes"] = all(hashes.get(k) == v for k, v in manifest["source_hashes"].items())
    source_matches = {}
    for original, digest in provenance["files"].items():
        file = Path(original)
        if file.suffix == ".py":
            copied = path / ("source-" + file.name)
            candidates = [key for key in hashes if key.endswith("/" + file.name)]
            source_matches[file.name] = (copied.exists() and sha(copied) == digest and
                                        any(hashes[key] == digest for key in candidates))
    checks["runner_source_copies_match_launch_and_remote_receipt"] = all(source_matches.values())
    checks["pad_audit_source_in_prelaunch_archive"] = "doorbench/dexterous/isaac_pad_audit.py" in hashes
    checks["source_capture_precedes_steps"] = manifest["captured_at_unix"] < provenance["captured_before_steps_unix"]
    checks["robot_hash_matches_layout_and_motor_contract"] = (
        sha(robot_xml) == layout["robot_xml_sha256"] == motors["source_xml_sha256"])

    with np.load(path / "acquisition-physics.npz") as z:
        physics = {k: z[k] for k in z.files}
    with np.load(path / "sensors/actor-sensors.npz") as z:
        actor = {k: z[k] for k in z.files}
    with np.load(path / "sensors/actor-rgb.npz") as z:
        rgb = {k: z[k] for k in z.files}
    times = physics["time_s"]
    dt = float(result["physics_dt_s"])
    expected = np.arange(1, round(result["duration_s"] / dt) + 1) * dt
    checks["complete_physics_clock"] = np.array_equal(times, expected)
    checks["actor_clock_matches_physics"] = np.array_equal(actor["time_s"], times)
    checks["rgb_clock_matches_render_steps"] = np.array_equal(rgb["time_s"], times[::20])
    checks["finite_arrays"] = all(np.isfinite(x).all() for data in (physics, actor, rgb) for x in data.values())
    checks["all_actor_streams_valid"] = actor["sensor_valid"].all()
    checks["actor_keys_exclude_privileged_state"] = set(actor) == {
        "joint_position", "joint_velocity", "imu_gyro", "imu_accelerometer", "tactile",
        "previous_action", "sensor_time_s", "sensor_valid", "time_s"}
    checks["rgb_shape_type"] = all(rgb[k].shape == (len(times[::20]), 128, 128, 3) and
                                   rgb[k].dtype == np.uint8 for k in ("rgb_left", "rgb_right"))
    checks["action_order"] = layout["action_order"] == [x["name"] for x in motors["actuators"]]
    checks["tactile_dimensions"] = sum(x["dimension"] for x in layout["sensors"]) == actor["tactile"].shape[1] == layout["tactile_dimension"]
    checks["tactile_mount_dimensions"] = all(x["dimension"] == 3*x["width"]*x["height"] for x in layout["sensors"])
    order = [config["robot_joint_names"].index(name) for name in layout["joint_order"]]
    sensor_joint_error = float(np.max(np.abs(actor["joint_position"] - physics["joints"][:, order])))
    checks["joint_sensor_reordering_exact"] = sensor_joint_error == 0.
    caps = np.asarray([a["force_range"] for a in motors["actuators"]])
    normalized = 2*(physics["motor_forces"] - caps[:, 0])/(caps[:, 1] - caps[:, 0]) - 1
    action_error = float(np.max(np.abs(normalized - actor["previous_action"])))
    checks["previous_action_matches_bounded_applied_motor_force"] = action_error < 1e-6
    checks["all_native_motor_caps"] = bool(np.all(physics["motor_forces"] >= caps[:, 0]-1e-7) and
                                           np.all(physics["motor_forces"] <= caps[:, 1]+1e-7))
    sensor_ages = times[:, None] - actor["sensor_time_s"]
    expected_camera = np.repeat(times[::20], 20)[:len(times)]
    checks["sensor_capture_timing"] = (np.max(np.abs(sensor_ages[:, :5])) < 1e-9 and
        np.array_equal(actor["sensor_time_s"][:, 5], expected_camera) and
        np.array_equal(actor["sensor_time_s"][:, 6], expected_camera))

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    ids = [model.joint(name).id for name in config["robot_joint_names"]]
    bounds = model.jnt_range[ids]
    limited = model.jnt_limited[ids].astype(bool)
    joint_excess = float(max(0., np.maximum(bounds[:, 0]-physics["joints"],
                                            physics["joints"]-bounds[:, 1])[:, limited].max()))
    loopback = {}
    for side in ("lh", "rh"):
        for digit in ("FF", "MF", "RF", "LF"):
            indices = [config["robot_joint_names"].index(f"{side}_{digit}J{i}") for i in (1, 2)]
            loopback[f"{side}_{digit}"] = float(max(0., (physics["joints"][:, indices[0]] - physics["joints"][:, indices[1]]).max()))
    checks["every_step_native_joint_bounds_20mrad"] = joint_excess <= .02
    checks["every_step_loopbacks_20mrad"] = max(loopback.values()) <= .02
    checks["recorded_torque_delivery_50Hz"] = all(np.max(np.abs(np.asarray(r["joint_torque_command"])-r["joint_torque_sent"])) < 1e-4 for r in trace)

    with gzip.open(path / "acquisition-pad-steps.json.gz", "rt") as f:
        pads = json.load(f)
    checks["complete_2ms_pad_clock_including_reset"] = np.array_equal(
        np.asarray([r["sim_time_s"] for r in pads]), np.arange(len(times)+1)*dt)
    local_mismatches = 0
    for row in pads:
        for c in row["contacts"]:
            p = c["body_position_m"]; n = c["hand_outward_normal_body"]
            qualified = (c["body"].endswith("distal") and p[1] < -.001 and
                .002 <= p[2] <= .040 and -n[1] > .5 and c["on_lever_cylindrical_side"] and
                c["inward_radial_normal_alignment"] > .8)
            local_mismatches += bool(qualified) != c["pad_qualified"]
    checks["recorded_local_pad_formula_consistent"] = local_mismatches == 0
    start = result["operation_reference"]["operation_start_s"]
    active = [r for r in pads if r["sim_time_s"] >= start]
    bad = [r for r in active if any(not c["pad_qualified"] and c["normal_force_N"] > 1e-6 for c in r["contacts"])]
    tail = [r for r in pads if r["sim_time_s"] >= result["duration_s"]-.5-1e-8]
    checks["final_grasp_report_matches_500Hz_tail"] = result["checks"]["sustained_pad_grasp"] == all(r["valid_pad_grasp"] for r in tail)
    checks["invalid_pad_count_matches_report"] = len(bad) == result["operation_invalid_pad_patch_samples"]
    counts = Counter(c["body"].rsplit("/", 1)[-1] for r in bad for c in r["contacts"] if not c["pad_qualified"])
    first_bad = None
    if bad:
        nearby = min(trace, key=lambda r: abs(r["time_s"]-bad[0]["sim_time_s"]))
        first_bad = dict(time_s=bad[0]["sim_time_s"], patches=[c for c in bad[0]["contacts"] if not c["pad_qualified"]],
                         closest_50Hz_teacher_sample=dict(time_s=nearby["time_s"], **nearby["teacher"]))

    tactile = []
    offset = 0
    for mount in layout["sensors"]:
        values = actor["tactile"][:, offset:offset+mount["dimension"]]
        offset += mount["dimension"]
        peak = float(np.abs(values).max())
        if peak > 0:
            active_times = times[np.any(np.abs(values) > 1e-7, axis=1)]
            tactile.append(dict(name=mount["name"], peak_abs_N=peak,
                nonzero_steps=len(active_times), first_nonzero_s=float(active_times[0]),
                saturated_steps=int(np.any(np.abs(values) >= 100., axis=1).sum())))
    derivatives = {key: dict(first_to_last_mean_absolute_pixel_change=float(np.abs(rgb[key][-1].astype(float)-rgb[key][0]).mean()),
        distinct_frames=len({hashlib.sha256(x.tobytes()).digest() for x in rgb[key]})) for key in ("rgb_left", "rgb_right")}
    run_time = __import__("datetime").datetime.fromtimestamp(provenance["captured_before_steps_unix"], __import__("datetime").timezone.utc).isoformat()
    important_files = ["operation-report.json", "acquisition-physics.npz", "acquisition-pad-steps.json.gz",
        "sensors/actor-sensors.npz", "sensors/actor-rgb.npz", "sensors/layout.json", "motor-contract.json",
        "provenance.json", "launch-source/manifest.json", "launch-source/source.tar.gz"]
    return dict(run=path.name, archive=str(path), run_started_utc=run_time,
        scope="Independent archive consistency audit; privileged teacher, no sensor-only policy or full traversal claim",
        archive_checks={k: bool(v) for k, v in checks.items()}, archive_consistent=all(checks.values()),
        task_passed=result["passed"], failed_task_gates=[k for k, v in result["checks"].items() if not v],
        samples=len(times), physics_dt_s=dt, rgb_frames_per_eye=len(rgb["time_s"]),
        joint_sensor_max_error_rad=sensor_joint_error, previous_action_max_error=action_error,
        camera_age_range_s=[float(sensor_ages[:, 5:].min()), float(sensor_ages[:, 5:].max())],
        max_joint_stop_excess_rad=joint_excess, max_loopback_excess_rad=loopback,
        operation=dict(max_handle_rad=result["maximum_handle_rad"], final_leaf_rad=result["final_leaf_rad"],
            operation_start_s=start, opening_start_s=result["operation_reference"]["opening_start_s"],
            invalid_patch_steps=len(bad), invalid_patch_bodies=dict(counts), first_invalid_patch=first_bad,
            final_qualified_loads_N=result["final_pad_grasp"]["qualified_pad_forces_N"],
            final_reference=result["operation_reference"]["final_goals"]),
        tactile_active_mounts=tactile, rgb=derivatives,
        provenance=dict(prelaunch_source_members=len(hashes), run_local_source_copies=len(source_matches),
            pad_audit_sha256=hashes.get("doorbench/dexterous/isaac_pad_audit.py"),
            runner_local_pad_source_omitted=True, complete_prelaunch_package_included=True),
        limitations=["Derived pad geometry is cross-checked; raw synchronized contact/body transforms were not separately archived.",
            "Joint/motor/timing checks use every 2 ms row; command-versus-sent joint torque is available only at 50 Hz.",
            "Foot tactile channels saturate at the declared 100 N per-channel sensor limit.",
            "Cameras use the disclosed fixed manipulation profile (45 degree downward pitch, 100 degree FOV), not the default upstream view.",
            "A complete actor recording is not evidence that the privileged teacher was controlled by those sensors."],
        file_sha256={name: sha(path/name) for name in important_files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+")
    parser.add_argument("--robot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [audit(path, args.robot) for path in args.archives]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(schema_version="doorbench.isaac-operation-archive-review.v1", runs=rows), indent=2)+"\n")
    for row in rows:
        print(row["run"], "archive checks", row["archive_consistent"], "task", row["task_passed"],
              "failed consistency checks", [k for k,v in row["archive_checks"].items() if not v])


if __name__ == "__main__":
    main()
