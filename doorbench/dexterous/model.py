"""Pinned external robot model; generated robot assets are never committed."""

from pathlib import Path
import hashlib
import json
import subprocess

import mujoco
import numpy as np

UPSTREAM_URL = "https://github.com/carlosferrazza/humanoid-bench.git"
UPSTREAM_REVISION = "cb1189039151c8aadaaa987b442da54383c87fab"


def check_upstream(root):
    root = Path(root).resolve()
    rev = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if rev != UPSTREAM_REVISION:
        raise ValueError(f"Expected HumanoidBench {UPSTREAM_REVISION}, got {rev}")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", "humanoid_bench/assets", "data"], text=True)
    if dirty:
        raise ValueError("Upstream robot assets/checkpoints have local modifications")
    return root


def prepare_robot(upstream, destination):
    root = check_upstream(upstream)
    source = root / "humanoid_bench/assets/envs/h1touch_pos_stand.xml"
    spec = mujoco.MjSpec.from_file(str(source))
    native = spec.compile()
    initial = native.key_qpos[0].copy()
    nominal = {native.joint(j).name: float(initial[native.jnt_qposadr[j]])
               for j in range(native.njnt) if native.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE}
    # Keep the articulation unchanged, but use the DoorBench floor and lighting.
    for geom in list(spec.worldbody.geoms):
        spec.delete(geom)
    for texture in list(spec.textures):
        if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            spec.delete(texture)
    for key in list(spec.keys):
        spec.delete(key)
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = str((source.parent / mesh.file).resolve())
    for texture in spec.textures:
        if texture.file:
            texture.file = str((source.parent / texture.file).resolve())
    spec.add_sensor(name="imu_gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="imu_accelerometer", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(spec.to_xml())
    # Compile the serialized model, not just the in-memory spec.
    robot = mujoco.MjModel.from_xml_path(str(destination))
    if robot.nu != 61 or not np.all(robot.actuator_forcelimited):
        raise ValueError("Unexpected actuator structure or unbounded actuators")
    audit = {
        "upstream": UPSTREAM_URL, "revision": UPSTREAM_REVISION,
        "mujoco": mujoco.__version__, "robot_xml_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "nq": robot.nq, "nv": robot.nv, "actuators": robot.nu,
        "left_hand_actuators": sum(robot.actuator(i).name.startswith("lh_") for i in range(robot.nu)),
        "right_hand_actuators": sum(robot.actuator(i).name.startswith("rh_") for i in range(robot.nu)),
        "mass_kg": float(robot.body_mass.sum()), "geom_count": robot.ngeom,
        "tactile_sensors": sum(robot.sensor(i).name.endswith("_touch") for i in range(robot.nsensor)),
        "mocap_bodies": robot.nmocap, "equality_constraints": robot.neq,
        "nonzero_gravity_compensation": int(np.count_nonzero(robot.body_gravcomp)),
        "free_root_height": float(initial[2]), "nominal_joint_positions": nominal,
        "actuator_names": [robot.actuator(i).name for i in range(robot.nu)],
        "limits": {robot.actuator(i).name: {"control": robot.actuator_ctrlrange[i].tolist(),
                   "force": robot.actuator_forcerange[i].tolist()} for i in range(robot.nu)},
        "status": "model loaded; task capability not yet established",
    }
    destination.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return destination, audit
