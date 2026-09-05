"""JSON state bridge from native MuJoCo poses to an independent vision renderer.

Importing this module requires only Python's standard library. MuJoCo is loaded
inside capture/export functions. No function steps physics, releases locks, or
solves an authored inspection pose to make it appear mechanically valid.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

SCHEMA_VERSION = 1
CAMERA_CONVENTION = "right_handed_x_right_y_up_z_backward"


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(value, size, label):
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(f"{label} must contain {size} values") from exc
    if len(values) != size:
        raise ValueError(f"{label} must contain {size} values")
    return [_number(v, label) for v in values]


def _matrix(value, label, *, rotation=False):
    rows = list(value)
    if len(rows) == 9 and not hasattr(rows[0], "__iter__"):
        rows = [rows[i:i + 3] for i in (0, 3, 6)]
    if len(rows) != 3:
        raise ValueError(f"{label} must be a 3 by 3 matrix")
    rows = [_vector(row, 3, label) for row in rows]
    if rotation:
        for i in range(3):
            for j in range(3):
                dot = sum(rows[k][i] * rows[k][j] for k in range(3))
                if abs(dot - (1.0 if i == j else 0.0)) > 1e-6:
                    raise ValueError(f"{label} must be an orthonormal rotation")
        a, b, c = rows
        det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
        if abs(det - 1.0) > 1e-6:
            raise ValueError(f"{label} must be a proper rotation (determinant +1)")
    return rows


def _quaternion(values, label="quaternion"):
    q = _vector(values, 4, label)
    norm = math.sqrt(sum(v * v for v in q))
    if abs(norm - 1.0) > 1e-6:
        raise ValueError(f"{label} must be a unit wxyz quaternion")
    q = [v / norm for v in q]
    # Equivalent q and -q serialize identically, including 180-degree rotations.
    sign = next((1.0 if v > 0 else -1.0 for v in q if abs(v) > 1e-15), 1.0)
    return [v * sign if v else 0.0 for v in q]


def _rotation_quaternion(value):
    r = _matrix(value, "rotation matrix", rotation=True)
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        s = 2 * math.sqrt(trace + 1.0)
        q = [s / 4, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s]
    else:
        i = max(range(3), key=lambda k: r[k][k])
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(max(0.0, 1 + r[i][i] - r[j][j] - r[k][k]))
        q = [0.0] * 4
        q[0], q[i + 1] = (r[k][j] - r[j][k]) / s, s / 4
        q[j + 1], q[k + 1] = (r[j][i] + r[i][j]) / s, (r[k][i] + r[i][k]) / s
    return _quaternion(q)


def _pose(position, *, quaternion=None, rotation=None):
    return {"pos": _vector(position, 3, "world position"),
            "quat_wxyz": _quaternion(quaternion) if quaternion is not None else _rotation_quaternion(rotation)}


def _resolution(value):
    values = _vector(value, 2, "camera resolution")
    if any(v <= 0 or v != int(v) for v in values):
        raise ValueError("Camera resolution must contain two positive integer pixel counts")
    return [int(v) for v in values]


def validate_camera(camera: Mapping) -> dict:
    """Normalize an explicit pinhole camera, in MuJoCo/Blender camera coordinates.

    Pixel coordinates are right/down with their origin at the top-left image
    boundary. For camera-local p, project K @ [p.x, -p.y, -p.z]. Resolution is
    [width, height]. Exactly one of quat_wxyz or rotation_matrix is required.
    """
    if not isinstance(camera, Mapping):
        raise ValueError("Camera must be a mapping")
    if camera.get("pixel_origin", "top_left_boundary") != "top_left_boundary":
        raise ValueError("Unsupported pixel_origin; expected top_left_boundary")
    if camera.get("projection", "perspective") != "perspective":
        raise ValueError("Only perspective pinhole cameras are supported")
    resolution = _resolution(camera.get("resolution", []))
    has_quat, has_matrix = "quat_wxyz" in camera, "rotation_matrix" in camera
    if has_quat == has_matrix:
        raise ValueError("Camera needs exactly one of quat_wxyz or rotation_matrix")
    convention = camera.get("convention", CAMERA_CONVENTION)
    if convention != CAMERA_CONVENTION:
        raise ValueError(f"Unsupported camera convention: {convention}")
    pose = _pose(camera.get("pos", []), quaternion=camera.get("quat_wxyz"), rotation=camera.get("rotation_matrix"))
    k = _matrix(camera.get("intrinsics", []), "camera intrinsics")
    if k[0][0] <= 0 or k[1][1] <= 0 or abs(k[0][1]) > 1e-12 or abs(k[1][0]) > 1e-12 or any(abs(a - b) > 1e-12 for a, b in zip(k[2], (0, 0, 1))):
        raise ValueError("Intrinsics need positive focal lengths, zero skew, and pinhole last row [0,0,1]")
    result = {**pose, "intrinsics": k, "resolution": resolution,
              "convention": CAMERA_CONVENTION, "pixel_origin": "top_left_boundary",
              "projection": "perspective"}
    if "name" in camera:
        if not isinstance(camera["name"], str) or not camera["name"]:
            raise ValueError("Camera name must be a nonempty string")
        result["name"] = camera["name"]
    return result


def _capture_camera(model, data, camera, mujoco):
    if camera is None:
        return None
    if isinstance(camera, str):
        camera = {"name": camera}
    if not isinstance(camera, Mapping):
        raise ValueError("Camera must be a camera name or camera mapping")
    if "pos" in camera or "quat_wxyz" in camera or "rotation_matrix" in camera:
        return validate_camera(camera)
    if set(camera) - {"name", "resolution"}:
        raise ValueError("Named camera accepts only name and resolution")
    name = camera.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Named camera requires a nonempty name")
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if camera_id < 0:
        raise ValueError(f"Unknown MuJoCo camera: {name}")
    native_resolution = list(model.cam_resolution[camera_id])
    default_resolution = native_resolution if all(int(v) > 1 for v in native_resolution) else [640, 480]
    width, height = _resolution(camera.get("resolution", default_resolution))
    scene = mujoco.MjvScene()
    view = mujoco.MjvCamera()
    view.type, view.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, camera_id
    mujoco.mjv_updateCamera(model, data, view, scene)
    lens = scene.camera[0]
    if lens.orthographic:
        raise ValueError("Orthographic cameras are not supported by the pinhole state schema")
    bottom, top, near = float(lens.frustum_bottom), float(lens.frustum_top), float(lens.frustum_near)
    half_width = float(lens.frustum_width) or ((top - bottom) / 2 * width / height)
    if top <= bottom or half_width <= 0 or near <= 0:
        raise ValueError("MuJoCo camera produced an invalid projection frustum")
    k = [[near * width / (2 * half_width), 0.0, width / 2 - float(lens.frustum_center) * width / (2 * half_width)],
         [0.0, near * height / (top - bottom), top * height / (top - bottom)], [0.0, 0.0, 1.0]]
    return validate_camera({"name": name, **_pose(data.cam_xpos[camera_id], rotation=data.cam_xmat[camera_id]),
                            "resolution": [width, height], "intrinsics": k})


def capture_mujoco_state(model, data, door_id=None, camera=None) -> dict:
    """Capture authoritative native poses without changing live simulation data.

    MuJoCo can leave position-dependent arrays one integration step behind qpos.
    Refresh kinematics/cameras on a private MjData copy, without dynamics or
    constraint solving. body_world composes with authored IR geometry directly;
    geom_world is the compiled native frame, including mesh compiler alignment.
    """
    import mujoco

    if door_id is not None and (not isinstance(door_id, str) or not door_id):
        raise ValueError("door_id must be a nonempty string or None")
    qpos = _vector(data.qpos, model.nq, "qpos")
    time = _number(data.time, "simulation time")
    copied = mujoco.MjData(model)
    mujoco.mj_copyData(copied, model, data)
    mujoco.mj_kinematics(model, copied)
    mujoco.mj_comPos(model, copied)
    mujoco.mj_camlight(model, copied)
    body_world, geom_world, joint_qpos = {}, {}, {}
    unnamed = {"body_ids": [], "geom_ids": [], "joint_ids": []}
    for index in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or ("world" if index == 0 else None)
        if name is None:
            unnamed["body_ids"].append(index)
            continue
        body_world[name] = _pose(copied.xpos[index], quaternion=copied.xquat[index])
    aliases = {}
    if "world_env" not in body_world:
        body_world["world_env"] = {k: list(v) for k, v in body_world["world"].items()}
        aliases["world_env"] = "world"
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        if name is None:
            unnamed["geom_ids"].append(index)
            continue
        owner = int(model.geom_bodyid[index])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, owner) or ("world" if owner == 0 else None)
        mesh_id = int(model.geom_dataid[index]) if int(model.geom_type[index]) == int(mujoco.mjtGeom.mjGEOM_MESH) else None
        geom_world[name] = {**_pose(copied.geom_xpos[index], rotation=copied.geom_xmat[index]),
                            "body_name": body_name, "body_id": owner, "geom_id": index,
                            "geom_type": mujoco.mjtGeom(int(model.geom_type[index])).name.removeprefix("mjGEOM_").lower(),
                            "size": _vector(model.geom_size[index], 3, "geom size"), "mesh_id": mesh_id}
        if mesh_id is not None:
            geom_world[name]["mesh_alignment"] = {
                "pos": _vector(model.mesh_pos[mesh_id], 3, "mesh alignment"),
                "quat_wxyz": _quaternion(model.mesh_quat[mesh_id]),
                "scale": _vector(model.mesh_scale[mesh_id], 3, "mesh scale"),
                "convention": "compiled_to_scaled_source_mesh"}
    for index in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        if name is None:
            unnamed["joint_ids"].append(index)
            continue
        start = int(model.jnt_qposadr[index])
        end = int(model.jnt_qposadr[index + 1]) if index + 1 < model.njnt else model.nq
        joint_qpos[name] = qpos[start] if end - start == 1 else qpos[start:end]
    result = {"schema_version": SCHEMA_VERSION, "door_id": door_id, "time_s": time,
              "state_kind": "simulation_snapshot", "kinematic_inspection": False,
              "body_world": body_world, "body_aliases": aliases, "geom_world": geom_world,
              "qpos": joint_qpos, "qpos_vector": qpos, "unnamed_source_objects": unnamed,
              "camera": _capture_camera(model, copied, camera, mujoco),
              "coordinate_system": "right_handed_z_up_meters", "source": {"engine": "mujoco", "version": mujoco.__version__}}
    # Final strict serialization is also a guard against accidental NumPy values.
    json.dumps(result, allow_nan=False)
    return result


capture_state = capture_mujoco_state


def _load_initial_model(door_dir):
    import mujoco

    directory = Path(door_dir)
    xml = directory / "door.xml"
    model_json_path = directory / "model.json"
    description = json.loads(model_json_path.read_text()) if model_json_path.exists() else None
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    if description:
        for body in description["bodies"]:
            joint = body.get("joint")
            if joint and "initial" in joint:
                _apply_named_qpos(model, data, {joint["name"]: joint["initial"]}, mujoco)
    return directory, model, data, description


def _apply_named_qpos(model, data, values, mujoco):
    if not isinstance(values, Mapping):
        raise ValueError("qpos must be a mapping of known scalar joint names to finite values")
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("qpos keys must be nonempty joint names")
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"Unknown MuJoCo joint: {name}")
        if int(model.jnt_type[joint]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError(f"Joint {name} is not scalar; supply free/ball state through live MjData")
        scalar = _number(value, f"qpos[{name}]")
        if model.jnt_limited[joint] and not (float(model.jnt_range[joint, 0]) - 1e-12 <= scalar <= float(model.jnt_range[joint, 1]) + 1e-12):
            raise ValueError(f"qpos[{name}]={scalar} is outside the current native joint limits")
        data.qpos[model.jnt_qposadr[joint]] = scalar


def _offline_state(door_dir, qpos=None, camera=None):
    import mujoco

    directory, model, data, description = _load_initial_model(door_dir)
    if qpos is not None:
        _apply_named_qpos(model, data, qpos, mujoco)
    mujoco.mj_forward(model, data)
    door_id = (description or {}).get("meta", {}).get("door_id", directory.name)
    result = capture_mujoco_state(model, data, door_id, camera)
    result["state_kind"] = "kinematic_inspection" if qpos is not None else "authored_initial"
    result["kinematic_inspection"] = qpos is not None
    result["source"]["initialization"] = "native_qpos0_then_authored_joint_initial"
    result["source"]["qpos_overrides"] = dict(qpos or {})
    for filename in ("door.xml", "model.json", "spec.json"):
        path = directory / filename
        if path.exists():
            result["source"][filename + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if description:
        # The MJCF exporter flattens static IR bodies onto native world. Their
        # IR geometry is already authored in world coordinates, so alias identity.
        for body in description["bodies"]:
            if body.get("static") and body["name"] not in result["body_world"]:
                result["body_world"][body["name"]] = {k: list(v) for k, v in result["body_world"]["world"].items()}
                result["body_aliases"][body["name"]] = "world"
    # Normalize mapping values (e.g. numpy scalar overrides) and verify finiteness.
    result["source"]["qpos_overrides"] = {name: _number(value, f"qpos[{name}]") for name, value in (qpos or {}).items()}
    return result


def capture_initial_state(door_dir, *, camera=None) -> dict:
    """Deterministic authored initial pose; generally closed, with authored lock states."""
    return _offline_state(door_dir, camera=camera)


def export_state(door_dir, out, *, qpos=None, camera=None) -> dict:
    """Write one strict JSON snapshot; explicit qpos stays within native limits.

    Prescribing a primary angle does not solve its closer/cam/lock couplings.
    Use capture_mujoco_state(env.m, env.d) for a physically simulated open pose.
    """
    result = _offline_state(door_dir, qpos=qpos, camera=camera)
    path = Path(out)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized)
    return result


def _named_mapping(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object keyed by registered source names")
    for name in value:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} keys must be nonempty source names")
    return value


def _joint_values(value, label, *, scalar_only=False):
    if isinstance(value, Real) and not isinstance(value, bool):
        return _number(value, label)
    if scalar_only:
        raise ValueError(f"{label} must be a finite scalar joint value")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a finite scalar or a 4/7-value ball/free joint state")
    if len(value) == 4:
        return _quaternion(value, label)
    if len(value) == 7:
        return _vector(value[:3], 3, label) + _quaternion(value[3:], label)
    raise ValueError(f"{label} must be scalar, a 4-value ball quaternion, or a 7-value free joint pose")


def _json_copy(value, label="snapshot"):
    """Copy supported JSON data, with actionable paths for invalid extra metadata."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Real):
        number = _number(value, label)
        # Preserve integer schema/ID fields rather than converting all numbers to float.
        from numbers import Integral
        return int(value) if isinstance(value, Integral) else number
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{label} object keys must be strings")
        return {key: _json_copy(item, f"{label}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_copy(item, f"{label}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{label} contains unsupported JSON value {type(value).__name__}")


def validate_snapshot(state, expected_door_id=None) -> dict:
    """Validate external schema-v1 snapshots without loading MuJoCo or changing poses.

    Returns a strict JSON-safe deep copy. Body/geom quaternions are canonicalized
    only within the existing 1e-6 unit-norm tolerance; malformed rotations are
    rejected. Optional telemetry can be omitted, but every supplied field must
    be well-formed. This checks the transport contract, not physical feasibility
    or that source hashes match files (the job preparer verifies those files).
    """
    import re
    from numbers import Integral

    if not isinstance(state, Mapping):
        raise ValueError("Snapshot must be a JSON object")
    version = state.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, Integral) or version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported snapshot schema_version {version!r}; expected {SCHEMA_VERSION}")
    if expected_door_id is not None and (not isinstance(expected_door_id, str) or not expected_door_id.strip()):
        raise ValueError("expected_door_id must be a nonempty string")
    door_id = state.get("door_id")
    if door_id is not None and (not isinstance(door_id, str) or not door_id.strip()):
        raise ValueError("Snapshot door_id must be a nonempty string or null")
    if expected_door_id is not None and door_id != expected_door_id:
        raise ValueError(f"Snapshot door_id {door_id!r} does not match expected door {expected_door_id!r}")
    if state.get("coordinate_system", "right_handed_z_up_meters") != "right_handed_z_up_meters":
        raise ValueError("Unsupported snapshot coordinate_system; expected right_handed_z_up_meters")
    result = dict(state)
    bodies = _named_mapping(state.get("body_world"), "body_world")
    if not bodies:
        raise ValueError("Snapshot body_world must contain at least one registered body pose")
    normalized_bodies = {}
    for name, pose in bodies.items():
        if not isinstance(pose, Mapping) or set(pose) != {"pos", "quat_wxyz"}:
            raise ValueError(f"body_world[{name!r}] must contain exactly pos and quat_wxyz")
        try:
            normalized_bodies[name] = _pose(pose["pos"], quaternion=pose["quat_wxyz"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid body_world[{name!r}]: {exc}") from exc
    result["body_world"] = normalized_bodies
    aliases = _named_mapping(state.get("body_aliases", {}), "body_aliases")
    for alias, target in aliases.items():
        if alias not in normalized_bodies:
            raise ValueError(f"Body alias {alias!r} has no body_world pose")
        if not isinstance(target, str) or target not in normalized_bodies:
            raise ValueError(f"Body alias {alias!r} targets missing body {target!r}")
    for alias, target in aliases.items():
        seen, current = set(), alias
        while current in aliases:
            if current in seen:
                raise ValueError(f"Body alias cycle involving {alias!r}")
            seen.add(current)
            current = aliases[current]
        if any(abs(a - b) > 1e-9 for key in ("pos", "quat_wxyz")
               for a, b in zip(normalized_bodies[alias][key], normalized_bodies[target][key])):
            raise ValueError(f"Body alias {alias!r} pose differs from its target {target!r}")
    if "time_s" in state:
        result["time_s"] = _number(state["time_s"], "time_s")
        if result["time_s"] < 0:
            raise ValueError("Snapshot time_s cannot be negative")
    if "state_kind" in state and state["state_kind"] not in ("simulation_snapshot", "authored_initial", "kinematic_inspection"):
        raise ValueError(f"Unsupported snapshot state_kind: {state['state_kind']!r}")
    if "kinematic_inspection" in state:
        if not isinstance(state["kinematic_inspection"], bool):
            raise ValueError("kinematic_inspection must be a JSON boolean")
        if "state_kind" in state and state["kinematic_inspection"] != (state["state_kind"] == "kinematic_inspection"):
            raise ValueError("state_kind and kinematic_inspection disagree")
    if state.get("camera") is not None:
        try:
            result["camera"] = validate_camera(state["camera"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid snapshot camera: {exc}") from exc
    if "qpos" in state:
        result["qpos"] = {name: _joint_values(value, f"qpos[{name!r}]")
                          for name, value in _named_mapping(state["qpos"], "qpos").items()}
    if "qpos_vector" in state:
        if not isinstance(state["qpos_vector"], (list, tuple)):
            raise ValueError("qpos_vector must be a flat JSON array of finite numbers")
        result["qpos_vector"] = [_number(value, f"qpos_vector[{i}]") for i, value in enumerate(state["qpos_vector"])]
        named_dofs = sum(len(value) if isinstance(value, list) else 1 for value in result.get("qpos", {}).values())
        if named_dofs > len(result["qpos_vector"]):
            raise ValueError("Named qpos values exceed the size of qpos_vector")
    if "geom_world" in state:
        result["geom_world"] = {}
        for name, geom in _named_mapping(state["geom_world"], "geom_world").items():
            if not isinstance(geom, Mapping) or "pos" not in geom or "quat_wxyz" not in geom:
                raise ValueError(f"geom_world[{name!r}] requires pos and quat_wxyz")
            try:
                normalized = {**geom, **_pose(geom["pos"], quaternion=geom["quat_wxyz"])}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid geom_world[{name!r}]: {exc}") from exc
            owner = geom.get("body_name")
            if owner is not None and (not isinstance(owner, str) or owner not in normalized_bodies):
                raise ValueError(f"Geom {name!r} references missing body {owner!r}")
            for id_field in ("geom_id", "body_id", "mesh_id"):
                value = geom.get(id_field)
                if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 0):
                    raise ValueError(f"geom_world[{name!r}].{id_field} must be a nonnegative integer or null")
            if "size" in geom:
                normalized["size"] = _vector(geom["size"], 3, f"geom_world[{name!r}].size")
                if any(v < 0 for v in normalized["size"]):
                    raise ValueError(f"Geom {name!r} dimensions cannot be negative")
            if "mesh_alignment" in geom:
                alignment = geom["mesh_alignment"]
                if not isinstance(alignment, Mapping) or not {"pos", "quat_wxyz", "scale"} <= set(alignment):
                    raise ValueError(f"Geom {name!r} has incomplete mesh_alignment")
                normalized["mesh_alignment"] = {**alignment, **_pose(alignment["pos"], quaternion=alignment["quat_wxyz"]),
                                                "scale": _vector(alignment["scale"], 3, f"geom_world[{name!r}].mesh_alignment.scale")}
                if any(v == 0 for v in normalized["mesh_alignment"]["scale"]):
                    raise ValueError(f"Geom {name!r} mesh scale cannot be zero")
            result["geom_world"][name] = normalized
    if "source" in state:
        if not isinstance(state["source"], Mapping):
            raise ValueError("Snapshot source provenance must be a JSON object")
        result["source"] = dict(state["source"])
        for filename in ("door.xml", "model.json", "spec.json"):
            field = filename + "_sha256"
            if field in result["source"]:
                value = result["source"][field]
                if not isinstance(value, str) or re.fullmatch(r"[a-fA-F0-9]{64}", value) is None:
                    raise ValueError(f"source.{field} must be a 64-character SHA256 hexadecimal digest")
                result["source"][field] = value.lower()
        if "qpos_overrides" in result["source"]:
            overrides = _named_mapping(result["source"]["qpos_overrides"], "source.qpos_overrides")
            result["source"]["qpos_overrides"] = {name: _joint_values(value, f"source.qpos_overrides[{name!r}]", scalar_only=True)
                                                       for name, value in overrides.items()}
            if overrides and not state.get("kinematic_inspection", False):
                raise ValueError("Source qpos_overrides require kinematic_inspection=true")
    result = _json_copy(result)
    json.dumps(result, allow_nan=False)
    return result
