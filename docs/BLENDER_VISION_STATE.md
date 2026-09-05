# MuJoCo state bridge for Blender vision

The appearance renderer consumes a snapshot of the existing MuJoCo articulation. It does not reconstruct motion from joint names, animate geometry independently, or alter physics to make an image plausible. The bridge lives in `doorbench.appearance.state`; importing it requires only Python's standard library. MuJoCo is imported inside capture/export functions.

```python
import json
from doorbench.appearance.state import capture_mujoco_state, export_state

# Use a live DoorEnv after any number of simulation steps.
snapshot = capture_mujoco_state(env.m, env.d, door_id=env.spec['id'],
                               camera={'name': 'robot_view', 'resolution': [960, 720]})
with open('frame.json', 'w') as stream:
    json.dump(snapshot, stream, allow_nan=False)

# Deterministic authored initial state for a standalone appearance job.
export_state('assets/doors/db0012_swing_single', 'initial.json', camera='iso')
```

`capture_initial_state(door_dir, *, camera=None)` returns the same initial snapshot without writing a file. Initial states start with native `qpos0`/joint reference values, then apply the authored IR `joint.initial` values. This usually means a closed leaf with the authored latch/lock configuration. It is identified as `authored_initial`, rather than a simulated successful opening.

`export_state(door_dir, out, *, qpos=None, camera=None)` also accepts explicit **scalar** joint overrides, such as `qpos={'leaf_hinge': 0.2}`. Names, finiteness, scalar joint types, and current native joint limits are checked. The result is marked `kinematic_inspection`; it is not a physically actuated state. An engaged lock is never silently released, and changing a primary angle does not solve its closer/cam/lock couplings. Use a live simulation snapshot for vision training involving a dynamically opened door.

## Schema version 1

| Field | Meaning |
|---|---|
| `schema_version` | Integer `1` |
| `door_id` | Caller-provided identity; offline factories derive it from the IR |
| `time_s` | Native simulation time, never wall-clock time |
| `state_kind` | `simulation_snapshot`, `authored_initial`, or `kinematic_inspection` |
| `kinematic_inspection` | True when the caller explicitly supplies offline qpos overrides |
| `body_world` | Named native body poses: `{name: {pos: [x,y,z], quat_wxyz: [w,x,y,z]}}` |
| `body_aliases` | Flattened static IR names mapped to native `world` |
| `geom_world` | Named compiled MuJoCo geom world poses, dimensions, body ownership, and mesh alignment metadata |
| `qpos` | Named joint values; hinge/slide values are scalar, free/ball values are vectors |
| `qpos_vector` | Complete native qpos array, including unnamed joints |
| `unnamed_source_objects` | IDs that cannot be represented by a registered source name |
| `camera` | Optional calibrated pinhole camera, described below |
| `source` | MuJoCo version; offline exports also record source-file hashes, initialization, and explicit overrides |

Coordinates are right-handed, Z-up, in meters. Quaternions are world rotations in **wxyz** order. Equivalent quaternion signs are normalized deterministically. Every number must be finite and every value serializes as strict JSON. There are no random values or timestamps, so repeated exports of the same source state are byte-identical.

MuJoCo may leave position-dependent arrays one integration step behind `qpos`. Capture copies `MjData` and runs native kinematics, center-of-mass positioning, and camera positioning on that copy. It does not step simulation, resolve constraints, invoke a new passive-force calculation, or modify the live model/data.

## Use body poses with authored meshes

The Blender importer should assign `body_world[body.name]` to each rendered body and compose the **authored IR geom local position/quaternion** below that body. This handles nested joints and MuJoCo reference positions without manually reconstructing their motion.

Do **not** apply a compiled `geom_world` mesh pose directly to an original OBJ. MuJoCo recenters and rotates meshes into principal-axis coordinates while compiling them. This alignment is already incorporated into the compiled geom pose. Applying it to the original OBJ a second time introduces an offset/rotation. The `mesh_alignment` record is diagnostic and maps compiled mesh coordinates back to the scaled source mesh frame; the body-plus-IR route needs no alignment correction.

`world_env` is always aliased to native `world` if it is absent as a named native body. Offline factories discover any other flattened static IR bodies and alias them to world as well. Unknown dynamic bodies are not fabricated. A reduced physics tier that omits a moving visual mechanism cannot by itself specify that mechanism's pose: the renderer must reject the missing state or use a separately validated visual articulation mapping. This bridge does not invent that mapping. Deforming skins/flex meshes and additional robot render geometry require their own geometry streams.

## Cameras

A named camera can be requested as `'iso'` or `{'name': 'iso', 'resolution': [960, 720]}`. The camera's native declared resolution is used when available; MuJoCo's default 1×1 declaration falls back to 640×480. Native rendering frustum values determine focal lengths and principal point, including asymmetric calibrations, without requiring an OpenGL context.

An explicit camera manifest uses:

```json
{
  "pos": [1.3, -2.1, 1.7],
  "quat_wxyz": [1, 0, 0, 0],
  "resolution": [640, 480],
  "intrinsics": [[500, 0, 320], [0, 510, 240], [0, 0, 1]]
}
```

A proper orthonormal 3×3 `rotation_matrix` may replace the unit quaternion. The camera frame uses **+X right, +Y up, −Z forward**, matching MuJoCo and Blender. Image coordinates run right/down, with their origin at the top-left image boundary; for a camera-local point `p`, project `K @ [p.x, -p.y, -p.z]` and divide by depth. MuJoCo documents camera orientation and calibrated projection parameters in its [camera reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera).

Only perspective, zero-skew pinhole cameras are supported. Focal lengths must be positive, resolution must contain positive integer pixels, and the intrinsics' last row must be `[0,0,1]`. Camera poses with invalid/non-unit quaternions, reflections, non-finite values, unknown native camera names, or unsupported projections are rejected.

## Verification

Ten focused tests cover native body/geom pose equivalence after physics steps on hinged, sliding, and repaired cold-room doors; a nested telescoping/ref-position fixture; unchanged live data; deterministic initial exports; invalid/unknown joint overrides; invalid/skewed calibration; native asymmetric camera intrinsics; optional dependency imports; and a non-centered asymmetric mesh proving body-plus-IR placement matches the compiler-corrected native mesh.

All 1000 generated doors also produced valid deterministic initial snapshots during the implementation audit. Independent numerical projection checks in Blender 5.2.1 covered square, portrait, and landscape frames, unequal focal lengths, shifted principal points, and a rotated/transformed camera. The largest error was below 0.00014 pixels with correctly scaled pixel aspect settings.

This is an offline serialization bridge, with no real-time throughput claim. It carries poses and calibration; it does not certify appearance realism, simulation quality, or a vision model's performance.
