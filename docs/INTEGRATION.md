# Simulator integration

## MuJoCo (reference)

```python
import mujoco
m = mujoco.MjModel.from_xml_path("assets/doors/db0002_swing_single/door.xml")   # or door_simple.xml / door_minimal.xml
```
* `scene.xml` includes `door.xml` and sets statistics for the viewer.
* Meshes resolve via `meshdir="../../hardware"`; keep the folder structure or edit `<compiler meshdir>`.
* Merge with a robot using `mujoco.MjSpec` (see `doorbench/benchmark/env.py::DoorEnv._build`).
* MJX: use `door_simple.xml` / `door_minimal.xml` (primitives only, no tendon limits in `minimal`).

## Isaac Sim / Isaac Lab (USD)

* `door.usda` defines `/World/<door_id>` with `UsdPhysics.ArticulationRootAPI`, rigid bodies with explicit mass
  properties, collision shapes (boxes/capsules/spheres + convex-hull meshes referenced from `assets/hardware/*.usdc`),
  revolute/prismatic joints with limits, joint friction (`physxJoint:jointFriction`), armature and drives carrying
  closer springs (`stiffness`, `damping`, `targetPosition`).
* Couplings (thumbturn → deadbolt, wheel → bolts, bifold panels) are exported as `physxMimicJoint:*` attributes and
  as JSON in `doorbench:couplings` on the root prim; static environment prims carry `CollisionAPI` only.
* One-sided latch coupling and maglock breakaway are environment logic; port `DoorEnv._lock_logic` or use the
  `simple` tier for RL.
* Import as an `ArticulationCfg` in Isaac Lab with `spawn=UsdFileCfg(usd_path=...)`; set `actuators` on the door
  joints if you want to drive automatic doors.

## URDF (Genesis, PyBullet, Gazebo, Drake …)

* `door.urdf`: root link `world_env` (frame + walls), leaf joints (`revolute`/`prismatic`) with `<dynamics
  damping friction>` and `<limit>`.  Joint zero is the spec's initial state (`doorbench:zero_offset`).
* Springs (`doorbench:spring`), closers (`doorbench:closer`), ratchets, welds and loop closures are extension
  elements — apply them with your simulator's joint-spring API.
* `<mimic>` couplings are bilateral; the latch bolt follows the operator.

## Genesis

```python
import genesis as gs
scene = gs.Scene()
door = scene.add_entity(gs.morphs.URDF(file="assets/doors/db0002_swing_single/door.urdf", fixed=True))
```

## Photoreal rendering

`blender/render_door.py` (Blender 4.x/5.x, headless) rebuilds a door from `model.json` with Poly Haven CC0 PBR textures
referenced by `spec.leaf.finish.texture` / material ids, glass & mirror shaders, and an HDRI, and renders the
catalogue "hero" images.
