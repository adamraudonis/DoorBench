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
* Import / physics test over the generated dataset (one door per family + 20 seeded random doors; all tiers,
  `scene.xml`, `door.urdf`, 500 free steps, QA hold / actuate / relatch / closer, `DoorEnv` episode, clearance gate):

  ```bash
  pytest -q tests/test_mujoco_import.py            # ~5 s; DOORBENCH_ASSETS=<dir> to test another dataset directory
  python scripts/demo_mujoco.py --ids db0002_swing_single --out /tmp/demo    # video of a programmatic hand opening it
  ```

## Isaac Sim / Isaac Lab (USD)

* `door.usda` (full fidelity): default prim `/<door_id>` with `Env` (static colliders + sites) and `Articulation`
  (`UsdPhysics.ArticulationRootAPI`, fixed `base` link joined to the world, one tree). Rigid bodies carry explicit
  mass properties; collision shapes are boxes/capsules/spheres + convex-hull meshes referenced from
  `assets/hardware/*.usdc`; revolute/prismatic joints have limits, force drives carrying closer / latch / return
  springs (`stiffness`, `damping`, `targetPosition`), armature and Coulomb friction as
  `physxJointAxis:angular|linear:staticFrictionEffort` (N·m / N, PhysX ≥ 5.6; the legacy load-dependent
  `physxJoint:jointFriction` coefficient is authored 0). MJCF position servos of spring-less joints (automatic
  sliders) are the PhysX drive itself (`doorbench:servo_in_drive`); see `docs/ISAAC_LAB.md` for the full
  MuJoCo → PhysX parameter mapping.
* `door_rl.usda` (canonical): the same 8 links / 7 joints for every door (`door_slide`, `door_hinge`,
  `operator_hinge`, `operator_slide`, `latch_slide`, `leaf2_slide`, `leaf2_hinge`; unused slots locked) so Isaac Lab's
  `MultiUsdFileCfg` can put a different door in every environment; `doorbench:rl` on the root describes the live
  slots, thresholds, grip points and sites.
* Couplings (thumbturn → deadbolt, wheel → bolts, riser ← hinge) are `PhysxMimicJointAPI` on the driven joint and JSON
  in `doorbench:couplings`; the one-sided latch coupling, closer asymmetry and maglock breakaway are environment logic
  (`doorbench_isaaclab.mdp.DoorMechanismAction` / `DoorEnv._lock_logic`).
* Ready-made Isaac Lab tasks, training / evaluation / hero-shot scripts and a one-command GPU-box setup:
  [docs/ISAAC_LAB.md](ISAAC_LAB.md), [isaaclab/README.md](../isaaclab/README.md). Static validation of every USD:
  `python scripts/isaaclab/validate_usd_static.py` (1000/1000 pass); Isaac Sim import validation:
  `bash isaaclab/cloud/validate.sh` (GPU).

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
