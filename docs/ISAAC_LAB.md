# Isaac Lab integration

DoorBench ships as Isaac Lab tasks: `DoorBench-Open-Hand-v0` (6-DoF gantry hand) and `DoorBench-Open-G1-v0`
(Unitree G1), each with a **different door in every environment**. This page explains the USD design decisions,
the environment, and the exact commands. Everything on this page targets **Isaac Sim 5.1.0 + Isaac Lab v2.3.2**
(rsl-rl-lib 3.1.2; `scripts/isaaclab/check_api_names.py --source ~/IsaacLab` audits every symbol and config keyword
argument against that tree) and was **not executed on the author's machine (no NVIDIA GPU)** — see
[`isaaclab/STATUS.md`](../isaaclab/STATUS.md) for what was and was not verified.

## 1. Quick start on a GPU box

```bash
git clone https://github.com/adamraudonis/DoorBench.git && cd DoorBench
bash isaaclab/cloud/setup.sh && source isaaclab/cloud/env.sh
bash isaaclab/cloud/validate.sh
bash isaaclab/cloud/train.sh --task DoorBench-Open-Hand-v0 --num_envs 1024 --max_iterations 300
bash isaaclab/cloud/hero.sh
bash isaaclab/cloud/eval.sh logs/rsl_rl/doorbench_hand/<run>/model_300.pt
```

Already have Isaac Lab? `./isaaclab.sh -p -m pip install -e /path/to/DoorBench/isaaclab` and run the scripts in
`scripts/isaaclab/` with `./isaaclab.sh -p`. Set `DOORBENCH_ASSETS=/path/to/DoorBench/assets` if the extension is
installed outside the checkout.

## 2. USD design (what `doorbench/export/usd.py` writes)

Two files per door, both validated by `scripts/isaaclab/validate_usd_static.py` (pxr only, 1000/1000 pass):

### `door.usda` — full fidelity

```
/<door_id>                        default prim; doorbench:door_id, doorbench:meta, doorbench:couplings (JSON)
  /Looks                          UsdPreviewSurface materials + PhysicsMaterialAPI friction materials
  /Env                            static colliders: frame, walls, floor, strikes, stops + benchmark sites
  /Articulation                   PhysicsArticulationRootAPI + PhysxArticulationAPI (fixed base)
    /base                         fixed link; Joints/base_fixed = FixedJoint to the world (body0 unset)
    /<body> ...                   RigidBodyAPI + MassAPI (mass, COM, principal inertia) per moving body
    /Joints/<joint>               Revolute / Prismatic (limits, force drive = spring/closer, PhysxJointAxisAPI friction
                                  efforts, armature) / Fixed; PhysxMimicJointAPI for polynomial couplings
/PhysicsScene                     outside the default prim (present standalone, dropped when referenced)
```

* **Fixed base, single tree.** Every moving chain hangs off `base`, so pairs of leaves, wall-mounted buttons, gate
  pins etc. form one PhysX articulation instead of several roots (the previous layout had joints to the world from
  several bodies, which PhysX does not parse as one articulation).
* **Coordinates.** USD joint value 0 is the spec's initial state; limits are the IR range shifted by the MJCF `ref`
  (`doorbench:zero_offset`). Revolute limits and drive targets are in degrees (UsdPhysics), prismatic in metres.
  Positive = opening / actuating, as in the MJCF.
* **Springs / closers** are `force` drives: `drive:angular:physics:stiffness` in N·m/deg (Isaac Lab converts to
  N·m/rad), `targetPosition` = the spring reference (negative for preloaded closers and latch springs).
* **Coulomb friction.** `physxJointAxis:rotX|transX:staticFrictionEffort` / `dynamicFrictionEffort` carry the hinge
  friction torque (N·m) / roller friction (N) from the physics model (PhysX ≥ 5.6, Isaac Sim ≥ 5.0). For older PhysX the
  legacy unitless `physxJoint:jointFriction` coefficient is set to `torque / estimated joint reaction force` (the
  legacy model multiplies the coefficient by the joint's spatial force; using the torque directly would over-brake the
  door 100×).
* **Couplings.** Bilateral polynomial equalities (thumbturn → deadbolt, wheel → bolts, riser ← hinge) become
  `PhysxMimicJointAPI` (`q_driven + gearing·q_driver + offset = 0`, radians / metres — PhysX-native units; verify on
  the GPU box). The one-sided latch tendon (`bolt ≥ scale·operator`) is environment logic (`DoorMechanismAction`).
* **Collision.** Primitives as Cube/Cylinder/Capsule/Sphere with `PhysicsCollisionAPI` + `PhysxCollisionAPI`;
  meshes referenced from `assets/hardware/*.usdc` with `MeshCollisionAPI approximation=convexHull` and
  `PhysxConvexHullCollisionAPI`. Self-collisions are disabled (parent/child hardware overlaps by design).

### `door_rl.usda` — canonical articulation (multi-door RL)

Isaac Lab's `Articulation` over N environments is one PhysX articulation *view*; the view requires every instance
to have the same links, joints and joint types. DoorBench doors have 1–18 moving bodies, so the exporter maps every
door onto one fixed structure:

| link | joint (parent → child) | used by |
|---|---|---|
| `base` | `base_fixed` (world → base) | all |
| `carriage` | `door_slide` prismatic (base → carriage) | sliders, garage / roll-up curtains |
| `leaf` | `door_hinge` revolute (carriage → leaf) | hinged leaves, rotors, flaps |
| `operator_pivot` | `operator_hinge` revolute (leaf → pivot) | levers, knobs, thumb latches, wheels |
| `operator` | `operator_slide` prismatic (pivot → operator) | touch bars, paddles, slide bolts |
| `latch` | `latch_slide` prismatic (leaf → latch) | spring latch bolts |
| `carriage2` / `leaf2` | `leaf2_slide` / `leaf2_hinge` (base → …) | second leaf of pairs, saloons, bypass and bi-parting sliders |

Unused joints are locked (±0.5 mm / ±0.05°, stiff drive). Every other moving part in the primary leaf's subtree
(deadbolts, thumbturns, keypad keys, closer arms, folded panels …) is welded into `leaf` / `operator` at the initial
state — engaged locks stay engaged (so `locked_recognize` doors stay locked), latch-like parts that would block the
door are welded released. World-mounted parts (gate lift pins, REX buttons) become static; leaf-like panels beyond the
second (strip curtains, accordions) are omitted. Slot statistics over the dataset:

```
347  door=hinge operator=hinge latch=slide          217  door=hinge (push plates, pulls, free-swinging)
123  door=slide                                      51  hinge + second hinged leaf     49  slide + second sliding leaf
 40  hinge + hinge operator, no latch                34  hinge + latch, no operator      29  hinge + slide operator (touch bars) + latch
```

The root prim carries `doorbench:rl` (JSON): live slots, per-joint source/range/spring, latch coupling scale, secondary
coupling, thresholds (open 10° / 0.10 m, clear 60° / 0.55 m, closed 3°), grip/push sites in link frames,
approach / goal / pass-plane points, opening size, push/pull side, lock state, damage thresholds and automatic-door
servo gains. `DoorState` reads it for every env after spawning — no dataset files are needed at run time.

Limitations of the canonical file (by design, documented per door in `doorbench:rl["notes"]`): keypad / card /
thumbturn unlocking is not possible (the lock parts are welded), bifold/accordion chains move as one panel, only two
leaves per door, no loop-closure closer arms (they are visual).

## 3. The environment

* **Scene.** Ground plane + dome light + `door` (`MultiUsdFileCfg` over the selected `door_rl.usda` files,
  round-robin so `num_envs ≥ len(doors)` shows every door; `replicate_physics=False`) + the agent + contact sensors on
  `Door/Articulation/leaf` and `.../operator` filtered by the agent's bodies. `env_spacing = 6 m` (walls are 5 m wide).
* **Door physics at run time.** `DoorMechanismAction` (a 0-dim action term, runs every physics step) restores the
  USD spring targets (Isaac Lab otherwise writes zeros as drive targets), couples the latch bolt to the operator
  (`bolt_target += scale·operator_q`), adds the asymmetric closer damping / backcheck as feed-forward effort, and
  servos automatic doors open while the agent is within 1.8 m of the door plane.
* **Labels.** `DoorState.update()` reproduces `doorbench/benchmark/labels.py`: touched (contact or tip within 10 cm
  of the grip point), operator actuated (≥ 70 % travel), latch released (≥ 80 %), opened / clear, passed through
  (base crosses the wall plane inside the opening), closed after, slammed (closing speed at the stop), damaged (agent
  contact force above the dent / glass threshold). `DoorState.success()` is the benchmark predicate per task.
* **Observations / actions / rewards / terminations / events**: see [`isaaclab/README.md`](../isaaclab/README.md)
  and `isaaclab/doorbench_isaaclab/door_task_env_cfg.py` (everything is a standard manager-based cfg you can subclass).
* **Door subsets.** `--doors easy-100` (default: curated, balanced over families; unlocked, simple operators, RL-friendly
  tasks) · `easy-300` · `all` · `random-50` · `family:saloon,pivot` · `db0002_swing_single,...` · `@ids.txt`;
  `--door_seed` shuffles, `--door_random_choice` uses Isaac Lab's random pick instead of round-robin.

## 4. Scripts

| script | purpose |
|---|---|
| `scripts/isaaclab/validate_usd_static.py` | pxr-only checks of every door.usda / door_rl.usda (structure, frames, drives vs model.json, meshes, materials, JSON) → `assets/usd_validation.json` |
| `scripts/isaaclab/validate_usd_isaacsim.py` | inside Isaac Sim: spawn batches of doors as `Articulation`s, compare joints/limits/gains, settle 200 steps, push + actuate 400 steps, report → `assets/usd_validation_isaacsim.json` |
| `scripts/isaaclab/train.py` / `play.py` | RSL-RL PPO train / play with `--doors`; logs in `logs/rsl_rl/<experiment>/` |
| `scripts/isaaclab/record_hero.py` | `--num_envs 512 --doors all`: wide screenshot, orbiting video, detail shot → `docs/media/isaaclab_hero*.{png,mp4}` |
| `scripts/isaaclab/eval_all_doors.py` | checkpoint over all doors × seeds → `results/isaaclab_<task>_<date>.json` (`per_door`, `aggregate` by family / task; adopts `results/schema.json` if present) |
| `scripts/isaaclab/make_hand_usd.py` | regenerates the gantry hand USD |
| `scripts/isaaclab/check_api_names.py` | offline checklist of every Isaac Lab symbol the extension uses (v2.3.0 reference list) |

## 5. Known unknowns for the first GPU run

1. `PhysxMimicJointAPI` gearing units (rad vs deg) for the full `door.usda` couplings.
2. `MultiUsdFileCfg` + `articulation_root_prim_path="/Articulation"` resolution for referenced prims (fallback: drop
   the argument; Isaac Lab then searches the child with `ArticulationRootAPI`, which is unique).
3. Contact-sensor filter expressions with `{ENV_REGEX_NS}/Robot/.*` (G1); the hand uses a single body.
4. The 0-dim action term (`DoorMechanismAction`); if the action manager rejects it, move `apply_actions` into an
   interval event term (`mode="interval", interval_range_s=(0, 0)`), which runs once per policy step.
5. Isaac Lab / rsl-rl versions: the PPO cfgs use the v2.3.2 / rsl-rl-lib 3.1.2 field set (`obs_groups`,
   `actor_obs_normalization`; the runner-level `empirical_normalization` is deprecated).  v2.3.2 removed
   `isaaclab.utils.io.dump_pickle` (isaaclab 0.47.0) — `scripts/isaaclab/_common.py` carries the replacement.
