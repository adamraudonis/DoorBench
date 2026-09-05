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
/<door_id>                        default prim; doorbench:door_id, doorbench:meta, doorbench:couplings,
                                  doorbench:env_release, doorbench:filtered_pairs (JSON)
  /Looks                          UsdPreviewSurface materials + PhysicsMaterialAPI friction materials
  /Env                            static colliders: frame, walls, floor, strikes, stops + benchmark sites
  /Articulation                   PhysicsArticulationRootAPI + PhysxArticulationAPI (fixed base)
    /base                         fixed link; Joints/base_fixed = FixedJoint to the world (body0 unset)
    /<body> ...                   RigidBodyAPI + MassAPI (mass, COM, principal inertia) per moving body
    /Joints/<joint>               Revolute / Prismatic (limits, force drive = spring/closer, PhysxJointAxisAPI friction
                                  efforts, armature) / Fixed; PhysxMimicJointAPI for ROTATIONAL polynomial couplings,
                                  doorbench:coupling_* for the rest; breakable env-release FixedJoints (loop joints)
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
* **Coulomb friction.** `physxJointAxis:angular:staticFrictionEffort` / `dynamicFrictionEffort` (revolute) and
  `physxJointAxis:linear:*` (prismatic) carry the hinge friction torque (N·m) / roller friction (N) from the physics
  model (PhysX ≥ 5.6, Isaac Sim ≥ 5.0). The instance name matters: the PhysX USD parser reads the per-axis API only
  on the drive's instance (`angular` / `linear`) of a single-DoF joint; `rotX` / `transX` are D6 tokens and are
  silently ignored — round 1 of the parity gate shipped them and read back 0 friction on every joint. The legacy
  load-dependent `physxJoint:jointFriction` coefficient is authored **0** (it would add `coeff × |joint force|` on top
  of the efforts; its old value is kept in `doorbench:legacy_friction_coeff` for reference).
* **Velocity cap.** Every link carries `physxRigidBody:maxAngularVelocity = 5729.58` deg/s (= 100 rad/s, the PhysX
  default). The schema unit is degrees per second, and so is Isaac Lab's `RigidBodyPropertiesCfg.max_angular_velocity`
  (`DOOR_RIGID_PROPS`): the round-1 value 100.0 clamped every leaf at 1.75 rad/s. MuJoCo has no cap.
* **Automatic doors.** The MJCF position servo (`meta.actuators`: `f = clip(kp (ctrl − q) − kv v, forcerange)`) is
  the same law as a PhysX PD drive. On spring-less joints (automatic sliders, elevators) the servo *is* the joint drive
  (`stiffness = kp`, `damping = kv + joint damping`, `targetPosition = ctrl`, `maxForce = forcerange`;
  `doorbench:servo_in_drive = true`). Joints that also carry a closer spring (automatic swing operators) keep the
  spring in the drive and the servo remains a feed-forward emulation (one drive per axis; a shared `maxForce` would
  clip the closer with the servo's force range).
* **Couplings.** Bilateral polynomial equalities (`q_driven = c₀ + c₁·q_driver`) get one of three treatments,
  recorded in `doorbench:coupling_mode` on the driven joint prim and in `doorbench:couplings`:
  * `mimic` — **rotational → rotational only** (187 equalities on 93 doors: hook ← handle, dog ← wheel, panel ←
    pivot). `PhysxMimicJointAPI:rotX` with `referenceJoint`, `gearing = −c₁`, `offset = −c₀` (`q_driven + gearing·
    q_driver + offset = 0`, radians). PhysX articulation mimic joints **support rotational axes only**: a mimic
    authored on a prismatic joint, or one whose `referenceJointAxis` is prismatic, is parsed without an error and
    then ignored — which is why the exporter no longer writes one.
  * `emulated` — hinge → slide and slide → slide (181 equalities on 124 doors: thumbturn → deadbolt, wheel → shoot
    bolts, dogs → bolts, cremone, helical riser ← hinge). No PhysX representation exists, so the driven joint prim
    carries the whole law and both consumers apply it **bilaterally**, never as a bare kinematic write:
    `doorbench:coupling_c0 / c1`, `coupling_driver` (+ `coupling_driver_joint` rel), `coupling_driven_inertia`
    (the driven DOF's effective inertia: subtree mass for a slide, subtree inertia about the axis for a hinge),
    `coupling_gravity_bias` (its constant generalised gravity force at the authored pose),
    `coupling_reflected_inertia = c₁²·I_driven`, `coupling_chain_order`, and on the **driver**
    `doorbench:coupling_reflected_armature` = Σ of the reflected inertias it drives.
  * `servo` — both joints carry their own MJCF position servo in their PhysX drive (14 doors: the two leaves of an
    automatic bi-parting slider or elevator). The drives already move them together; nothing is authored or emulated.

  The one-sided latch tendon (`bolt ≥ scale·operator`) stays environment logic (`DoorMechanismAction`).
* **Environment-released locks.** A mag lock / delayed egress / electric bolt / interlock is a MuJoCo `<weld>`
  leaf → world. It is exported as a real breakable `UsdPhysics.FixedJoint` `base → leaf` under `Joints/<weld name>`
  with `physics:excludeFromArticulation = true` (PhysX solves it as a maximal-coordinate **loop joint** between two
  articulation links, so the articulation stays a tree with one parent per link), `physics:breakForce =
  breakTorque = ` the latch model's `holding_force_N` (2670 N for a 600 lbf magnet, 5340 N for 1200 lbf) and
  `physics:jointEnabled`, which the environment clears on badge / REX / delayed-egress timeout —
  `mdp.release_env_lock`, the counterpart of `DoorEnv._lock_logic` setting `d.eq_active[eid] = 0`.
  `doorbench:env_release` on the default prim names them (`doorbench:role = "env_release"` on the joint).
* **Collision.** Primitives as Cube/Cylinder/Capsule/Sphere with `PhysicsCollisionAPI` + `PhysxCollisionAPI`;
  meshes referenced from `assets/hardware/*.usdc` with `MeshCollisionAPI approximation=convexHull` and
  `PhysxConvexHullCollisionAPI`. **Self-collision is enabled** (`physxArticulation:enabledSelfCollisions = true`):
  PhysX then skips joint-adjacent link pairs, which is exactly MuJoCo's parent/child default, and every further pair
  MuJoCo suppresses — same weld body, weld parent/child, `<contact><exclude>` — is authored as
  `PhysxFilteredPairsAPI` (`physxFilteredPairs:filteredPairs`; 2091 pairs over the dataset, at most 18 on one door,
  at most 185 extra broad-phase pairs). With self-collision off, any latch that holds one *moving* link against
  another passed straight through in PhysX: swing pairs latched into the inactive leaf, gate and baby-gate lift
  pins, sliding drop bolts.

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
(deadbolts, thumbturns, keypad keys, closer arms, folded panels …) is welded into `leaf` / `operator`. **In which
state** is decided per part and recorded, not guessed:

| welded | when | example |
|---|---|---|
| **released** (joint moved to its travel end) | the part is a spring latch (`role == "latch"`), **or** — provided the spec grants a robot-side release — it is latch hardware by semantic, the operator drives it through a bilateral equality / tendon (transitive closure from `meta.operator_joint`), or its lock is not engaged | hook bolts of a hook-slider, cremone shoot bolts on the espagnolette handle, dogs driven by a ship wheel, world-mounted lift pins |
| **engaged** (initial state) | an engaged lock part with no canonical slot and no operator coupling — the robot would have to work it separately, which this file cannot represent — **or** any engaged lock part of a door with `lock.robot_side_release = false` (keyed outside only, padlock, no inside trim): the real door stays locked, so `locked_holds` must hold here too | a thrown slide bolt, a keypad-locked deadbolt, a second dog with its own lever, the hook of an outside-keyed hook lock |

`holding` marks the parts that can actually hold the leaf shut: engaged, with a `latch` / `lock` role or semantic,
not a sensor or decoration, and **not press-only** — a slide whose axis is the leaf normal (keypad keys, REX and call
buttons, privacy buttons) presses into the face and can never reach the frame, so welding it in either state changes
nothing. A bolt, rod, hook or pin moves in the plane of the leaf, toward an edge.

Welding a hook or a cremone bolt engaged locked the door **by construction** while the protocol expected it to open
(11 hook sliders, 2 cremone pairs and ~35 doors flagged `RL_CANON` in parity round 2). `doorbench:rl` records every
decision — `welded` (all parts, with `role`, `semantic`, `shift`, `was_engaged`, `holding`, `reason`),
`released_parts`, `released_holding`, `welded_engaged`, `operator_driven_joints` — and
`doorbench.parity.protocol.expected_outcomes` reads that ground truth and combines it with the joints the protocol
itself works to release the door (`thumbturn_joint`, `aux_joints`, `dog_joints`): a door whose only holding part is
welded released gets `hold = "na:rl holding part welded released (…)"` (15 doors: hook sliders, vault and blast
bolt rings, ship dogs, cremone pairs), and a door where one of those release joints is welded
**engaged** gets `operate = "stays_closed"` (`_rl_blocking`). World-mounted parts (gate lift pins, REX buttons) become static; leaf-like panels beyond
the second (strip curtains, accordions) are omitted.

`door_rl.usda` carries the same self-collision setting, filtered pairs (at link level: `leaf`/`operator` is filtered
because the handle is a child of the leaf in MuJoCo, `leaf`/`leaf2` is **not**, which is what latches a pair),
env-release joints and coupling metadata as `door.usda`. Slot statistics over the dataset:

```
347  door=hinge operator=hinge latch=slide          217  door=hinge (push plates, pulls, free-swinging)
123  door=slide                                      51  hinge + second hinged leaf     49  slide + second sliding leaf
 40  hinge + hinge operator, no latch                34  hinge + latch, no operator      29  hinge + slide operator (touch bars) + latch
```

The root prim carries `doorbench:rl` (JSON): live slots, per-joint source/range/spring, latch coupling scale, secondary
coupling, thresholds (open 10° / 0.10 m, clear 60° / 0.55 m, closed 3°), grip/push sites in link frames,
approach / goal / pass-plane points, opening size, push/pull side, lock state, damage thresholds and automatic-door
servo gains. `DoorState` reads it for every env after spawning — no dataset files are needed at run time.

Limitations of the canonical file (by design, documented per door in `doorbench:rl["notes"]` and, per part, in
`doorbench:rl["welded"]`): keypad / card / thumbturn unlocking is not possible (those lock parts are welded engaged
and the door is expected to stay closed), bifold/accordion chains move as one panel, only two leaves per door, no
loop-closure closer arms (they are visual).

### MuJoCo → PhysX parameter mapping

The MJCF is the reference; every USD quantity below is derived from the same IR joint so that the door behaves the
same in both engines (checked by `scripts/isaaclab/validate_usd_static.py` against `model.json` and by the parity
gate, `docs/ISAAC_PARITY.md`). Where the two engines model a quantity differently the mapping reproduces MuJoCo's
behaviour, not the nominal number.

| MJCF (`<joint>` / `<actuator>`) | USD (per joint prim) | derivation |
|---|---|---|
| `stiffness k`, `springref q₀` (N·m/rad, rad) | `drive:angular:physics:stiffness = k·π/180`, `targetPosition = (q₀ − ref)·180/π` (`force` drive) | UsdPhysics angular drives are per degree; USD joint 0 is the MJCF `ref` pose. Prismatic: 1:1. Isaac Lab zero-initialises drive targets, so `DoorMechanismAction` / the parity runner restore `doorbench:target_si` every step. |
| `damping b` | `drive:*:physics:damping = b·π/180` (revolute), `b` (prismatic) | both engines integrate joint damping implicitly (`implicitfast` / PhysX drive), so the same coefficient gives the same decay. |
| `frictionloss F` (N·m or N) | `physxJointAxis:angular|linear:staticFrictionEffort = dynamicFrictionEffort = F`, `viscousFrictionCoefficient = 0`, `physxJoint:jointFriction = 0` | MuJoCo's `frictionloss` is one Coulomb bound for stick and slip → static = dynamic effort. The legacy PhysX coefficient multiplies the joint's spatial reaction force (load dependent: the round-1 turnstile columns dragging on the floor showed 7–10× the authored friction), so it must be 0. |
| `armature` (kg·m² / kg) | `physxJointAxis:*:armature` and `physxJoint:armature` | identical semantics (added to the joint-space inertia diagonal). |
| `range`, `solreflimit` | `physics:lowerLimit / upperLimit` (deg / m), hard | PhysX articulation limits have no compliance; MuJoCo's soft limit (`solref 0.005`, gasket doors `0.02`) rebounds ~17 % of the impact velocity. Not mappable: the parity protocol grades the peak angle, not the post-rebound rest angle. |
| `<position kp kv forcerange ctrl=0>` on a spring-less joint | `stiffness = kp`, `damping = kv + b`, `targetPosition = ctrl − ref`, `maxForce = |forcerange|`, `doorbench:servo_in_drive = true` | `f = clip(kp(ctrl−q) − kv·v, ±F)` is the PhysX PD drive law. The passive `b·v` (2–8 N·s/m) is clipped along with the servo: < 3 % of the 150 N saturation at 0.5 m/s. Environments move only the target (`set_joint_position_target`). |
| `<position>` on a joint with its own spring (automatic swing + closer) | drive = spring; servo as feed-forward effort (`servo_effort`, `in_drive = false`) | one drive per axis: a shared `maxForce` would clip the closer spring with the servo's force range. |
| no velocity cap | `physxRigidBody:maxAngularVelocity = 5729.58` deg/s (100 rad/s), Isaac Lab cfg `max_angular_velocity = 5729.58` | PhysX default; far above the fastest door motion (pet flap ≈ 65 rad/s under the QA push). The unit is deg/s in both places — 100.0 there clamps a leaf at 1.75 rad/s. |
| `<equality joint>` `q_a = c₀ + c₁·q_b`, both **rotational** | `PhysxMimicJointAPI:rotX` on the driven joint (`gearing = −c₁`, `offset = −c₀`, `referenceJointAxis = rotX`) | PhysX articulation mimic joints are a linear relation between two joint axes — the same law, in radians. |
| `<equality joint>` with a **prismatic** axis on either side | `doorbench:coupling_mode = "emulated"` + `coupling_c0/c1`, `coupling_driven_inertia`, `coupling_gravity_bias`, `coupling_reflected_inertia`, and `coupling_reflected_armature` on the driver | PhysX mimic joints support rotational axes only; a prismatic mimic is parsed and dropped. Consumers apply the constraint `c = q_a − c₀ − c₁q_b = 0` with its Lagrange force: the driven joint tracks `q_a`, `v_a = c₁v_b`, the driver gets `Q_b = c₁·τ_a^ext` (τ_a^ext = drive spring/damping + Coulomb bound + gravity bias, all authored on the driven prim) and the driver's armature absorbs the inertial term `−c₁²I_a·q̈_b`; the driven joint's own PhysX drive is switched off at run time so its spring is not counted twice. A bare kinematic write leaves `Q_b = 0`: the driver loses the coupled part's weight and sags. |
| `<equality joint>` between two joints that both carry a position servo | `doorbench:coupling_mode = "servo"`, nothing authored | both PhysX drives *are* the MJCF servos (same `kp`, `kv`, `forcerange`, same target), so they move the two leaves together as the equality does. |
| `<equality weld>` leaf → world (mag lock, delayed egress, electric bolt, interlock) | `UsdPhysics.FixedJoint` `base → leaf`, `physics:excludeFromArticulation = true`, `breakForce = breakTorque = holding_force_N`, `physics:jointEnabled`, `doorbench:role = "env_release"` | MuJoCo activates / deactivates the weld with `d.eq_active` and breaks it when any constraint row exceeds the holding force (three force rows and three torque rows against the same number, `DoorEnv._lock_logic`), so PhysX gets the force on both channels. `excludeFromArticulation` keeps the articulation a tree; `mdp.release_env_lock` clears `jointEnabled`. |
| MuJoCo contact filter (same weld body, weld parent/child, `<contact><exclude>`) | `physxArticulation:enabledSelfCollisions = true` + `PhysxFilteredPairsAPI` for every filtered pair | PhysX's self-collision skips joint-adjacent links only. With self-collision off (the round-2 setting) a latch holding one moving link against another never touched: swing pairs latched into the inactive leaf, lift pins, drop bolts. |
| helical hinge: `<joint equality rise = c₁·hinge>` on a vertical riser slide | `door_rl.usda`: riser locked, `doorbench:rl["rise_coupling"] = {coeff_m_per_rad c₁, carried_mass_kg m, gravity_torque_Nm = −m·g·c₁}` | opening lifts the leaf by `c₁` m/rad, i.e. potential `m g c₁ q` → constant closing torque `−m g c₁` on the hinge (3.5 N·m on a 47 kg cold-room door, 1.2 N·m on a stall gravity hinge). `DoorMechanismAction` and the rl parity runner apply it as feed-forward effort; `door.usda` keeps the riser and its mimic joint (`doorbench:meta["rise_coupling"]` carries the same numbers for a runner that emulates the mimic kinematically, which loses this reaction). |

## 3. The environment

* **Scene.** Ground plane + dome light + `door` (`MultiUsdFileCfg` over the selected `door_rl.usda` files,
  round-robin so `num_envs ≥ len(doors)` shows every door; `replicate_physics=False`) + the agent + contact sensors on
  `Door/Articulation/leaf` and `.../operator` filtered by the agent's bodies. `env_spacing = 6 m` (walls are 5 m wide).
* **Door physics at run time.** `DoorMechanismAction` (a 0-dim action term, runs every physics step) restores the
  USD spring targets (Isaac Lab otherwise writes zeros as drive targets), couples the latch bolt to the operator
  (`bolt_target += scale·operator_q`), applies the `emulated` couplings bilaterally (`_apply_couplings`: the driven
  joint tracks `q_a = c₀ + c₁q_b` and the driver takes the reaction `c₁·τ_a^ext`), adds the asymmetric closer damping
  / backcheck and the rising-hinge gravity torque (`rise_coupling`, the same reaction for the one coupling whose
  driven joint has no slot) as feed-forward effort, and servos automatic doors open while the agent is within 1.8 m of
  the door plane (moving only the drive target when the servo is already the drive, `actuators[*].in_drive`).
  `DoorState` verifies at start-up that PhysX holds the authored Coulomb efforts and writes them through
  `write_joint_friction_coefficient_to_sim` if it does not, and adds each driver's
  `coupling_reflected_armature` to its armature and switches the drive of every emulated coupling's DRIVEN joint off
  (that joint's state is written each step and its spring / damping are reflected onto the driver analytically, so
  its own drive would double-count; the authored gains stay in the USD, which is what the validator compares).
* **Environment-released locks.** `DoorState.set_env_lock(env_ids, enabled)` toggles `physics:jointEnabled` on the
  breakable FixedJoint of a mag lock / delayed egress / electric bolt / interlock; `mdp.release_env_lock` is the
  event term for badge / REX / timer, and `mdp.reset_door` re-engages the lock for the next episode.
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

0. Three representations added after parity round 2 and checked statically only (`validate_usd_static.py`,
   `tests/test_usd_export_fixes.py`), never yet stepped in PhysX: the env-release loop `FixedJoint`
   (`excludeFromArticulation` between two articulation links, `breakForce` on both channels), self-collision with
   `PhysxFilteredPairsAPI` (does PhysX read the union of both directions of the relationship? the exporter authors
   one side per pair), and the bilateral coupling emulation (`write_joint_armature_to_sim` for the reflected
   inertia + the per-step reaction). If PhysX rejects the loop joint inside an articulation, the fallback is
   `--emulate-weld`; if it ignores the one-sided filtered pair, author both sides.
1. `PhysxMimicJointAPI` gearing units (rad vs deg) for the `door.usda` couplings that stayed mimics (rotational →
   rotational only now — the prismatic ones were being dropped silently and are emulated instead). (The joint
   friction and
   velocity-cap unknowns of the first runs are settled: per-axis efforts must sit on the `angular` / `linear`
   instance and `max_angular_velocity` is deg/s — see the parameter-mapping table above; the parity runner now reads
   both back and fails the structure check when PhysX disagrees.)
2. `MultiUsdFileCfg` + `articulation_root_prim_path="/Articulation"` resolution for referenced prims (fallback: drop
   the argument; Isaac Lab then searches the child with `ArticulationRootAPI`, which is unique).
3. Contact-sensor filter expressions with `{ENV_REGEX_NS}/Robot/.*` (G1); the hand uses a single body.
4. The 0-dim action term (`DoorMechanismAction`); if the action manager rejects it, move `apply_actions` into an
   interval event term (`mode="interval", interval_range_s=(0, 0)`), which runs once per policy step.
5. Isaac Lab / rsl-rl versions: the PPO cfgs use the v2.3.2 / rsl-rl-lib 3.1.2 field set (`obs_groups`,
   `actor_obs_normalization`; the runner-level `empirical_normalization` is deprecated).  v2.3.2 removed
   `isaaclab.utils.io.dump_pickle` (isaaclab 0.47.0) — `scripts/isaaclab/_common.py` carries the replacement.
