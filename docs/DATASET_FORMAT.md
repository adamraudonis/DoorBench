# Dataset format

```
assets/
  manifest.json                 index of all doors (see below)
  hardware/<key>.obj|.usdc      shared procedural hardware meshes (levers, knobs, bars, wheels, hinges ...)
  doors/<door_id>/
    spec.json                   full parametric specification + derived physics (auditable, with sources)
    model.json                  simulator-agnostic articulated model (bodies, joints, geoms, couplings, materials)
    door.xml                    MJCF, full fidelity (every mechanism body, mesh visuals, tendons, loop closures)
    door_simple.xml             MJCF, leaf + primary operator + latch bolt, primitive collision only
    door_minimal.xml            MJCF, leaf only (hinge friction/damping/closer spring)
    scene.xml                   MJCF wrapper for `python -m mujoco.viewer --mjcf scene.xml`
    door.urdf / door_simple.urdf / door_minimal.urdf
    door.usda                   UsdPhysics articulation, full fidelity (references ../../hardware/*.usdc)
    door_rl.usda                canonical 8-link / 7-joint articulation for Isaac Lab multi-door RL (docs/ISAAC_LAB.md)
    qa.json                     automated sign-off record
    thumb_*.jpg                 catalogue thumbnails (robot view, far side, isometric, handle detail, open)
```

## Coordinate conventions

* Z up, metres, kilograms, seconds, radians.
* The wall plane is `y = 0`; the opening is centred at `x = 0`; the floor is `z = 0`.
* The robot approaches from `-y` (site `approach_point` at `(0, -1.5, 0)`) and must reach `goal_point` at `(0, +1.5, 0)`.
  Site `door_plane_center` marks the opening.
* For hinged leaves `meta.u` is the sign of the leaf's local x direction (hinge → latch edge) and `meta.v` the swing
  direction sign (`+1`: opens toward `+y`, i.e. the robot pushes; `-1`: the robot pulls).
* **Positive joint values always mean "opening" or "actuating"** (door opens, lever pressed, bolt retracts, bar pushed).
* `meta.primary_joint` is the leaf joint to observe; `meta.operator_joint` is what the robot must actuate (may be null).

## Joint semantics

| role | examples | notes |
|---|---|---|
| `primary` | `leaf_hinge`, `leaf_slide`, `rotor_hinge`, `flap_hinge` | carries hinge friction (`frictionloss`), damping, closer spring (`stiffness` + `springref`), closer asymmetry in `damping_closing/opening` |
| `operator` | `leaf_handle_hinge`, `leaf_exit_device_slide`, `wheel_hinge` | return spring, backlash range when locked |
| `latch` | `leaf_latch_bolt_slide` | spring bolt; `0` = extended, `+` = retracted; one-sided tendon to the operator (MJCF) |
| `lock` | `leaf_deadbolt_slide`, `leaf_deadbolt_thumbturn_hinge`, `dog_k_hinge`, keypad keys, `rex_button_slide` | polynomial equality couplings |
| `mechanism` | closer arms, riser slides for helical hinges | driven, not robot-interactive |
| `secondary` | second leaf of a pair, fold panels, pet flap in a door | |

`spec.json → physics` holds every derived number with the formula and source used
(mass breakdown, hinge friction model, EN 1154 closer parameters, latch/lock parameters, compliance flags,
damage thresholds).

## Lock state

`spec.lock.engaged` and `spec.lock.robot_side_release` define the initial state.  When engaged without a robot-side
release, the operator joint range is limited to the lock backlash ("jiggle") and deadbolts are fixed.  The benchmark
environment (`doorbench.benchmark.DoorEnv`) implements release logic for keypads (button sequence), REX buttons,
card readers (`env.badge()`), delayed egress timers, maglock breakaway, elevator call buttons and turnstile credentials.

## Format-specific notes

* **MJCF** is the reference. Joint `ref` equals the authored geometry configuration; `qpos0` therefore is the spec's
  initial state. The spring latch uses a *one-sided* fixed tendon `bolt_q ≥ scale · operator_q`, so a bolt can also be
  pushed in by the strike lip when the door slams (re-latching works).
* **URDF** joints have `q = 0` at the spec's initial state (`doorbench:zero_offset` gives the MJCF offset).
  Springs, closers, ratchets, welds and loop closures are exported as `doorbench:*` extension elements; couplings use
  `<mimic>` (bilateral).
* **USD** (`door.usda`): default prim `/<door_id>`, static `Env`, `Articulation` with a fixed `base` link; `UsdPhysics`
  revolute/prismatic joints (q = 0 at the spec's initial state, `doorbench:zero_offset` = MJCF `ref`) with force drives
  carrying closer / latch / return springs (`stiffness`/`damping`/`targetPosition`), Coulomb friction efforts
  (`physxJointAxis:*:staticFrictionEffort`), `physxJoint:armature`, `PhysxMimicJointAPI` couplings and a JSON string
  `doorbench:couplings` on the root prim.  Mass properties are explicit (`MassAPI`).  `door_rl.usda` maps every door onto
  the same canonical articulation (see [ISAAC_LAB.md](ISAAC_LAB.md)); `assets/usd_validation.json` is the static
  validation report of both files for all doors.

## manifest.json

`doors[]` entries carry the catalogue fields: id, family, context, use case, task, difficulty (1–5), mass, leaf dims,
operator / latch / lock / closer / hinge ids, condition, swing/side, extras, tags, body/joint counts, QA sign-off,
thumbnail paths, file paths and a physics summary.
