# H1 walking development

The official **H1**, not G1, actor from Unitree RL Gym moves DoorBench's unchanged full H1/dual-Shadow robot under native motor forces. Forward walking and a phase-adapted stop/restart are working in CPU MuJoCo development trials. **The strict seeded stop protocol passes 30/40, with all ten held-raised-arm starts failing; a [physical Door55 waypoint approach](DEXTEROUS_DOOR_APPROACH.md) has separate 9/9 native evidence, while doorway clearance remains open. A [continuous lowered-stance primitive](DEXTEROUS_LOCOMOTION_TRANSITION.md) now has separate native development evidence; Isaac results are reported by the integration work package.** This does not complete the [full-sequence plan](DEXTEROUS_NEXT_STEPS.md).

## Measured evidence

September 8, 2026 UTC; each row is one 12-second development rollout on an unobstructed plane, with a constant 0.4 m/s body-forward command. These are not randomized robustness or door benchmark scores.

| Reset pose | Distance | Maximum torso tilt | Maximum joint-limit penetration | Result |
|---|---:|---:|---:|---|
| Near-straight walking posture | 4.022 m | 1.99° | 2.74 mrad | Forward walking checks passed |
| Original nominal crouch | 4.033 m | 3.76° | 2.58 mrad | Forward walking checks passed |
| Deep opening-reference body and hand pose | 4.607 m | 3.07° | 2.07 mrad | Forward walking checks passed |

Both feet lifted and alternated contact. All 61 original motor force caps were respected; no nonfoot ground collision or self collision occurred. Masses, inertias, joint limits, damping, friction, collision masks and native motor parameters were checked unchanged after each rollout. No root writes occur after reset, and no external wrench, foot anchor, weld or root support is applied.

The opening-pose trial starts with that pose on an empty plane, with root yaw zero and collision soles placed on the floor at reset. It is **not** a continuous transition from an actual door interaction. Its raised arm and hand are held in their initial posture, so passage clearance still needs separate verification.

## Phase-adapted stopping: 30/40 strict development trials

The original actor continues stepping at a zero velocity request because its gait-phase channels continue oscillating. DoorBench can fade the amplitude of those two channels from one to zero over one second after the command becomes zero, then restore the phase when movement is requested. No weights or physical parameters change. This is an empirically tested input adaptation outside the official deployment recipe, not a newly trained policy or a general guarantee.

The [frozen protocol](../configs/dexterous/h1-locomotion-development.json) uses ten seeds per group with independent uniform ±0.005 rad initial leg-angle noise. Alongside the physical checks, the final second must have root speed below 0.02 m/s, excursion below 1 cm, and at least 30 N on each foot. These narrow initial-state variations are not broad environment robustness or a door-coverage score.

| Sequence | Full checks | Worst peak tilt | Worst final-second root speed |
|---|---:|---:|---:|
| Walk → stop | 10/10 | 2.58° | 0.00276 m/s |
| Walk with a turn → stop | 10/10 | 2.54° | 0.01263 m/s |
| Walk → stop → restart → stop | 10/10 | 3.74° | 0.01406 m/s |
| Deep opening pose, arm held raised → walk → stop | **0/10** | 3.15° | 0.02686 m/s |

All 40 complete their 14- or 16-second run without falls, self collisions, nonfoot ground contacts, external forces, or changed plant parameters. The raised-arm group fails the stricter quiet-speed and 1 cm excursion criteria and stays in the denominator. A separate standard-arm 60-second test stayed upright through approximately 50 seconds of hold and ended below 0.000035 m/s. The analogous 60-second raised-arm test **fell at 41.86 seconds**, so waiting longer does not solve the asymmetric-posture problem. Natural arm retraction or a posture-aware controller remains necessary.

[Machine-readable seeded report](../results/dexterous/2026-09-08/h1-locomotion-phase-stop.json). The uncut replay renderer uses diagnostic body/hand colors and fixed 1 m ground marks; numerical and visual evidence remain tied to the same recorded physics timestamps.

## Failures retained

The 14-second walk/stop test walked successfully but failed its quiet-stop check: the official actor continues alternating its feet at zero velocity command, drifts slowly and reaches 0.214 m/s instantaneous root speed in the final second. It remains upright (<2.58° tilt), but is not a satisfactory stationary manipulation posture.

Hard-constraint, softened-constraint and pose-IK stance handoffs were tested. They either did not solve at independently landed foot frames or subsequently fell. `--settle-stance` retains the last unsuccessful pose-IK experiment and is **not a recommended controller**. Attempts remain in the local experiment archive with source snapshots and generated traces/outcomes; the first QP prototype failed during report serialization and retains only its last diagnostic state and error record. Earlier hand-built alternating-support QP trials also failed. The existing HumanoidBench two-hand reaching checkpoint moved approximately 0.95 m toward a distant target and then fell; it is not treated as a walking policy.

## Reproduce

Use the same Python environment as the [dexterous setup](DEXTEROUS_REPRODUCTION.md), with MuJoCo, NumPy, SciPy, Torch and OSQP. The robot must be generated by `setup_robot.py`; assets are not committed.

```bash
python scripts/dexterous/setup_h1_walking.py --output out/h1-walking-upstream
python scripts/dexterous/probe_locomotion.py \
  --robot out/dexterous/robot/h1-shadow.xml \
  --checkpoint out/h1-walking-upstream/deploy/pre_train/h1/motion.pt \
  --output out/h1-forward-001 --mode forward --seconds 12
python scripts/dexterous/render_locomotion.py \
  --robot out/dexterous/robot/h1-shadow.xml --trial out/h1-forward-001
```

For phase-adapted stopping, use `--mode walk-stop --phase-stop --seconds 14`. Execute all 40 frozen development trials with:

```bash
python scripts/dexterous/evaluate_locomotion.py \
  --robot out/dexterous/robot/h1-shadow.xml \
  --checkpoint out/h1-walking-upstream/deploy/pre_train/h1/motion.pt \
  --output out/h1-locomotion-evaluation-001
```

The current protocol returns nonzero because the raised-arm group fails. It does not omit those trials or relax their thresholds.

Select `--initial-pose nominal`, or `--initial-pose opening --reference configs/dexterous/isaac-door55-reference.json`, for the other starts. `--mode walk-stop`, `turn-stop` and `restart` request additional development tests; their failures stay in the report. Every invocation requires a fresh output directory, stores the exact source and dependency manifest, and returns a nonzero exit code when a required check fails. Forward mode deliberately does not claim that stopping was tested.

The actor file, configuration, license and source are fetched from the pinned official revision. Its SHA-256 is checked before loading. The TorchScript actor is outside git. See the [retained BSD-3-Clause license](licenses/UNITREE_RL_GYM_LICENSE.txt).

## Exact interface and model differences

Source: [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym/tree/276801e46c5d433564f24658bac64f254b7d2d4b), official `deploy/pre_train/h1/motion.pt`, SHA-256 `44a0fbceb81f3877833ae9a398d039bea1759cb0d3c8188181013885f70589eb`.

`H1WalkingPolicy` receives 41 values in this order: pelvis-frame gyro × 0.25 (3), projected gravity (3), desired body velocity × [2,2,0.25] (3), leg angles minus defaults (10), leg velocities × 0.05 (10), previous action (10), sine/cosine of a 0.8-second gait phase (2). It outputs 10 leg targets: defaults plus 0.25 × action. It runs at 50 Hz.

Leg order is left then right: hip yaw, hip roll, hip pitch, knee, ankle. Defaults are `[0,0,-0.1,0.3,-0.2]` per leg. The torque law uses `kp=[150,150,150,200,40]`, `kd=[2,2,2,4,2]` per leg, evaluated each 2 ms physics step. `NativeH1MotorAdapter` inverts the existing affine servo equation each step to realize those torques while retaining its original control and force limits. Isaac can apply `H1WalkingPolicy.torques(...)` through its audited bounded torque adapter; native scalar leg axes and ordering must be preserved.

| Property | Official deployment example | DoorBench tested plant |
|---|---|---|
| Mass | 51.649896 kg | 53.239896 kg |
| Motors | 10, upper body rigidly combined | 61, torso/arms/two Shadow hands articulated |
| Leg axes/order | Official named H1 joints | Same axes/order, resolved by name |
| Hip-pitch limits | [-1.57, 1.57] rad | Existing [-3.14, 2.53] rad |
| Leg joint damping | 0.001 | Existing 1.0 |
| Leg armature | 0.01 | Existing 0.1 |

These differences were **not** erased to obtain a result. Compatibility is empirical, not an exact-model parity claim. Run `audit_h1_locomotion.py --robot ... --upstream <pinned full Unitree clone> --output ...` for the detailed joint/motor comparison.

The actor itself is proprioceptive: it sees neither world position nor linear velocity, vision, touch, object identity, or door state. The plane test supplies a predefined velocity schedule. A future waypoint planner or stationary stance teacher can still be privileged; this low-level actor does not establish sensor-only door opening.

## Immediate continuation

1. Integrate the [continuous lowered-stance primitive](DEXTEROUS_LOCOMOTION_TRANSITION.md) with the real handle controller, including natural arm retraction where declared. Preserve the ten raised-arm failures and the long-hold fall.
2. Preserve original force caps and correct 20 ms/2 ms clocks in Isaac; validate any additional stance/reaching handoffs there separately.
3. Integrate the grasp/release controller, add actual whole-body clearance checks and score a continuous approach/open/traverse sequence.
