# Dexterous experiment ledger

All results on this page are development experiments. **No complete door-opening policy has been achieved.** The acceptance target remains in [the approved plan](DEXTEROUS_PLAN.md); platform/reproduction instructions are in [the migration guide](DEXTEROUS_REPRODUCTION.md).

## Body reaching, September 7, 2026

Same development seeds 10000–10029, 0.25 m target-offset curriculum, six-second horizon. A reach requires both forearm target errors below 8 cm, torso tilt below 12 degrees and pelvis height above 0.8 m for 0.5 seconds. A fall or numerical instability fails the trial. These criteria are only a body-skill gate, not the final opening/traversal protocol. Evaluations below ran on local macOS CPU.

| Controller | Training transitions | Upright reaches | Falls | Evidence |
|---|---:|---:|---:|---|
| Frozen upstream body skill | External pretrained checkpoint | 0/30 | 9/30 | [Trials](../results/dexterous/2026-09-07/upstream-body-reach.json) |
| Reach 001, standing reward weight 1 | 430,000 | 0/30 | 9/30 | [Trials](../results/dexterous/2026-09-07/reach-001-body-reach.json) |
| Reach 002, standing reward weight 4, intermediate | 820,000 cumulative | 1/30 | 12/30 | [Trials](../results/dexterous/2026-09-07/reach-002-intermediate.json) |
| Reach 002, later checkpoint | 2,120,000 cumulative | 0/30 | 4/30 | [Trials](../results/dexterous/2026-09-07/reach-002-late.json) |
| Reach 002, final checkpoint | 2,432,944 cumulative | 0/30 | 7/30 | [Trials](../results/dexterous/2026-09-07/reach-002-final.json) |
| Reach 003, continue after training success | 2,936,752 cumulative | 1/30 | 2/30 | [Trials](../results/dexterous/2026-09-07/reach-003-final.json) |

Reach 001 stopped around 432,000 transitions because a fall exhausted the native constraint arena. Its last checkpoint is retained; it did not complete the 500,000-transition budget. Increasing solver memory is an infrastructure repair, not a change to collisions or success thresholds. Reach 002 resumes that checkpoint with 128 MiB arena memory per environment and a stronger standing reward. The intermediate result does not establish improvement in reliability.

The later Reach 002 checkpoint reduced falls but still failed the upright-reaching gate. Final pelvis heights ranged from 0.713 to 0.794 m, with median 0.767 m. Inspection identified an incentive problem: successful episodes terminated while unsuccessful episodes could continue receiving positive dense reward. This is a plausible explanation for settling just below the 0.8 m threshold, not a proven causal result. The v3 configuration tests continued training episodes after reaching, with the original evaluation gate unchanged. A fall after reaching still fails the training episode. Historical v1/v2 configurations preserve their original termination behavior.

Reach 003 completed 503,808 additional transitions. Its final median pelvis height was 0.777 m and it passed only 1/30 trials, with two falls. The termination correction alone has not solved body control, and these small development samples do not establish a reliable improvement. Further posture/target curriculum work is required before connecting this body skill to approach and traversal. All three GPU-run checkpoints and logs are archived; the owned pod was terminated and its absence confirmed after copying them.

A short throughput profile at unchanged physics settings measured approximately 251, 467 and 781 transitions/sec for 4, 8 and 16 environments respectively. [Raw profile](../results/dexterous/2026-09-07/native-profile.json). This profile includes native CPU physics and actor inference, excludes rendering/PPO updates, and establishes the best of those tested counts only. It is not an Isaac Sim or maximum-GPU-capacity measurement.

## Grasp initialization and contact probe

The current grasp spike is specific to the right Shadow Hand and `db0055_swing_single`. It is not an arbitrary-robot or arbitrary-door adapter.

1. Fit finger-pad positions, pad directions and a palm target within joint limits. Seed optimization reserves 4% of each joint range and permits a bounded nearby base pose.
2. Penalize robot collisions explicitly. A pose with accurate fingertips but self-intersection is rejected. The inspected candidate has about 1.1 mm maximum pad-target error and no recorded robot penetrations deeper than 2 mm; this is kinematic evidence only.
3. Execute the pose in native physics with a free body and bounded motor position commands. No runtime pose writes, welds or external supporting forces are used. Starting already near the handle is explicitly non-qualifying for the task.
4. Initial constant targets load four fingers but lose thumb contact. Optimize thumb/finger preload using solved forces and opposing contact directions; retain failures and longer-horizon tests.

The short optimized probe established all five loaded digits on opposing sides for most of its 0.4-second window. Extending one selected command to two seconds retained opposition for only 34 of 100 control frames. That constant command did not provide a reliable grasp, and the lever did not complete its release travel. This motivated the tactile-feedback experiment below. Approach, full release, opening and traversal remain outstanding.

The subsequent tactile PPO experiment trained for 201,216 transitions on six local CPU environments. Development seeds 20000–20029 perturb the initialized hand joints by ±0.002 radians. Each trial requires one continuous second of loaded opposition within a three-second horizon, pelvis height above 0.8 m and torso tilt below 12 degrees.

| Controller | Initialized grasp successes | Falls | Evidence |
|---|---:|---:|---|
| Constant optimized preload | 0/30 | 0/30 | [Trials](../results/dexterous/2026-09-07/grasp-constant.json) |
| Tactile PPO, final checkpoint | 30/30 | 0/30 | [Trials](../results/dexterous/2026-09-07/grasp-001-final.json) |

An intermediate checkpoint at 160,032 transitions also passed 30/30 under this narrow validation. Close-up native-state replays were inspected at the beginning, middle and end of a trial, and a wide view of the final checkpoint was inspected. These results establish a local contact-hold skill only: one robot, one door, a supplied near-handle pose, small perturbations, no approach or complete lever release. Integration with approach and body movement remains unvalidated. The final policy uses hand proprioception and touch; it has not yet learned vision-based task execution.

Additional initialization stress tests of the unchanged final checkpoint:

| Uniform per-joint perturbation | Successes | Falls | Evidence |
|---|---:|---:|---|
| ±0.02 rad (about 1.15°) | 30/30 | 0/30 | [Trials](../results/dexterous/2026-09-07/grasp-001-noise-002.json) |
| ±0.1 rad (about 5.73°) | 19/30 | 1/30 | [Trials](../results/dexterous/2026-09-07/grasp-001-noise-010.json) |

These tests clip initialized joint positions to native joint limits. They are separate from the original ±0.002-radian validation, and do not change its result. The larger perturbation exposes limited acquisition tolerance and a body fall; the policy is not generally robust.

A separate three-second continuous-hold test within a five-second horizon, at the original ±0.002-radian perturbation, passed 30/30 with no falls. [Trials](../results/dexterous/2026-09-07/grasp-001-hold-3s.json). Reproduce using `evaluate_grasp.py --hold-steps 150 --horizon 250` with the same checkpoint and input files. This longer initialized hold still does not establish opening or traversal.

`scripts/dexterous/fit_grasp_seed.py`, `optimize_grasp_contact.py` and `probe_grasp_seed.py` save inputs and source manifests for subsequent experiments. Their output directories contain the exact initialized pose, preload, contact traces and native trajectories. Rendered close-ups are diagnostic views; the gold operator color changes visualization only.

## Isaac import and controller development, September 7–8

These are development attempts, not benchmark results. Each attempt uses one initialized H1/Shadow grasp on Door55. The final sensor-only policy and traversal task remain unsolved.

- The pinned Isaac 5.1 / Lab 2.3.2 environment starts successfully on L40S. A full automated readiness invocation passed on September 8 at 01:00:21 UTC; see [one-click setup](ISAAC_ONE_CLICK.md).
- Import attempts exposed nested joint-default errors, an incorrectly converted free joint, and concurrent mesh-conversion races. The corrected native/imported link comparison has micrometre-scale position error. Automated tests cover free-root and nested-axis preservation.
- Opening attempts 006–008 showed door movement, but their recording path inserted extra physics steps. They are retained as invalid timing diagnostics, excluded from any controller score. Later attempts enforce the 2 ms clock explicitly.
- The initial open-loop motor reference did not transfer reliably. Closed-loop task-space control is being tested with actual Isaac body/door state. This is a privileged teacher; MuJoCo is used for analytic FK/IK only, while PhysX alone advances the plant.
- Actuator transmission bounds are authored explicitly after import. Actual torque commands sent to PhysX are recorded alongside the native motor calculation. Hand forces come from the solved rigid-contact tensor view.
- The first correct-clock teacher could depress the handle and retract the latch but reached its arm limit before opening sufficiently. A lower reset and gravity-feedforward stance are being investigated; they are not yet validated successful policies.

The native opening probes and stance-controller prototypes in this development branch include unsuccessful experiments. They are opt-in diagnostic scripts, not default benchmark policies. Retain their source and failed trajectories when reproducing development decisions.
