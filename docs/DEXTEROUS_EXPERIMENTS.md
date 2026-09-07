# Dexterous experiment ledger

All results on this page are development experiments. **No complete door-opening policy has been achieved.** The acceptance target remains in [the approved plan](DEXTEROUS_PLAN.md); platform/reproduction instructions are in [the migration guide](DEXTEROUS_REPRODUCTION.md).

## Body reaching, September 7, 2026

Same development seeds 10000–10029, 0.25 m target-offset curriculum, six-second horizon. A reach requires both forearm target errors below 8 cm, torso tilt below 12 degrees and pelvis height above 0.8 m for 0.5 seconds. A fall or numerical instability fails the trial. These criteria are only a body-skill gate, not the final opening/traversal protocol. Evaluations below ran on local macOS CPU.

| Controller | Training transitions | Upright reaches | Falls | Evidence |
|---|---:|---:|---:|---|
| Frozen upstream body skill | External pretrained checkpoint | 0/30 | 9/30 | [Trials](../results/dexterous/2026-09-07/upstream-body-reach.json) |
| Reach 001, standing reward weight 1 | 430,000 | 0/30 | 9/30 | [Trials](../results/dexterous/2026-09-07/reach-001-body-reach.json) |
| Reach 002, standing reward weight 4, intermediate | 820,000 cumulative | 1/30 | 12/30 | [Trials](../results/dexterous/2026-09-07/reach-002-intermediate.json) |

Reach 001 stopped around 432,000 transitions because a fall exhausted the native constraint arena. Its last checkpoint is retained; it did not complete the 500,000-transition budget. Increasing solver memory is an infrastructure repair, not a change to collisions or success thresholds. Reach 002 resumes that checkpoint with 128 MiB arena memory per environment and a stronger standing reward. The intermediate result does not establish improvement in reliability.

A short throughput profile at unchanged physics settings measured approximately 251, 467 and 781 transitions/sec for 4, 8 and 16 environments respectively. [Raw profile](../results/dexterous/2026-09-07/native-profile.json). This profile includes native CPU physics and actor inference, excludes rendering/PPO updates, and establishes the best of those tested counts only. It is not an Isaac Sim or maximum-GPU-capacity measurement.

## Grasp initialization and contact probe

The current grasp spike is specific to the right Shadow Hand and `db0055_swing_single`. It is not an arbitrary-robot or arbitrary-door adapter.

1. Fit finger-pad positions, pad directions and a palm target within joint limits. Seed optimization reserves 4% of each joint range and permits a bounded nearby base pose.
2. Penalize robot collisions explicitly. A pose with accurate fingertips but self-intersection is rejected. The inspected candidate has about 1.1 mm maximum pad-target error and no recorded robot penetrations deeper than 2 mm; this is kinematic evidence only.
3. Execute the pose in native physics with a free body and bounded motor position commands. No runtime pose writes, welds or external supporting forces are used. Starting already near the handle is explicitly non-qualifying for the task.
4. Initial constant targets load four fingers but lose thumb contact. Optimize thumb/finger preload using solved forces and opposing contact directions; retain failures and longer-horizon tests.

The short optimized probe established all five loaded digits on opposing sides for most of its 0.4-second window. Extending one selected command to two seconds retained opposition for only 34 of 100 control frames. It is therefore **not yet a reliable grasp**, and the lever did not complete its release travel. Tactile correction, approach, full release, opening and traversal remain outstanding.

`scripts/dexterous/fit_grasp_seed.py`, `optimize_grasp_contact.py` and `probe_grasp_seed.py` save inputs and source manifests for subsequent experiments. Their output directories contain the exact initialized pose, preload, contact traces and native trajectories. Rendered close-ups are diagnostic views; the gold operator color changes visualization only.
