# From initialized opening to complete sensor-driven traversal

Execution plan approved by the owner on September 8, 2026 (UTC). This is the next work package under [the full research plan](DEXTEROUS_PLAN.md), not a replacement for its catalogue coverage targets.

## Starting point

The H1/dual-Shadow teacher has opened `db0055_swing_single` in live Isaac Sim in four fixed-start repetitions. Two fresh prepared environments reached 95 degrees. These are initialized, privileged demonstrations: the hand starts near the handle, the teacher uses simulator geometry/state, and the robot does not approach or traverse. They are not a robustness score or a vision/tactile policy. See [the evidence and reproduction commands](ISAAC_HANDLE_DEMO.md).

The previous native H1 reaching experiment passed only 1/30 development trials with two falls. The G1 walking checkpoint belongs to a different embodiment. Stable reaching and locomotion are immediate risks, not completed prerequisites.

## Ordered milestones

| Stage | Implementation | Acceptance evidence |
|---|---|---|
| 1. Complete one physical task | Start with hands away from the handle; approach, acquire an opposed grasp, operate the lever, open, traverse, release safely and settle. Coordinate reaching, balance and walking on the audited H1 model. Release timing depends on the door rather than a universal fixed order. | An uncut Isaac rollout, wide and hand close-up views, and independent checks of mechanism operation, motor limits, collisions, balance and whole-body passage. No runtime root pose writes, supports, welds or direct door torques. |
| 2. Establish repeatability | Randomize declared starting positions, handle locations, friction and door resistance. Add recovery from missed grasps, slips and closing doors. | Proposed development gate: at least 95/100 complete trials. Freeze feasible ranges, seeds, timeouts and numerical success checks before scoring. Keep every failure. This is a one-door development gate, not catalogue generalization. |
| 3. Remove privilege early | Build finite local tactile observations, camera streams, proprioception and recurrent student training. Begin on stage-1 trajectories rather than waiting for all-door teacher coverage. | Audit the entire runtime stack: no exact handle/door transforms, door angles, object IDs, contact identity labels or hidden oracle routing. Teacher/critic/reward may use state during training only. Score sensor-only performance separately. |
| 4. Learn from student errors | Imitate the teacher, collect teacher corrections on states visited by the student, then fine-tune with RL. Train recovery, observation delays and contact/visual uncertainty. | Matched-seed teacher/student reports, sensor ablations and failure videos. No replacement of unsuccessful rollouts by attractive selected examples. |
| 5. Expand by mechanism | Push/pull, opposite hinge sides, knobs, closers, sliding, double doors and unlocking procedures, followed by broader coverage. | Per-family opening and traversal scores; CAD/mechanism-lineage holdouts. Human-feasible robot failures remain in the denominator. Pet doors remain a downloadable supplement. |

The final proposed targets remain at least 95% reliable coverage of the independently frozen human-openable set, 9/10 seeded successes per required scenario, at least 90% coverage in each populated family, and separately reported traversal and unseen-door results. These are ambitious research targets, not promised outcomes.

## Why this approach

Separating physical motor capability from partial-observation learning makes failures diagnosable and reduces the initial exploration problem. Useful low-level walking/reaching skills are supported by [HumanoidBench](https://arxiv.org/abs/2403.10506). [DoorMan](https://arxiv.org/abs/2512.01061) provides a relevant privileged-teacher to visual-policy strategy, while [DAgger](https://proceedings.mlr.press/v15/ross11a.html) motivates correcting student-visited states. Their results do not establish success for our robot, hands or catalogue.

## Portability, resources and progress

Use the [one-click Isaac environment](ISAAC_ONE_CLICK.md); preserve source revision, dependency versions, robot adapter, task configuration, inputs, checkpoint, seeds and evaluation commands. Profile state-only, RGB and RGB+tactile throughput before large allocations. Keep bounded owned-pod allocations, teardown guards and off-pod evidence. Do not interfere with another agent's node.

Each implementation milestone must update the [execution ledger](DEXTEROUS_EXPERIMENTS.md), with measured results and limitations. GPU runs must be registered in Run Center with their stage, heartbeat, cost and evidence paths. Hand inspection must use close-ups as well as wide footage. A geometric fit is a candidate, not a physical success; an initialized opening is not traversal.

## Work log

- September 8, 2026 UTC: plan recorded before execution. First engineering task is moving from the verified near-handle pose to contact-free reaching and acquisition, without changing the known-good opening baseline. Full-sequence evaluation and the sensor audit follow. No new full-sequence or sensor-only result exists yet.
