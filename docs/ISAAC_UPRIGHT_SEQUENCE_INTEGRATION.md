# Port the upright sequence to Isaac

Status: implementation inspection on 2026-09-10. This is not evidence of an
Isaac opening or traversal. Native passage003 finished and failed physical support/contact checks.
The complete objective remains in [the research plan](DEXTEROUS_PLAN.md).

## Actual integration gap

`scripts/isaac/run_standing_operation.py` invokes a 36-second native prerequisite
and then `scripts/dexterous/isaac_opening.py` for acquisition and partial opening.
Its command and acceptance report do not cover the upright transfer/return/
withdrawal/panel sequence. Extending the duration alone will not add those stages.

The native implementation wraps `DoorOperationTeacher` in
`StandingTransferTeacher`, `StandingReturnTeacher`, then
`StandingWithdrawalTeacher`. Their `force` signatures use the same time, root,
joint positions/velocities, handle/leaf poses, mechanism angles, hand loads and
grip qualification as the existing Isaac operation call, plus `left_panel_load`.
See `scripts/dexterous/probe_acquisition_operation.py` for the actual construction
and call order. The native wrappers generate capped motor commands; their
MuJoCo geometry models must remain unstepped when used with PhysX.

## Required implementation and evidence

1. Obtain a mechanically qualified Isaac acquisition/partial-opening state.
   Reconstruct and screen transfer targets from that measured state. Native
   source-bound plans must not be relabelled as Isaac plans or accepted by
   weakening initial-state checks.
2. Add a distinct upright-sequence path in `isaac_opening.py`, with validated
   route inputs and all source/geometry hashes. Instantiate the three wrappers
   in order. Preserve the existing standalone operation mode and identify the
   expanded scope explicitly in the output.
3. Feed measured left-surface load into the wrappers. The existing full-opening
   branch already consumes `surface['total_normal_load_N']`; use the same
   measured contact accounting contract, including frame/epoch checks. Do not
   substitute an IK estimate, target force, or geometry penetration for load.
   Preserve tangential hand-force compensation and the original motor limits.
4. Record whole-handle and whole-body contacts throughout transfer, return,
   withdrawal and wider opening. In particular, elbow/leaf contact is forbidden
   support. Dense planned clearance cannot replace actual PhysX contact checks.
   Keep latch release, opposed grip before release, final palm support and
   aperture hold as separate requirements.
5. Replan each subsequent source-bound segment from the actual attained Isaac
   state. Feed successful opening into actual-state hand stow and passage
   planning; then verify a single uninterrupted motor-driven episode with a
   stable finish. Component passes and initialized continuations do not qualify
   the complete sequence.
6. Only after qualification, run declared repeats/held-out scenarios and train
   the vision+tactile+proprioception actor. Teacher geometry/state access remains
   privileged and must never be reported as sensor-only success.

Before restarting the owned pod, rediscover its endpoint and arm fresh local
and remote teardown guards. Retain the prepared environment, immutable input
hashes, command lines, compact reports and restore receipts. Apply storage
admission before collecting full-rate evidence.

## Standalone force-accounting correction

The standalone operation branch previously passed only normal contact loads to the teacher, while native compensation and the longer Isaac branches included friction. It now uses the shared validated normal-plus-friction pair reducer for both hands. Portable source capture includes the measurement helper; reports explicitly identify this force convention.24 focused CPU tests pass, including known signed friction vectors, duplicate/truncated buffer rejection and launcher checks. This has not yet been tested on a live GPU and does not repair or reclassify historical runs. See [accounting evidence](evidence/isaac-standalone-friction-accounting.json).
