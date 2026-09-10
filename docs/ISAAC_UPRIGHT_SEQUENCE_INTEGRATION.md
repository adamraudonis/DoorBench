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

The standalone operation branch previously passed only normal contact loads to the teacher, while native compensation and the longer Isaac branches included friction. It now uses the shared validated normal-plus-friction pair reducer for both hands. Portable source capture includes the measurement helper; reports explicitly identify this force convention.24 focused CPU tests pass, including known signed friction vectors, duplicate/truncated buffer rejection and launcher checks. Live friction037 completed36s with correct raw force accounting but failed sustained fingertip grasp; it does not qualify partial opening or repair historical runs. See [accounting evidence](evidence/isaac-standalone-friction-accounting.json).


## Measured continuation velocities

Standalone acquisition now records `joint_velocity` beside `joints` at every
post-step archive epoch, in `configuration.json` robot joint order. The values
come directly from Isaac articulation state; no finite differences or zero
substitution. Continuous sequence recording retains its existing values. This
adds about5MB uncompressed per36s/69-joint float32 recording. Older standalone
archives without this field must not be treated as complete continuation states.
The recording change does not alter motor commands, physics or acceptance gates;
verify the field and epoch count in the next actual Isaac archive before relying
on it for source-state planning.

The root-state metadata now uses the same mode predicate as the recorder:
continuous, calibrated locomotion and standing acquisition use actor-origin
world velocity. Other modes preserve the legacy COM-velocity convention.
Previously standing acquisition was mislabeled as COM velocity in configuration;
its numeric recording was actor-origin. This correction applies to future source
snapshots only. Frozen tangent038 predates it and the joint-velocity addition;
do not use that archive as a complete continuation state or rewrite its evidence.

## Bounded handle reference experiment

Isaac tangent039 failed: the first excluded index-middle patch appeared at
17.388s, with the nearest recorded palm reference about0.173rad ahead of the
actual handle. Close-ups confirm contact migration and final grasp loss.
`--operation-operator-lead-limit-rad` now optionally clips the palm's operator
reference around the measured handle angle, after compliance and follow blending.
The default remains unchanged. The initial native test uses0.06rad with the same
tangent feedback, motor limits, distal-pad contract and actual release thresholds.
This is privileged controller feedback; it is neither a direct door command nor
a sensor-only policy. Requested and clipped angles are recorded separately.
Unit tests verify both lead and lag bounds and that an unretracted bolt cannot
be bypassed. The0.06rad native candidate stalled; the0.10rad candidate passed
native runtime and independent contact checks. Isaac040 is testing the latter;
its native prerequisite also passed. No Isaac result is yet established.


## Extracting a measured standalone state

`scripts/dexterous/extract_isaac_attained_state.py --run RUN_DIRECTORY --time-s EXACT_RECORDED_TIME --output NEW_JSON`
exports the complete measured root13, all69 joint positions and velocities, and
all declared door coordinates and velocities. It reads the coordinate order and
explicit root convention from the same run configuration, binds the original
motor/door identities, and hashes the input files before and after extraction.
The output contains a `binding` accepted by the destination-state identity API.
It rejects missing velocities, nonfinite or differently sized arrays, ambiguous
clocks, interpolation and legacy COM-velocity metadata. It does not reconstruct
missing values. Extraction is read-only and existing outputs cannot be overwritten.

This adapter does not qualify a failed grasp, prove imported FK equivalence, or
restore a running simulator. Only use a mechanically qualified attained state
for continuation planning, followed by the independent geometry and contact
gates above. Twenty-six focused adapter/binding tests pass; actual040 archive
extraction remains pending completion and verified collection.


## Index-posture candidate after handle-lead040

The measured040 first excluded index-middle contact occurs at16.36s, just after
lever return. A source-verified static sweep suggested proximal offset−0.02rad
and summed FFJ1/2 tendon offset+0.06rad. Native index-posture004 now passes all19
runtime checks and both independent contact audits over36s, ending at0.077877rad.
It is only a qualified native partial-opening candidate.

The coordinator and Isaac CLI now accept `--operation-index-proximal-offset-rad`
and `--operation-index-tendon-offset-rad`, defaulting to zero. They enforce the
teacher's existing0.1/0.12rad bounds and forward the same values to both backends.
Isaac permits nonzero offsets only for standalone operation. The existing
one-second teacher ramp and original motor/contact gates remain unchanged.
Forty-seven focused pipeline, launcher and teacher tests pass. A GPU comparison
requires a fresh immutable source snapshot; do not edit frozen040 or claim these
options were present in its command.
