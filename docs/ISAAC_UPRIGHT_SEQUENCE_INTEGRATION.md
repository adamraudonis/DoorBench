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
native runtime and independent contact checks. Isaac040 completed36s but failed sustained pad grasp and the independent
contact qualification (11 excluded patches). Its native prerequisite passed.
See [the final evidence](evidence/isaac-handlelead040-final-failure.json).


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
gates above. Twenty-six focused adapter/binding tests pass; actual040 final archive
extraction now passes at exactly36s after293 files were finally verified. All
69 measured positions/velocities, root13 and declared door positions/velocities
are present. [Extraction identity](evidence/isaac040-final-state-extraction.json).
This failed grasp endpoint is diagnostic only and cannot seed a qualified route.


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

## Imported collider inventory

The six importer warnings about default sphere types prompted a read-only audit
of the verified040 import. All974 source collider identities (body, geometry
name and type) match the USD:958 meshes,12 cylinders and4 boxes, with no sphere
colliders. The prepared XML has six default geometry declarations without type,
but every body geometry has an explicit type. This is consistent with warnings
about defaults; it does not prove their exact importer call site.

`scripts/dexterous/audit_imported_collision_inventory.py --mjcf PREPARED_XML
--usd IMPORTED_ROBOT_USDA --output NEW_JSON` repeats the check without starting
physics or modifying inputs. It detects wrong types and wrong identities, not
just matching counts. Three tests include replacing a cylinder with a sphere
and renaming its body geometry. This inventory does not validate mesh vertices,
cooked convex hull dimensions, transforms or task success. Frozen041 is unchanged;
its resulting import should receive the same check after verified collection.
[040 inventory](evidence/isaac040-collider-inventory.json).

## Qualified041 endpoint and coordinate mapping

041 passes all19 runtime checks and the complete18,000-interval independent
contact audit: zero excluded loaded contacts and zero failed final-hold samples.
This is one privileged standing partial opening, not a full sequence or repeatability
result. Its292 result files and17 hand-review files are verified locally.
[Qualification](evidence/isaac041-qualified-partial-opening.json).

`destination_planning_coordinates.planning_coordinates` maps the immutable
measured binding to fresh planner qpos/qvel arrays. It checks the original motor
and door identities, exactly one free root and complete scalar-joint coverage.
Root angular velocity is converted from world to body axes; linear velocity stays
in world axes. It never steps or writes an active plant. Fifteen mapping/extraction
tests pass, including a90-degree rotated-root case and rejection of altered state
and coordinate inventories.

On actual041, all72 scalar coordinates map, and hand poses reconstructed using
731 hash-verified source assets agree within1.081um/1.663urad. The original
source XML is retained; only referenced asset paths are relocated in a separate
planning copy. [Mapping evidence](evidence/isaac041-planning-coordinates.json).

The standalone archive records hand-body transforms but lacks attained foot and
torso body poses. Therefore this is not yet the complete measured-body admission
required by `admit_destination_return_kinematics`. Future standalone recordings
need same-epoch feet, torso, both palms and handle poses before that admission can
be demonstrated. Do not substitute reconstructed values for original measured
poses. The native transfer rebase CLI also expects native manifest/trajectory
inputs; a separate bound Isaac input path is still required.

## Same-epoch standing body recording

Standalone standing acquisition now records both ankle bodies, torso, both palms
and the handle after each physics step, alongside the existing joint/root state.
The configuration declares their exact order and world body-origin XYZ/WXYZ
convention; provenance includes the recorder source. Six float32 poses add about
3 MB uncompressed for a 36-second run. Shape, finite-value and quaternion checks
reject invalid records without silently normalizing measurements.

Sixteen focused recorder, coordinate-mapping and pipeline tests pass. This is
prospective instrumentation: frozen041 does not contain these measurements. A
new qualified physical recording is required before full-body planner admission.
No controller, motor limit, physics timestep or acceptance threshold changed.

The attained-state extraction CLI now includes `measured_bodies` when the archive
declares the new ordered body recording. It requires the exact bound state epoch,
complete array shape and valid measured quaternions. Legacy041 remains extractable
without fabricated body evidence. Sixteen recorder/extraction tests pass; this
still does not establish full-body kinematic admission on a live run.

## Measured-body repeat042 dispatched for preparation

Fresh source `64c79febf35c999bc44f30fcd6803a7a26b22b0928c82331cd0935bbe1b0405f`
contains the body recorder and extractor. The queued trial repeats041 control:
index proximal -0.02 rad, tendon +0.06 rad, handle lead 0.10 rad and the original
2 N middle preload. It must pass its native prerequisite and full Isaac contact
audits before its endpoint is eligible for continuation planning. Preparation
and an attached collector are running; no042 result is claimed here.

Failed037 raw records were privately archived, independently downloaded and
hash-verified before84 local raw files were removed (652,666,082 logical bytes).
Reports and visual diagnostics remain locally; qualified041 raw evidence stays
local. [Offload receipt](evidence/isaac037-verified-offload.json).

Both replacement guards were acknowledged on the owned A40 before the old
guards stopped; teardown is now September10 at21:42:27 UTC. The local preparation
passed the20GiB retention admission with72s plus128MiB reserved. The dispatcher
checks storage again after readiness. [Preparation](evidence/isaac-body-poses042-preparation.json).

Failed039 was also archived with independent download/hash verification before
removing84 local raw files, leaving additional space for042 collection. Its
reports and visual diagnostics remain. Any new037/039 raw reanalysis must first
restore the corresponding private archive. [039 offload](evidence/isaac039-verified-offload.json).

## Combined destination planner admission

`admit_destination_planner` now requires both the bound69-joint/root/door state
and same-epoch measured body poses before returning an unstepped planning state.
It preserves measured velocities and exposes the existing coordinate and2um/2urad
body checks together. Eighteen focused tests cover valid admission, stale or
missing bodies, changed joint data and wrong door identity. Legacy041 cannot
pass this combined admission because it lacks the required body recording.

The CLI below runs on a machine where the original robot assets resolve. It
checks bound robot XML and door USD hashes, records source-design identities
including referenced assets, and rechecks inputs after admission. Door MJCF/USD
collision cooking parity and physical contact qualification are separate.

```sh
PYTHONPATH=. python scripts/dexterous/audit_destination_planner.py \
  --robot ORIGINAL_ROBOT.xml --door ORIGINAL_DOOR.xml \
  --door-usd ORIGINAL_DOOR.usda --extracted MEASURED_STATE.json \
  --motors motor-contract.json --output NEW_ADMISSION.json
```

A pose mismatch produces a failed evidence receipt; missing or changed input
identities fail without a passing receipt. This source is newer than frozen042
and will be used as a separately identified read-only audit after collection.

The file-based admission command is also tested end to end on a generated
69-joint scene: original model/motor/state files produce a passing receipt,
then a1mm measured ankle displacement produces exit1 and a retained failure
receipt. Seven CLI/combined-admission tests pass. This synthetic boundary test
does not replace the pending measured042 audit.
