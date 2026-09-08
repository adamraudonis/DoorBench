# Native walking, grasp acquisition and partial opening

One **55 s uninterrupted native episode** now walks 0.7 m to Door55, lowers,
prepares the right hand, grasps the handle, retracts the latch and holds the
door partly open. The exact executed run passes all sixteen declared gates.
Additional checks on its complete 500 Hz evidence confirm contact-free arm
preparation and no wrong-side or non-volar handle contact at any time.

This is a privileged teacher, not a vision/tactile actor, an Isaac result, full
door opening or traversal. There are no physical root, joint or foot pose writes
after the initial reset. No door motor is commanded. The hand and door move
through contact driven by the original capped robot motors.

| Actual event or measurement | Run 001 |
|---|---:|
| Initial walking distance | 0.7 m |
| Lowered, quiet body handoff | 13.502 s |
| Contact-free hand readiness complete | 23.504 s |
| Qualified grasp starts lever operation | 31.294 s |
| Actual lever/bolt release allows leaf goal | 36.294 s |
| Final leaf angle | 0.084583 rad / 4.85° |
| Maximum lever angle | 0.842512 rad |
| Maximum latch retraction | 12.202 mm |
| Maximum joint-limit excursion | 6.353 mrad |
| Maximum Shadow loopback excursion | 0.907 mrad |
| Maximum non-foot penetration | 0.526 mm |
| Maximum torso tilt, including walking | 2.519° |
| Stance solver failures | 0 |

The final half-second retains all five qualified pads. Five isolated 2 ms
samples during lever/opening motion briefly unload a digit; none loads an
invalid contact patch. The final FF/MF/RF/LF/TH loads are
1.496/1.582/1.936/2.371/10.670 N. Six body views across the episode and nine
close hand views were personally inspected. The thumb opposes the four fingers,
and the torso stays upright.

## Connecting the physical components

The [approach/lowering controller](DEXTEROUS_APPROACH_LOWERING.md) uses the
pinned Unitree walking actor, then a landed-foot stance QP. That same physical
stance controller continues through hand preparation and operation, preserving
its real landed foot frames. Its numerical iteration budget is 100000 with the
same constraints and 1e-4 tolerances; no constraint is inserted into the plant.

The original approach request's **2° heading target remains missed**: the low
stance reaches a 12.212° heading error. That original component failure is
retained. The continuous controller uses the actual reached root and legs for
the arm handoff, so grasp success does not depend on claiming that heading
target was achieved. The reached body is about 13 mm from the original target
in horizontal position.

Naive neutral-arm interpolation brushed the slab, and a compact-finger route
still grazed the handle with the little finger's middle link. The frozen
[readiness reference](../configs/dexterous/door55-readiness-v2/reference.json)
first compacts the fingers and positions the thumb, raises the arm outward,
then reaches above the handle and unfolds to the existing pre-curl start.
Its 501 samples were checked for contact. The raised waypoint is a bounded IK
pose, not a claim of exact intermediate orientation tracking. Final static
pre-grasp residual on the source landed body is 0.304 mm / 1.03°.

At runtime, planning copies receive the actual root/leg state and actual first
joint goal. The committed driver re-screens this route at the actual landed
posture before sending motor commands. This read-only screen was added after
run 001 and separately passed against its saved actual preparation reference.
All executed preparation steps were already contact-free. After readiness,
the unchanged [acquisition route](DOOR55_PRE_CURL_ACQUISITION.md) and
[operation wrapper](DOOR55_CONTINUOUS_OPERATION.md) follow the measured hand
and handle. The physical state is never reinitialized at a phase boundary.

All phases share one explicit-force adapter normalized **once at reset**.
Walking servo targets are converted each 2 ms through the original affine
gain/bias and original control limits before force clipping. The landed QP
reads the normalized model and emits force units directly. Acquisition emits
the same original capped forces used in its separate qualification. This
changes the controller interface without changing force limits, contact,
mechanics or adapter mode during the episode.

## Reproduce

Use the corrected V2 robot and matching motor contract from the mechanical
setup, plus the pinned Unitree H1 checkpoint:

```bash
python scripts/dexterous/probe_walk_grasp.py \
  --robot out/dexterous/robot/h1-shadow-loopback-v2.xml \
  --door out/dexterous/assets/doors/db0055_swing_single \
  --reference configs/dexterous/door55-precurl-v2/reference.json \
  --preparation configs/dexterous/door55-readiness-v2/reference.json \
  --motors out/dexterous/import-v2/h1-import.motors.json \
  --checkpoint out/locomotion/download-check/deploy/pre_train/h1/motion.pt \
  --output out/dexterous/walk-grasp-001 --seconds 55
```

The implementation requires `ApproachLoweringController` and its optional
solver settings from commit `8cdfd93c9`. Frozen readiness numeric values match
the tested source; only irrelevant inherited metadata was removed. Run 001's
exact driver remains in its evidence alongside the captured imported source
tree, motor contract, original and actual readiness references, 500 Hz audit,
50 Hz states and exact terminal state. The committed driver additionally makes
the post-run preparation/anatomy checks explicit and rejects invalid durations.

Full evidence: `/tmp/doorbench-shadow-loopback/walk-grasp-001/`, copied to
`~/Desktop/Projects/DoorBench-runs/2026-09-08-shadow-loopback/`.
The [compact receipt](evidence/door55-continuous-walk-grasp-2026-09-08.json)
records exact hashes, limitations and all additional checks. Failed preparation
screens remain in the same archive's `arm-preparation/` directory.

For bimanual continuation, keep the actual reached stance and handle grip
active until measured left-palm contact supports the leaf. The final right
elbow is 2.45661 rad with a 2.61 rad limit: only 0.1534 rad remains, so a simple
straight-back wrist withdrawal is not assumed feasible. Left-hand transfer,
right release, full opening and traversal require their own continuous tests.

## Portable full-sequence interface

[`FullSequenceTeacher`](../doorbench/dexterous/full_sequence_teacher.py) now
connects the measured-state body teacher, readiness reference, acquisition and
operation behind one force-only API. A separate 55 s native run of this
interface passes **all nineteen gates**. Final leaf travel is 0.084899 rad,
stance solver failures are zero, preparation stays contact-free, and no handle
contact patch is anatomically invalid. Nine isolated operation samples unload a
digit; the final half-second qualifies all five pads. The measured low-stance
heading error is 13.119° and is retained in its
[separate receipt](evidence/door55-full-sequence-teacher-2026-09-08.json).

```python
teacher = FullSequenceTeacher(
    robot_xml, motor_contract, acquisition_reference, readiness_reference,
    reset, unitree_checkpoint, joint_geometry, door_xml=door_xml,
    operation_options={},
)
motor_forces, info = teacher.force(
    time_s, root13, measured_joint_positions, measured_joint_velocities,
    measured_foot_loads, measured_handle_pose, measured_leaf_pose,
    {"operator": lever_rad, "leaf": leaf_rad, "latch": bolt_m},
    measured_world_hand_body_forces,
    grasp_qualified=actual_strict_pad_audit,
    hand_contact_count=actual_right_hand_contact_count,
)
```

`reset` contains `initial_root` (world xyz+wxyz), named `joints`, named original
servo `motor_targets`, `goal_xy` and `goal_yaw_rad`. It describes the active
simulator's initial state; constructing the teacher does not set that simulator.
`root13` contains world xyz+wxyz, world linear velocity, and world angular
velocity. Foot loads are measured world-up support forces `[left, right]`.
All poses, mechanism units and joint geometry follow the
[operation interface](DOOR55_CONTINUOUS_OPERATION.md). Call once per active
2 ms step. The caller still owns the actual physics and all safety audits.

The body controller retains its measured landed foot frames through arm
loading. Measured hand-body forces contribute to its analytic stance equations
at body origins, matching the existing acquisition approximation; these are
never externally applied robot forces. This loaded-body approximation was
tested in the complete portable native run. The MuJoCo calculators are never
stepped. A separate scene compiled from the same `door.xml` and robot XML checks
readiness geometry at the actual measured root and mechanism angles. It rejects
an asset-frame mismatch or any right-hand contact in the readiness samples;
it does not assume that an old canonical root was reached.

`operation_options` passes optional settings to `DoorOperationTeacher` without
changing its defaults. This qualified run used an empty options dictionary.
Any later compliance/timing options need their own physical qualification.
If `info["blocked_reason"]` is non-null, the readiness screen failed and no arm
preparation is started; the body controller continues to hold its actual stance.

Add `--portable-sequence` to the reproduction command above. This requires
`ApproachBodyTeacher` with optional measured hand loads, commit `55016fc9d` on
the body adapter's `2e98e5008`/`8cdfd93c9` history. Its full evidence is
`/tmp/doorbench-shadow-loopback/walk-grasp-portable-001/`. The direct committed
driver was also rerun with the frozen readiness configuration as
`walk-grasp-002`; all nineteen gates pass with the same final state as run 001.
Readiness unit tests reject a 0.5 mm accidental precontact, which would fit
under the older generic 3 mm penetration limit, and a mismatched asset frame.
