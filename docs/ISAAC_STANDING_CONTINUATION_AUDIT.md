# Actual standing-continuation observation accounting

`isaac_standing_continuation_audit.py` independently checks the completed capture
added to the initialized-standing withdrawal producer. It reads saved files only;
it neither initializes a simulator nor resets state or submits motor commands.
The receipt and detached admission always have `authorized_stages: 0` and
`physical_task_qualification: false`. A successful observation audit does not
qualify withdrawal, walking, approach, or passage.

```powershell
python scripts/dexterous/audit_isaac_standing_continuation.py `
  --trial out/NEW-COMPLETED-ISAAC-WITHDRAWAL/trial `
  --output out/NEW-COMPLETED-ISAAC-WITHDRAWAL/independent-continuation-audit.json
```

This command preserves existing receipts and failed evidence. Malformed or
incomplete inputs produce a failed receipt and exit 1; a complete correctly
recorded invalid motor submission also fails admission. Checkpoints cannot
substitute for the finalized export. Historical runs without these new streams
cannot pass this audit, including the successful transfer005 prerequisite.

## What is checked

The decoder streams the NPZ fields in blocks and the two compressed JSON arrays
one bounded record at a time. Memory use does not grow with episode duration.
Every interval must appear exactly once from 0.002 s through the declared final
epoch. The actual body origins, leaf pose, foot-load records, transfer surface
record, and contact interval share that epoch. Original body/joint ordering is
explicitly required. Missing, extra, truncated, nonfinite, duplicate-key, or
misaligned records are rejected.

The saved USD scene determines all enabled collision participants, including
instance proxies, nearest rigid-body parents, and static shapes. Every robot
participant and every scene counterpart must appear in the recorded sensor/filter
inventory. All used file-backed USD layers are hashed. Original source Python
hashes resolve to immutable captured `source-*.py` files, while this independent
auditor and its helpers bind their own current source hashes. Input hashes are
checked again after streaming.

Normal and friction slots are independent inventories. The auditor decodes the
recorded ordered pair/slice lists without calling the live packer, checks their
capacity, bounds and disjointness, and recomputes foot normal loads, hand force
vectors, contact counts, projected palm/panel loads, and release direction.
The original normal normalization tolerance is 1e-5; the original independent
normal-patch versus force-matrix tolerance is 1e-3 N. Derived reductions use
1e-8 absolute comparison, allowing numerical summation roundoff without changing
any physical contact threshold. Raw ordered evidence is not relabeled as an
unordered contact multiset.

## Submitted motor inputs, not delivered torque

The original 61-by-69 transmission is reconstructed in the actual recorded robot
joint order. The original passive profile, rank, motor contract, and caps remain
bound. Each backend generalized input is combined with the original explicit
passive terms at its measured **pre-step** velocity and independently inverted.
Residual and cap checks retain 1e-5 tolerance; command/readback comparison retains
the producer's strict error <1e-4. The legacy `actual_joint_effort` and
`actual_motor_forces` arrays mean submitted generalized input and its
motor-equivalent reconstruction. Neither measures delivered actuator torque.

The first pre-step velocity is an actual saved measurement, but there is no
separate reset-time velocity archive against which to compare it. The receipt
reports this limitation explicitly. Every later pre-step velocity must equal
the preceding recorded actual joint velocity exactly; no zero initial velocity
is invented. The reset motor-input field is an unstepped diagnostic only.

The row field named `evidence.physics_qualified` in this capture means the
producer's sticky reconstruction/cap submission-valid flag. It is checked as
such and is **not** full task qualification. A future controller must combine
observations with its separate current physical safety and source qualification.

## Detached API and future consumer

`audit_standing_continuation(trial)` returns the independent receipt, including
`endpoint`, `input_sha256`, exact `source_physics_sha256` and
`source_terminal_time_s`. The endpoint includes actor-origin root13, joint and
mechanism positions/velocities, actual body poses, contact reductions, commanded
motor input and its command epoch, backend submitted generalized input, and the
motor-equivalent reconstruction.

`admit_standing_continuation_observations(trial, audit_path,
expected_epoch_s, expected_physics_sha256)` freshly repeats the accounting,
requires the saved receipt to match, and binds the caller's separately qualified
source epoch and state archive. It returns detached data only. A future
same-episode postopening stage still needs qualified physical source admission,
live exact prefix comparison, actual inherited feedback state and a separately
admitted controller route. This helper grants none of those permissions.

CPU fixtures are explicitly synthetic. The read-only actual transfer005 scene
inventory check resolved all six USD layers and exactly matched the existing
producer's 84 participants (64 robot participants); it did not audit a nonexistent
transfer005 continuation stream or promote that run into a released source.
