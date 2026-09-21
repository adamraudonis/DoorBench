# Recorded standing-reference analysis

`isaac_standing_reference_audit.admit_standing_reference_tail(trial,
continuation_audit, expected_epoch_s, expected_physics_sha256)` independently
accounts for the final three accepted withdrawal commands against a freshly
audited continuation observation archive. It reads saved evidence only. It
does not evaluate a reference, run an IK/QP/controller, restore state, or grant
a stage permission.

The source must contain the explicit `accepted-command-tail-v1` capture and
the minimal actual-Isaac withdrawal runtime. The exact runtime, prerequisite
source binding, original motor contract, captured capture-helper source, and
all observation inputs are hashed. The prerequisite source-state hash is
distinct from the completed episode's physics NPZ hash. The observer audit is
recomputed; its original submitted-input reconstruction and cap tolerances
remain unchanged. A failed physical report can still be analyzed, but can
never acquire physical or bridge qualification through this analysis.

For endpoint **T**, the three commands are at **T−0.006, T−0.004, T−0.002**.
Each is paired with its completed physical interval ending two milliseconds
later. The observer compares captured command inputs with archived state at
the command epoch, and the returned 61-vector's dtype and bytes with the
following interval's submitted motor command. Motor readback remains a
reconstruction of submitted generalized input, not delivered actuator torque.
Missing, pending, failed, reordered, truncated, nonfinite, changed, or
inconsistent evidence is rejected. NPZ replay retains only the four relevant
state rows in addition to the bounded streaming buffer.

Algebraic checks relate the accepted 75-coordinate reference to the captured
root, stance biases, LH nominal joints, RH post-feedback targets, arm history,
and original hand transmission. Captured actual enhanced arm/hand gains,
support histories and model bias remain explicitly identified. Their presence
does not constitute a replay of internal controllers or a restorable solver
checkpoint. No unavailable state is replaced with invented zeros.

The receipt reports nominal positions, two finite-difference velocities,
finite-difference accelerations, original coupled speed/acceleration limits,
and the prior nominal's difference from measured endpoint T. Rotation-vector
coordinate rates are reported separately from world angular interval averages.
Neither is claimed to be an instantaneous measured velocity. The first stored
reference velocity cannot be independently reconstructed without an earlier
reference outside the three-row tail. A RH touch-site goal is not compared as
if it were the measured palm body origin.

`passed` requires source/reference accounting and the original coupled motion
limits over the available tail. `reference_tail_accounting_passed` distinguishes
valid evidence with an excessive nominal rate. Every result keeps
`physical_task_qualification=false`, `bridge_feasibility_qualified=false`,
`controller_state_restoration_supported=false`, and `authorized_stages=0`.
A future continuous bridge still needs a separately qualified actual release,
live controller continuity, source-bound geometry/domain screening, and actual
physical validation. Three recorded references cannot prove that bridge.

```powershell
python scripts/dexterous/audit_isaac_standing_reference.py `
  --trial <completed-run>/trial `
  --continuation-audit <saved-observation-audit.json> `
  --expected-epoch-s <actual-endpoint> `
  --expected-physics-sha256 <actual-NPZ-SHA256> `
  --output <new-reference-analysis.json>
```

Existing receipts are never overwritten. Corrupt evidence produces a failed
receipt and exit code 1. Tests use a complete synthetic observation/capture
fixture and label it as such; no successful physical withdrawal is fabricated.
