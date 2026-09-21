# Opt-in actual-Isaac standing withdrawal

This mode adds a motor-driven right-hand withdrawal after a **qualified actual
Isaac standing transfer**. It uses the newly recorded episode through the source
epoch, a fresh source-bound coupled map, and the existing capped motor
controllers. It does not load an archived state into the plant. Native
withdrawal defaults remain unchanged.

The controller/launcher integration has CPU coverage; an actual Isaac release
has not been physically qualified by this implementation. Planning convergence,
a passing static audit, and a completed controller clock do not establish a
successful release or traversal. Failed transfer sources remain ineligible.

## Required evidence and runtime document

First complete the [actual-Isaac planning sequence](ISAAC_COUPLED_RELEASE_PLANNING.md):
fresh actual-source admission, RH candidate, distinct passing 2,001-sample RH
geometry audit, source configuration, fresh coupled map, and its distinct
passing dense envelope audit. The supported runtime profile is currently
`volar-phalange-v1`. Old native envelopes or audits cannot substitute for these
Isaac receipts.

Create a separate JSON document with this schema. All three paths must be
absolute and each SHA-256 must match the named file's bytes. The values below
are placeholders, not an admitted experiment:

```json
{
  "schema": "doorbench.isaac-standing-withdrawal-runtime.v1",
  "source_config_path": "C:/absolute/new-plan/withdrawal-source.json",
  "source_config_sha256": "<SHA-256>",
  "coupled_envelope_path": "C:/absolute/new-plan/map/envelope.json",
  "coupled_envelope_sha256": "<SHA-256>",
  "coupled_audit_path": "C:/absolute/new-plan/map-audit.json",
  "coupled_audit_sha256": "<SHA-256>",
  "capture_returned_motor_command": true,
  "inherit_transfer_support": true,
  "left_arm_only": true,
  "left_full_orientation": true,
  "coupled_motion_projection": "fixed-poses-v1"
}
```

Optional `scope` metadata and the explicitly experimental
[`withdrawal_palm_load_profile`](ISAAC_WITHDRAWAL_PALM_PROFILE.md) may be added.
The required options cannot be disabled. Source qualification, actual normalized endpoint, original assets,
prospective profile, full motor contract, and all proof hashes are rechecked.
The map keeps its static status and zero authorized stages; live entry has a
separate witness.

## Launcher and exact episode length

Use `scripts/isaac/run_local_operation.py --standing-withdrawal-route <runtime.json>`.
Repeat the qualified transfer's launcher arguments exactly, including its
operation source, transfer route, support settings, experimental offsets,
render profile and live transfer-prefix witness. Change only output and total
duration before adding the new withdrawal option. The launcher compares the
resulting producer arguments with the qualified source's recorded `launch.json`.
Jev is not enabled in this deterministic continuation mode.

The total `--seconds` must equal the **actual source terminal epoch plus the
independently audited withdrawal duration**, exactly. For example, a qualified
44-second source and a 16-second audited route require 60 seconds; this example
does not claim that a particular 44-second source has qualified.

In PowerShell, with `$sourceLauncherArgs` holding the exact original launcher
arguments except `--seconds`, `--output`, and `--execute`:

```powershell
$totalSeconds = $actualSourceTerminalSeconds + $auditedWithdrawalSeconds
python scripts/isaac/run_local_operation.py @sourceLauncherArgs `
  --seconds $totalSeconds --output $freshRunDirectory `
  --standing-withdrawal-route $runtimeJson
```

Without `--execute`, the launcher performs read-only admission and prints the
concrete command. Add `--execute` to run the local physical experiment. The
launcher supplies the source-bound authored door XML to the private geometry
calculator; it does not change the live door or restore source coordinates.

## Live entry, support and failure behavior

The optional launcher flag `--standing-transfer-stop-on-rest` can produce a
fresh transfer source at the first measured joint resting hold. In a standalone
transfer, `--seconds` remains the hard deadline; it does not become the recorded
successful duration. The observer requires 251 consecutive 500 Hz samples,
including both endpoints of the half-second window. The window starts no earlier
than transfer start plus eight seconds, so termination cannot precede start plus
8.5 seconds. Every row must retain the selected RH grasp, qualified loaded
patches, zero non-digit handle force, complete synchronized raw evidence,
solved stance, at least 2 N measured palm support, and the original resting
mechanism bounds. A failed row resets the whole window. Clock corruption or
nonfinite evidence stops the attempt; a deadline without a trigger fails.

`standing-transfer-rest-stop.json` records the unchanged maximum deadline and
the frozen first qualifying endpoint. Final checks use the actual recorded
endpoint and retain every preceding interval. The independent rest-stop audit
reconstructs that first window in addition to the original raw contact and
physical audits. This option applies prospectively to a new trial; finding a
window inside a previously failed longer run does not qualify that failed run.

When a later withdrawal repeats such a qualified source, keep
`--standing-transfer-stop-on-rest` in the identical source recipe. The detector
then acts in `prefix-only` mode: it freezes at the same first endpoint but does
not terminate the new episode. That endpoint must equal the admitted withdrawal
start before any withdrawal motor command is submitted. The total duration
remains **actual source endpoint plus audited withdrawal duration**, not the old
source's maximum deadline plus duration. Both exact source-prefix witnesses
remain mandatory.

Every 500 Hz source interval must match the original physical/command archive:
time, root, joints, joint velocity, commanded motor forces, mechanism positions
and velocities, and recorded standing-body poses. `motor_forces` represents
**commanded motors**, not measured delivered torque. A separate synchronized
leaf-pose witness compares exact bytes after explicit float64 canonicalization;
the original historical JSON tensor dtype is not claimed to be known.

At the exact source epoch, both witnesses must be complete and authorize entry.
Original source contact/physical qualification remains mandatory. Runtime entry
also retains the original half-second resting opposed-grip and palm-only support
requirements. No missing, divergent, repeated, stale or late witness can start
the new stage. Rejection is terminal for that attempt, and partial evidence is
retained instead of recording completion.

The first withdrawal motor command captures the actual preceding command
returned by the transfer controller. Its nested operation cache is diagnostic
only. The original one-second bounded motor handoff and motor caps remain.

Palm feedback inherits the actual transfer's existing `StandingSupportFeedback`
object, filter state, blend, surface-velocity history and start epoch. The support
target comes from the qualified source recipe (for example, 6 N); the runtime
document cannot replace it. Entry requires the preceding feedback update at
exactly `start - 0.002`, and every subsequent interval uses the current measured
palm-only load and leaf pose with the unchanged target. The original bounded
support range and 2 N physical support gate remain.

The private coupled reference retains the original rate/acceleration limits,
fixed-pose projection and post-limiter geometry guards, plus the audited final
40 mm RH/environment clearance. A measured mechanism state outside the admitted
map, missing feedback, invalid geometry or failed rate projection stops the
attempt. Static geometry is never treated as a contact-force measurement.

## Evidence and Python seams

`admit_isaac_withdrawal_runtime(path, motors=None)` is the shared read-only
launcher/factory admission. It exposes `source_context.data`, `source_admission`,
`input_sha256`, `runtime_path`, `start_time`, and `duration`. The context contains
runtime, source-config, map, audit and original input identities.

`create_isaac_withdrawal_controller(standing_transfer, motors, path)` preserves
the predecessor object. Call `authorize_source_prefix(combined_receipt)` before
the first stage force. The controller uses the existing transfer `force`
signature and exposes `started_withdrawal`, `release_started`, and `info`.
`info['withdrawal_progress'] >= 0.999` indicates reference-clock completion;
the independent physical audits still decide trial success.

`withdrawal_runtime_source_paths()` returns absolute controller/admission helper
paths. Producer and launcher deduplicate these with their existing source lists
and separately capture the combined witness and evidence helpers. The new run
honestly records its current code; historical source provenance is unchanged.

The run records `live-withdrawal-prefix-witness.json`, the synchronized leaf
array, `standing-withdrawal-steps.json.gz`, source admission and runtime inputs.
Per-step evidence includes inherited support history and actual-command capture
diagnostics. Offline prefix comparison, independent raw contact accounting,
and `isaac-withdrawal-audit.json` remain separate required checks. Transient
contact failures are retained; final clearance alone cannot qualify the run.

The explicit withdrawal mode additionally captures future continuation inputs
throughout the same episode. `standing-continuation-contract.json` declares the
contact row/filter layout and body-origin order: both ankles, both palms, leaf
and handle. `standing-continuation-steps.json.gz` contains the completed-interval
foot loads, full hand force vectors, external-contact counts, measured release
normal, body poses and separate sparse normal/friction patch buffers. Empty
buffer capacity is omitted; occupied slot indices and values are retained.
These observations do not execute or authorize a post-opening controller.

The physics archive also retains `continuation_body_poses`, `actual_foot_loads`,
`actual_joint_effort`, `actual_motor_forces`, `pre_step_joint_velocity` and the
legacy root diagnostic. The motor fields retain the existing legacy names:
they reconstruct the backend's **submitted actuation input**, not an independent
measurement of delivered physical torque. Reconstruction uses a copied velocity
from before that exact physics step. The recorded contract and maximum residual
make that distinction explicit; normal force-cap and transmission checks apply.
Failure finalization exports the captured prefix without granting stage authority.
