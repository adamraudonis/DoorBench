# Same-live transfer pause: protocol and integration cut

The new helpers are detached. They do not read a PhysX object, call a controller,
step/render Kit, admit a source, or install a plan. A passing handshake is not a
physical or geometric qualification. The existing exact offline replay route
remains unchanged.

## Producer cut and retained state

For the prospective live mode, stop control execution immediately after the
**first actual qualified transfer rest window** reported by the existing sticky
`TransferRestStop`. Keep its original 251-sample predicates and original phase
checks. At `isaac_opening.py:1536-1540` the detector has just consumed the
same-epoch pad and transfer rows, but the interval's remaining bookkeeping is
not finished. Therefore enter the pause only at the end-of-interval seam after
the nonfinite/fall checks (`1661-1664` in the inspected producer), before the
existing standalone rest-stop break (`1665-1668`). This follows:

* One submitted command and `sim.step(render=False)` (`1366`), the normal
  post-step articulation updates and actual clock check.
* Complete physics arrays, pad raw contact record and transfer record through T;
  all optional continuation/reference capture completion, motor-delivery checks,
  normal frame recording and interval wall-timing closure.
* Existing earlier source witnesses. Do not disable one to reach the pause.

At the cut, N physical intervals have completed: `step_index=N`, `epoch_s=T`,
and the next loop command is at T. The previous submitted 61-motor command and
its nominal/filter observations are at T-.002. Never call `force(T,...)` to
collect the pause state. Retain the actual transfer, operation/acquisition,
stance/QP warm-start (`stance.last`), LH controller, support filters, hand/arm
tracking, passive model data, observers and command-capture objects in memory.
Rest-window and predecessor-command collectors needed by a later-created
withdrawal controller must run prospectively before this cut; the pause is not
permission to manufacture their missing history.

The producer must introduce a separate live phase/suffix budget. `a.seconds`
remains the configured maximum transfer acquisition deadline. A new staged
cursor can extend the same loop after a validated plan sets its own T+duration
deadline; it must neither repeat indexN-1 nor submit an unadmitted hold command.
The rest detector's sticky first T remains recorded. Distinguish paused,
aborted, and actually continued states; do not mark continuation before entry.

## Immutable prefix copy without finalization

`snapshot_live_evidence(writers, fresh_directory, pause_token=...)` uses real
`BoundedEvidence.checkpoint` at a fresh token-specific trial-root filename. It
flushes pending serialized rows, copies each hash-checked closed chunk into the
new bundle, and writes a separate compressed JSON array for pure reducers.
Every checkpoint remains `complete=false, passed=false`. It never calls
`writer.export`, deletes a chunk, or marks the process completed. Append and
final export remain available; the copied chunks survive final working-chunk
deletion. Failure preserves both live evidence and partial new files.

The producer separately copies the exact N-row physical NPZ, N+1 pad records
(including reset), N transfer records, the rest receipt, original configuration,
motor contract, provenance/assets/current source captures and a **phase-scoped**
report. It must use `acquisition_states['time_s'][-1]`, not a lower-rate trace
epoch. Publish the distinct paused snapshot accepted by the live-source
inspector: `doorbench.paused-isaac-transfer-snapshot.v1`,
`source_kind=paused-live-isaac-transfer-v1`, `source_engine=isaac-physx`,
`closed_prefix=true`, `episode_complete=false`. Do not fabricate successful
`result.json`, a finished process, or an old completed-source receipt.

The snapshot's `files` maps role to exact absolute pause-local path and SHA256.
Its `live_pause` carries the token, episode ID, controller identity, N, T, .002,
and measurement/controller fingerprints. The live-source inspector owns exact
body-coordinate/raw/phase admission; the file-copy helper grants none.

The agreed mandatory source roles are configuration, motor_contract, provenance,
physics, pad_steps, transfer_steps, rest_stop, phase_report,
grasp_profile_definition, mechanical_audit, reset_state, transfer_route,
source_prefix_witness, and observer_state. The copied transfer route keeps its
`standing-transfer-route.json` basename for the original asset mapping reducer.
The phase report uses `doorbench.paused-isaac-transfer-phase.v1`; the rest wrapper
uses `doorbench.paused-transfer-rest-stop.v1`. Both state episode_complete=false.
Their original gates and complete raw accounting are re-run by the separate
phase auditor; arbitrary true flags in a snapshot do not pass this handshake's
trusted runtime-admission callback.

## Fingerprint and worker handshake

Create `LivePlanningPause(fresh_directory, episode_id=..., retained_objects=...,
timeout_seconds=..., monotonic=...)` first to obtain its 64-hex random token.
The helper keeps strong references to the retained objects. It uses their
name/type/Python-ID inventory for same-process identity, not serialized
controller reconstruction or cross-process identity.

`capture_pause_anchor(...)` fingerprints explicit supplied values only:

* `episode_id`, completed `step_index=N`, `epoch_s=T` and the fixed .002 clock;
* `physics_clock`: real simulation time/index, independently counted physics
  callbacks, articulation/sensor cache clocks and target state;
* `measured`: fresh raw PhysX robot/root/joint/door positions and velocities,
  relevant body poses/velocities, the prior motor command and actual command
  target/effort buffers, with exact named orders;
* `controller`: complete relevant mutable controller/model/filter/observer
  state and clocks, including actual native serialization bytes where supported;
* `retained_objects`, logical evidence counts, and an explicit
  `pending_unaccepted_command=False` observation.

Arrays bind dtype, shape and exact C-order bytes; scalar types, signed zero and
native serialized bytes remain distinct. Unsupported objects or nonfinite
numeric arrays reject. The helper does not inspect objects, invoke arbitrary
properties, or claim that an arbitrary caller subset is complete. The root
producer/anchor adapter must define and test the full named inventory. Hash
only logical evidence state after checkpointing; working storage layout has
intentionally changed, while controller and plant state must not.

Call `pause.publish(snapshot_path, anchor)` after snapshot files are closed. It
writes the fresh request and binds snapshot/source-file hashes. Poll with
`pause.poll()` using ordinary wall-clock sleeps outside Kit. The worker writes
a token-specific response atomically (write/close a temporary, then publish the
new final filename). Ready response schema:

```
schema: doorbench.live-isaac-planning-response.v1
pause_token, episode_id, snapshot_path, snapshot_sha256
decision: ready
source_context_sha256
files: {runtime: {path,sha256}, phase_audit: {path,sha256},
        context: {path,sha256}, ...additional fresh proof files}
```

Abort uses the same identity keys, `decision: abort`, and a nonempty `reason`
instead of context/files. Wrong tokens, stale/corrupt/duplicate fields, missing
files, path escape, input mutations and expired/backward clocks abort without
deleting evidence. Pause-local files are required, not historical file aliases.

Finally call `validate_resume(capture_anchor=..., validate_plan=...)`. The first
callback is the producer's read-only backend+controller inventory capture. The
second must freshly validate original phase gates and the next plan/runtime;
worker `passed` flags are insufficient. It returns a trusted receipt with
`passed=true`, matching `snapshot_sha256` and `source_context_sha256`, plus
`input_sha256` binding every source, response and proof file actually consumed.
The helper checks the exact full live anchor before and after that callback,
rechecks all hashes/deadline, and returns a one-use handshake. Its own
`authorized_stages` stays0. Actual stage authorization and the first post-limit
geometry/rate/motor guards still belong to the runtime adapter. First next
force/submit/step occurs once at T on the retained objects.

## Calls excluded from the wait

No `app.update`, `sim.render`, `sim.step`, timeline pause/play, robot/door
`update`, `write_data_to_sim`, sensor update or controller force/reference
evaluation. Several installed Kit/IsaacLab render/play/update paths pump the
application or change caches; a CPU main-thread file wait avoids those paths.
The root anchor adapter should use explicit raw backend getters and existing
cache buffers, with no lazy property that updates an articulation. A newly
recorded physics callback counter must also remain unchanged.

Timeout/abort raises into normal failure finalization with partial evidence
preserved. The outer launcher watchdog must budget physical run plus bounded
planning time; report pause wall time separately from simulation time. No state
restore, time-origin adjustment, reset, zeroing velocities, or renewed old Jev
wall-clock lease is allowed.

Tests cover exact bytes/dtypes/clock/object identity, one-use response, stale
token/files, timeout and failed admission, mutation during admission, pending
command rejection, and real250+tail evidence copying followed by resumed append
and final export. They are CPU tests. An instrumented short actual pause/resume
invariance trial remains necessary before claiming Kit holds this seam.

## Implemented qualification and actual diagnostic

The snapshot assembler now packages active evidence without ending the episode.
The distinct paused-source inspector, full transfer-phase auditor and release
planning context preserve the original raw-contact, rest, body/material, asset
and motor checks. A synthetic integration test passes actual writer checkpoint
and copy through inspection, full phase reduction and request publication; it
substitutes only the prior operation-source witness at its explicit boundary.
These helpers are not yet wired into a complete late-created withdrawal runtime.

The opt-in producer argument `--pause-readback-probe-at-seconds` performs a
separate two-wall-second suspension after a complete interval. Local actual
probe002 at2s passes exact before/after controller, backend/cache, submitted
buffer and clock equality with1000 unchanged physics callbacks. The same episode
then records intervals2.002 and2.004. This contact-free probe is not qualification
of loaded suspension, release, complete opening or traversal. Probe001 rejected
a stricter newly introduced clock check before waiting; the correction preserves
the existing producer clock tolerance while comparing raw pause clocks exactly.

The readback inventory includes inactive zero-valued direct-wrench buffers.
Robot invariant getters are captured; extension to an explicit door-property
inventory is still required before claiming those properties are checked across
a loaded planning pause. No serialized MuJoCo state is restored to a plant.
