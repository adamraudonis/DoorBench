# Captured withdrawal references for a later panel bridge

The explicit Isaac `--standing-withdrawal-route` producer now enables
`accepted-command-tail-v1` and writes
`trial/standing-continuation-reference-tail.json`. Native and default controller
paths do not allocate snapshots or call the observer. The earlier acquisition
and transfer prefix returns before the capture hook. No reference evaluation,
force calculation, filter update, model step, plant query, or state installation
is added by this observer.

The force-local hook copies the existing requested RH goal and full final
post-feedback target, which previously disappeared when `force()` returned.
These are distinct from the private model's achieved palm pose. The helper also
copies accepted coupled position/velocity/time, stance targets and biases,
the actual primal QP warm-start vector, arm target/velocity history, RH palm
correction, LH IK target/seed/nominal and normal offset, and inherited support
filter/surface history. It records hand preload/reference state, motor handoff
offset/clock, the final outer returned 61-vector, and the existing measured
inputs to that command. Original named motor caps and the static LH path are
copied once; actual enhanced arm/hand gains are captured once when those trackers
exist. Each command copies only the changing final LH nominal, not the full
101-node static path. Model `qfrc_bias` is labeled private inverse-dynamics data,
not measured motor or contact force.

The schema is `doorbench.isaac-standing-continuation-reference-tail.v1`:

- `tail` contains at most three accepted snapshots; `completed_reference_intervals`
  counts accepted withdrawal intervals.
- Each row distinguishes `command_time_s = t` and the observed command input at
  `t` from `post_step_time_s = T = t + 0.002`. The returned vector must match the
  producer's submitted vector by dtype and bytes.
- At terminal physical epoch `T`, the latest stored reference was computed at
  **T−0.002**. It is not a newly evaluated target at `T`.
- `static_controller_data` binds motor names/caps and the construction-time LH
  path; `attained_tracking_contract` records actual enhanced tracking gains.
- A row enters the tail only after the step and the existing backend submission
  failure guard. `pending_unaccepted_command` preserves an attempt that did not
  reach this acceptance point; it may have been unexecuted or followed by a
  failed interval. Consult the separately preserved physical failure report.
- Optional target-velocity state is explicitly marked absent if the selected
  controller never created it. Required state is never replaced with zero.
- No acceleration state is invented. The last three accepted values and stored
  velocities support explicit finite-difference diagnostics with their actual
  coordinate conventions.

The observer retains zero stage authority and no task qualification. It does
not provide controller restoration or a full serialized OSQP checkpoint. Future
panel admission must bind this receipt's bytes alongside its actual source
physics/observation audits, compare the live predecessor objects and captured
tail at handoff, then independently screen the bridge from the preceding nominal
reference to the new source-bound panel path. A measured source knot alone does
not establish reference or acceleration continuity.
