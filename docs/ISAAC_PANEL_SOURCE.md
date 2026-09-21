# Detached actual Isaac panel source

`admit_isaac_panel_source(source, robot=..., door_xml=..., door_usd=...)`
admits only the recorded endpoint of a completed, independently passing Isaac
withdrawal. It returns `doorbench.isaac-released-panel-source.v1` data for a new
unstepped panel planner. This is separate from the grasp-source adapters:
intentional release has no final-grasp predicate, while every original
phase-specific withdrawal check remains required.

Admission recomputes the full independent withdrawal audit and requires exact
agreement with its saved receipt. That audit retains actual runtime/source
admission, the exact live prefix, raw contact accounting and per-interval
authored geometry reconstruction. The adapter also binds archived source code,
original robot/XML/USD assets and the prospective contact declaration. It
rechecks the inclusive 251-sample endpoint: actual palm-only load at least 2 N,
all-RH/environment clearance at least 40 mm, solved stance and an open leaf.
Force labels cannot replace their recorded world-space palm force vectors.
Both original accepted stance statuses (`solved` and `solved inaccurate`) remain
accepted. The adapter also requires the successfully finalized original local
launch/result conjunction, zero process and audit return codes, and no early
stop or error. It verifies explicit arguments against recorded configuration,
captured historical code, all launch input identities, and independently repeats
the recorded transfer and withdrawal-prefix comparisons. Later edits to working
source cannot replace captured historical bytes.

`source_time_s`, `initial_qpos`, `initial_qvel`, `extracted_state`,
`coordinate_admission` and `actual_body_poses_xyz_wxyz` retain the exact measured
endpoint. The original 2 micrometre/2 microradian coordinate check includes feet,
torso, both palms, handle, leaf and every recorded RH/environment body required
by the original clearance model. Only the existing bounded float32 quaternion
normalization is admitted; no native state or optimized initial pose is used.

`motor_handoff` records the original 61-motor order and the submitted command for
the final interval, at `source_time_s - .002`, with its complete motor-contract
identity and original caps. It is evidence, not command playback or measured
delivered torque. `predecessor_controller_diagnostics` are recorded diagnostics,
not a reconstruction of integral, support or stance state. A later continuous
controller must inherit those live states and the actual returned predecessor
command. Optional `standing-continuation-steps.json.gz` is hash-bound if present;
this adapter does not qualify its additional later-stage measurements.

No actual successful Isaac withdrawal is currently claimed. Synthetic tests
exercise source rejection, immutable identities, motor/epoch ordering, body
inventory, support and clearance boundaries. The output authorizes zero stages,
performs zero physics steps, and never qualifies a panel path, full opening or
passage. A distinct new dense panel audit and physical runtime admission remain
necessary.
