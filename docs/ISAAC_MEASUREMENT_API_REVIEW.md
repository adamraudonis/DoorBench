# Isaac 5.1 measurement contract review

This review read source files from the existing owned pod without importing
Isaac, stepping physics, changing a running job or launching a GPU process.
The local source copies and complete hashes are retained in
`out/continuous/api-source-review-001/` in the controller worktree. This review
does not qualify a live traversal.

## Submitted effort and measured reaction are different APIs

The installed core articulation wrapper's `get_applied_joint_efforts` delegates
to `get_dof_actuation_forces` and documents that it returns values set by the
effort setter. Its separate `get_measured_joint_efforts` delegates to
`get_dof_projected_joint_forces`, which projects solver joint reaction into each
DOF's direction.

The traversal inverse therefore checks **submitted actuation-input readback**.
It verifies that the backend received the expected input in the correct order,
including floating-point conversion. It does not measure independently sensed
motor output. Projected joint reactions include other solver loads and must not
be interpreted as the original 61 motor commands.

The runner submits `transmission.T @ motor_input - native_passive_terms`.
Restoring passive terms must use the matching pre-step velocity. Inverting the
full-rank transmission recovers the backend's motor-equivalent input; the
residual checks otherwise hidden effort in differential finger coordinates.
Original 61 motor caps remain independently enforced in software. The API notes
that these external DOF inputs are separate from implicit PD drive forces; the
robot's implicit stiffness/damping remain zero in this adapter.

The new `motor-readback-contract.json` makes these semantics explicit. Existing
new-array names `actual_joint_effort` and `actual_motor_forces` are retained for
compatibility, with their submitted-input meaning recorded alongside them.
Actual contact impulses, integrated motion and collision/limit audits remain
independent physical evidence.

## Correct the root linear-velocity origin

The installed `ArticulationData.root_state_w` concatenates actor-frame pose and
COM-frame world velocity. Its positions and orientations are valid actor/body
origins; its linear velocity is measured at a different point.

`root_link_state_w` concatenates actor pose with actor-origin world velocity.
The SDK converts the latter as

```text
v_actor = v_COM + omega_world × (-R_actor * COM_offset_in_actor)
```

Both angular velocities are world-frame and equal for a rigid body. The native
MuJoCo free-joint calculators require actor-origin world linear velocity and
body-local angular velocity, so their existing `R.T @ omega_world` conversion is
correct. They must receive the actor-origin linear velocity directly.

The new traversal mode now uses `root_link_state_w` for control, state archives
and quiet-stop checks. It separately records the unmodified mixed-origin values
as `legacy_root_state_w`. Older modes and archived evidence remain unchanged.

For the archived `isaac-full-v2-003` trajectory, using the authored pelvis COM
offset `[-0.0002, 0.00004, -0.04522]` m gives:

| Quantity | Result |
| --- | ---: |
| Maximum origin-dependent linear-velocity difference | 0.0244431 m/s |
| Final-second maximum velocity difference | 0.00031335 m/s |
| Archived final-second maximum horizontal COM speed | 0.00162107 m/s |
| Corrected actor-origin estimate of that speed | 0.00137793 m/s |

This calculation is source-derived, assuming imported COM placement matches
the authored robot; runtime COM readback should confirm that assumption during
the smoke. The archived poses, joint/contact trajectories and angular velocities
do not change. Prior root-speed labels describe COM speed. The final quiet
result still lies well below 0.03 m/s, but the older controller's linear feedback
used a different origin, so it is not an exact replay of the corrected adapter.

## Contact ordering, buffers and reset timing

`get_contact_data(dt)` returns six arrays: scalar normal force, world contact
point, world normal, separation, per-pair count and per-pair start. A stale doc
summary calls this five arrays, but the implementation and tuple both contain
six. `get_friction_data(dt)` returns four independent patch arrays. Both convert
the preceding step's impulses using the provided 0.002-second timestep.

The SDK reuses its internal count/start tensor storage between normal and
friction getters. The runner's immediate detached copies prevent one getter
from overwriting the other's slice metadata. Normal contacts and friction
patches must never be paired by raw slot index. Sensor and filter names are
recorded with the actual tensor order, including the full scene needed for
self/environment checks. Every occupied slot is archived by the new traversal
stream.

`body_state_w[..., :7]` correctly returns link/actor origins and world wxyz
orientations. The SDK invalidates these buffers after reset pose/joint writes
and performs articulation forward kinematics when refreshing link poses. A
touch SITE pose still needs its explicit offset from the palm BODY origin.

At t=0 there is no completed episode interval. Reset input/contact buffers carry
the interval `[0,0]` and cannot establish executed force/contact evidence. The
first actual interval is `[0,.002]`, read after `sim.step`, then
`robot.update(dt)`/`door.update(dt)`. Subsequent control consumes that interval
with integrated endpoint poses; this does not assert that contact points and
endpoint body poses describe one instantaneous configuration.

Two helper details are tracked separately: left touches must include positive
load as well as nonpositive separation, and foot support should exclude forces
from other robot bodies while retaining those contacts in the mechanical audit.
These are false-support/clearance guards, not established causes of a past
physical failure.

## Exact inspected sources

| Installed source | Relevant lines | SHA-256 |
| --- | --- | --- |
| `omni.physics.tensors` 107.3.26 / `impl/api.py` | 1830, 2034, 2063, 5841, 5877, 5961 | `5dd16f8a37eccc94ac82338d6c1127e785cccf761cae1f6f18ef03d55b0f325f` |
| Isaac Sim core `prims/impl/articulation.py` | 1217, 1285 | `65901a45bd1852c0e126581ed2bb75105b6e221128e24f2202d138c822b119a5` |
| Isaac Lab `assets/articulation/articulation.py` | 218, 260, 561 | `fb179aacdd848906367d01105020e1d36412bd0f1da7187be09f0781ff589309` |
| Isaac Lab `assets/articulation/articulation_data.py` | 483, 538, 551, 583, 657 | `d69495c66c409fb20823e4e467f9d90e63163b2050b77d2ac8e433221d1d208a` |
