# Prospective IK and command observation

`SelectedWindowIkObserver` is a detached observer, not a controller. It has no
reference to a teacher, articulation, model, solver, API client, or plant. It
copies caller-supplied values and checks clocks, shapes, finite data, final
motor caps, and exact final-command byte accounting. It does not evaluate PD,
run IK/FK, update a filter, infer an internal target from a torque, or smooth
anything. No producer hook is installed by this change.

This is intended to replace the incomplete internal-target evidence in actual
Isaac transfer005. The earlier diagnosis inferred a torso target step at
command time 18.380 s (recorded physical interval end 18.382 s), and a left
elbow target step at 31.840 s (interval end 31.842 s). Those inferences remain
inferences; no new record is backfilled into that source.

## Snapshot contract

Construct one observer with the complete original joint order, motor order,
motor caps, component IDs, explicit episode/input hashes, and prospective
inclusive command-time windows. For example, `[18.20, 18.60]` and
`[31.65, 32.05]` contain 402 samples at 500 Hz. These are illustrative windows
for a later run, not retrospective source evidence. The capacity is bounded
at construction; at most 10,000 records can be selected. Each selected tick
must appear once. Skipped, duplicate, off-grid or stale command/input epochs
make the observation incomplete. Calls outside the windows store nothing.

The single `capture(...)` call receives:

- Actual pre-command root13, all joint positions/velocities, and phase, with
  measured time exactly equal to command time `t`.
- Every declared component, or explicit `None` when inactive. Each component
  contains the **entire held solve vector and its coordinate order**, actual
  last-refresh time, full affected motor order, servo target, measured motor
  positions/velocities, and any reference velocity used by the original code.
- Original pre-PD arrays: `kp`, three affine bias coefficients, impedance
  gain, extra damping, separate reference-velocity coefficient, mapped model
  bias contribution, other feedforward, and already evaluated unclipped
  servo output. The observer copies these arrays; it does not reconstruct
  their sum or certify their decomposition.
- The actual held task goal. A full orientation is marked `rotation-matrix`;
  the original left reach's normal-only constraint is marked `axis-z`. No
  missing wrist orientation is manufactured from a normal vector.
- Ordered full motor-vector copies at the relevant return/override
  boundaries, followed by the exact final returned motor vector. The final
  stage must match its dtype and bytes; all final values must respect the
  supplied original caps. Stages may replace channels, apply projections,
  clip, or blend: they are not mislabeled as additive torque contributions.

The final command remains a returned/submitted command, **not delivered
torque**. The recorded interval `[t,t+.002]` is intended; it is not marked
completed. A separate later audit must cross-bind this command to the
physical archive row ending at `t+.002` and verify the actual pre-step state
at `t`, including the producer's pre-step velocity copy. Supplied source
hashes are retained but are not independently re-admitted by this observer.
Every receipt explicitly grants zero task/stage authority.

`servo_unclipped` specifically denotes the local affine/impedance servo after
its mapped model bias and active reference-velocity term, **before** contact,
hybrid, stance replacement, wrapper overrides or clipping. Capture those
later results as distinct full-vector pipeline stages. For the current
acquisition and LH base servo, `other_feedforward` is explicitly zero; do not
place a projection or overwritten channel into that additive field. Any
future additive term must be copied from the actual calculation and named in
the producer's captured source. In acquisition, the model bias contribution
has zeros on channels where the existing code never adds model bias (including
the torso). These are declared branch facts, not missing values filled in by
the observer.

## Exact future hook locations

These are a design for a separately captured, opt-in producer change. None
should be added to the currently running withdrawal experiment.

1. **`AcquisitionTeacher.force`, refresh block at lines 116–140.** After the
   existing solve and target clipping, copy the complete `w.qpos` coordinates
   selected by `self.arm_q`, their current `self.arm_joints` names, `pos`,
   `rot`, and `self.last_update`. Capture every coordinate, including torso
   while it is an active solve coordinate. Keep these copied goals between
   refreshes; do not call the solver again to recover them. Separately retain
   the full 61-element `self.target`, not the producer's `ctrl` alias.
2. **`AcquisitionTeacher.force`, lines 157–179.** Capture the existing
   `length`, `speed`, `kp`, `bias`, `gain`, `damping` and base force expression
   at 500 Hz before later modifications. Copy the already evaluated force at
   model-bias, grip-pressure, stance replacement and final clipping
   boundaries. The mapped bias/pressure arrays needed for the component
   record must be materialized observationally from those existing locals;
   do not replace the controller's original expression or summation order.
   The stance replacement covers leg channels; it is not the torso servo.
3. **`LeftPalmContact.update_targets`, lines 191–217.** After the existing
   `fit.x` assignment, retain the **full `self.previous` in `solve_names`
   order**, `self.last_update`, `goal`, and actual orientation constraint.
   `self.target` contains only the seven LH servo coordinates and is not the
   entire coupled solve vector. Preserve any actual transition from the
   combined solve to the isolated seven-coordinate solve as a new inventory.
4. **`LeftPalmContact.apply_forces`, lines 227–268.** Copy original `q`, `v`,
   `kp`, `bias`, `damping`, `self.target`, and the actual target-velocity value
   used in line 237 (including the original zero default, materialized once).
   Preserve the model bias slice and the pre-clip vector at servo,
   reference-velocity, push, hybrid projection/blend, and final return
   boundaries. Never recompute a Jacobian, mass matrix, gravity or projection
   merely to populate the observer. If a value cannot be obtained from the
   actual existing computation, that proposed producer hook is not ready.
5. **`StandingTransferTeacher.force`, lines 138–154**, and the operation
   wrapper: preserve complete vectors after operation/pad/hub control, LH
   replacement, attained RH tracking, and motor handoff. These overwrite and
   blend stages explain why a local pre-PD vector need not equal final torque.
6. **`scripts/dexterous/isaac_opening.py`, after the selected outer
   controller returns and before line 1355's motor-to-joint transformation.**
   Invoke the observer once with all supplied snapshots, exact pre-step
   `state/joints/velocities`, and final `forces`. Do not read those values
   again after `sim.step`. A later source audit can also check the backend
   submitted-input reconstruction; this observer does not claim to do so.

Future producer snapshots must be opt-in, defensive copies taken before
mutable scratch arrays are reused. Copying raw operands is preferable to
rewriting the force equation around derived coefficients: changing floating
summation order can change the exact replay prefix. Native/default returns
must be separately verified unchanged before any physical capture trial.
All new producer/runtime sources must enter the existing capture registry.

## Coefficients and finite differences

The offline `window_finite_differences` helper derives the signed measured
velocity coefficient as `bias[:,2] - extra_damping`. It keeps the supplied
reference-velocity coefficient separate:

| Current channel | Measured velocity coefficient | Reference velocity coefficient |
| --- | ---: | ---: |
| Acquisition torso | −26 | 0 (no reference-velocity term) |
| LH shoulder/elbow | −12 at original scale | +10 at original scale |
| LH wrist | −1 at original scale | +0.8 at original scale |

These values follow from the captured primitives and original motor contract;
the observer never hardcodes them. Damping scaling or different source gains
must appear in the data. Adding `26 * target_velocity` or replacing the LH
coefficient 10 with 12 would be a prospective controller experiment, not
observation and not an algebraically equivalent baseline.

Backward differences are computed only between consecutive selected ticks
with the same component and coordinate order. Gaps, inactive components and
solve-inventory changes break the difference chain. A 2 ms held-target step
is not an instantaneous velocity or a 10 ms solve velocity. Preserve full
vectors and refresh clocks when comparing both notions. The diagnostic is
bound to the exact observation document's canonical SHA256 and retains its
source-binding declaration, without promoting that declaration to proof.
Finite supplied inputs whose subtraction or division overflows are rejected;
derived damping coefficients and differences must also remain finite. This
diagnostic rejection does not modify the original observations or commands.

No interpolation duration is qualified here. A later analysis can compare
vector steps, update spacing, command jumps, cap activity, measured motion and
task errors before choosing explicit duration/rate/acceleration inputs to
`GuardedIkReference`. Nonlinear foot/palm/contact constraints still require a
fresh guard on every 500 Hz sample. Only a new physical trial can establish
smoother motion and retained task success.
