# Local tactile digit-force comparison

The retained [palm translation experiment](SENSOR_PALM_TOUCH_COORDINATION.md)
realized its small arm correction but barely redistributed contact load.
This new opt-in experiment uses local tactile feedback to add bounded digit
motor effort, preserving the original motor transmission, caps and anatomy.
Its high-level route remains instance-specific and scripted. It is not a
learned or vision policy, and it receives no runtime door/root/object state.

## Original coupled actuation

For original transmission matrix `A`, motor effort `u` produces joint moment
`A.T @ u`. The desired pad virtual work is `tau = J_pad.T @ F`. The mapper
projects it into the original motor span with
`u = solve(A @ A.T, A @ tau)` and records the unactuated residual
`tau - A.T @ u`. It never pretends that the J1/J2 pair has two independent
motors. No torque is applied to the rejected passive difference coordinate.

At the attained index-only endpoint, a nominal 2 N index correction requests
J2/J1 moments of 0.07887/0.03400 Nm. The shared J0 motor can apply 0.05643 Nm to
each; the unavailable ±0.02243 Nm split remains an explicit residual. The
largest additional motor bias across all five nominal 2 N corrections at that
pose is 0.13508 Nm. This is a kinematic mapping result, not actual pad pressure
or physical qualification. Original passive loopbacks still act in the plant.

The force line passes through each authored distal tactile site along its
calibrated outward `-Z` axis (palmar body `-Y`). Translating the anchor along
that same normal to the pad surface would leave virtual-work moments unchanged.
The site is a fixed robot calibration, not an inferred contact point. The
finite local tactile projection is used as feedback; shear-heavy tactile
resultants are not mislabeled as geometric contact normals.

## Frozen 36-second protocol

The baseline is the index-only profile, with its existing smooth stiffness
and bounded index preload. The separate unsuccessful palm-shift option is not
enabled. Acquisition through 19 seconds must stay bitwise identical.

After 19 seconds, the force correction has these fixed parameters:

| Parameter | Declared value |
|---|---:|
| Original local target per finger / thumb |2 N / 3 N|
| Proportional / integral gain |0.5 / 1 s⁻¹|
| Integral and virtual correction bound |±2 N per digit|
| Virtual force-state slew bound |2 N/s|
| Initial ramp |Quintic smoothstep, 19–21 s|
| Contact weight |`clip(raw local palmar projection / 0.2 N, 0, 1)`|

The error uses the preceding accepted filtered local load (the first contact
decision initializes from its available raw load). Clamped anti-windup is
explicit. At zero measured local palmar load, the correction is exactly zero
and that digit's integrator resets. This is sensor-level contact gating; it
does not identify the contacted object or qualify anatomical contact. Changes
in contact weight may change applied correction faster than the internal
state's slew bound, so actual command continuity is measured independently.

Only original finger motor rows receive the additional policy bias. It enters
before the existing capped 61-motor adapter and previous-action update; no
returned force is edited. The active model's coefficients, masses, passive
mechanics and force caps stay fixed. Most finger caps remain ±1 Nm, while the
original thumb THJ4/THJ5 caps remain ±2/±3 Nm. All prior pose-offset bounds,
0.8 N finger / 1.2 N thumb progression gates, joint/loopback checks, original
distal anatomy and actual 500 Hz physical criteria remain unchanged. This new
force profile additionally requires the original opposed-pad grasp predicate
at every actual contact interval from 19 seconds through the end, rather than
only a final hold. Prior profiles and their frozen reports are unchanged.

The same source-bound nominal index geometry screen is required because the
joint-goal route is unchanged. It cannot predict the loaded displacement from
additional motor effort. One finite native trial, with every failure retained,
will measure whether this extra feedback achieves the original handle/latch
and sustained opposed-grasp requirements.

Run `audit_robot_digit_force.py --trial ... --output ...` to inspect the motor
projection over archived robot poses without stepping physics. For the new
physical comparison, add this flag to the full index-only probe command:

```sh
--digit-force-protocol configs/dexterous/sensor-digit-force-v1.json
```

Do not pass `--palm-protocol`. The probe freezes all source and profile bytes
and writes the projected joint moments, unavailable passive moments, requested
motor biases, actual motor forces and original independent contact evidence.

## Retained native trial 001

`out/continuous/sensor-digit-force-001` ran 36 seconds / 18,000 physical steps
from source `120b86ee6` and **failed 25/28 checks**. It reached 0.442599 rad
of lever rotation and 6.391 mm latch retraction, below the required 0.8 rad /
11 mm. The virtual press clock stalled at 23.444 s; the index-only comparison
had reached 0.520626 rad. The route-completion, requested-excursion and sustained
operation checks remain failed.

Every actual interval from 19–36 s retained the original opposed distal grasp.
There were zero invalid loaded patches, unintended hand contacts, external
assistance, motor delivery errors, QP failures or native warnings. Maximum
pelvis/torso tilt was 0.3325°, joint-stop penetration 2.040 mrad, passive-loop
violation 0.498 mrad and nonfoot penetration 0.233 mm. The first 19 seconds of
qpos, qvel, commands, time and joint goals match the earlier acquisition bitwise.
All 18,000 raw contact classifications and qualified loads independently match.
The independent contact auditor correctly remains failed overall because the
underlying task report failed.

The independent mapping replay reconstructs every additional motor bias exactly;
the largest requested bias is 0.136186 Nm and its largest 2 ms change is
0.000343 Nm. The internal virtual force changes by at most 0.004 N per step;
the contact-weighted/ramped output changes by up to 0.005396 N per step. These
are distinct quantities, as declared before the run. The largest unavailable
J1/J2 difference moment is 0.022591 Nm and remains unactuated.

### Why force feedback stalled

`diagnose_digit_force_regulation.py` independently reconstructs every finger
command from recorded encoders, the controller's saved sensor-derived robot
estimate, named goals and original coefficients. It matches all 18,000 issued
commands to 4.56e-15 Nm; the separate actual-step audit shows exact delivery.
No finger motor saturates. This calculation evaluates the policy algebra and
robot bias dynamics; it is not a new physical contact measurement.

| Motor, final-second mean | Added force bias | Position response | Total actual effort |
|---|---:|---:|---:|
| Index FFJ3 | +0.13319 Nm | −0.04386 Nm | +0.08952 Nm |
| Index shared FFJ0 | +0.05639 Nm | −0.02821 Nm | +0.02818 Nm |
| Thumb THJ5 | −0.13520 Nm | +0.13305 Nm | +0.01769 Nm |
| Thumb THJ2 | −0.10281 Nm | +0.24028 Nm | +0.14007 Nm |

Small gravity/Coriolis and damping terms explain the remaining differences.
The index moves beyond its fixed nominal coordinate: FFJ3 1.127647 versus
1.124906 rad, and J1+J2 0.921559 versus 0.918033 rad. Position feedback then
opposes the new normal-force effort. Thumb unloading is likewise opposed by
its fixed posture feedback. Final mean local/qualified-normal load is
0.665/0.776 N at the index and 4.186/6.201 N at the thumb. All contacts are on
the permitted distal patches; this failure is not load secretly applied to a
middle link. The actual index split remains individually audited (J2 0.460637,
J1 0.460922 rad); the small passive-limit solver error is within the frozen gate.

The prior index-only endpoint had FFJ0 effort 0.03214 Nm with no added bias.
The new bias does not guarantee a larger delivered normal pressure. These are
different closed-loop equilibria and press phases, so their difference is not
a matched-state intervention or an isolated causal gain measurement.

### Next declared design direction

Use a contact-normal force task with posture feedback projected out of that
same actuated direction, retaining damping and posture regulation in its
orthogonal motor subspace. This addresses the observed competition directly.
Separating constrained force directions from motion directions is established
in [Raibert and Craig's hybrid control formulation](https://fab.cba.mit.edu/classes/865.15/classes/measurement/hybrid-position-force.pdf);
[operational-space control](https://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf)
also frames motion and force tasks at the end effector. These papers motivate
the architecture; they do not establish that this underactuated hand will pass.
A motor-space projection is not automatically a dynamically decoupled Cartesian
controller. Its contact-transition, passive-split and collision behavior must
be screened and physically tested with the original limits.

For acquisition after the corrected Isaac reach, distal-only feedback cannot
repair an already loaded middle link by simply adding pressure. The existing
packet includes each proximal, middle and distal tactile site. A future local
contact state machine should pause the approach on the first touch, distinguish
robot-side contact regions, unload a middle-link contact before bounded
reorientation, and only advance after opposed distal loads persist. Own-robot
FK may express approximate taxel surface locations in a palm-relative frame;
no true object pose or simulator contact ID may enter. Coarse taxels cannot
certify millimeter axial clearance or identify the object. Such uncertainty
must stop or trigger a bounded search, with the independent original anatomy
auditor retaining authority. The measured base displacement across backends
also motivates palm-relative Cartesian goals using the sensor-estimated base,
rather than replaying joint angles as if the base pose were identical. These
are proposed new components, not completed acquisition or policy results.

Receipts and close-up images are in the trial directory. The final
`touch-operation-hand-az150-0002.png` was personally inspected. It is a render
of recorded physical geometry, not an additional live simulation or video.

| Evidence | SHA256 |
|---|---|
| Provenance | `d1b25963c9058180f376fd0b90921752dc9f8d77f1bdecae71ef4b83e7a35eba` |
| Failed report | `d785cdea8b8bd82c500cc8e728ff30e9fc76f80c71d8df6b779017f6d888f936` |
| Trajectory | `2a4fa27dea5a14e3cfb2b86d64f00dd69d421977e636682c5b1fa143ba5a7af5` |
| Actual-step archive manifest | `18a65ce65c518a64de5579bea67b69b576f884fc3443b0c01d98b8617f55727b` |

Mapping receipt 001 predates an optional reporting-field-name correction and
contains a null local-projection summary; receipt 002 supplies that summary.
Neither receipt changes the physical evidence or its outcome.
