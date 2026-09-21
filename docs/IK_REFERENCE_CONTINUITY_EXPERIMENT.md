# Prospective IK reference continuity experiment

This is a detached helper and integration proposal. It has not been connected
to an existing controller, replayed as a physical policy, or run in Isaac or
MuJoCo. Existing recipes and the qualified Isaac transfer005 remain unchanged.

## What the measured evidence supports

The source005 command diagnosis binds the actual archive, motor contract,
configuration and captured controllers. At command time18.380 s, the torso
target inferred from the unsaturated servo changes by -0.032011325 rad at a
10 ms IK refresh. Its stiffness contribution is -96.033974 Nm; measured-state
feedback changes the resulting submitted-command jump to -86.413597 Nm.
At31.840 s, the left-elbow inferred target changes by +0.024381480 rad. The
unclipped command is18.335591 Nm, capped at the original18 Nm. The observed
command jump is21.798945 Nm. Neither event is a stage handoff; the elbow has no
measured left-hand panel load during the surrounding60 ms.

The held-target behavior and servo algebra are established by archived values.
The complete internal IK target vectors were not independently recorded, and
the evidence does not identify a singularity or prove that interpolation will
improve physical behavior. The torso target is also following a rapidly
spring-returning measured operator, so some reference motion is real demand.
Interpolation cannot make that demand disappear without lag or another control
change.

## Exact prospective integration seams

For operation, `AcquisitionTeacher.force` solves the complete right-arm/torso
IK at100 Hz, then assigns `self.target` through the motor transmission. Its
500 Hz servo consumes that held value. Retain the complete raw joint-reference
vector at this assignment, interpolate the whole active IK group together
(torso plus seven right-arm/wrist joints when present), reconstruct the full
body reference and run fresh geometric admission before converting it through
the unchanged motor transmission. Do not interpolate the torso alone. Do not
alter finger/contact compensation, stance-QP output, gravity/bias terms or
motor-force caps. Preserve the real pre-activation target at the first sample;
do not seed from the measured joint coordinate and thereby discard preload.

For the separate left-approach study, `LeftPalmContact.update_targets` assigns
the seven-joint IK target after `least_squares`; `apply_forces` consumes it at
500 Hz. The full seven-joint vector is the corresponding seam. Its panel-normal
and hybrid-support force terms remain unchanged. This is a different experiment
from smoothing the operation group and must not be enabled simultaneously in
the first comparison. Mode transitions need their own continuous handoff; do
not silently reset the interpolator when support or another tracker activates.

The untouched native/Isaac public CLI currently exposes neither experiment.

## Standalone primitive and limits

`GuardedIkReference` in `guarded_ik_reference.py` is a pure coordinate utility.
It takes ordered names, a captured initial reference, original joint-box
bounds, motor timestep, and explicit prospective velocity, acceleration and
segment-duration limits. There are no default gains or new physical limits.

At a real IK refresh, supply the newly solved vector to `sample`; call
`sample` at every motor tick thereafter. A single minimum-jerk progress scalar
moves the whole vector between endpoints. The segment duration respects the
exact peak factors15/8 for speed and10/sqrt(3) for acceleration and is rounded
up to whole motor ticks. This preserves the convex joint box without
independent coordinate clipping. New IK solutions arriving during an active
segment replace the queued goal; they do not reset the active segment or use
future information. Rest-to-rest segments meet with continuous position,
velocity and acceleration. Their goal timestamps and queue state expose the
introduced lag.

Every output requires `validate(sample)` to return literal `True` after
checking the complete reconstructed reference against the same-epoch measured
root, mechanism, feet, palm and contact geometry. All existing tolerances must
remain unchanged. This callback is mandatory even for a held reference because
the plant and leaf move. Any clock, joint/rate, duration or geometric rejection
is sticky and emits no new reference. No catch-and-resume using stale geometry
is permitted.

The helper preserves a convex joint box, **not** nonlinear foot/palm/contact
constraints. A test deliberately rejects an interpolated midpoint between two
valid unit-circle endpoints. A production guard must use the real whole-body
FK and authored collision/anatomical geometry; a `lambda: True` test callback
does not qualify a runtime. An optional projection would need to recheck
geometry, rate and acceleration after projection, as the existing coupled
release reference already does. This helper contains no such projection and
cannot silently claim that authority.

## Velocity feedforward must match the actual servo

Preserve measured-state feedback and add reference velocity only in the
opt-in mode. For the current acquisition equation, the matching velocity
coefficient is `damping - bias[:,2]`, not `damping` alone. The torso coefficient
is26 Nm/(rad/s). The left-elbow coefficient is12. `LeftPalmContact` currently
adds `damping * target_velocity` when that optional field is present; a matched
experiment must replace that term, not add another copy of it. All unchanged
bias/gravity, contact and final force-cap logic still applies afterward.

Ten-millisecond smoothing alone is not a satisfactory parameter choice.
A rest-to-rest minimum-jerk segment for the observed torso displacement has
peak reference speed6.002 rad/s and a possible156.1 Nm velocity-feedforward
contribution. The elbow displacement has peak4.572 rad/s and a54.9 Nm
contribution. Those are individual reference terms, not predicted total
torque or delivered torque. Thus adding feedforward can introduce new peaks.

As an illustrative, **unqualified** rate-only constraint,2 rad/s would require
at least30.011 ms for that torso change and22.858 ms for the elbow, rounded to
32 and24 ms at500 Hz; an acceleration bound may lengthen both. These are not
approved experiment parameters. The measured operator is returning around
2.8 rad/s at the torso event, so added delay can readily threaten millimetre
palm/contact tolerances. Select a prospective contract from the complete
logged IK vectors and reference guard margins; do not select it solely to
make a torque plot look smooth.

## Smallest useful next comparison

First capture the actual raw IK refresh vector and final admitted/interpolated
vector, names, exact epochs, reference velocity/acceleration, measured state,
unclipped spring/damping/feedforward/bias/contact terms, cap activity and final
returned motor command in a separately captured baseline. The source005
`motor_targets` producer field is not this internal reference.

Then enable only the operation IK group at a declared phase boundary, with
unchanged earlier commands and a prospectively stated motion/lag contract.
Keep every original final task gate. Compare command-step distributions/cap
counts together with actual joint speed/acceleration, palm/foot tracking,
grasp/anatomical patch validity, stage timing and final success. Test the left
approach separately only after the first result is understood. A recorded-state
algebra calculation can validate the seam and derivative plumbing; it cannot
establish physical smoothness or contact preservation.

The current28 standalone tests check analytic references, common-vector motion,
causal coalescing, exact initial servo algebra, rate/acceleration contracts,
continuous rest endpoints, mandatory per-tick guards, nonlinear constraint
failure, defensive outputs and terminal rejection. No physics is stepped.
