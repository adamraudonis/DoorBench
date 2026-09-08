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
