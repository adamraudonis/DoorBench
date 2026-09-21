# Local coupled receiving-palm and release reference

The prospective native release reference addresses two measured failures in
`local-native-release-001`: the receiving left wrist ran out of reach as the leaf
opened, and a fixed-world right hand scraped the moving handle. Release002 fixed
the discontinuous predecessor motor capture but retained those geometric
failures. Neither recording is a qualified release or traversal.

The exact initial state remains the qualified terminal state of
`out/local-native-transfer-003` at 50 s. The source-bound right-hand route is
`out/local-planning/profiled-radial-release-003/report.json`. Its original frozen
geometry audit passes 2,001 samples. The coupled body/hand path first passed an
independent 2,001-sample moving-mechanism screen in
`out/local-planning/coupled-release-support-004`.

The consumed live domain is
`out/local-planning/coupled-release-envelope-006/envelope.json`, bound by its
adjacent `geometry-audit.json`. It passes all 61,338 sampled configurations,
including operator extrema/interior slices and both latch extrema. These are
unstepped geometry checks, not contact-force or physical tracking evidence.

| Reference elapsed time, s | Maximum admitted measured leaf angle, rad |
| --- | --- |
| 0–8.0 | 0.12 |
| 8.6 | 0.16 |
| 9.0 | 0.26 |
| 9.6 | 0.30 |
| 10.4–16.0 | 0.40 |

The upper bound interpolates linearly between the listed knots; the lower leaf
bound is 0.08 rad. The measured operator domain is −0.005 to +0.06 rad, and the
latch domain is ±1 mm. The original operator ≤0.05 rad qualification at measured
rest entry is unchanged. The wider subsequent operator domain was selected
prospectively because the actual capture-only release002 reached +0.050365 rad
during release. Envelope005 failed the combination of large aperture and the
maximum operator angle around 9.44–9.76 s; envelope006 excludes that unreachable
combination by reducing its aperture bound in that period.

Within the final sampled domain, maximum torso tilt is 2.684°, root displacement
12.395 mm, COM displacement 14.691 mm, and foot position error 0.312 mm. The
minimum left wrist physical margin is 8.929 mrad. Every right-hand collider has
at least 4.385 mm clearance from every handle collider once world blending is
permitted. All original collision, anatomy, side-normal, joint and loopback
thresholds remain in the independent audit; the unchanged distal-only anatomy
score is retained alongside the declared volar profile.

## Runtime contract

`StandingWithdrawalTeacher` loads the coupled helper only when its config
explicitly supplies both envelope and independent audit hashes. Default
withdrawal behavior does not import or use it. Admission binds the exact source,
duration, evaluator, domain and original limits. Relative and absolute path
spellings may identify the same hashed file; conflicting aliases are rejected.

The controller still uses the original capped motors at 500 Hz. A separate,
unstepped native model indexes the map with actual measured leaf/operator/latch
coordinates. The left palm stays in the measured leaf frame. The right palm
follows the measured handle through right-route time 3.5 s, then blends to the
screened retreat in the resting world. Reference elapsed time is `t − start`;
the quintic map to the 8.5 s route clock is applied exactly once. Runtime asserts
that this clock matches the teacher's clock.

Only nominal body/joint references are returned. No plant root/joint state or
mechanism coordinates are written. Original receiving contact feedback,
reference offsets, attained motor preload and bounded tracking feedback remain
active. The new nominal reference limiter preserves 1.2 rad/s scalar joint
speed, 3 rad/s² scalar acceleration, 0.02 m/s root translation and 0.03 rad/s root
rotation limits. Discrete braking prevents overshoot at stationary references.
Geometry is checked again after limiting. A rejected update is terminal and is
recorded before any new motor submission; it cannot be caught and resumed.

This is a privileged evaluator/controller experiment. The sampled map does not
establish real force continuity, sensor-only perception, successful release,
wide opening, walking or traversal.

## Prepared experiment and preserved diagnostics

The ready configuration and exact original-prefix command are in
`out/local-planning/coupled-release-runtime-003/withdrawal.json` and
`command.json`. The command selects a fresh `out/local-native-release-003`
directory, retains the original acquisition/operation/transfer recipe, starts
transfer at 36 s and measured-rest withdrawal at 50 s, and runs to 66 s. It adds
the qualified coupled reference, actual predecessor motor capture and isolated
full-orientation left-arm tracking. It does not add Jev or a progress governor.

The archived capture002 observation replay has exact clock agreement but stops
at 57.810 s when acceleration-limited nominal palm tracking exceeds the original
1 mm gate. Its failed receipt remains in
`out/local-planning/coupled-release-runtime-002/capture002-reference-replay.json`.
This is a counterfactual reference evaluation against a different controller's
recording, not a prediction of the new physical trajectory. The bounded physical
experiment must be assessed from its own complete raw recording and independent
contact/support audits.

An actual-controller command replay reproduced the first 2,000 recorded motor
commands through 3.998 s with exactly zero difference, then was explicitly
interrupted to prioritize the physical diagnostic. It is not a completed
controller replay. The interruption is retained in
`out/local-planning/coupled-teacher-replay-002/interrupted.json`.

The focused motion/domain/admission/path/bridge/capture regression suite passes
77 tests. No physical release success is inferred from these tests or screens.
