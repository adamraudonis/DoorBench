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

## Physical diagnostic and preserved failures

The first configuration and exact original-prefix command are in
`out/local-planning/coupled-release-runtime-003/withdrawal.json` and
`command.json`. The command selects a fresh `out/local-native-release-003`
directory, retains the original acquisition/operation/transfer recipe, starts
transfer at 36 s and measured-rest withdrawal at 50 s, and requested 66 s. It adds
the geometrically admitted coupled reference, actual predecessor motor capture and isolated
full-orientation left-arm tracking. It does not add Jev or a progress governor.

The actual run stopped before the next motor submission at **54.296 s**. The
independent coordinate limiter moved the nominal right foot 1.003737 mm from its
fixed pose, exceeding the original 1 mm gate. The unlimited admitted target had
only 0.139008 mm error. Root rotation had reached the 0.03 rad/s limit and lagged
its desired rotation by 0.982 mrad, while the right knee lagged by 3.798 mrad.
Thus independently clamping a coupled body pose broke the fixed-foot constraint.
The exact failed state is reproduced without physics in
`out/local-native-release-003/coupled-stop-diagnosis.json`.

The recorded physical prefix also contains loaded invalid right-hand patches;
its final-half-second minimum receiving palm-only load is 1.547 N, below 2 N.
The nominal reference failure is not the only failure, and fixing it does not
qualify the recorded release. All original failed contact/support evidence stays
unchanged.

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

## Prospective coupled motion projection

`coupled_motion_projection: fixed-poses-v1` explicitly enables a small constrained
projection of the body reference. Default behavior remains the historical
independent limiter. The projection jointly considers both fixed feet and both
measured-frame palms. It changes root, torso, leg, arm and wrist references only;
finger references and measured mechanism coordinates are not optimization
variables. The independently admitted geometry evaluator and envelope006 bytes
remain unchanged.

An archive comparison executes the exact release003 reference and limiter next
to the current default implementation. All 2,148 accepted target sets, root
poses, motion states, velocities and diagnostics are bit-identical, followed by
the same 54.296 s terminal gate. The receipt is
`out/local-planning/coupled-projection-default-regression.json`.

Each 500 Hz reference step is restricted to the original root speed balls and
scalar joint velocity/acceleration intervals. A private-model Jacobian maps
world-relative rotation-vector derivatives through MuJoCo's quaternion velocity
convention. Two local convex solves minimize pose error while requesting the
original pose bounds with a 1% linearization reserve. Finite approximate solver
results are normalized back into the same motion set; optimizer status is logged
and does not constitute admission. The motion limiter independently rechecks all
rates, then the existing exact nonlinear pose, collision, joint, anatomical,
COM, panel-gap and handle-clearance guards decide whether the reference may be
returned. Any rejection remains terminal. No plant state is set or stepped by
the projection.

The final implementation passes 88 focused tests, including analytic
acceleration integrals, rejected-step atomicity, vector speed balls, projected
velocity history, nonzero-pose Jacobian finite differences, finger isolation,
and finite/nonfinite solver outcomes. The complete closed release003 prefix,
including its terminal failed state, passes all 2,149 reference evaluations:
maximum nominal foot error 0.385 µm and palm error 0.134 µm. The replay is
`out/local-planning/coupled-projection-006/replay.json`; it is counterfactual
reference evidence and does not alter physical contact classifications.

A full replay attempt against the older capture-only release002 remains failed:
at 58.292 s the right-palm orientation error reaches 0.010171 rad, exceeding the
unchanged 0.01 rad bound. Position errors remain at most 0.221 mm. The terminal
diagnostic is preserved in
`out/local-planning/coupled-projection-005/replay.json`; no suffix after that
rejection is claimed to pass. This does not prove mathematical infeasibility or
predict the new controller's physical mechanism motion.

The isolated prospective native004 command is
`out/local-planning/coupled-release-runtime-004/command.json`. Its only controller
configuration change from native003 is the explicit projection option. It
retains the original preload taper, phase timing, map, physical and motor gates,
and 66 s requested duration. The config hash and exact runtime helper hashes are
recorded alongside the command. Preparing the command does not launch physics or
claim a qualified release.
