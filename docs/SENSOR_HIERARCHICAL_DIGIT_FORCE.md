# Contact-normal effort with retained posture directions

The [additive force comparison](SENSOR_DIGIT_FORCE_FEEDBACK.md) failed because
fixed position feedback opposed its normal-force correction. This separately
versioned experiment removes that competition in the original motor-effort
space. It preserves the existing sensor-only estimator, static press route,
original transmissions, passive loops, masses, collision shapes and motor caps.
It is still an instance-specific scripted component, not a learned vision policy.

For one digit, let `p` be the original motor vector produced by a unit virtual
force along its calibrated distal-pad normal. The available normal projector
is `P = p p.T / (p.T p)`. No unavailable J1/J2 difference torque is introduced.
The position-feedback motor effort is split into `P @ u_position` and
`(I-P) @ u_position`. Only the first part is replaced as contact control blends
in. All original damping and robot bias compensation remain; the final original
motor cap and previous-command ownership remain in the same underlying adapter.
This Euclidean effort projector is not a claim that inertia and passive contact
dynamics are perfectly decoupled in Cartesian space.

A direct removal of all position effort could erase the already established
preload. At the first 19-second force decision, the controller therefore captures
`f_initial = dot(p,u_position)/dot(p,p)` from its actual sensor encoders and the
fully assembled named joint goal. This is an equivalent motor-effort scalar,
not a measured contact force. It is admitted only in [0,2] N, then held fixed.
The transferred normal request is `clip(f_initial + f_PI, 0, 4)` N. The same
bounded inner PI from the prior force profile remains unchanged (±2 N correction,
2 N/s internal slew; original 2 N/finger and 3 N/thumb local targets).

The blend is the same quintic 19–21 s ramp times the same calibrated positive
local-touch weight. The adjustment added before the unchanged adapter is
`alpha * (p * (clipped_total - f_PI) - P @ u_position)`; the inner controller has
already added `alpha * p * f_PI`. At full blend the normal position contribution
is replaced by the fixed initial effort plus feedback, while orthogonal posture
is unchanged. With no local contact the blend is zero and the original posture
controller remains. The normal request is unilateral; no requested tensile
pad force is hidden in the transfer.

The wrapper intercepts the already assembled joint goals, after local preload
and index coordination have selected them, and before the original capped motor
adapter executes. It never edits a returned force. Invalid input or a failed
underlying decision is terminal. Reset restores the original policy bias and
clears the captured effort. The first 19 seconds are unchanged.

`screen_hierarchical_digit_force.py` performs a detached algebra screen over
all archived decisions. Its receipt binds the controller, profile, robot, motor
contract and original nominal geometry receipts. It checks the normal-transfer
identity, preserved orthogonal motor posture, original force caps at those
recorded poses and unchanged prefix. It does not predict the newly compliant
trajectory or qualify its loaded contacts. Those require a fresh 36-second
physical trial with all existing 500 Hz joint, loopback, collision, contact,
tracking, balance and sustained operation gates. Continuous original opposed
pad qualification remains required throughout 19–36 seconds.

The preserved screen in `out/continuous/sensor-hierarchical-force-plan-001/`
passes over 18,000 archived decisions. Initial FF/MF/RF/LF/TH equivalents are
0.50627/0.56664/0.91154/0.99092/0.25123 N. The maximum tangent algebra residual is
6.94e-17 Nm, normal residual 5.81e-13 N, and no projected command at the archived
poses exceeds an original cap. This is not a physical success result.
Screen attempt 001 encountered a NumPy-boolean JSON serialization error before
writing a receipt; the error log is retained. Receipt 002 passed; receipt 003
adds explicit robot/motor identity fields required by the probe.

For the finite comparison, append these flags to the complete prior digit-force
probe invocation:

```sh
--hierarchical-force-protocol configs/dexterous/sensor-hierarchical-digit-force-v1.json \
--hierarchical-force-screen out/continuous/sensor-hierarchical-force-plan-001/screen-003.json
```

Every prior failed run and every qualified stationary/reach/acquisition default
is preserved. This new protocol does not address the separate actual-base palm
correction or middle-link contact recovery needed by the corrected Isaac grasp.

## Retained finite trial 001

Source `2d09de4d3` completed all 36 seconds / 18,000 steps but **failed 22/28
checks**. Maximum lever rotation was 0.368501 rad; latch retraction 5.316 mm.
All motor caps, original mechanics, joint stops, passive loopbacks, collision
and anatomical patch checks, balance, warning and assistance checks passed.
Tracking, requested excursion, sustained/continuous opposed grasp, route
completion and sustained operation remain failed. Peak motor-coordinate error
was 47.60 mrad against the unchanged 40 mrad gate. There was no invalid loaded
pad patch or unintended hand contact. The unchanged first19-second prefix and
all raw contact classifications/loads were independently verified.

The hierarchy initially improves index loading: its actual qualified normal
load is 1.58 N at21s, 1.60 N at23s and 1.83 N at24s. At25.074s it still has
1.534 N with a permitted distal patch and roughly10mm axial margin. Exact
pre-integration geometry then crosses from7.13µm overlap to4.93µm separation
at25.076s. Both the actual raw collision record and detached geometry show no
index contact during that interval. This is normal separation, not an endcap
escape or invalid thumb/middle-link contact.

The feedback still uses its preceding valid tactile sample at25.076s, as required
by the causal sensor timing. When zero touch becomes available at25.078s, the
instantaneous contact weight switches the normal mode off and restores the
accumulated positional restoring effort. Index FFJ3 changes from+0.17151 to
−0.40140Nm in2ms. The first physical loss occurs before this command change;
the abrupt reversal then causes continued separation and prevents recontact.
The contact-weighted virtual force has a2N step even though the internal force
state still respects its0.004N/step bound. That is a mode-switching defect.

The thumb tracking gate fails earlier, at the25.010s endpoint: THJ5 is0.868993
versus the0.909084rad nominal target, a40.091mrad error. Removing its normal
posture effort allows opposition/orientation drift. A smooth index-loss response
alone cannot erase this preceding failure. The two repairs need separate,
explicit comparisons: bounded contact-mode retention and slew; and a thumb
pressure subspace that preserves its opposition posture, justified by the
original joint axes and reachable normal effort.

The independent hierarchy algebra matches the recorded additional biases to
3.01e-15Nm and preserves orthogonal motor posture to1.39e-16Nm. Exact finger
command reconstruction matches to4.77e-15Nm; actual-step motor delivery error
is zero. Therefore this failure does not arise from missing effort or a runtime
projection mismatch. The endpoint hand image was personally inspected; visual
proximity alone does not demonstrate index contact.

Full evidence is under `out/continuous/sensor-hierarchical-force-001`.
`first-loss-geometry-002.json` keeps every unnamed mesh separately by geometry
ID and records the first failed tracking motor. The earlier receipt's signed-
distance dictionary collapsed unnamed meshes; its distance field must not be
used. Its actual contact list and detached contact reconstruction are unaffected.
The corrected receipt does not modify any physical data.

| Evidence | SHA256 |
|---|---|
| Provenance | `dfcfa5aa95ccaa2f4ea8f4815d7eb163bf8516b5b32c99d700eb1029522f6025` |
| Trajectory | `881dc733a17a15022949d1f0ffa8cee5f5a49afcc28c4244c2c66d27a26bd4fe` |
| Actual-step archive manifest | `673eedff82ec8b28dbccc2970524a712df6dddc390ff16861af7ec7860c2d987` |
