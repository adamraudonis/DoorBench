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
