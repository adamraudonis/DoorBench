# A sensor-based locomotion foundation

The complete vision/touch door policy remains unfinished. Learning 61 raw motor
forces from one full trajectory and short corrections still falls quickly.
This component reuses the pinned official Unitree H1 walking network and derives
its orientation inputs from the robot's own IMU and encoders.

The runtime receives no root pose, base velocity, door state, contact identities
or environmental geometry. The private robot model removes articulated torso
motion from the mounted IMU. Native instantaneous samples and Isaac finite
delta-angle samples have separate, explicit timing/composition paths. The only
orientation prior is a declared upright reset, with arbitrary local yaw and XY.
The network's gait oscillator is an internal controller clock, not a task phase.

| MuJoCo component | Duration | Forward displacement | Maximum root tilt | Independent checks |
|---|---|---|---|---|
| Zero body command | 5.000 s | −0.217 m | 2.015° | 12/12 |
| Constant 0.1 m/s forward command | 5.000 s | +0.220 m | 1.975° | 12/12 |

Both use actual physics, original motor limits and passive hand mechanics. Every
submitted force is delivered exactly. The audits replay all 2,500 decisions,
compare the estimated gravity with evaluator-only actual orientation, reconstruct
the gyro, and check warnings and loaded scene contacts. Maximum gravity-vector
error is about 0.0011 and gyro reconstruction error below 3e-8 rad/s.

Zero command **does not hold position**: the robot drifts backward. Forward speed
also differs from the request. These are stable locomotion components, not precise
navigation, learned vision control, acquisition or opening. Upper-body targets
are a disclosed fixed motor posture. RGB and tactile packets are recorded and
validated but do not drive this baseline.

## Reproduce

`TEACHER` identifies the admitted native capture and its exact physical reset.
`H1_POLICY` is the original pinned Unitree checkpoint documented in
[the locomotion module](../doorbench/dexterous/locomotion.py).

```bash
python scripts/dexterous/evaluate_native_sensor_policy.py \
  --teacher-run "$TEACHER" --locomotion-checkpoint "$H1_POLICY" \
  --body-command .1 0 0 --seconds 5 --output out/sensor-forward
python scripts/dexterous/audit_sensor_locomotion.py --run out/sensor-forward
```

The separate five-second Isaac adapter requires
`--sensor-locomotion-calibration`, `--sensor-locomotion-robot` and
`--sensor-locomotion-checkpoint`, together with the frozen reset proof and actual
sensor layout. It rejects simultaneous teacher controls or another actor mode.
The calibration binds only motor posture, constant command, robot/motor identity,
checkpoint identity and the upright assumption. Actual reset encoders are sampled
before the first step; no IMU/touch history is fabricated.

## Actual Isaac result — September 9, 2026, 07:10 UTC

`sensor-locomotion-isaac-002` ran 2,500 physical steps on the L40S using
`backend-dry-v2`. It passed 15/15 runtime physical checks and
[19/19 independent replay checks](evidence/isaac-sensor-locomotion-002.json).
The five-second trial moved 0.187 m in the floor plane, with maximum torso tilt
1.934°. Every replayed motor command and gyro component matched exactly;
maximum gravity-vector error was 2.44e-7 and actual-body FK rotation error was
6.38e-7 rad. No runtime robot pose writes or teacher actions occurred.

The first attempt failed reset-container preflight before physics. The second
used two identical standing-reset rows, as required by that existing container;
no configuration trajectory was replayed. Its coordinator's historical gyro
audit failed because that auditor expects grasp-specific files. A separate
locomotion auditor now consumes the actual locomotion records and stores its
receipt separately; the coordinator failure remains preserved.

```bash
python scripts/dexterous/audit_isaac_sensor_locomotion.py \
  --run "$TRIAL" --robot "$ROBOT_XML" --checkpoint "$H1_POLICY" \
  --calibration "$CALIBRATION" --original-layout "$SENSOR_LAYOUT" \
  --output out/independent-locomotion-audit.json
```

The audit never steps physics and supplies only recorded actor packets to the
controller. It independently reconstructs mounted-IMU rotations, encoders,
gravity estimates and every motor command. Contact and force-delivery checks
remain bound runtime evidence; accelerometer and tactile values are not
independently reconstructed by this audit. This is a stable five-second
component, not a complete door task or a learned vision/touch policy.

The previous grasp's `legacy-tanh-v1` result is a different passive-joint profile.
The next architecture should learn perception, steering and dexterous interaction
above validated locomotion/balance components, with explicit tested transitions.
