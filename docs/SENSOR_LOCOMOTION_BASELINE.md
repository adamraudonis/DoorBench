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

## Walking to a quiet stance

`--stop-after-s 3` selects a separate ten-second native component. It requests
braking at the next appropriate gait phase, sets body velocity to zero and
reduces the H1 oscillator amplitude over one second. Only after three seconds
of braking, sustained local foot touch and estimated speed below 0.02 m/s does
it transfer to the sensor balance controller. The new stance initializes only
its private unstepped estimator and motor targets from encoders and the prior
IMU estimate. No active robot pose is written. It is a timed stop experiment,
not visual approach control.

Trial001 failed after one stance interval: fleeting two-foot support did not
establish a valid stop. Trial002 never transitioned because its walking gait
continued. Both are preserved. Trial003 included gait braking and passed
[18 independent checks](evidence/native-sensor-walk-stop-003.json) over 5,000
physical steps: 0.133 m forward travel, stance handoff at 6.582 s, maximum root
tilt 1.976°, final-second speed below 0.000051 m/s and minimum actual foot
support 213.65 N. Every motor command replayed exactly with no QP failures.
Original joint-stop and loopback development tolerances remained unchanged.

The Isaac adapter uses `--sensor-locomotion-stop-after-seconds 3 --seconds 10`
with the same bound locomotion calibration and checkpoint. It has separate
handoff/quiet-state checks. The actual `sensor-walk-stop-isaac-002` trial completed on September 9, 2026 at
07:36:32 UTC. It passed 18/18 runtime checks and
[22/22 independent checks](evidence/isaac-sensor-walk-stop-002.json).
It moved 0.0883 m in the floor plane, handed off at 6.494 s, and stayed below
1.950° torso tilt. All 5,000 motor commands and gyro components replayed exactly.
The original `backend-dry-v2` passive profile and all physical checks remained
active. The first launch failed before physics on an older system Python;
using the pinned environment interpreter resolved setup without changing the
controller. Its failed receipt is retained.


Further lowering experiments remain unqualified. A first native descent reached
0.872 m but rotated about 24°; its older height/balance audit did not check
heading. A stricter heading objective held orientation but failed to descend.
A wider-stance attempt lost support and accumulated QP failures. Robot-only IK
posture guidance is under development with original joint bounds and explicit
foot-frame residual checks. None of these results qualify handle interaction.

The optional native lowering flags (`--lower-to-m`, `--stance-yaw-weight`,
`--hip-spread`, `--stance-leg-path`) are explicit development experiments.
Seven attempts have not established a qualified lowering transition. The
robot-only leg planner tests original joint bounds and foot-pose residuals;
optimizer convergence is disclosed separately from geometric feasibility.
A geometrically valid wider-stance path still lost support during actual
execution. These options do not change the default qualified stopping protocol.

A standing grasp alternative is now promising: the original 401-pose hand route
can be fitted at the measured standing height by using the existing torso-yaw
joint. It passes a separate 1,001-pose full-scene collision screen. A native
physical trial using preserved actual foot frames passed 12 grasp checks, and
a fresh trial with full transition recording passed 14, including every-interval
stance solves and warning checks. Peak torso tilt was 0.4673° and final palm
tracking error below 0.577 mm. Body and close-up hand views were inspected.
This is privileged acquisition from a pregrasp reset, not continuous walking,
opening, independent replay qualification or a sensor-only hand policy.

Reproduce candidate generation from a qualified standing capture:

```bash
python scripts/dexterous/prepare_standing_acquisition.py \
  --stance-run "$STANDING_RUN" \
  --reference configs/dexterous/door55-precurl-v2/reference.json \
  --output out/standing-candidate
python scripts/dexterous/rescreen_acquisition_reference.py \
  --robot "$ROBOT_XML" --door "$DOOR_DIR" \
  --reference out/standing-candidate/reference.json --output out/standing-screen
python scripts/dexterous/probe_acquisition_teacher.py \
  --robot "$ROBOT_XML" --door "$DOOR_DIR" --motors "$MOTORS" \
  --reference out/standing-screen/reference.json --stance-profile landed-foot-v1 \
  --record-transitions --seconds 10.6 --output out/standing-physical
```

Use the pinned environment interpreter on the GPU; its system Python does not
provide the same hashing/runtime APIs. Rescreen and physically test against the
exact destination robot bytes before claiming cross-engine results.

The complete recorded native standing grasp also passes the
[independent contact audit](evidence/native-standing-grasp-004.json): all 5,300
intervals agree with recomputed pad classifications and forces, no loaded patch
uses the wrong surface, and the opposed hold lasts 3.522 s through the endpoint.
Final-half-second minimum forces are 1.900/1.545/1.226/1.682 N on the four finger
pads and 7.460 N on the opposed thumb. This verifies the standing grasp, not an
opening. The first standing operation trial retracts the latch but fails its
held partial-opening criterion; bounded compliance compensation is being tested.
