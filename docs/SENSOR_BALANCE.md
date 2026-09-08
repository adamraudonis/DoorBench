# Sensor-only lowered balance component

The opt-in `SensorBalanceController` holds the corrected H1/Shadow robot in a
constant lowered posture using encoders, torso IMU and local ankle touch. It is
an independent stabilization component. It has not acquired or opened a door,
and native success does not establish Isaac or arbitrary-pose robustness.

## What enters the controller

```python
from doorbench.dexterous.sensor_balance import SensorBalanceController
controller = SensorBalanceController(
    robot_xml, motor_contract, sensor_layout, calibration['desired_posture'],
    physics_dt_s=.002, gravity_correction=calibration['gravity_correction'])
controller.reset_episode()
forces, diagnostics = controller.force(actor_packet, now_s=0.0)
```

Constructor inputs are fixed robot calibration and exactly 69 desired joint
angles. The supplied robot XML must match the motor/sensor model hash. Native
motor gain, bias, control/force caps, joint ordering and tactile sites/bins must
agree. The frozen calibration is
[`sensor-balance-v1.json`](../configs/dexterous/sensor-balance-v1.json); its motor
identity is the canonical `motor_contract_fingerprint`, not the JSON file hash.
The calibration takes only the first hands-away joint posture from the actor006
reference. It excludes its world root pose, the door, and every later motion row.

Runtime input is the exact numeric `doorbench.sensors.v2` packet plus a local
2 ms clock. It contains no simulator object, root/body/door pose, contact object
identity, teacher state/action, phase or target. RGB is validated but unused.
The previous-action field must equal the preceding submitted force divided by
its original maximum absolute cap. Force returns a `(61-forces, diagnostics)`
pair. Diagnostics are recorder output, not additional actor inputs.

The cold t=0 packet has every stream invalid. Its command comes only from fixed
upright calibration and desired posture. Afterwards encoders, IMU and tactile
must be valid and at most 6 ms old. Any rejected call makes the episode terminal;
call `reset_episode()` before trying another episode. After 50 ms, both ankle
grids must report at least 5 N force norm to support the stationary assumption.

## Feedback and state estimation

A separate **robot-only model is never stepped**. It supplies calibrated FK,
Jacobians, mass and gravity terms. Initial yaw/XY are arbitrary local coordinates;
upright is an explicit initialization assumption. Pelvis height comes from the
calibrated foot meshes at the desired angles, not a recorded root height.

The torso gyro propagates IMU orientation. Accelerometer gravity correction is
weak and used only near one g. Encoder FK removes the moving torso-to-pelvis
transform and its angular velocity. Encoder FK and ankle tactile weighting
estimate root position/velocity under a no-slip two-foot hypothesis. Neither
true root pose nor true foot poses enter the estimator.

A 100 Hz inverse-dynamics QP uses that estimated robot state and finite sole
support polygons. It emits ten leg motor forces. Constant upper-body posture
feedback and gravity compensation run at 500 Hz. Original 61 motor forces are
capped every step. The foot support equations exist only inside the calculator:
there are no physical anchors, extra wrenches, pose writes or changes to masses,
joint limits, passive hand loopbacks or collision shapes.

This first component assumes a known upright, supported start and fixed arms.
It has no slip-recovery, flight, walking, arbitrary-orientation initialization or
moving-arm qualification. Sensor noise, delays beyond the tested native timing,
and sustained external contact remain untested. A noisy IMU can drift in yaw.
The QP is not a new source of privileged state: it uses only the sensor-derived
estimate, with its external-contact generalized force explicitly zero.

## Reproduce the physical test

```bash
python scripts/dexterous/probe_sensor_balance.py \
  --robot /path/to/h1-shadow-loopback-v2.xml \
  --motors /path/to/h1-import.motors.json \
  --door /path/to/db0055_swing_single \
  --reference /path/to/actor006/frozen-source/reference.json \
  --calibration configs/dexterous/sensor-balance-v1.json \
  --seconds 5 --output /fresh/path/balance-001
```

The evaluator alone owns the full scene and resets it to the same actor reference.
It verifies the reference's first joint posture matches the frozen calibration.
The evaluator changes only the motor *command representation* at reset to
explicit force, preserving native transmissions, passive dynamics and caps.
There are no subsequent pose writes. Native actual-step IMU/touch samples are
captured before any dynamics refresh, timestamped at interval start and made
available at interval end. Encoders are captured at the integrated endpoint.
The packet builder provides the same clipping/validity/cold-start contract.

Every 2 ms, separate physics evidence checks actual delivered actuator force,
all joint and eight passive-loopback bounds, nonfoot penetration, both hands
away, original plant arrays, all native warnings, upright torso, lowered height
and no external assistance. Final quiet/support checks cover the final second.
The 12° tilt, 20 mrad soft-limit and 3 mm nonfoot allowances are existing
benchmark-development tolerances, not claims about physical hardware accuracy.
The probe stops on a 15° tilt/.65 m pelvis safety bound and keeps failed prefixes.

`actual-transitions.jsonl.gz` preserves the actual contact-force interval and
matching pre-integration geometry. `physics.jsonl.gz` stores endpoint mechanics
separately; `actor-inputs.npz` freezes only numeric sensor packets and commands.
Sources, exact reset, calibration, input hashes and trajectory are retained.
No generated evidence is committed to git.

Run boundary tests with `pytest tests/test_sensor_balance.py`. For authored-model
tests on another host, set `DOORBENCH_H1_V2_XML` and
`DOORBENCH_H1_V2_MOTORS`; the pure API/calibration tests run without those assets.

## Native measured results

All results below are stationary components using the actor006 hands-away reset.

| Trial | Duration | Initial disturbance | Outcome | Maximum tilt |
|---|---:|---|---|---:|
| 001 | 1 s / 500 steps | None | 19/19 checks | 0.3112° |
| 002 | 5 s / 2,500 steps | None | 19/19 checks | 0.3114° |
| 003 | 5 s / 2,500 steps | +0.05 m/s world-X velocity at reset | 19/19 checks | 0.3911° |
| 004 | 5 s / 2,500 steps | None; stronger admission checks | 19/19 checks | 0.3114° |
| 005 | 5 s / 2,500 steps | Final source, frozen joint calibration | 19/19 checks | 0.3114° |
| 006 | 5 s / 2,500 steps | Final source, +0.05 m/s world-X reset velocity | 19/19 checks | 0.3911° |

Trials005/006 retest the final source after calibration binding and terminal
rejection changes. Trial005 provenance SHA256 is
`c940afa6399ad64b94e4539f3cea8c02e66497fc05c948b57f253f64214f82f6`;
trial006 is
`f88b567797dd41925e086606cfa0a0bc7c9a35f206620d0278c6f710eade4ed1`.
The local evidence root is `/tmp/doorbench-continuous/out/continuous/`, with
`sensor-balance-NNN` directories. All six trials
are retained; none failed the declared physical checks. The 25 boundary tests
include deliberate bad-calibration, privileged-input, stale-sensor and failed
support controls, each of which must be rejected.
Trial002 stayed at .87068–.87247 m pelvis height. Maximum original joint-stop
penetration was .116 mrad and passive-loopback difference .211 mrad, with zero
nonfoot penetration and no hand contacts. Final-second horizontal speed stayed
below 0.0000357 m/s; minimum actual per-foot floor normal load was 252.16 N.
An independent evaluator-only comparison found at most .00210° discrepancy
between the sensor-derived attitude and the withheld true root attitude.
These findings do not establish a working sensor-only door-opening policy.

## Inspect and independently replay the evidence

```bash
python scripts/dexterous/audit_sensor_balance_estimate.py \
  --trial /path/to/balance-005 --output /fresh/path/estimate-audit.json
python scripts/dexterous/render_sensor_balance.py --trial /path/to/balance-005
```

The audit replays all 2,500 decisions using only the frozen packets and clock.
Trial005 reproduced motor forces and estimated state exactly (maximum numerical
difference zero), with calculator time still zero. The true root is separately
used only to measure estimator error: maximum orientation error .002095° and
position error 1.736 mm; final errors .000325° and .259 mm. The initial vertical
offset follows the soft ground contact versus calibrated sole-plane height.
The replay checks exact controller-source identity and records all input hashes.

The video renders recorded native states; it does not resimulate or qualify
contact forces. Beginning/middle/end images were personally inspected: torso
upright, both feet on the floor, and both hands away from contact. No rendering
or annotation enters the controller sensor packet.
## Shared runtime and independent Isaac evaluation

`SensorBalanceRuntime(robot_xml, motors, layout, calibration_json)` validates the
frozen calibration and delegates to the same sensor controller. Call
`reset_episode()` before the first command, then `force(packet, now_s)` every 2 ms.
It returns 61 original bounded motor forces. `previous_action` is the normalized
last command; `last_info` is detached controller diagnostics. A rejected
inference ends the episode until an explicit reset. It accepts no simulator,
root pose, object state, target trajectory or learned checkpoint.

The calibration schema is `doorbench.sensor-balance-calibration.v1`.
It binds the robot XML SHA256, canonical motor-contract fingerprint,
exact69-name `desired_posture`, timestep, gravity correction and upright
local gauge. Source provenance identifies the constant joint posture only.
Additional root/door coordinate fields are rejected.

`evaluate_sensor_balance(rows, physics_checks)` is evaluator-only. Each actual
post-step row supplies `time_s`, `root13_actororigin` (xyz, wxyz, world linear
and angular velocity), `torso_tilt_deg`, two floor-only `foot_floor_loads`,
`hand_contact_count`, and the preceding command's `controller_info`.
It requires2500 uninterrupted2ms steps, height0.82–0.92m, tilt≤12°, no hand
contact or QP failure, an unstepped calculator and the cold first command.
Throughout the final second, root XY speed must remain<0.03m/s, angular
speed<0.05rad/s and both floor support loads>30N. Every supplied original
physical check must pass. Missing, malformed, short or gapped evidence
fails. This stationary protocol claims neither acquisition nor door opening.

The [adapter receipt](evidence/sensor-balance-runtime-001.json) replays all2500
actual sensor packets from final-source native005 with **zero motor-force
difference**. The independent evaluator passes that original run under all 12
grouped checks while retaining its 19 underlying physical checks. Replay steps
no physics and does not constitute an Isaac result. The adapter/controller
boundary suites pass 64 tests. The canonical calibration SHA256 is
`d98e7c1a84ca463e4023ef588aed178e5b8d0d909522a75779313768d965cf72`.

The shared Isaac runner accepts `--sensor-balance-calibration` and
`--sensor-balance-robot` for a separate five-second stationary experiment.
It requires the same bound contact-free reset and sensor layout as the learned
actor tests, and rejects teacher control options or a simultaneous checkpoint.
`balance-steps.json.gz` contains every actual post-step actor-origin pose and
velocity, floor-only loads, hand contacts and preceding controller diagnostics.
`balance-contacts.jsonl.gz` retains the occupied raw contact slots separately.
The scoped result is `balance-report.json`; its acquisition diagnostic remains
incomplete. RGB is recorded for inspection and is unused by this controller.
