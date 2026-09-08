# Sensor balance with a scripted, contact-free reach

The H1/Shadow robot can execute the first 45% of the frozen Door55 acquisition
joint route while its pelvis/leg controller uses only encoders, local IMU and
local touch. Native trials005 and006 each passed23/23 checks over11 seconds
(5500 physical2ms steps). Their state, velocity and force arrays are bitwise
identical. This is an initialized, stationary reaching component. It does not
acquire the handle, open the door, use vision, learn its joint goals, or qualify
an Isaac port or another reset.

The high-level script explicitly commands30 right-arm/hand/torso joints through
26 original motor coordinates. Pelvis/leg and left-side targets remain fixed.
The controller retains the original physical masses, forces, passive loopbacks,
joint stops and contact geometry. Nothing writes the active root, feet or joints
after reset. The existing stationary and12-joint arm controllers are unchanged.

## Reproduce

Use the prepared native environment and the versioned Shadow loopback-v2 robot.
The output directory must be new. The default is the passing moving-reference
finger-feedback profile, frozen in
[`sensor-reach-balance-feedforward-v3.json`](../configs/dexterous/sensor-reach-balance-feedforward-v3.json).

```sh
python scripts/dexterous/probe_sensor_reach_balance.py \
  --robot /path/to/h1-shadow-loopback-v2.xml \
  --motors /path/to/h1-import.motors.json \
  --door /path/to/db0055_swing_single \
  --reference configs/dexterous/door55-precurl-v2/reference.json \
  --output out/sensor-reach-fresh
python scripts/dexterous/render_sensor_reach_balance.py --trial out/sensor-reach-fresh
python scripts/dexterous/render_sensor_reach_balance.py --trial out/sensor-reach-fresh --hand
```

No GPU is required. The measured native run took36.6 seconds wall time. Renders
replay the attained native states; they are separate from the physical test.
Both body and close-up hand views were inspected for trial005. The fingers stay
above and away from the handle because this deliberately stops before grasping.

## Runtime boundary

`SensorReachBalanceController(robot_xml, motor_contract, sensor_layout,
desired_posture, ...)` validates the authored robot and uses the unchanged
sensor-only estimator described in [SENSOR_BALANCE.md](SENSOR_BALANCE.md).
Its `force(packet, now_s=t, joint_goals=goals)` accepts the exact numeric
sensor.v2 packet, local clock, and separately declared named joint commands.
It returns61 originally capped forces and diagnostic metadata. No world root,
door pose, active simulator, reference object or geometry handle is a runtime
input. A separate robot-only analytic calculator belongs to the estimator.
`reset_episode()` clears the estimator, previous force and accepted goals.
Rejection is terminal until reset. Omitting goals holds the last accepted goal.

The pure `ReferenceReachSchedule` extracts only static named joint columns. Its
clock follows1 second of initial hold,8 seconds through the45% source prefix,
and2 seconds of final settle, using quintic timing and linear source-row
interpolation. Source reference byte SHA256 is
`ff02d42d9000aa6061d317668d3186c3d85ca7b5ee787602707f0d0088775b3e`.
The complete source reference/root is available to the evaluator and offline
scene screen; a runtime adapter should discard it after extracting the path.

Torso yaw is explicitly enabled. Freezing it instead displaces the nominal
palm by232mm at this prefix and405mm at the full original endpoint. The actual
passing torso-yaw excursion was0.37833rad (21.68 degrees), while torso tilt
remained below0.333 degrees. The controller's goal slew bound is1.5rad/s;
sampled route maximum is1.2201rad/s. The older modest-arm test's0.5rad/s limit
was not changed.

## Coupled fingers and tracking

Each non-thumb finger has a single motor controlling the J1+J2 coordinate, plus
the authored passive J1<=J2 loopback. A named joint goal specifies a nominal
split; it cannot independently actuate each half of that sum. The new protocol
therefore measures tracking in the original motor-transmission coordinates for
coupled fingers, and in the independent joint coordinate for independent motors.
Every actual joint angle, joint limit and passive loopback remains audited.
Individual nominal-angle tracking is retained as a diagnostic, not hidden.

For this opt-in feedback profile, right finger controller stiffness is4 times
its original policy coefficient and velocity damping is0.05Nm*s/rad. Damping
acts on target motor velocity minus measured motor velocity. Target velocity is
the difference between accepted joint-command targets divided by2ms; it has no
privileged state input. The original1Nm finger caps remain unchanged and final
capping/previous-action ownership stays inside the balance controller. These
are policy arrays, not active-plant gain or passive damping edits. The default
wrapper constructor retains1x stiffness and zero extra damping unless requested.

## Measured result and retained failures

All files below are local generated evidence under
`/tmp/doorbench-continuous/out/continuous/`; no assets or trajectories are committed.

| Native trial | Protocol | Result |
|---|---|---|
| sensor-reach-balance-001 | Individual-joint tracking, original feedback | Failed at1.014s: interpolation produced a value2.78e-17 below the wrist's exact lower bound. Corrected constant-column interpolation; no bound relaxation. |
| sensor-reach-balance-002 | Individual-joint tracking, original feedback | Failed21/23 despite completing11s safely. Independent passive finger split and motion tests failed. Kept under its original checks. |
| sensor-reach-balance-003 | Motor-coordinate tracking,4x stiffness, damping toward zero velocity | Failed22/23: LF sum transient error0.05025rad. |
| sensor-reach-balance-004 | Same feedback with12s reach /15s total | Failed22/23: LF sum transient error0.04280rad. |
| sensor-reach-balance-005 | Motor-coordinate tracking, damping relative to target velocity | Passed23/23,11s. |
| sensor-reach-balance-006 | Same physical controller/path, explicit static loopback screen and passing CLI default | Passed23/23; exact physical-array repeat of005. |

Final006 metrics:

| Measurement | Observed | Criterion |
|---|---:|---:|
| Maximum motor-coordinate error |0.017165rad|<0.04rad|
| Maximum independent nominal-angle error |0.068824rad|Diagnostic; passive split is underactuated|
| Actual commanded-coordinate excursion |Every requested change>0.01rad exceeds70%|>70%|
| Maximum torso tilt |0.332077 degrees|<=12 degrees|
| Pelvis height |0.870684–0.872467m|0.82–0.92m|
| Joint-stop penetration |2.34232mrad|<=20mrad|
| Passive loopback difference |0.18853mrad|<=20mrad|
| Hand contacts / nonfoot penetration |0 /0m|0 /<=3mm|
| Actual final palm vs nominal offline endpoint |3.30169mm|<20mm|
| Final-second maximum horizontal speed |0.0000951m/s|<0.03m/s|
| Final-second maximum angular speed |0.0002613rad/s|<0.05rad/s|
| Minimum final-second per-foot floor support |257.24N|>30N|
| QP failures / native warnings / external assistance |0 /0 /0|0 /0 /0|

The501-pose static screen uses full compatible collision masks, including
receiving-only shapes. It checks contacts/limits/loopbacks at every sample and
positive hand clearance at101 samples. The minimum sampled hand/environment
gap is6.5423mm, above the declared5mm requirement. This is a **nominal planned
path** screen: the passive split and dynamic body pose differ during execution.
The separate actual2ms contact and anatomy evidence qualifies physical safety;
the static result does not certify5mm dynamic clearance.

`NativeTransitionRecorder` captures the actual interval's force/contact buffer
before any dynamics refresh and records endpoint kinematics separately. Actor
packets use the actual native local IMU/touch sensor outputs with their interval
capture/availability times; endpoint encoders and the exact previous force are
causal. Initial sensor values are invalid and ignored. Final quiet/support tests
use the last full second. The probe freezes all imported project source files,
robot/door/motor/reference/calibration/layout/schedule hashes, reset numbers,
controller inputs,61 forces,69 joint states and raw contact intervals.

Final006 provenance SHA256:
`e31cf347f32297a60b9015aef857538d76da0bae383d37ff44bb2e73ba6bcaba`.
Trial005 body/hand videos are `scripted-reach-body-az150.mp4` and
`scripted-reach-hand-az150.mp4`. The next physical step is an independently
qualified port or a carefully screened later approach segment; this result does
not establish that open-loop goals will handle object variation or touch.
