# Sensor-balanced scripted lever press: first candidate failed

A separate30-second candidate extended the qualified19-second acquisition with
an8-second static joint-space press and3-second settle. Native operation001
**failed** and stopped at25.780 seconds. Acquisition was reproduced exactly,
but the position-held grasp lost contact and loaded the wrong thumb surface
before reaching the release angle. The original acquisition remains qualified;
this operation does not.

## Reproduce and inspect

```sh
python scripts/dexterous/plan_sensor_handle_press.py \
  --acquisition out/qualified-sensor-acquisition \
  --independent-audit out/native-independent.json \
  --output out/press-plan-fresh
python scripts/dexterous/probe_sensor_handle_operation.py \
  --robot /path/to/h1-shadow-loopback-v2.xml \
  --motors /path/to/h1-import.motors.json \
  --door /path/to/db0055_swing_single \
  --reference configs/dexterous/door55-precurl-v2/reference.json \
  --press-plan out/press-plan-fresh/plan.json \
  --output out/sensor-operation-fresh
python scripts/dexterous/audit_sensor_acquisition_contacts.py \
  --trial out/sensor-operation-fresh \
  --output out/sensor-operation-fresh/independent-contact-audit.json
python scripts/dexterous/render_sensor_handle_operation.py \
  --trial out/sensor-operation-fresh --hand
```

Local evidence is `/tmp/doorbench-continuous/out/continuous/`:
` sensor-press-plan-001/plan.json` and `sensor-handle-operation-001/`.
The hand replay is `scripted-operation-hand-az150.mp4`; failure details are in
`failure-reduction.json`. Input robot/door/motor/reference assets are the same as
[SENSOR_SCRIPTED_ACQUISITION.md](SENSOR_SCRIPTED_ACQUISITION.md).

## Protocol and boundary

The offline planner reads the attained acquisition state, fits8 torso/arm/wrist
joint angles to rotate the palm rigidly with the known lever to0.85rad, and keeps
the attained base/feet/finger geometry fixed. It screens121 actual-shape poses
against the original distal-pad surface and collision criteria. A constant
measured arm tracking offset makes the first command equal the previous
acquisition command. This is an instance-specific geometric hypothesis, not
loaded-grasp qualification.

`ScriptedHandleOperationSchedule` receives only the static named-joint arrays
and durations. It retains every acquisition command through19 seconds and holds
finger goals constant during the press. The sensor balance controller receives
only this joint command, encoders, local IMU, local touch and clock. No actual
lever angle, world pose, contact object ID or inverse-kinematics result is passed
at runtime. A separate evaluator stops execution if the original0.5-second
opposed-pad acquisition gate fails at19 seconds. No plant reset occurs there.

Original robot forces, masses, joint/coupling limits and contacts are unchanged.
The planned operation pass requires the actual lever>=0.8rad and latch>=11mm
through the final0.5 seconds, together with the original distal-pad grasp,
all-episode anatomical contact checks, complete physics evidence, and balance.
The old acquisition palm position is retained only as a displacement diagnostic
after pressing; it is no longer an operation target criterion.

## Actual failure sequence

| Event | Time / measurement |
|---|---|
| Qualified acquisition prefix |0–19s; all state, velocity and61-force arrays bitwise identical to acquisition001|
| Original acquisition handoff audit |12/12 checks passed;9500 physical steps|
| First lost opposed grasp |19.626s; middle-finger pad unloaded; actual lever0.00141rad|
| First invalid loaded surface |22.546s; thumb middle segment about17.48N total; actual lever0.08919rad|
| Safety stop |25.780s; palm/lever contact60.65N at1.559mm penetration|
| Maximum actual lever / latch |0.53765rad /7.804mm, below required0.8rad /11mm|
| Maximum torso tilt |0.46605 degrees|
| Maximum motor-coordinate error |0.03370rad, below0.04rad|
| Maximum joint-stop / loopback violation |2.040 /0.418mrad|
| Original motor caps / external assistance / QP failures |Preserved /0 /0|

The robot remained upright. Static rigid-grasp planning did not predict the
loaded finger/lever interaction: the lever lagged the commanded hand trajectory,
and the contact pattern deteriorated. Low initial pad force and passive finger
redistribution are plausible contributors; this one trial does not isolate them.
The next declared test uses only calibrated local distal touch to preload the
grip and stop route progression on pad unloading. It must retain the original
anatomical and operation gates and cannot treat static planning as physical
qualification.

Operation001 provenance SHA256:
`f1690375deb969b0be0c749e63cacd5cf2987990f251a3698351536b4af17323`.
The acquired prefix, loaded-contact failure and all force/contact records are
retained. The independent contact reducer also handles interrupted failures;
an absent requested final window is explicitly failed rather than treated as
an empty successful hold.

New press plans use schema v2 and bind the exact acquisition calibration, schedule, motor contract, initial velocity, gravity compensation and controller source closure. The independent acquisition audit must hash-bind the physical trajectory and controller evidence. Changing tactile preload or another acquisition setting requires regenerating the plan. Historical v1 attempts remain archived; they are not admitted by new operation runs.
