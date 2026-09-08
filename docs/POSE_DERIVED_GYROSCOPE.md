# Opt-in ideal delta-angle gyroscope

`pose-delta-angle-v1` is a separate sensor-producer experiment. The default
`backend-angular-velocity-v1` remains unchanged. No physical robot trial is
qualified by this document or by its offline results.

The motivation is the independently reproduced
[PhysX pose/velocity integration discrepancy](PHYSX_POSE_VELOCITY_FIXTURE.md).
The existing gyroscope correctly reports the backend angular-velocity stream;
that stream does not exactly integrate to the pose that contacts and cameras
see. This profile instead measures the net rotation executed by the robot's own
IMU frame over each completed 2 ms interval.

If `R0` and `R1` are the own-IMU rotations at the interval endpoints, the producer
computes `log(R0.T @ R1) / dt`. This is a finite delta angle expressed in the
IMU's local axes. Its rotation axis has identical components in the interval's
start and end local frames. It reconstructs successive sampled rotations,
including noncommuting motion, without needing any absolute world orientation in
the actor. A constant change of world gauge leaves its output invariant.

It is **not an instantaneous rate measurement**, a repaired momentum state, or a
validated hardware noise model. In particular, numerical pose corrections from
constraint solving appear as finite angular increments. Treating those changes
as observable motion makes the sensor agree with the simulated pose and camera;
it does not make those corrections physically continuous. The official
[PhysX solver description](https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/Simulation.html)
explains why biased position integration and final unbiased velocities can differ.

## Producer and actor boundary

`OwnImuPoseGyroscope` in `doorbench/dexterous/pose_gyro.py` accepts the recorder's
existing one-body tensor view, the expected IMU rigid-body path, the actual robot
body inventory, and the fixed mounting quaternion. It validates the singleton
view against the declared robot hierarchy before each read. Translation is read
as part of the backend transform tensor but is unused. It never calls a physics
writer or step function.

```python
producer = OwnImuPoseGyroscope(
    recorder.imu._view,
    expected_body_path=declared_imu_body_path,
    robot_body_paths=actual_robot_body_paths,
    imu_quaternion_wxyz_body=layout["imu"]["quaternion_wxyz_body"],
    physics_dt_s=0.002,
)
producer.reset_episode(now_s=0.0)  # producer-only baseline after physical reset
# After each completed physical step:
local_rate = producer.observe(now_s=step_end_s)  # float32[3]
```

Only that three-element rate replaces `imu_gyro`. The actor receives no body
pose, absolute orientation, camera pose, object identity, or extra oracle field.
The producer's metadata receipt stays in the evidence/provenance output. The
first actor packet remains invalid at t0. The first delta becomes available at
2 ms. It covers the preceding interval, so its effective midpoint is 1 ms old;
there is no additional transport delay. Missing, repeated or nonfinite producer
epochs and changed body bindings fail terminally until reset.

The emitted sensor layout and report must name this profile, in addition to
saving the producer receipt. This must change a learned checkpoint's bound
sensor-layout fingerprint; old velocity-based data/checkpoints must not silently
be described as delta-angle data. The actor packet keys and physical mechanics
are unchanged. For the first isolated runtime experiment, preserve the balance
controller's existing first-valid-sample skip; changing its initialization is a
separate experiment.

## Detached recorded-state results

The Isaac recorder exposes this experiment through
`--sensor-gyro-profile pose-delta-angle-v1` alongside the existing sensor-layout
and controller arguments. It resets the producer after the physical robot reset,
then reads exactly one completed interval before enqueuing each sensor packet.
The original layout file hash and effective emitted layout hash are both saved;
the latter includes `imu.gyro_profile`. A conflicting explicit profile in an
input layout is rejected. Historical layouts without that field retain their
original default checkpoint fingerprint, while their report explicitly identifies
`backend-angular-velocity-v1`.

The separate `gyro-producer.json` receipt is checkpointed during recording.
`gyro-producer-evidence.npz` contains the reset and subsequent own-body quaternion
reads for an independent numerical audit. This evaluator evidence is not exposed
to the actor. An unchanged pose-estimator or a passed producer unit test does not
qualify the resulting closed-loop physical controller.

The assessment replays only the estimator over frozen native and corrected-passive
Isaac acquisition records. Actual root/body poses are used on the sensor-producer
and evaluator sides. Joint encoders, foot touch, accelerometer and calibration
are retained. The estimator receives the same numeric packet fields; its private
estimation function is evaluated without calling its force method, solving its
QP, advancing physics, or substituting a teacher command.

| Recorded engine / estimator | Attitude error RMS | Estimated palm position RMS | Late-hold palm RMS |
|---|---:|---:|---:|
| Native, original gyro | 0.04579 mrad | 0.18067 mm | 0.09838 mm |
| Native, pose gyro / unchanged estimator | 0.04581 mrad | 0.18079 mm | 0.09841 mm |
| Isaac, original gyro | 3.13062 mrad | 3.82701 mm | 3.48924 mm |
| Isaac, pose gyro / unchanged estimator | 0.08329 mrad | 0.38870 mm | 0.35098 mm |
| Isaac, pose gyro / explicitly integrate first interval | 0.04980 mrad | 0.39932 mm | 0.37055 mm |

The extra first-interval variant is diagnostic only. Removing gravity correction
allows almost exact orientation reconstruction by construction; it is not an
independent accuracy validation and is not adopted. Residual palm error remains
because foot-landmark inference, contact motion and numerical geometry still
matter. These counterfactual errors do not establish physical grasp success or
controller stability under the altered stream.

Native captures are stamped at the preceding interval start, while the current
Isaac packets contain endpoint IMU data. The assessment uses each archive's
actual metadata and evaluates current palm estimates against current poses. It
retains native's 2 ms capture-to-decision delay. The first diagnostic attempt
incorrectly assumed matching timestamp sequences and rejected before scoring;
that failed log is preserved.

## Noise and accelerometer limitations

No random noise or bias is added: this is an ideal sensor profile. A detached
additional float32 own-IMU quaternion export causes about `2.09e-5 rad/s` RMS
rate error on the recorded Isaac sequence (`2.48e-5` during the late hold). This
measures numerical quantization sensitivity, **not** realistic IMU noise or a
calibrated variance. Any later noise/bias experiment needs an explicit profile,
seed and fresh physical evaluation.

The accelerometer stays the original velocity-derived specific-force stream.
A centred pose second difference disagrees with its same-centre acceleration by
0.2063 m/s² RMS overall, 2.3966 m/s² during the first100 ms of settling and
0.1585 m/s² late. Native disagreement is only0.00097 m/s² overall. The pose second
difference itself has an additional 2 ms centre-to-availability delay and
amplifies pose quantization; it is an offline consistency diagnostic, not a
proposed accelerometer replacement. The mixed gyro/accelerometer semantics are
therefore declared explicitly. No claim of an ideal full IMU or cross-engine
sensor parity is made.

## Evidence and next qualification

The compact [assessment receipt](evidence/pose-derived-imu-assessment-004.json)
binds the exact inputs, source and original reports. Full numerical results are
under `/tmp/doorbench-passive-review/out/pose-derived-imu-004`; earlier001–003
attempts remain. The producer tests reject changed/nonrobot body views, bad
rotations and epochs; test reset/sign flips, mounting rotation, world-gauge
invariance, translation independence, noncommuting increments and future-frame
independence. These tests validate boundaries and arithmetic, not dynamics.

The next allowed physical step is a fresh explicitly named reach/acquisition
experiment under this gyro profile, with unchanged TGS32/8, damping/friction,
actuator caps, anatomy gates and original actor boundary. Preserve the original
backend-stream failures and report both mechanics and actual grasp outcome.
