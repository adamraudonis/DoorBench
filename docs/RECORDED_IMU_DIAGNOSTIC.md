# IMU packets match velocities; Isaac poses evolve differently

The recorded Isaac IMU channels match their declared sensor frame and timestamps. The remaining orientation-estimation error cannot be fixed by an encoder reorder, quaternion-sign change, or one-step clock shift.

The independent audit reconstructs the original robot IMU site's orientation and velocity from each actual same-epoch root state and named joint state. Gyro packets agree with the reconstructed angular velocity to approximately 2×10⁻⁸ rad/s RMS. Using actor-origin root linear velocity, finite-differenced site velocity reproduces the Isaac accelerometer to approximately 2×10⁻⁶ m/s² RMS. Query poses, angular velocities, and reordered encoder values exactly equal the independently recorded causal balance/sensor rows.

| Measurement | Native | Isaac with backend dry friction |
| --- | ---: | ---: |
| Gyro-only orientation error, all samples RMS | 0.000593 mrad | 7.751 mrad |
| Existing gravity-corrected error at capture epoch, RMS | 0.04579 mrad | 3.131 mrad |
| Gyro-only error during final 0.5 s, RMS | 0.000734 mrad | 12.434 mrad |
| Existing gravity-corrected error during first 0.1 s, RMS | 0.02705 mrad | 0.28057 mrad |

Native IMU capture timestamps refer to the preceding interval start; Isaac packets refer to the endpoint. The audit obeys each recorded timestamp and separately scores decision and capture epochs. Replaying the current estimator reproduces its archived orientation to less than 4×10⁻¹⁶ rad. Including Isaac's first skipped 2 ms gyro span has negligible impact on the later error. Gravity correction improves the Isaac result; disabling it is not a supported fix.

The discrepancy remains when bypassing the IMU adapter entirely and integrating the actual recorded root angular velocity. Isaac's accumulated final root orientation error is approximately [-11.35, -5.10, -0.72] mrad. Using previous, current, or trapezoidal angular-velocity samples produces nearly the same result. Native endpoint integration reconstructs the root orientation to numerical precision. Isaac's mean pose-increment-minus-endpoint-velocity residual is approximately [1.193, 0.539, 0.073] microradians per 2 ms step. The torso-joint final integral error is much smaller, approximately -0.060 mrad.

These observations isolate the problem to reported velocity versus pose evolution, but do not prove a particular PhysX solver mechanism. The appropriate next test is a small backend fixture recording synchronized transforms and velocities through contact/constraint resolution. Deriving a gyro rate from successive own-sensor orientations would also be a different, explicitly qualified sensor-emulation profile; it must not silently replace the existing observations or inject a world-pose oracle into the actor.

Reproduce with `scripts/dexterous/diagnose_recorded_imu.py --robot XML --calibration CALIBRATION --native NATIVE_ARCHIVE --isaac ISAAC_ARCHIVE --output NEW_DIRECTORY`. It blocks simulator stepping. The [receipt](evidence/recorded-imu-diagnostic-003.json) binds the inputs and all retained diagnostics. Earlier 001/002 acceleration comparisons used COM linear velocity in an actor-origin Jacobian; 003 corrects that scoring calculation. Gyro results were unaffected. The inspected installed Isaac Lab IMU source computes local angular velocity from backend velocities and acceleration from velocity differences; its post-run source hash is retained separately from prelaunch provenance.

An independent review identified that the optional finite IMU-increment residual compared a previous-frame rotation increment with a current-frame gyro vector. Fresh [diagnostic 004](evidence/recorded-imu-increment-frame-004.json) transports that endpoint gyro into the previous frame and retains the old field explicitly as untransported. The main root-integration result already used a fixed world frame and is unchanged, as are exact same-epoch gyro/FK agreement and estimator replay.
