# Own-sensor data from the continuous teacher

`probe_walking_release.py --continuous-traversal --record-actor-sensors` records finite robot observations alongside the privileged teacher. The recorder reads the live plant and returns no actions. The development run `continuous-opening-traversal-003-sensors` must reproduce the verified dynamics through 127.5 seconds before its task result can be accepted.

Each 2 ms packet contains only joint positions/velocities, local IMU rates/acceleration, finite tactile cells, the previous normalized motor action, and sensor times/validity. Fixed native eye cameras capture 128×128 RGB at 25 Hz. Object poses, door angles, body identity labels, teacher phase and exact gravity orientation do not enter the packet. The teacher still uses privileged state.

Timestamps distinguish the actual capture epochs. Encoders and rendered cameras use the refreshed post-step state; IMU and tactile values retain the preceding dynamics epoch. MuJoCo computes those sensors during forward dynamics before integration, so stamping every value with the post-step clock would misrepresent their age. See the [MuJoCo simulation pipeline](https://mujoco.readthedocs.io/en/latest/programming/simulation.html#forward-dynamics). The explicit layout profile is `native-preintegration-inertial-tactile-v1`.

Numeric packets are saved in bounded, hashed chunks under `own-sensors/`, with an RGB archive and final capture receipt. The recorder's four-step physical fixture passes ordering, validity and forbidden-field checks. A loaded-contact spot check independently reconstructs all tactile cells exactly from saved contact positions/forces and sensor-local frames. The full-stream audit is:

```bash
python scripts/dexterous/audit_native_sensor_capture.py --run "$RUN"
```

It checks every encoder, previous-action and timestamp row against the actual transition archive, and samples independent gyro and tactile reconstruction every 100 steps. It never steps physics. It does not yet independently reconstruct the accelerometer.

This is **not yet a qualified student-training bundle**. The initial decision before the first physics step is absent and explicitly flagged. A training adapter must add audited initial-observation coverage, preserve episode continuity, use packet `i` to label action `i+1`, and require both the task and sensor audits. No retrospective relabeling of the privileged teacher as a sensor policy is permitted. The current full-sequence result remains [the native teacher milestone](CONTINUOUS_NATIVE_TRAVERSAL.md).
