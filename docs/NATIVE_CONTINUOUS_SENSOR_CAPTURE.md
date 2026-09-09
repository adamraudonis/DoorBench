# Own-sensor data from the continuous teacher

`probe_walking_release.py --continuous-traversal --record-actor-sensors` records finite robot observations alongside the privileged teacher. The recorder reads the live plant and returns no actions. The run `continuous-opening-traversal-003-sensors` reproduced all 63,995 verified physical transitions through 127.990 seconds, byte for byte, and passes 47/47 task and 16/16 independent physics checks.

Each 2 ms packet contains only joint positions/velocities, local IMU rates/acceleration, finite tactile cells, the previous normalized motor action, and sensor times/validity. Fixed native eye cameras capture 128×128 RGB at 25 Hz. Object poses, door angles, body identity labels, teacher phase and exact gravity orientation do not enter the packet. The teacher still uses privileged state.

Timestamps distinguish the actual capture epochs. Encoders use the refreshed post-step state; corrected camera variant v1 explicitly refreshes camera transforms before rendering that state; IMU and tactile values retain the preceding dynamics epoch. MuJoCo computes those sensors during forward dynamics before integration, so stamping every value with the post-step clock would misrepresent their age. See the [MuJoCo simulation pipeline](https://mujoco.readthedocs.io/en/latest/programming/simulation.html#forward-dynamics). The explicit layout profile is `native-preintegration-inertial-tactile-v1`.

Numeric packets are saved in bounded, hashed chunks under `own-sensors/`, with an RGB archive and final capture receipt. The recorder's four-step physical fixture passes ordering, validity and forbidden-field checks. A loaded-contact spot check independently reconstructs all tactile cells exactly from saved contact positions/forces and sensor-local frames. The full-stream audit is:

```bash
python scripts/dexterous/audit_native_sensor_capture.py --run "$RUN"
```

It checks every encoder, previous-action and timestamp row against the actual transition archive, and samples independent gyro and tactile reconstruction every 100 steps. It never steps physics. It does not yet independently reconstruct the accelerometer.

## Camera correction and training admission

The original capture's numeric stream passes **15/15 independent join checks**. Every encoder, action and time row matches the actual physics archive; 640 independent gyro and tactile reconstructions have zero error. Accelerometer reconstruction remains outstanding.

The original RGB must **not** be used for training. Inspection found both an unsuitable field of view and stale camera transforms: refreshing body kinematics alone did not update the camera transform. The separate `continuous-opening-traversal-003-camera-v1` archive uses fixed body-mounted eyes pitched down 45 degrees with a 100-degree field of view, and calls `mj_camlight` before rendering exact recorded states. It adds no physics steps and does not track the handle. Original evidence remains unchanged.

This variant passes **9/9 independent checks**: sampled left/right frames at six epochs reproduce pixel for pixel from raw physical states, and the initial decision matches the actual reset encoders, fixed cameras and first delivered motor force. Initial IMU/touch is explicitly invalid; previous action is zero. These are ideal simulated camera observations, not a new physical rollout.

`NativeSensorDemonstration(run, camera_variant)` in `doorbench/dexterous/native_sensor_demonstrations.py` joins this evidence without impersonating the separate Isaac archive format. It requires all four qualification reports, validates the raw/chunk/image hash chains, checks camera causality, and prepends the audited reset observation. Its 63,995 examples pair the current observation with the next actual motor force; labels and privileged evaluator state never enter actor inputs. The reusable `packet`/`sequence` interface preserves recurrent episode history.

```python
from doorbench.dexterous.native_sensor_demonstrations import NativeSensorDemonstration

episode = NativeSensorDemonstration(
    "out/continuous-opening-traversal-003-sensors",
    "out/continuous-opening-traversal-003-camera-v1",
)
inputs, next_actions = episode.sequence(0, 32)
```

This is teacher imitation data from **one slow native episode**, not a trained sensor policy, varied-start result, Isaac success or generalization score. Full sensor-only rollouts remain required. See [the native teacher milestone](CONTINUOUS_NATIVE_TRAVERSAL.md) and [the remaining plan](DEXTEROUS_NEXT_STEPS.md).
