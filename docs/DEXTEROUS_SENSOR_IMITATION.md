# Training the sensor actor

September 8, 2026: the data reader, recurrent model and optimizer are implemented. Four boundary/causality tests and a two-update synthetic optimizer smoke pass. **There is no trained or evaluated robot actor result yet.** This is preparation for stage 3 of [the execution plan](DEXTEROUS_NEXT_STEPS.md); the current physical teacher still uses simulator state.

The actor consumes stereo robot-camera pixels, local tactile force bins, joint encoders, local IMU measurements, its previous motor command, and sensor age/validity. Absolute timestamps become relative ages, so the network receives no absolute episode clock. Its recurrent state resets each episode. Extra packet keys, future observations, incorrect dimensions and nonfinite inputs are rejected. The network emits normalized native motor forces; a fixed actuator contract maps these to the original asymmetric effort bounds.

The shared stereo CNN, touch encoder and proprioception encoder feed a GRU. This is a compact baseline for imitation and subsequent correction/RL, not a claim that the architecture solves the task. There is no simulator object, door identity, exact handle pose, door angle, task phase or oracle routing argument in the actor interface.

## Data and reproduction

Record a corrected-hand run with `isaac_opening.py --sensor-layout PATH`. The recorder saves a packet **after** each physics step. Packet `i` contains the force just applied as `previous_action`; the imitation target is the force applied at step `i+1`. Images are selected using their exact recorded acquisition timestamp. A future image cannot replace a missing past frame. Windows never cross an episode boundary; validation must use separate episode archives.

The reader requires a passing acquisition or operation report, passing mechanical audit, eight actual passive-tendon backend readbacks and matching sensor/motor calibration. V1 opening recordings with missing hand mechanics are ineligible. The task report and next-action labels are training metadata; neither enters the actor packet.

```bash
PYTHONPATH=. python scripts/dexterous/train_sensor_imitation.py \
  --episode out/qualified-training-run \
  --validation-episode out/qualified-separate-run \
  --output out/sensor-imitation/run-001 --device cuda:0
```

Use `--qualification acquisition-report.json` for an acquisition-only curriculum. The default is `operation-report.json`. Output includes the frozen sensor layout, source/dependency manifest, episode hashes, progress and `actor.pt`. Training uses truncated recurrent windows with a sensor-only burn-in. The reported loss measures next-force prediction; it is not a task success rate.

Before reporting a sensor-only result, execute the checkpoint in the physical simulator with the privileged teacher disabled, then run the same independent physical and task gates. Record matched seeds, complete failures and camera/tactile ablations. Imitation on successful teacher states alone is expected to suffer from errors accumulated during execution; teacher corrections on student-visited states and recovery training remain required work.

On a different cluster, preserve the checkpoint, sensor layout, actuator ordering, robot asset hashes, physics timestep and exact evaluation protocol. A different robot needs a new validated mechanics/sensor/action adapter and new training data; changing array dimensions alone does not establish a port.


## Same-run initial observation

New teacher recordings capture `sensors/teacher-initial-decision.npz` before the
first physics step. Its packet comes from the same sensor builder used by the
actor, with unavailable streams left invalid. The teacher motor action is a
separate label. The sensor report binds this file's SHA; the demonstration reader
checks time zero, motor bounds and agreement with the first post-step recorded
action before prepending it. Actor-history training can then start from an actual
recorded reset without substituting later frames or another run.

September10: qualified Isaac042 supplies17999 causal examples, but lacks this
initial packet. Its attempted actor-history fit was rejected before training;
no checkpoint was created. Run045 is already immutable and also predates the
capture change. A future physics run must validate the new capture before any
actor-history training claim.32 relevant tests pass; these are not a live sensor
policy result.
