# Running the sensor actor

`SensorPolicyController` loads the `doorbench.sensor-actor.v1` checkpoint written
by `scripts/dexterous/train_sensor_imitation.py`. It consumes only the numeric
`doorbench.sensors.v2` packet and a local acquisition clock. Its output is 61
native motor forces, in the actual import contract's actuator order.

This adapter does not establish policy success. The September 8 qualification
below used random weights and no physical robot episode. A trained checkpoint
still needs independent closed-loop trials with the existing mechanical gates.

## Integration

Load the **actual runtime** motor contract and sensor layout independently of the
checkpoint. Do not copy its layout into the runtime to make a mismatch pass.

```python
import json
from pathlib import Path
from doorbench.dexterous.sensor_policy_controller import SensorPolicyController

controller = SensorPolicyController(
    Path("run/actor.pt"),
    motor_contract=json.loads(Path("runtime/h1-import.motors.json").read_text()),
    sensor_layout=json.loads(Path("runtime/layout.json").read_text()),
    physics_dt_s=0.002,
    image_shape=(128, 128, 3),
    device="cpu",  # a torch device such as "cuda:0" is also accepted
)
controller.reset_episode()
# Reset the sensor observation builder separately at this same episode boundary.

# At each physical tick, after causally available sensors have been delivered:
packet = observation_builder.observe(now_s=local_time_s, previous_action=controller.previous_action)
motor_forces = controller.force(packet, local_time_s)
# Apply these through the existing audited native motor transmission adapter.
# Step the plant once, retaining its independent every-step physical audit.
```

`force(packet, now_s)` has no scene, door, teacher, root-pose, FK, reference-path,
phase, target or success input. The module neither imports a simulator nor applies
forces itself. Local time checks sensor causality and becomes relative sensor
ages; absolute episode time is not an actor feature. The recurrent model may
still learn temporal patterns from observations and its memory.

The caller must apply the returned forces without teacher overrides. The copied
`previous_action` is the normalized command just returned; it is suitable for the
observation builder only when that command was actually applied. If any downstream
controller alters it, record the applied command and declare the intervention.
Do not call the intervention an autonomous sensor policy success.

## Contracts and failure behavior

- Checkpoints load with `weights_only=True`, strict finite tensor weights and an
  explicit architecture schema. There is no unrestricted-pickle fallback.
- Construction checks the v2 hand-mechanics profile, 69 unique ordered joints,
  61 unique ordered actuators, exact source XML hash, physics timestep, complete
  sensor layout/calibration equality, tactile size and image shape. The layout
  includes sensor order, camera poses/profile and touch channel order.
- Forces use every original finite increasing motor range, including asymmetric
  ranges. The checkpoint schema does not duplicate these ranges: they come from
  the actual import whose XML hash must match the checkpoint layout. The caller
  must still verify the live backend realizes that contract, including original
  transmission, limits, passive damping/friction and all eight v2 loopbacks.
- Every plant reset requires an explicit `reset_episode()`. Inference before it
  raises. The first clock may have any nonnegative origin; subsequent calls must
  advance exactly one configured physics timestep. A changed origin does not
  silently reset memory.
- Packets require exactly the declared finite numeric arrays; extra privileged
  fields, object arrays, malformed validity flags, unnormalized prior actions,
  future observations and malformed acquisition times raise. RGB must be uint8.
  Invalid sensor payloads are masked to zero by the shared actor preparation.
- Validation failures leave the recurrent state and clock unadvanced. Propagate
  the failure to the trial runner; the adapter contains no teacher fallback.

## Verification

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_sensor_policy_controller.py tests/test_sensor_actor.py
```

September 8, 2026: **46 tests passed** (42 adapter tests and four existing actor
boundary tests). Cases include unsafe checkpoint-object rejection, ordering and
camera-calibration mismatches, exact reset replay, shifted clock origin, invalid
packet recovery, absent sensor payload masking and each asymmetric force cap.

A separate 1,000-step CPU smoke used the exported native v2 layout (53 tactile
sensors, 1,344 values), 69 joints, 61 motors, two 128×128 RGB images and hidden
size 192. Random weights produced finite forces within every original cap.
Single-thread inference took median **1.37 ms**, p95 **1.99 ms**, max **2.52 ms**
after ten warm-up ticks on the current Mac. This excludes rendering, sensor
delivery and plant stepping; it is not a real-time guarantee or a GPU benchmark.

Local evidence: `/tmp/doorbench-sensor-inference/out/sensor-inference-smoke-001/`
contains the runnable smoke script, exported layout, untrained checkpoint and
report. Native model SHA-256:
`3148cbbefa04e65ae813080275008619a3d4ef94f1ca0887c8f51e79630c3c21`.
Generated weights and assets are not committed. **Robot episodes: 0; task success:
unmeasured.**


## Live Isaac execution

The simulator runner accepts `--sensor-policy-checkpoint actor.pt --sensor-layout actual-layout.json --reset-from-acquisition-path --sensor-objective partial-opening` with the same actual `--robot-usd`, `--door-usd`, `--motors`, frozen reset `--reference`, and fresh `--output` used by the teacher trial. Do not pass `--native-robot`, `--acquisition`, or any teacher/extra feedback option: the runner rejects their combination with actor mode before creating the plant. At reset the recurrent state is cleared; the initial empty sensor packet is marked invalid. Subsequent calls receive only the latest causal numeric robot sensor packet and local time. The renderer follows fixed robot camera mounts. Separate diagnostic cameras do not enter the actor packet.

The actor's 61 force outputs pass through the same original transmission and passive joint damping/friction model as the teacher. Every 2 ms the independent evaluator checks actual motor delivery, joints, coupled tendons, contacts and balance. `actor-report.json` reports the declared acquisition or partial-opening curriculum objective. Neither is a traversal result. A failed run retains its evidence and cannot fall back to the teacher. Unexpected runtime errors preserve the executed physics/sensor prefix. The simulator adapter has been implemented and its mode/packet boundaries unit tested; a trained closed-loop robot trial remains to be run.


The live runner also requires `--reset-from-acquisition-path` and a bound
`--sensor-reset-preflight` receipt. [Build that receipt on CPU](SENSOR_RESET_PREFLIGHT.md)
using the actual reference, motor contract, native models and USD inputs before
launching the actor. Only the resulting validated reset numbers enter reset;
no native physics model is instantiated by the actor execution path. The first
all-invalid sensor packet and its actual bounded force action are saved in
`sensors/actor-initial-decision.npz` before stepping. Periodic sensor checkpoints
are explicitly incomplete; final sensor reports hash both numeric archives.
