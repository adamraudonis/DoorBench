# Isaac Door55 approach adapter

The native H1 waypoint teacher passed nine nearby-start trials on both the
original and the corrected versioned hand plant. Its Isaac port is
still under qualification. Two-second physics smokes completed on both the
original hand and the corrected versioned hand. The v2 smoke verifies all eight
passive constraints in the live solver. Both runs correctly fail the full
approach/quiet-stop criteria because of their short duration. One full frozen-protocol Isaac approach has completed and failed stopping;
the corrected controller is under test.

This uses the official **H1** actor, the full free-base H1/dual-Shadow robot, and
the complete Door55 scene. The waypoint controller reads privileged world pose
and velocity. This is a component teacher, not a vision/tactile policy and not
an uninterrupted grasp/open/traverse demonstration.

## Run after the ready environment

On the prepared Isaac node, from the repository root:

```bash
/workspace/venv/bin/python scripts/isaac/run_approach_demo.py
```

Add `--matrix` for the same nine starts as the native protocol, or `--record` for
live RTX video. The wrapper requires the receipt from `scripts/isaac/prepare.sh`,
exports the exact native resets with the asset Python, fetches the pinned actor
if missing, and checks the resulting report rather than relying on process exit
status. It leaves a fresh evidence directory for every invocation.

Before a shared-node run, obtain the current owner's GPU slot and keep its
existing teardown deadline. No part of this component provisions a pod.

For an explicit versioned plant:

```bash
export PYTHONPATH=.
/workspace/asset-venv/bin/python scripts/dexterous/export_h1_door_approach.py \
  --robot /path/to/h1-shadow.xml --door assets/doors/db0055_swing_single \
  --reference configs/dexterous/h1-door55-approach-target.json \
  --output out/approach-resets

/workspace/venv/bin/python scripts/dexterous/isaac_door_approach.py \
  --headless --robot-usd /path/to/robot.usda \
  --door-usd assets/doors/db0055_swing_single/door.usda \
  --motors /path/to/h1-import.motors.json \
  --checkpoint /path/to/deploy/pre_train/h1/motion.pt \
  --reset out/approach-resets/case-1-seed-0.json \
  --seconds 25 --output out/isaac-approach-single
```

Use the prepared launch environment (including its EULA settings). Full matrix:
replace the fixture with `evaluate_isaac_door_approach.py` and `--reset` with
`--reset-dir out/approach-resets`. The aggregate retains initialization errors,
timeouts, nonzero exits, and failed reports. Its default per-trial wall limit is
900 seconds; profiling suggests roughly eight minutes for a 25-second trial,
plus initialization. Set a justified `--timeout` within the owned GPU deadline.

## Mechanics and checks

The adapter preserves the imported motor control and force caps, affine-servo
biases, transmission matrix, passive damping/friction and armature. It writes
leg motor efforts every 2 ms. Neutral arm/hand motors retain native reset
controls. The robot is reset once; no root pose or velocity writes, foot
constraints or external supports occur during the rollout. The door receives
no task-controller commands. Its authored native springs and passive latch
coupling remain active.

Every solved 2 ms step checks commanded/delivered torques, joint stops, finite
state, torso tilt and all robot/scene contacts. All non-foot floor, wall, door
and self contacts are retained. Fixed plant parameters are compared before and
after the rollout. Quiet-state metrics use the final second at 50 Hz; the exact
last solved step is also retained. Passing needs target position <3 cm, heading
<2 degrees, horizontal speed <2 cm/s, excursion <1 cm, and both feet supporting
>30 N, alongside the physical checks.

The port verifies that native reset and motor-contract XML hashes match,
and rejects differences in passive tendon names, joint coefficients or ranges.
It then imports `doorbench.dexterous.isaac_tendons.author_passive_tendons`
after stripping obsolete imported tendon schemas and before constructing the
articulation. The versioned hand contract records passive tendons and checks
their joint-difference limits every step, with a 20 mrad numerical tolerance.
Every v2 tendon is read back from PhysX: count, unilateral stiffness and
limits must match, and bilateral spring stiffness, damping, rest lengths and
offsets must be zero. These backend settings are compared again after the
rollout. An empty contract is labeled **upstream-v1**, never silently presented
as a corrected hand. A short v2 whole-body physics smoke has now verified all eight backend
constraints; full approach qualification remains pending.

`trace.jsonl` survives interrupted runs. `report.json`, `manifest.json`,
`landed-state.json` and `pipeline.json` record the result, source hashes, exact
final robot/door state and terminal status. Terminal status is written before
`app.close()` because this Isaac build can exit immediately during shutdown.
A process exit of zero alone does not mean the trial passed.

## Measured development evidence — 2026-09-08 UTC

[Profile report](../results/dexterous/2026-09-08/isaac-door-approach-v1-profile.json)
records the original v1 USD, motor-map, checkpoint and source hashes.

| Run | Result |
| --- | --- |
| Native original nine-start protocol | 9/9 approach + quiet stop; original v1 hand |
| Native corrected-hand nine-start protocol | 9/9; maximum passive joint-difference excursion 0.306 mrad |
| Isaac approach-001 | Launch environment failed before physics |
| Isaac approach-002 | Partial rendered run interrupted; report serialization bug retained |
| Isaac approach-003 | Partial numerical run interrupted to return GPU; no completed score |
| Isaac profile-v1-005 | 2 s physics smoke; approach checks false, as expected |
| Isaac profile-v2-006 | 2 s corrected-hand physics smoke; every physical check passed |
| Original v2 long case0/seed0 | Full 25s; physical checks passed, target/quiet-stop failed |

The smoke stayed upright (2.424 degrees maximum tilt), with zero observed
self/non-foot/scene penetration and 3.76e-6 Nm maximum motor-delivery error.
It used 38.58 wall seconds: 34.56 seconds (89.6%) were spent in physics stepping
and articulation updates. Host-state/contact reads used 1.90 seconds.
The packed readback helper preserves float32/64 values and bounded integer
indices exactly, returns independent arrays, and rejects unsafe integer
rounding; six CPU tests cover its buffer/value contract. Its GPU path ran in
the smoke. This does not isolate a particular PhysX kernel as the bottleneck.

The [v2 smoke report](../results/dexterous/2026-09-08/isaac-door-approach-v2-smoke.json)
adds the explicit tendon backend verification, exact final-step capture and
pre-shutdown terminal marker. It completed exactly 2.0 s with all physical
checks passing, maximum joint-difference excursion 2.36e-5 rad, and maximum
torso tilt 2.445°. All eight live unilateral constraints had the expected
limits and 10000 numerical limit stiffness, with zero bilateral spring,
damping, rest length and offset. It took 41.59 wall seconds. The full nine-start
approach matrix is a separate longer run.

The original imported Shadow hand omitted its documented J1 <= J2 passive
loopback constraint. Existing v1 locomotion evidence is preserved under its
original hashes; it does not qualify the corrected plant. Full Isaac approach
and its matrix should resume on the versioned corrected robot once the
standalone unilateral-tendon fixture passes. Do not retune physics to make the
walking policy pass. See [native approach](DEXTEROUS_DOOR_APPROACH.md) and
[actor provenance](DEXTEROUS_LOCOMOTION.md) for the existing development
protocol and model differences.


## Stopping-controller development

The [first full original v2 Isaac trial](../results/dexterous/2026-09-08/isaac-door-approach-v2-original-failure.json)
completed all 25 simulated seconds and passed its physical gates, but failed
quiet stopping. After its first brake settled about 40 mm away, the teacher
restarted. Its short velocity filter retained the side-to-side motion of an
in-place gait and prevented a second predicted-position brake. Final-second
speed reached0.2224 m/s. This is a teacher failure; Door55 should not be excluded
from evaluation because of it. One complete failed case and the following
interrupted initialization remain archived; the other eight starts were not
scored on that controller.

The optional `--brake-velocity-window .8` uses pelvis displacement over one full
0.8 s gait cycle for brake prediction. Steering retains its original short
velocity filter, actor weights and motor/plant settings. The original behavior
remains the default. A [separate frozen development protocol](../configs/dexterous/h1-door55-approach-cycle-development.json)
passed [9/9 native v2 starts](../results/dexterous/2026-09-08/h1-door55-approach-v2-cycle.json):
worst final XY 20.612 mm and heading 0.708°. Two focused tests verify that the
estimator removes periodic sway, preserves net linear travel, handles uneven
sampling, and does not alias reused simulator state buffers. Its first live
Isaac retry is pending. The matrix runner accepts `--stop-on-failure` so a
systematic failure can stop further spending while retaining the complete case.
