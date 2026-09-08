# Isaac Door55 approach adapter

The native H1 waypoint teacher passed nine nearby-start trials on both the
original and the corrected versioned hand plant. Its Isaac port is
still under qualification. The only completed live port run is a **two-second
neutral-hand performance smoke**, which correctly fails the approach/quiet-stop
criteria. There is no completed Isaac door-approach result yet.

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

The port imports `doorbench.dexterous.isaac_tendons.author_passive_tendons`
after stripping obsolete imported tendon schemas and before constructing the
articulation. The versioned hand contract records passive tendons and checks
their joint-difference limits every step, with a 20 mrad numerical tolerance.
An empty contract is labeled **upstream-v1**, never silently presented as a
corrected hand. The v2 authoring and whole-body rollout still need live tests.

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

The smoke stayed upright (2.424 degrees maximum tilt), with zero observed
self/non-foot/scene penetration and 3.76e-6 Nm maximum motor-delivery error.
It used 38.58 wall seconds: 34.56 seconds (89.6%) were spent in physics stepping
and articulation updates. Host-state/contact reads used 1.90 seconds.
The packed readback helper preserves float32/64 values and bounded integer
indices exactly, returns independent arrays, and rejects unsafe integer
rounding; six CPU tests cover its buffer/value contract. Its GPU path ran in
the smoke. This does not isolate a particular PhysX kernel as the bottleneck.

The smoke predates three current adapter changes: explicit v2 tendon authoring,
exact-final-step trace capture, and pre-shutdown pipeline finalization. Those
changes compile but are not claimed as live-tested by this report.

The original imported Shadow hand omitted its documented J1 <= J2 passive
loopback constraint. Existing v1 locomotion evidence is preserved under its
original hashes; it does not qualify the corrected plant. Full Isaac approach
and its matrix should resume on the versioned corrected robot once the
standalone unilateral-tendon fixture passes. Do not retune physics to make the
walking policy pass. See [native approach](DEXTEROUS_DOOR_APPROACH.md) and
[actor provenance](DEXTEROUS_LOCOMOTION.md) for the existing development
protocol and model differences.
