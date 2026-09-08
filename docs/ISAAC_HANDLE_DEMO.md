# Isaac: H1/Shadow uses a lever and opens a door

This development demo uses a free-base H1 with Shadow Hands in **Isaac Sim 5.1 / Isaac Lab 2.3.2, CUDA PhysX on an L40S**. The hand turns the lever, retracts the passive latch, pushes the leaf, releases and withdraws while the robot remains standing.

**Scope:** one door (`db0055_swing_single`), a supplied near-handle initialization, and a privileged controller using exact simulator state. This is not approach, traversal, a vision/tactile-only policy, human ground truth, or an all-door benchmark. The robot starts with the hand placed around the handle; it must establish loaded opposing contacts before turning it.

[Watch the wide recording](https://github.com/adamraudonis/DoorBench/releases/download/isaac-h1-handle-20260908/isaac-h1-handle-wide.mp4) · [Hand close-up](https://github.com/adamraudonis/DoorBench/releases/download/isaac-h1-handle-20260908/isaac-h1-handle-closeup.mp4) · [Traces, audits and source snapshots](https://github.com/adamraudonis/DoorBench/releases/download/isaac-h1-handle-20260908/isaac-h1-handle-evidence.zip)

The wide and close-up videos are separate live repetitions, not synchronized views of one trajectory.

**Repeatability issue found September 8, 08:25 UTC:** a subsequent sensor-enabled run failed the unchanged final-opening gate, ending at 25.7°. The teacher starts releasing at about 0.257 rad and withdrawing near 0.49 rad, before the required 0.7 rad opening. The historical successful runs relied on later door motion. A maintained-contact panel-push continuation is under development; it is not yet a verified replacement. [Retained failed run](../results/dexterous/2026-09-08/isaac-h1-manipulation-camera-capture.json).

## Run it

First launch the [ready Isaac environment](ISAAC_ONE_CLICK.md). On that node, from the checkout recorded in `connection.json`:

```bash
source isaaclab/cloud/env.sh
python scripts/isaac/run_handle_demo.py
```

The command uses the freshly imported robot and generated door from the readiness receipt, validates the configured reset/workspace, records a ten-second live simulation, and runs the opening audit. It exits unsuccessfully if a check fails. No old checkpoint, private input pose or manual asset transfer is required.

For an independent repetition with a close-up camera:

```bash
python scripts/isaac/run_handle_demo.py --view hand
```

Outputs are timestamped under `out/isaac-demos/`: `live-isaac.mp4`, `trace.json`, `configuration.json`, source snapshots, provenance and `opening-audit.json`. The initial-state/workspace configuration is [isaac-door55-reference.json](../configs/dexterous/isaac-door55-reference.json). It is specific to this robot and door.

## What the controller does

A bounded motor controller holds four finger pads against one side of the lever and the thumb against the other. It waits for measured opposing contact forces before depressing the operator. A whole-body inverse-dynamics controller supplies leg motor commands for standing; an eight-joint arm/waist workspace keeps the palm reachable during the turn and push. The opening command reduces its lead when measured hand reaction force grows.

Release holds the hand's own pose relative to the leaf instead of chasing the spring-returning lever. The digits unload before the wrist withdraws. This mattered: earlier releases touched the panel or caught the lever again and pulled the door back.

Only bounded robot motor torques move the plant. There are no runtime robot pose writes, fixed-base supports, grasp welds or direct opening forces on the door. The latch linkage is passive. MuJoCo supplies independent analytic kinematics and inverse dynamics; **Isaac alone advances physics**. Rendering is decoupled from the 2 ms motor timestep and checked against the simulator clock.

## Evidence and limits

| Trial / view | Started (UTC, September 8) | Final opening | Maximum torso tilt | Physical audit |
|---|---|---:|---:|---|
| 038 / occluded camera diagnostic | 03:56:14 | 72.1° | 1.73° | 18 / 18 passed |
| 039 / hand close-up | 04:00:52 | 60.3° | 1.69° | 18 / 18 passed |
| 040 / packaged command | 04:06:56 | 95.0° | 1.76° | 18 / 18 passed |
| 041 / final wide recording | 04:14:57 | 95.0° | 1.68° | 18 / 18 passed |

All four maintained five-digit opposition throughout the measured lever-turn window (handle above 0.2 rad, leaf below 0.03 rad). The last two used newly generated assets and fresh imports from separate desktop launches. [Machine-readable results](../results/dexterous/2026-09-08/isaac-initialized-opening.json).

Every qualifying trace must pass 18 checks, including initial closed state, lever travel above 0.8 rad and bolt retraction above 11 mm before opening, loaded five-digit opposition, native motor force limits, actual motor-command delivery, verified PhysX contact materials/offsets, standing, timing, and a final leaf angle of at least 0.7 rad (40.1 degrees). Wide and close-up inspection supplements these numeric checks.

Development repetitions use the same initial state. Their different final angles are a reminder that contact-rich GPU simulations are not guaranteed bitwise identical. These few runs do not measure robustness to new poses, objects or robots. Failed attempts remain in the [experiment ledger](DEXTEROUS_EXPERIMENTS.md#isaac-import-and-controller-development-september-78).

The next policy milestone is acquisition from outside the supplied grasp, followed by unsupported traversal. Replacing privileged state with camera and finite-resolution tactile observations requires a separate student policy and held-out evaluation; this demo does not establish that result. For another robot or cluster, repeat the [migration gates](DEXTEROUS_REPRODUCTION.md#isaac-sim--massive-node-migration-gates).
