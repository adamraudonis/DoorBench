# Run your policy on a G1 in Isaac Sim

This walkthrough runs DoorBench with a physically simulated Unitree G1 and Unitree’s existing locomotion checkpoint. It does not train a new policy. The checkpoint controls the legs; the upper body holds a fixed posture. This first adapter uses state observations; optional video recording does not supply camera observations to the policy. Door movement comes from robot contact or the automatic door’s sensor and motor.

The four cases are integration examples, not a score on the 985-door benchmark. The original [MuJoCo G1 demo](../robot_demo/README.md) uses the same checkpoint and observation adapter, but its results cannot be substituted for an Isaac Sim run.

## 1. Install the simulator

Use Linux with an NVIDIA RTX-capable GPU. This integration targets **Isaac Sim 5.1.0, Isaac Lab v2.3.2, and Python 3.11**. Use the same versions when reproducing the demo; Isaac Lab’s current default branch may require a different simulator.

On a fresh Ubuntu GPU machine, pin the tested source snapshot:

```bash
git clone https://github.com/adamraudonis/DoorBench.git
cd DoorBench
git checkout 85b4a81fe8f28d79ccaab34f730f3d4b1c763c9a
bash scripts/pod_bootstrap.sh
source isaaclab/cloud/env.sh
```

This installs into `/workspace` by default, downloads several large simulator packages, and launches Isaac once. Check for `RUNTIME_CHECK_OK`, `ISAACLAB_IMPORT_OK`, and `ISAACSIM_OK` in its output. See [GPU setup](RUNPOD.md) for details. Do not run this system bootstrap on a machine whose environment you need to preserve.

With an existing matching Isaac Lab installation, activate its Python environment and install the two packages:

```bash
python -m pip install -e .
python -m pip install -e isaaclab
```

All following commands run from the DoorBench repository root using that environment’s `python`. Alternatively, use `/path/to/IsaacLab/isaaclab.sh -p` in place of `python`.

The setup pins the tested utility packages. There is one documented upstream exception: Isaac Sim 5.1 pins FastAPI 0.115.7, while Isaac Lab 2.3.2 pins Starlette 0.49.1, which that FastAPI version does not support. This environment uses FastAPI **0.121.0** with Starlette **0.49.1**. `check_g1_runtime.py` reports that specific exception and rejects other dependency conflicts. The demo does not start an HTTP service.

## 2. Download the original policy

```bash
python scripts/isaaclab/fetch_g1_policy.py
```

The downloader fetches the checkpoint, deployment configuration, and BSD-3-Clause license from Unitree’s pinned commit `276801e46c5d433564f24658bac64f254b7d2d4b`. It verifies SHA-256 checksums and does not overwrite a file with different contents. The files remain in ignored `robot_demo/third_party/unitree_g1_policy/`.

The checkpoint SHA-256 is `cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d`. The robot USD uses Isaac Lab’s `G1_MINIMAL_CFG`, which has simplified collision geometry; its asset path is recorded with every result. This is a different robot asset from the Menagerie model used by the older MuJoCo demo.

## 3. Prepare the four doors

```bash
python scripts/isaaclab/prepare_g1_demo.py --isaac
```

This uses Isaac’s bundled USD library to generate JSON and USD assets without rendering images. For generation in a separate CPU environment with `usd-core`, omit `--isaac`. It writes `out/isaac-g1-demo/assets/demo-suite.json`, which specifies each task and its initial state.

| Door | Test | What opens the door? |
|---|---|---|
| `db0119_swing_single` | Traverse an already-open doorway | Set open once at initialization; no opening credit to the policy |
| `db0990_automatic_sliding` | Approach and traverse | Proximity sensor commands the door’s force-limited motor |
| `db0123_saloon` | Push through passive leaves | Physical contact with the robot |
| `db0705_swing_single` | Attempt a closed latched doorway | Requires manipulation; locomotion alone is expected to fail |

## 4. Run one door, then the suite

```bash
python scripts/isaaclab/demo_g1.py --headless --device cuda:0 \
  --door db0119_swing_single --out out/g1-single

python scripts/isaaclab/run_g1_demo_suite.py --device cuda:0 \
  --seeds 0 --out out/g1-suite
```

Use a new output directory for each suite run. Every case gets a fresh simulator process and a fresh recurrent policy. Add `--video` to save an RGB MP4, or omit `--headless` on a machine with a display. First startup can take several minutes while Isaac downloads the robot and compiles shaders.

The suite creates:

- `summary.json`: successes, policy failures, simulator errors, and exact commands.
- `<door>-seed<seed>.json`: task, final position, opening, fall status, source/checkpoint hashes, simulator versions, and elapsed time.
- `<door>-seed<seed>.trace.json`: observations and robot actions at 50 Hz.
- `<door>-seed<seed>.log`: complete simulator output.

A valid native result requires both a fresh JSON file and a `DOORBENCH_G1_RESULT` log marker, without a Python traceback. An Isaac shutdown exit code of zero alone is insufficient.

The demo calls traversal successful when the robot root reaches `y >= 1.2 m`, stays within the opening width with a 20 cm side margin, and remains upright for 0.5 seconds. Falling terminates the episode. This root-based metric does **not** certify full-body clearance, safe contact forces, or absence of door damage. It is deliberately separate from the full benchmark’s scenario metrics.

## 5. Replace the policy with yours

If your checkpoint uses the same Unitree 47-observation/12-action contract, pass its path with `--checkpoint` and keep the default adapter. For a different contract, create an importable Python module, for example `my_policy.py` at the repository root:

```python
import numpy as np

def make_policy(context):
    # Load your model and create fresh recurrent state here, once per episode.
    names = context['joint_names']
    default = np.asarray(context['default_joint_positions'], dtype=np.float32)

    def act(observation):
        # Replace this posture-only example with your model inference.
        # Map your training joint order to `names`; do not assume array ordering.
        return {'joint_positions': default.copy()}

    return act
```

Then run:

```bash
python scripts/isaaclab/run_g1_demo_suite.py --device cuda:0 \
  --policy my_policy:make_policy --checkpoint /absolute/path/to/your/checkpoint \
  --out out/my-policy-suite
```

`context` provides the robot’s ordered joint names and default angles, checkpoint/config paths, seed, device, physics timestep, and policy timestep. `observation` provides time, base position, base quaternion **wxyz**, base linear/angular velocity in the **body frame**, robot joint positions and velocities, named door joint positions, a body-frame velocity command, and the world-frame goal.

Return exactly one finite vector of length `len(context['joint_names'])`:

- `joint_positions`: absolute angles in radians, followed by the demo’s PD controller at every physics step.
- `joint_efforts`: torques in N·m, held until the next policy update.

Both modes obey the robot asset’s effort limits. The plug-in has no direct door-action channel. The default cadence is **500 Hz physics and PD, 50 Hz policy**. Use `--policy-config` with a YAML deployment configuration to change timestep, decimation, default leg angles, or PD gains. Match your training normalization, quaternion convention, joint order, action scaling, and recurrent-state reset behavior explicitly.

The Unitree adapter builds its original **47-value observation and 12-leg-joint action**. It ignores the door observations. `DoorBench-Open-G1-v0` is a separate training environment with a different observation/action contract; this checkpoint cannot simply be loaded into that PPO task. See [Isaac training integration](ISAAC_LAB.md) for that workflow.

## Scope before scaling up

This demo uses `door_rl.usda`, the canonical seven-joint representation. It covers the four listed integration cases. It does not implement the full catalogue’s lock, credential, latch-coupling, damage, and mechanism callbacks, and should not be used to claim manipulation success across all doors. Use the task environment and validate the relevant mechanism in native PhysX before expanding your evaluation. The full mechanical exports and ongoing [construction review](https://github.com/adamraudonis/DoorBench/blob/85b4a81fe8f28d79ccaab34f730f3d4b1c763c9a/docs/review/mechanical-foundations/README.md) have separate validation requirements. Pet doors remain outside the robotics benchmark.

A locomotion checkpoint is useful for proving the policy-to-simulator connection and exposing transfer failures. It is not a human demonstration or a door-manipulation baseline.

## Recorded Isaac run

Executed on **September 6, 2026**, on a RunPod **NVIDIA L40S**, driver **580.159.03**. Runtime: Isaac Sim **5.1.0.0**, Isaac Lab **v2.3.2** (package **0.54.2**), Python **3.11.15**, PyTorch **2.7.0+cu128**, and NumPy **1.26.0**. Isaac loaded its bundled Warp **1.8.2**.

**3 of 4 selected demo cases passed; zero simulator errors in the final suite.** One seed per case. These selected examples are not an estimate of full-dataset performance.

| Door | Native outcome | Simulated duration | Maximum primary opening |
|---|---|---:|---:|
| Already-open `db0119` | Traversed upright | 8.902 s | 100°; initialized open |
| Automatic slider `db0990` | Sensor opened; traversed upright | 8.902 s | 0.980 m |
| Saloon `db0123` | Contact opened one leaf; traversed upright | 10.348 s | 52.6° |
| Latched `db0705` | Failed; robot fell against the closed door | 10.588 s | About 0.10° |

[Machine-readable results and native receipts](review/isaac-g1/2026-09-06/results.json) include checkpoint, input, runner, robot USD layer, and trajectory hashes. The complete runnable source snapshot is [`85b4a81fe`](https://github.com/adamraudonis/DoorBench/commit/85b4a81fe8f28d79ccaab34f730f3d4b1c763c9a). All **162 generator source files** on the pod matched mechanical revision `2b61dee71d122819c950e7864c010fb3a6f8975e`; see the [source inventory](review/isaac-g1/2026-09-06/generator-source-hashes.json).

The exact final suite command was:

```bash
python scripts/isaaclab/prepare_g1_demo.py --isaac \
  --out out/isaac-g1-demo/native-assets
python scripts/isaaclab/run_g1_demo_suite.py --device cuda:0 \
  --assets out/isaac-g1-demo/native-assets --seeds 0 \
  --out out/isaac-g1-demo/suite-20260906
```

The initial integration attempts exposed an integer/float gain mismatch and a missing ground plane. Both were fixed before the final suite; their diagnostic logs were retained. The canonical USD intentionally omits the floor, so the runner now supplies Isaac Lab’s ground plane. The open-door trial also passed separately before the full suite. No checkpoint tuning or training was performed.
