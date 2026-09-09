# Start a ready Isaac environment

Double-click **[Start Isaac Sim.command](../Start%20Isaac%20Sim.command)** on macOS, or run:

```bash
python3 scripts/isaac/launch.py
```

The launcher opens a local Run Center, provisions or reuses its own RunPod allocation, installs the pinned runtime, prepares the H1 robot with Shadow Hands and one development door, and runs live physics checks. A fresh installation downloads several gigabytes and can take over 90 minutes on network storage; an installed runtime is reused. **Environment ready** means the checks passed, not that a door-opening policy has succeeded.

On the development Mac, the installed shortcut is **Start DoorBench Isaac.command** on the desktop. Its source lives under `~/Library/Application Support/DoorBench/Isaac Launcher`, independently of temporary development worktrees. The desktop launch and a cached restart passed on an L40S; the latest verification was **September 8, 2026 at 04:14:26 UTC** ([receipt summary](../results/dexterous/2026-09-08/isaac-desktop-readiness.json)).

First-time requirements: Python 3, Git, SSH, and a RunPod API key in `RUNPOD_API_KEY` or `~/.runpod/config.toml`. The key stays outside the repository. The Mac launcher requires executable permission; `chmod +x 'Start Isaac Sim.command'` fixes a checkout that loses that bit. Linux uses the same Python command.

For the corrected Shadow finger mechanics, use the opt-in [v2 environment and acquisition demo](ISAAC_V2_READY.md). The default above preserves the original v1 model for reproduction.

## What happens automatically

1. Connect to the owned GPU, with an allocation deadline and independent local/remote teardown guards. Existing unrelated pods are never selected or modified.
2. Copy a source bundle with hashes. Generated assets, policy checkpoints, credentials and local outputs are excluded.
3. Install **Isaac Sim 5.1.0, Isaac Lab v2.3.2 and PyTorch 2.7.0/cu128**. Asset generation uses a separate Python environment with MuJoCo 3.12.0 and USD 26.8, avoiding conflicting USD libraries inside Kit.
4. Generate `db0055_swing_single` and require its signed-off mechanical QA. Fetch the pinned, licensed robot source and preserve its 69 articulated joints, 61 motor transmissions and free base.
5. Import the robot, verify native sliding-friction coefficients and 1 mm collision margins directly in PhysX, run CUDA physics with rendering, compare all robot link poses and mass with independent native kinematics, and check standing stability, commanded wrist motion, delivered motor targets and the simulation clock. Instanced finger colliders are included in these checks.
6. Save a timestamped readiness receipt, exact installed packages, import audit, trace and video. Any failed check leaves the environment unready and appears in Run Center.

The configuration is [configs/isaac/runtime.json](../configs/isaac/runtime.json). This is a tested-version installation recipe; it does not claim an immutable, prebuilt container image exists. Record a container digest when your cluster supplies one.

Run Center selects a free localhost port starting at 5190. It shows the current stage, logs, GPU utilization, VRAM, price and teardown deadline. Launching the same source while preparation is active attaches to that process. Each attempt keeps its own record under `out/isaac-launch/`; `connection.json` has the SSH command, remote checkout and readiness-receipt location.

## Another cluster

Point the same launcher at an SSH-accessible Linux node:

```bash
python3 scripts/isaac/launch.py --host user@gpu-node --port 22 --key ~/.ssh/cluster --work /local-ssd/doorbench
```

This mode makes no RunPod calls and sets no teardown timer; use the cluster scheduler's wall-time limit. The node must provide a compatible NVIDIA driver, an RTX-capable GPU for Isaac rendering, writable storage and root or passwordless sudo for package installation. A site-managed node can have an administrator preinstall the runtime before running the checks.

Choose a sufficiently large local SSD for the runtime, caches and generated assets, then archive evidence to durable storage. On September 9, a fresh L40S setup spent more than 80 minutes installing dependencies on a FUSE network mount, including thousands of small Boost header files. Low CPU utilization during this stage did not mean the installer had stopped: process I/O counters and open destination files showed progress. Avoid relocating an active installation or deleting its caches mid-run. See [evidence collection](GPU_EVIDENCE_COLLECTION.md) for the independent off-pod collector.

The remote entry point is also usable directly by a scheduler:

```bash
DOORBENCH_WORK=/local-ssd/doorbench bash scripts/isaac/prepare.sh
```

A different robot needs an embodiment adapter, new motor/sensor mappings and fresh physical validation. Changing the robot name in the configuration does not retarget a checkpoint. See [the reproduction contract](DEXTEROUS_REPRODUCTION.md) for the larger training and migration protocol.

## Use the environment

After Run Center shows ready, connect using the saved command, then:

```bash
cd /remote/checkout/from/connection.json
source isaaclab/cloud/env.sh
python scripts/isaaclab/check_g1_runtime.py
```

The default readiness fixture contains one lever door. Generate the complete catalogue when needed using the separate asset environment:

```bash
/workspace/asset-venv/bin/python scripts/generate_dataset.py --out assets --workers 8 --no-thumbs
/workspace/asset-venv/bin/python scripts/isaac/check_assets.py assets
```

Use your configured work directory instead of `/workspace` on another cluster. Existing G1 instructions and benchmark commands are in [the Isaac policy guide](ISAAC_G1_DEMO.md). The [initialized H1/Shadow opening demo](ISAAC_HANDLE_DEMO.md) runs with `python scripts/isaac/run_handle_demo.py` after activation. It remains a privileged one-door experiment, not a vision/tactile policy.

RunPod allocations have a six-hour default deadline (configurable up to eight hours). Copy results off the pod before that deadline: termination deletes its ephemeral volume. To terminate this launcher's owned pod early, use `python3 scripts/dexterous/pod.py terminate`. A failed preparation does not immediately delete debugging evidence; its deadline remains armed.

For explicitly continued research, the owned-allocation renewal helper can set
a larger finite total ceiling while retaining independent local/remote teardown
and at most three hours per renewal. The default total ceiling remains eight
hours. For example, prepare a two-hour extension within a twelve-hour ceiling:

```bash
python3 scripts/dexterous/renew_owned_guard.py --pod-id YOUR_OWNED_POD \
  --hours 2 --max-total-hours 12 --output out/isaac-launch/renewal-plan.json
python3 scripts/dexterous/renew_owned_guard.py \
  --apply-plan out/isaac-launch/renewal-plan.json \
  --output out/isaac-launch/renewal-applied.json
```

The helper verifies the journal, API identity, exact old guard processes and both
new acknowledgements before replacing guards. It never signals GPU jobs or
selects another allocation. The total ceiling is explicitly recorded in the
plan and receipt, capped at24hours; it is not an automatic recurring renewal.
