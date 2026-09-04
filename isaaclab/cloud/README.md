# Running DoorBench in Isaac Lab on a rented GPU box

Isaac Sim needs an NVIDIA RTX-class GPU on Linux (Ubuntu 22.04 / 24.04, driver ≥ 535, glibc ≥ 2.35).
Everything below targets **Isaac Sim 5.1.0 + Isaac Lab v2.3.2** (Python 3.11 uv venv, torch 2.7 cu128). The setup
was executed on a RunPod Secure Cloud **L40S** on 2026-09-04; the replicable pod flow (API, pod spec, costs,
troubleshooting) is in [`docs/RUNPOD.md`](../../docs/RUNPOD.md). Status of each script: [`../STATUS.md`](../STATUS.md).

## Instance recommendations (Sept 2026 list prices, on-demand, approximate)

| Provider / instance | GPU | VRAM | ≈ $/h | Notes |
|---|---|---|---|---|
| RunPod community cloud | RTX 4090 | 24 GB | 0.35–0.70 | cheapest; fine for the Hand task at 1024 envs and the hero shot; consumer GPU, occasionally pre-empted |
| RunPod secure / Lambda | L40S | 48 GB | 0.80–1.10 | best value for G1 at 2048–4096 envs; RTX-capable (needed for rendering the hero shot) |
| Lambda / RunPod | A10 (24 GB) | 24 GB | 0.60–0.80 | works; ~half the throughput of an L40S |
| RunPod / Lambda | A100 80 GB | 80 GB | 1.20–1.70 | no RT cores: headless training fine, RTX rendering (hero shot, cameras) is slow but works |
| RunPod / Lambda | H100 80 GB | 80 GB | 2.00–3.30 | overkill for these tasks |

Pick a template with **Ubuntu 22.04 + CUDA 12.x driver** (RunPod "PyTorch" or plain "Ubuntu" templates, Lambda's
default Ubuntu image), ≥ 60 GB disk (Isaac Sim pip install ≈ 20 GB + Isaac Lab + logs), ≥ 32 GB RAM.
Container hosts that forbid `apt`/`sudo`: use the Dockerfile route instead.

Budget for the whole pipeline on an L40S: setup 15 min, USD validation of 1000 doors ≈ 20 min, Hand training
(1024 envs × 300 iterations) ≈ 15 min, hero shot ≈ 5 min, evaluation of 1000 doors × 3 seeds ≈ 30 min → **≈ $2**.

## Setup (pip route)

```bash
git clone https://github.com/adamraudonis/DoorBench.git && cd DoorBench
bash isaaclab/cloud/setup.sh          # ~25 min: uv venv (py3.11), isaacsim 5.1.0 wheels, IsaacLab v2.3.2, DoorBench (+extension)
source isaaclab/cloud/env.sh          # activates the venv, exports ISAACLAB_DIR / DOORBENCH_ASSETS, accepts the EULA
```

## Setup (container route)

```bash
docker login nvcr.io                  # free NGC account + API key
docker build -t doorbench-isaaclab -f isaaclab/cloud/Dockerfile .
docker run --gpus all --rm -it --network host -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
    -v $PWD/logs:/workspace/DoorBench/logs -v $PWD/docs/media:/workspace/DoorBench/docs/media -v $PWD/results:/workspace/DoorBench/results \
    doorbench-isaaclab
# inside: the cloud scripts detect /isaac-sim/python.sh automatically
```

## The five commands

```bash
bash isaaclab/cloud/validate.sh                                    # I1: pxr static check + headless Isaac Sim import of all 1000 doors
bash isaaclab/cloud/train.sh --task DoorBench-Open-Hand-v0 --num_envs 1024 --max_iterations 300     # I2: first training run (gantry hand)
bash isaaclab/cloud/train.sh --task DoorBench-Open-G1-v0 --num_envs 2048 --max_iterations 3000      #     humanoid
bash isaaclab/cloud/hero.sh [--checkpoint logs/rsl_rl/doorbench_hand/<run>/model_300.pt]           # I3: 512 doors in one scene -> docs/media/isaaclab_hero.{png,mp4}
bash isaaclab/cloud/eval.sh logs/rsl_rl/doorbench_hand/<run>/model_300.pt --seeds 3                 #     all 1000 doors -> results/*.json
```

All scripts accept the Isaac Lab flags (`--num_envs`, `--seed`, `--device cuda:0`, `--video`, `--headless` is
implied) plus `--doors easy-100 | easy-300 | all | family:saloon,pivot | db0002_swing_single,... | @ids.txt`.

## Troubleshooting

* `nvidia-smi` works but Isaac Sim fails to start: driver too old (< 535) or Vulkan ICD missing → `apt install libvulkan1` and check `vulkaninfo`; on pure-compute datacentre images you may need `--headless` (all our scripts pass it) and `--enable_cameras` only for rendering.
* `GLIBC_2.35 not found`: the box is Ubuntu 20.04 → use the Dockerfile.
* First start hangs at "extension cache": normal, the `extscache` wheels are unpacked once (~2 min).
* `PhysX error: ... articulation ...` on a specific door: run `bash isaaclab/cloud/validate.sh --ids <door_id>` and open an issue with `assets/usd_validation_isaacsim.json`.
* Out of GPU memory at 1024 envs: `--num_envs 512`; the doors carry ~20 convex colliders each.
