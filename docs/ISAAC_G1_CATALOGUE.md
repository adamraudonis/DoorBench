# Catalogue-wide Unitree G1 diagnostic in Isaac Sim

This is a native PhysX rollout of Unitree’s unchanged locomotion checkpoint, with one closed-start attempt for each of the **985 non-pet doors**. It is a diagnostic of approaching and passing through a doorway. It is separate from the assigned manipulation, locking, closing and safety tasks in the core DoorBench benchmark.

The G1 controls its legs and holds a fixed upper-body posture. It has no door perception, grasp planner or handle-turning controller. Manual doors move through physical contact; automatic sliders receive their own proximity-sensor and motor commands. The policy cannot write door joint state or directly command a door actuator.

[Recorded results and per-door table](review/isaac-g1-catalogue/README.md) · [Native evidence, frozen inputs and video](https://github.com/adamraudonis/DoorBench/releases/tag/g1-isaac-2026-09-06)

## Task and accounting

- Seed 0, at most 16 simulated seconds per case; physics and PD at 500 Hz, policy at 50 Hz.
- Initial root position `(0, -1.5, 0.79)` relative to each door, facing `+y`; the primary door joint starts closed. This fixed approach differs from some doors’ assigned benchmark approach.
- Raw goal: root reaches `y >= 1.2 m`, remains within the opening width with a 20 cm side margin, and stays upright for 0.5 seconds. Root height below 0.45 m or a tilt beyond 60° terminates the episode as a fall.
- The reported traversal additionally requires the saved root trajectory to cross the actual wall plane *inside* the opening width and height. Reaching the far-side goal by going around or below an opening receives no credit.
- All 985 fixtures are attempted. The **18 horizontal hatches** are not upright doorway-transit tasks; the applicable vertical collection contains **967 doors**. The **15 pet doors** are excluded before export.
- A simulator/export error remains a failed attempt. Missing or invalid native results receive one isolated retry; ordinary policy failures are not retried. Report unresolved errors and retry counts separately.
- The canonical USD representation has explicit limitations on spatial springs, cables and other mechanisms. Its unsupported cases remain visible in the inventory. A root crossing does not certify complete mechanism parity, full-body clearance, safe forces or successful unlocking.

The four-door [earlier integration demo](ISAAC_G1_DEMO.md) contains an already-open doorway and selected easy cases. Its 3/4 result is not comparable to this closed-start catalogue sweep.

## Watch progress locally

For new GPU runs, register the output directory in the [local Run Center](GPU_RUN_CENTER.md) before launching. The current runner emits a five-second heartbeat; the frozen historical runner remains available for exact reproduction.

## Reproduce the evaluation

Use Linux with an RTX-capable GPU and the [tested installation procedure](RUNPOD.md). The evaluation runner was frozen at `a0d8248cc`; the source hashes and frozen inputs accompanying the recorded run take precedence over a newly generated dataset.

```bash
git clone https://github.com/adamraudonis/DoorBench.git
cd DoorBench
git checkout a0d8248cc
DOORBENCH_WORK=/opt/doorbench-runtime bash "$PWD/scripts/pod_bootstrap.sh"
source isaaclab/cloud/env.sh
python scripts/isaaclab/check_g1_runtime.py
python scripts/isaaclab/fetch_g1_policy.py
```

The tested runtime is Isaac Sim 5.1.0, Isaac Lab v2.3.2, Python 3.11 and PyTorch 2.7.0+cu128. ONNX is pinned to 1.21.0 because a later ONNX release conflicts with Isaac’s pinned typing-extensions. Install the runtime on local disk; this run’s network-mounted `/workspace` volume produced stale-file-handle errors during package installation.

For a new geometry revision, generate a new, separate fixture directory in a CPU environment with DoorBench and `usd-core` installed:

```bash
python scripts/isaaclab/prepare_g1_catalogue.py --out out/g1-inputs
```

For an exact reproduction, download the recorded frozen inputs and verify the release checksum before extracting:

```bash
mkdir -p downloads/g1
(
  cd downloads/g1
  set -e
  curl -fLO https://github.com/adamraudonis/DoorBench/releases/download/g1-isaac-2026-09-06/isaac-g1-frozen-inputs.tar.gz
  curl -fLO https://github.com/adamraudonis/DoorBench/releases/download/g1-isaac-2026-09-06/SHA256SUMS
  sha256sum --ignore-missing -c SHA256SUMS
  tar -xzf isaac-g1-frozen-inputs.tar.gz
)
```

Then run with a new output directory:

```bash
python scripts/isaaclab/run_g1_catalogue.py \
  --assets "$PWD/downloads/g1/assets" \
  --batch 32 --out out/g1-catalogue
python scripts/isaaclab/summarize_g1_catalogue.py \
  --assets "$PWD/downloads/g1/assets" \
  --results out/g1-catalogue --out out/g1-catalogue/traversal-audit.json
```

The runner verifies input hashes before starting. Every batch gets a fresh simulator process and policy state. Receipts record exact source/checkpoint hashes, simulator versions and UTC timestamps; traces retain robot root poses and native door joint positions at 50 Hz. The audit verifies trace and source hashes. Keep native logs as well as the aggregate JSON: a process exit code alone is insufficient evidence of completion.

The current runner also audits automatically at completion. For researcher policy integration, start with the [single-door policy adapter](ISAAC_G1_DEMO.md#5-replace-the-policy-with-yours); the catalogue grid is specifically the Unitree checkpoint diagnostic.

## Native 4×4 recording

The hero view uses sixteen distinct successful vertical cases selected from the audited run, then **reruns them together** in Isaac Sim. It is an illustration of selected successes, not a random performance sample. Its receipts and trajectories must be checked separately from the original sweep.

The presentation changes lighting, the floor’s visual material and cell spacing. It retains the robot asset, door inputs, collision geometry, physical materials and policy. A runtime assertion verifies that the presentation material preserves the floor’s physics binding. The original and presentation runners have separate hashes. No human animation or door-joint playback substitutes for the native policy rollout.

The hero utilities were added after the frozen evaluation runner. After finishing the evaluation above, use `git checkout g1-isaac-2026-09-06` to access the published presentation and selection utilities, while keeping the downloaded inputs and previous evaluation output. On that source revision, adding `--hero` to `run_g1_catalogue.py` performs selection, records the grid, and audits the rerun separately in `hero/traversal-audit.json`. The published recording uses the explicit selection retained in the evidence archive's `hero/selection.json`:

```bash
python scripts/isaaclab/hero_g1.py --headless --device cuda:0 \
  --assets "$PWD/downloads/g1/assets" --out out/g1-hero \
  --video --batch-doors \
  db0010_swing_double db0031_saloon db0098_gate_swing db0108_revolving \
  db0130_automatic_sliding db0356_swing_double db0990_automatic_sliding db0332_baby_gate \
  db0350_strip_curtain db0123_saloon db0127_swing_double db0203_automatic_sliding \
  db0260_revolving db0279_swing_double db0301_gate_swing db0323_automatic_sliding
```

For a different selection, `select_g1_hero.py --audit <audit.json> --assets <assets> --out <selection.json>` chooses candidates from audited successes. Audit the new simultaneous rerun independently before describing it as successful.

GPU contact dynamics can vary between process/grid configurations even with the same seed; the pilot and presentation receipts document this variation. Source and seed pins identify the experiment, rather than promising bitwise-identical trajectories.

The camera records 2560×1600 presentation video at 25 fps, plus several still frames. Keep the complete recording when choosing a README image, so a pleasing pose cannot conceal a failed attempt.
