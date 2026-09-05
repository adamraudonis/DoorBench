# DoorBench

**1,000 articulated door models for robotics simulation.**

[Browse the catalogue](https://adamraudonis.github.io/DoorBench/) · [Download the dataset](https://huggingface.co/datasets/adamraudonis/DoorBench) · [Release guide](docs/DATASET_RELEASE.md)

DoorBench combines 30 door families with operators, locks, closers and configurable materials. It includes detailed model specifications, simulation exports, benchmark scenarios, Blender appearances, and reference motion for studying door interactions.

- **Simulation:** full, simple and minimal MJCF/URDF; full and canonical USD; shared OBJ/USDC hardware meshes.
- **Appearance:** 1,014 saved Blender renders, interchangeable finishes and lighting, CC0 textures, and available packed scenes.
- **Reference motion:** native scripted-hand door trajectories with a procedural humanoid reference. Outcomes and tracking errors are explicit; the avatar is kinematic, not a dynamically controlled humanoid policy.
- **Reproducibility:** versioned public downloads, per-file checksums, source provenance and deterministic generation.

![One barn door rendered with three material and room combinations](docs/review/blender/looks.jpg)

This is a research dataset with known construction and physics approximations. Automated QA is not physical or artistic certification. Read the [construction audit](docs/review/takeover/REVIEW.md) and [appearance review](docs/review/blender/REVIEW.md) before using it for training or evaluation.

## Try a door

```sh
git clone https://github.com/adamraudonis/DoorBench.git
cd DoorBench
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[all]" huggingface_hub
.venv/bin/python scripts/huggingface_release.py download \
  --revision v2026.09.05 --components assets --out data/doorbench
.venv/bin/python -m mujoco.viewer \
  --mjcf data/doorbench/assets/doors/db0002_swing_single/scene.xml
```

Use `DoorEnv` for benchmark scenarios and access-control behavior; loading MJCF directly does not reproduce every environment callback. The [release guide](docs/DATASET_RELEASE.md) explains downloads, motion arrays, licensing and local catalogue setup. To regenerate data, see [the dataset format](docs/DATASET_FORMAT.md) and `scripts/generate_dataset.py`.

## Documentation

| Topic | Guide |
|---|---|
| Data layout and simulator use | [Dataset format](docs/DATASET_FORMAT.md), [integration](docs/INTEGRATION.md), [physics](docs/PHYSICS.md) |
| Scenarios and historical baselines | [Benchmark](docs/BENCHMARK.md), [results](results/README.md) |
| Blender and vision data | [Appearance](docs/BLENDER_APPEARANCE.md), [state bridge](docs/BLENDER_VISION_STATE.md) |
| Reference motion and review records | [Motion schema and replay](docs/REFERENCE_MOTIONS.md), [human review](docs/HUMAN_REVIEW.md) |
| Humanoid experiments | [G1 demo](robot_demo/README.md), [Isaac Lab](docs/ISAAC_LAB.md), [current status](isaaclab/STATUS.md) |

Historical baseline scores use earlier dataset versions and broader scenario/seed sweeps. They are not results for the new one-seed reference-motion corpus, and a rendered avatar does not establish humanoid task success.

Code, generated doors and original reference motion are [MIT](LICENSE). Poly Haven textures retain CC0, including maps packed in Blender scenes. See [release licensing](docs/DATASET_RELEASE.md#provenance-and-licenses) and [CITATION.cff](CITATION.cff); record the release tag and checksum inventory when citing experimental data.
