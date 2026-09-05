---
pretty_name: DoorBench
license: mit
language:
- en
task_categories:
- other
tags:
- robotics
- simulation
- mujoco
- isaac-sim
- articulated-objects
- synthetic
- blender
size_categories:
- 1K<n<10K
configs:
- config_name: catalogue
  data_files:
  - split: catalogue
    path: metadata/doors.jsonl
---

# DoorBench

**{{DOORS}} procedural articulated doors for robot simulation, with MJCF, URDF and USD exports, Blender appearances, and reference motion.**

[Interactive catalogue](https://adamraudonis.github.io/DoorBench/) · [Source and tools](https://github.com/adamraudonis/DoorBench) · [Release guide](DATASET_RELEASE.md)

Version **{{RELEASE}}** contains {{RENDERS}} saved Blender images, including {{PHOTO_RENDERS}} images rendered with the higher sample preset, and reference clips/native trajectories for {{MOTION_DOORS}} doors. The release manifest, complete per-file SHA256 inventory, source hashes and immutable Hub revision identify the data used by an experiment. Generator source: [`{{SOURCE_COMMIT}}`](https://github.com/adamraudonis/DoorBench/tree/{{SOURCE_COMMIT}}).

![The same door in three rendered appearance combinations](preview.jpg)

## Download and use

Install `huggingface_hub`, download the verification helper from this revision, and run it. No access token is required for this public, ungated dataset.

```python
from huggingface_hub import hf_hub_download
import subprocess, sys

helper = hf_hub_download("{{REPO_ID}}", "download.py", repo_type="dataset",
                         revision="{{RELEASE}}")
subprocess.run([sys.executable, helper, "download", "--repo-id", "{{REPO_ID}}",
                "--revision", "{{RELEASE}}", "--out", "doorbench-data"], check=True)
```

The helper checks archive and individual file hashes, rejects unsafe archive paths, and installs only into a new directory. Add `--components assets` for the smaller simulation-only download, or select a comma-separated subset of `assets,appearance,textures,reference-motions`. Preserve the directory structure: exports reference shared hardware meshes.

```python
import mujoco
model = mujoco.MjModel.from_xml_path(
    "doorbench-data/assets/doors/db0002_swing_single/door.xml")
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
```

Install the [DoorBench source package](https://github.com/adamraudonis/DoorBench) to use its scenario logic, lock/access-control callbacks and benchmark environment. A direct MuJoCo load is useful for inspection but does not reproduce every `DoorEnv` behavior.

## Contents

| Component | Contents |
|---|---|
| `assets` | All {{DOORS}} specifications, articulated JSON models, MJCF/URDF fidelity tiers, full/canonical USD, OBJ and USDC hardware meshes, QA records and simulation thumbnails |
| `appearance` | Saved RGB images, render index, source/renderer/state/map provenance, and available Blender scenes with packed maps |
| `textures` | Ten Poly Haven CC0 raster material sets and the source/license/checksum/scale manifest |
| `reference-motions` | One primary core scenario at seed 0 per door: native MuJoCo arrays, browser clips, procedural humanoid reference and outcomes |

`metadata/doors.jsonl` is the searchable catalogue table shown by the Hub. It is **not** a train/test split. The archives contain the detailed records. Close procedural relatives should be grouped when designing train/test splits; random row splitting can leak geometry templates.

## What the motion means

Door motion is recorded from the actual `scripted_hand` MuJoCo baseline, with the native state arrays retained. The humanoid is an original **kinematic reference skeleton** aligned to that interaction. It is not a dynamically controlled humanoid, an expert teleoperation recording, or proof that a robot can execute the movement. Successes, failures, unreachable targets and hand-target errors remain explicit. Use the motion index and clip fields when filtering demonstrations; do not train on every clip as if it were successful.

The [reference-motion guide](https://github.com/adamraudonis/DoorBench/blob/{{SOURCE_COMMIT}}/docs/REFERENCE_MOTIONS.md) defines array schemas, physics versus actor timelines, replay and accuracy limits. The [human review guide](https://github.com/adamraudonis/DoorBench/blob/{{SOURCE_COMMIT}}/docs/HUMAN_REVIEW.md) explains how to keep your inspection findings separate from automated QA.

## Known limits

This is a procedural research dataset, not a certified reconstruction of real products. Automated `signed_off` means the implemented checks passed; it is not physical or artistic approval. Known source limitations include backed glazing/louvers, approximate masses, incomplete hardware/support, simplified garage/rollup motion and closer forces. See the [construction audit](https://github.com/adamraudonis/DoorBench/blob/{{SOURCE_COMMIT}}/docs/review/takeover/REVIEW.md) and [appearance review](https://github.com/adamraudonis/DoorBench/blob/{{SOURCE_COMMIT}}/docs/review/blender/REVIEW.md).

Blender output is offline display-transformed RGB. Its materials, room context, tiny bevels and bounded decorative foliage are visual derivatives; collision geometry does not provide exact pixel masks. Depth, segmentation, real-time vision observations and photorealistic robot bodies are not supplied. Snapshot cameras support calibrated perspective projection within the documented bounds. Scene appearance references are not measured material BRDFs for each door.

Historical benchmark results in the source repository used earlier assets and three seeds per listed scenario. This release's one-seed primary-scenario reference recordings are a different evaluation population and must not replace those results or be compared as the same benchmark.

## License and attribution

DoorBench code, generated doors, rendered compositions and original procedural reference motion are **MIT**; retain [LICENSE](LICENSE). Poly Haven raster maps, including maps packed in Blender scenes, remain **CC0-1.0**. Their source URLs, authors and license references are retained in the texture manifest and render provenance. See [THIRD_PARTY.md](THIRD_PARTY.md). No Unitree meshes, pretrained policy weights, MuJoCo binaries or Blender binaries are included.

The collection contains no human recordings or personal data. Human/robot motion is synthetic. Commercial use is allowed under the respective MIT and CC0 terms; no performance or fitness guarantee is made.

## Citation

Use the repository's [CITATION.cff](https://github.com/adamraudonis/DoorBench/blob/{{SOURCE_COMMIT}}/CITATION.cff) and record the dataset release tag, Hub commit and inventory hash with your experiment.
