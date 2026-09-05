# Versioned dataset releases

DoorBench's generated data are distributed through the public, ungated [Hugging Face dataset](https://huggingface.co/datasets/adamraudonis/DoorBench). Source code and detailed documentation remain in [GitHub](https://github.com/adamraudonis/DoorBench), and the [catalogue](https://adamraudonis.github.io/DoorBench/) is for browsing. Use a release tag or full Hub commit to identify experimental data; a moving `main` branch is not a dataset version.

The current catalogue separates **985 robotics doors** from **15 supplementary pet doors**. Pet assets remain downloadable, but the current runner and UI exclude them from all benchmark/evaluation suites. Previously published archives retain their historical pet metadata and recordings. Hugging Face releases are batched at most once per day; website corrections can ship independently without rewriting an existing Hub release.

Baby-gate overhead-wall corrections are distributed through the website's versioned correction archive, described in [website publication](WEBSITE_DEPLOYMENT.md). The original `v2026.09.05` Hub download remains an immutable earlier snapshot; use the corrected website files or regenerate from current source when those corrections are required.

## Download

From the source checkout:

```sh
uv pip install --python .venv/bin/python huggingface_hub
.venv/bin/python scripts/huggingface_release.py download \
  --repo-id adamraudonis/DoorBench --revision v2026.09.05 \
  --out data/doorbench-v2026.09.05
```

The destination must be new. The downloader resolves the requested tag to a fixed commit, verifies the release inventory and compressed archives, rejects unsafe archive entries, verifies every extracted file, then installs the completed directory. No token is required. For simulation only, add `--components assets`; otherwise all available components are downloaded.

| Component | Extracted directory | Purpose |
|---|---|---|
| `assets` | `assets/` | All generated specifications, models, fidelity tiers, shared OBJ/USDC meshes, QA and thumbnails |
| `appearance` | `appearance/` | RGB renders, provenance/index and available packed Blender scenes |
| `textures` | `textures/` | CC0 maps and original relative-path source/license manifest |
| `reference-motions` | `reference-motions/` | Native arrays and browser clips, plus outcome/accuracy index |

Keep shared hardware next to `doors/`; the MJCF, URDF and USD exports use relative references. To use the downloaded door:

```python
from doorbench.benchmark import DoorEnv

env = DoorEnv("data/doorbench-v2026.09.05/assets/doors/db0002_swing_single", tier="full")
env.reset(scenario=env.core_scenarios[0], seed=0)
env.step()
print(env.labels().to_dict())
```

For a local catalogue, place or link the downloaded `assets/` at the repository's `assets/`, `appearance/` at `out/appearance/`, and `reference-motions/` at `out/reference-motions/`, then start the viewer. The website's deployment uses its own checksummed distribution manifest; it is separate from this research download.

## Motion and training semantics

The reference corpus records one primary core scenario, seed 0, full fidelity, for each door. Native door states come from the actual scripted-hand baseline. The visible humanoid is an original procedural kinematic reference; it is not a trained or dynamically controlled humanoid. The motion index identifies each clip/NPZ, hashes its source spec/model/XML, and records success, outcome, duration, frames, hand-target error and unreachable frames. Preserve these failure and quality fields when selecting demonstrations.

The [reference-motion guide](https://github.com/adamraudonis/DoorBench/blob/master/docs/REFERENCE_MOTIONS.md) defines the native/actor array schemas, distinct physics/actor timelines, coordinate conventions and replay examples. In particular, sampled forces are not a complete control-rate log, and procedural foot-contact labels are not sensor measurements. Use the generator commit in `release.json` to select the matching documentation revision. Browser clips can be compressed as `.json.gz`; the release includes those derivatives alongside original JSON and native arrays when generated.

No train/test split is declared. Related procedural templates should be grouped to reduce geometry leakage. Historical benchmark tables span different assets/scenarios/seeds; the new reference corpus cannot be substituted for a rerun of that benchmark. Construction, appearance and camera limits are detailed in [the format guide](https://github.com/adamraudonis/DoorBench/blob/master/docs/DATASET_FORMAT.md), [state bridge](https://github.com/adamraudonis/DoorBench/blob/master/docs/BLENDER_VISION_STATE.md), and [review](https://github.com/adamraudonis/DoorBench/blob/master/docs/review/takeover/REVIEW.md).

## Provenance and licenses

The published `v2026.09.05` release resolves to Hub commit `6e17f0f588bf81fec0f04b2a329b471488164366`, with source commit `54a5c7c771d8419c69bd94d432b0ff75a2016daa`. Its `release.json` SHA256 is `b9c809bd405d72d1d5bc96a1611370926c3b245c02b6d8e7459199f7af329a73`. Anonymous download of the published helper and all 17,246 simulation asset files was verified, including loading and stepping a downloaded door in MuJoCo.

`release.json` identifies the release, generator commit, dataset/render manifest hashes, component archives and their counts. `inventory.json` records the SHA256, byte count, component and license of every archive member. It remains byte-identical to the release inventory for subset downloads; `installed.json` records which components were installed and their resolved Hub commit. Archive filenames include their inventory digest. Gzip/tar metadata and entry order are deterministic; unchanged inputs produce identical archives. Mutating a file while it is packed fails verification.

DoorBench's generated assets and original reference skeleton/motion are MIT. Poly Haven maps retain CC0-1.0, including when packed inside a Blender scene. Their manifest retains provider metadata, source URLs, authors, original scale and explicit calibration overrides. `LICENSE` and `THIRD_PARTY.md` travel with the download. No G1 model files, pretrained policy weights, simulator binaries or Blender binaries are redistributed.

## Prepare and publish a new version

Batch Hugging Face updates at most once per day, publishing a reviewed revision when ready. Local development and website updates can continue between releases; a daily upload is not required.

Freeze and validate generated inputs before preparing a release. This script verifies full source/render correspondence, required source hashes, USD asset references, one default render per door, complete motion coverage and per-clip/native/source checksums. The motion generator's own physical and visual checks remain separate; packaging does not turn a failed episode into a successful demonstration.

```sh
.venv/bin/python scripts/huggingface_release.py prepare \
  --assets assets --appearance out/appearance \
  --texture-manifest out/appearance-textures/manifest.json \
  --motions out/reference-motions \
  --release v2026.09.05 --source-commit FULL_40_CHARACTER_COMMIT \
  --out out/huggingface-release/v2026.09.05

.venv/bin/python scripts/huggingface_release.py publish \
  --folder out/huggingface-release/v2026.09.05 \
  --token-file /private/path/to/huggingface-token
```

Preparation can omit `--motions` while the base archives are built; publication requires the completed motion component. Re-preparation reuses only checksum-verified identical components. The publication command uses the official [Hugging Face Hub upload API](https://huggingface.co/docs/huggingface_hub/guides/upload), verifies public anonymous access and remote file hashes, and creates the release tag at the verified commit. It rejects attempts to reuse a tag for different release bytes. Keep credentials outside the repository, or use `HF_TOKEN`; the script never serializes the token into data or provenance.

The small source dataset-card template is `deploy/huggingface/README.md`. Generated archives, inventory, resolved card and upload receipts stay under ignored `out/`. Do not commit generated assets or upload transient logs, caches or credential files.
