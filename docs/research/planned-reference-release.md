# Experimental planned-reference releases

The first complete supplement is public: [planned-v2026.09.05 on Hugging Face](https://huggingface.co/datasets/adamraudonis/DoorBench/tree/planned-v2026.09.05/experimental/planned-reference/planned-v2026.09.05). It contains 61 traversal references, 51 locked-door checks, and reports for the other 888 doors. The immutable Hub commit is `e33db254c76712a2a9fcbfd97776b924dd473a15`; see [the baseline review](planned-reference-baseline.md) for the limits of these results.

The planned-reference supplement is separate from the native scripted-hand dataset. Every one of the 1,000 doors retains a status and reason. Only `accepted_kinematic` rows receive browser playback and research trajectories. Rejected and unresolved rows retain public audit summaries; an unresolved planner, timeout, or engine error is not a proof that a person cannot operate that door.

Acceptance means independent **sampled kinematic and actor-route evidence checks** passed for an original approximate adult rig. Door motion remains prescribed and retimed from the native recording. This does not certify forces, balance, causal humanoid control, grasp or lock semantics, the original benchmark clock, continuous collision clearance, natural appearance, or personal visual approval. The native primary recording outcomes (879 successes, 118 failures, 3 damaged) and historical benchmark scores remain separate evaluations. See [the detailed scope](planned-reference-scope.md).

`scripts/planned_reference_release.py` prepares a supplement locally, then publishes only through its separate explicit `publish` command. It does not alter the corpus, source assets, recordings, generator, or acceptance statuses. Do not publish a baseline snapshot if a replacement revision is still being selected.

## Local readiness and preparation

```sh
.venv/bin/python scripts/planned_reference_release.py prepare \
  --corpus out/reference-planned-corpus-v1 \
  --release planned-v1 \
  --out out/planned-release/planned-v1 --dry-run
```

The adjacent `planned-v1.plan.json` reports readiness, status counts, accepted source bytes, and browser compatibility. A live runner lock, pending attempt, missing door, stale source, changed generator/runtime, stale native recording index, or mismatched result prevents preparation. Once the corpus is complete, dry-run exports the exact browser derivatives into temporary storage to measure them; it does not build research archives or contact Hugging Face.

Every accepted browser derivative must fit the current MotionLab bounds: 64 MiB gzip, 256 MiB decoded JSON, 100,000 frames, and 16 million pose scalars for each native/actor group (`frames × bodies × 7`). Any failure reports the precise door and limit. Decoded sizes beyond the limit are reported as a lower bound from bounded streaming decompression. Nothing is truncated, silently dropped, or reclassified. A separate explicitly designed web-unavailable contract would be needed before publishing such a corpus.

After selecting the final corpus revision and committing its generator, validator, exporter, release script, helper, and license, prepare a fresh directory:

```sh
.venv/bin/python scripts/planned_reference_release.py prepare \
  --corpus out/reference-planned-corpus-v1 \
  --release planned-v1 --source-commit FULL_40_CHARACTER_SOURCE_COMMIT \
  --out out/planned-release/planned-v1
.venv/bin/python scripts/planned_reference_release.py publish \
  --folder out/planned-release/planned-v1 --dry-run
```

The preparation reads the runner's exact resume/integrity contract under a shared corpus lock. It requires all 1,000 terminal results, matching current generator/runtime and source hashes, and the published native index/manifest. The committed source must match every inventoried generator and release file. Preparation uses private staging and atomically exposes the finished bundle. The default research shards target 512 MiB of uncompressed source bytes; one door is never split across shards. Disk space is checked before streaming accepted NPZs directly into archives.

The bundle contains:

- `web/index.json`: all status rows, content-addressed accepted gzip clips, and audit descriptors with SHA-256 and byte counts.
- `status.jsonl`: the same 1,000 rows as the web index, including reasons and accepted research download locations.
- `archives/accepted-*.tar.gz`: only accepted clip JSON, full-rate NPZ, byte-exact independent report, and original actor MJCF extracted verbatim from its clip.
- `research-inventory.json`: every archive member hash and size. Accepted browser audits are cross-bound to these archive members.
- `release.json`: complete source, generator, corpus snapshot, browser compatibility, file, and archive provenance.
- `native-dependency.json`, `README.md`, `LICENSE`, `LIMITATIONS.md`, and verified download helpers.

Public result summaries are explicitly labeled projections carrying `original_result_sha256`. Rejected validation projections similarly retain their original hash while removing local paths and tracebacks. Accepted clip and validation bytes remain unchanged. Rejected trajectories and transient logs are omitted. This supplement contains no pretrained human assets, SMPL models, motion weights, or textures; its original rig and generated motion artifacts use the repository's MIT license.

## Explicit publication and deployment

Only the owner performs publication after reviewing the concrete prepared bundle:

```sh
.venv/bin/python scripts/planned_reference_release.py publish \
  --folder out/planned-release/planned-v1 --token-file PRIVATE_TOKEN_FILE
```

The publisher targets the existing public `adamraudonis/DoorBench` dataset, strictly under `experimental/planned-reference/planned-v1/`. It allowlists only verified bundle files and neither replaces the native root dataset card nor uploads transient local files. Tokens are read only for this command and never serialized. `planned-*` tags are immutable: an existing tag must identify exactly the same release, or publication fails. Uploaded Git/LFS content hashes are checked anonymously before tagging; reruns verify the same hashes.

`publication.json` records the actual Hub commit, release SHA-256, status counts, and **full commit-pinned web index URL plus SHA-256**. Deployment should consume that receipt into its checked-in manifest and set `VITE_PLANNED_REFERENCE_INDEX` to this immutable URL. Do not use the moving Hub `main` branch or a locally guessed URL. The web files remain remote; they need not be added to the existing Pages asset archive.

Native geometry and recordings are referenced at the already published `v2026.09.05` Hub commit `6e17f0f588bf81fec0f04b2a329b471488164366`. The dependency file records the exact release, archive, dataset manifest, and source recording index hashes. Native geometry is stored inside archives on Hugging Face, so there is no fabricated per-file `assets_base_url`. MotionLab continues to use the matching website `./assets` files, checking their source/hardware hashes against each clip. Research users can obtain the native assets using that release's existing verified downloader.

## Verified research download

The supplement includes `download.py` and `archive_helpers.py`. Install `huggingface_hub`, keep both helpers together, and select the experimental tag or full immutable Hub commit:

```sh
python download.py download --repo-id adamraudonis/DoorBench \
  --release planned-v1 --revision FULL_HUB_COMMIT --out planned-data
```

`--archives NAME,NAME` installs selected whole shards. The downloader resolves the immutable revision, verifies every downloaded file and archive member, rejects traversal/symlinks/duplicate or unlisted members, and installs atomically into a fresh directory. It retains the complete original `release.json`, `research-inventory.json`, and status rows even for a subset; `installed.json` records the resolved revision and selected shards. It does not fetch native archives or large unselected research shards implicitly.
