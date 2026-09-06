# Website publication

The Pages workflow builds the viewer from master and restores the exact dataset and Blender renders named in `deploy/site-assets.json`. The generated files are stored as a versioned GitHub release archive. They are not added to source commits. The release URL and SHA-256 are pinned; replacing the archive with different content makes deployment fail.

`scripts/site_assets.py` packages all door exports, simulation thumbnails, shared OBJ/USDC hardware, the appearance index and every image/metadata/Blender scene referenced by that index. With `--reference`, it includes the reference-motion index and one compressed browser clip per door. Native trajectories and full JSON clips remain in the Hugging Face release. It excludes texture caches, backups, prepared jobs and logs. It requires a default render for every door and verifies source-model, hardware and output checksums. Restoration verifies the entire archive before extracting regular files into `assets/`, `appearance/` and `reference-motions/`, rejects unsafe or duplicate paths, then rechecks the complete catalogue and motion coverage.

To publish a new generated collection, first finish rendering and reviewing it. Use a new release tag for each publication:

```sh
RELEASE_TAG=research-site-YYYYMMDD-COMMIT
python3 scripts/site_assets.py pack \
  --assets assets --appearance out/appearance \
  --reference out/reference-motions \
  --archive out/publication/doorbench-site-assets.tar.gz \
  --manifest deploy/site-assets.json \
  --release-url "https://github.com/adamraudonis/DoorBench/releases/download/$RELEASE_TAG" \
  --source-commit "$(git rev-parse HEAD)"

# Check the exact archive locally in a fresh output directory.
python3 scripts/site_assets.py restore --manifest deploy/site-assets.json \
  --archive out/publication/doorbench-site-assets.tar.gz --out out/publication/check

# Upload the archive to a draft, then publish it before pushing the manifest to master.
gh release create "$RELEASE_TAG" out/publication/doorbench-site-assets.tar.gz \
  --target "$(git rev-parse HEAD)" --draft --latest=false \
  --title "DoorBench website assets $RELEASE_TAG" --notes-file out/publication/release-notes.md
gh release edit "$RELEASE_TAG" --draft=false --latest=false
```

Write release notes identifying the source commit, door/render counts and review record before running the release command. After local viewer checks pass, commit only the small manifest and source/documentation changes, merge master, and push master. The existing Pages workflow downloads the public release and publishes automatically. A subsequent code-only deployment reuses the same pinned generated collection. Updating the generator does not silently regenerate or re-render the published dataset; publish a new reviewed bundle for that.

The workflow copies benchmark results from master and exposes `deployment.json` at the website root for the published bundle's provenance. Existing baseline scores predate the current geometry repairs; they remain historical results with their own recorded commits. A website deployment does not rerun robot evaluations.

## Small collection corrections

`deploy/collection-update.json` optionally names a second, immutable GitHub release archive. `scripts/site_asset_patch.py` binds each changed file to its exact previous checksum and its replacement checksum. Pages verifies the full original release and its historical motion index first, then applies this correction. The script rejects a different base release, changed source files, unsafe archive paths and corrupt payloads, and rechecks every corrected appearance against its source geometry. `collection-update.json` is also exposed at the website root alongside the original `deployment.json`.

The baby-gate correction removes the overhead wall from all 10 gates and includes regenerated simulator exports, thumbnails and Blender images. The corrected catalogue also classifies the 15 standalone pet doors as supplementary, leaving 985 benchmark-eligible doors. Historical clips retain their original checksums. Changed geometry explicitly disables those clips instead of assigning an old motion to a new model. Pet doors expose geometry and downloads without evaluation or motion controls.

`scripts/prepare_collection_update.py --base <verified-site> --baby-gates <review-output> --out <new-directory>` merges the ten generated gate rows and image-only render inventory while applying the central eligibility policy. It preserves all other model files. The ten gates received new QA; other signoffs are inherited from the original release, not a fresh all-door certification. See the [scoped review and outstanding findings](review/collection-correction/REVIEW.md).

To build another scoped correction, prepare a complete updated copy of the verified base under `out/`, then run `site_asset_patch.py pack --help`. Pack and restore it locally, publish the archive under a new GitHub release tag, and commit only its manifest, source and documentation. Never replace the contents of an existing release archive. A website correction is separate from the next batched Hugging Face release.

After deployment, verify the workflow success and the live `deployment.json`, `assets/manifest.json`, `appearance/index.json`, a PNG, render metadata and a Blender scene download. Verify `reference-motions/index.json` against the pinned checksum and open a recorded animation: playback verifies the clip against the deployed source model, specification and MJCF. Confirm the catalogue offers Blender previews for every door, the barn-door page has four saved appearances (default preview plus three photo variants), and the review workspace loads. Reload an already-open page to fetch the new index.

The published site must remain below the [GitHub Pages size limit](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits). Keep the uncompressed dataset, appearance media, viewer and result files comfortably under 1 GB. The workflow checks this before uploading the Pages artifact. The packed Blender files already contain their image maps, so no separate texture cache is required.

## Viewer tests and dataset versions

Pages runs mechanism regression tests against JSON freshly generated from the checked-out source, using `scripts/generate_viewer_test_fixtures.py` and `DOORBENCH_TEST_ASSETS`. It separately supplies the restored release as `DOORBENCH_PUBLISHED_ASSETS` to the all-door compatibility test. Both inputs are required in CI; missing files fail the corresponding checks. The current-source tests retain their detailed mechanism assertions, and the release test checks finite existing joint targets, native ranges, deadbolt couplings and inspection guides across all 1,000 published doors.

This distinction matters because a versioned older dataset does not gain new metadata when viewer code changes. The release archive's original integrity, appearance and reference-source checks still run before either test input is used. Fixture generation does not replace the published assets or certify their physics.
