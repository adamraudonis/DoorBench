# Website publication

The Pages workflow builds the viewer from master and restores the exact dataset and Blender renders named in `deploy/site-assets.json`. The generated files are stored as a versioned GitHub release archive. They are not added to source commits. The release URL and SHA-256 are pinned; replacing the archive with different content makes deployment fail.

`scripts/site_assets.py` packages all door exports, simulation thumbnails, shared OBJ hardware, the appearance index and every image/metadata/Blender scene referenced by that index. It excludes texture caches, backups, prepared jobs and logs. It requires a default render for every door and verifies source-model, hardware and output checksums. Restoration verifies the entire archive before extracting regular files into `assets/` and `appearance/`, rejects unsafe or duplicate paths, then rechecks the complete catalogue.

To publish a new generated collection, first finish rendering and reviewing it. Use a new release tag for each publication:

```sh
RELEASE_TAG=blender-site-YYYYMMDD-COMMIT
python3 scripts/site_assets.py pack \
  --assets assets --appearance out/appearance \
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

After deployment, verify the workflow success and the live `deployment.json`, `assets/manifest.json`, `appearance/index.json`, a PNG, render metadata and a Blender scene download. Confirm the catalogue offers Blender previews for every door and the barn-door page has four saved appearances (default preview plus three photo variants). Reload an already-open page to fetch the new index.

The published site must remain below the [GitHub Pages size limit](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits). Keep the uncompressed dataset, appearance media, viewer and result files comfortably under 1 GB. The workflow checks this before uploading the Pages artifact. The packed Blender files already contain their image maps, so no separate texture cache is required.
