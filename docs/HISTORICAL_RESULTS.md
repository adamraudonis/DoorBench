The September 4 baseline files describe the doors at their recorded source
revisions. Door mechanics and scenarios have changed since those runs. Checking
their episodes against today's scenario list incorrectly rejects valid older
records and can also mislabel historical coverage.

`results/provenance/historical-baselines.json` binds the four shipped result
files to their exact SHA-256 checksums and full source commits. The registry
also records each original `assets/manifest.json` Git blob ID, SHA-256, dataset
timestamp, version and inventory size. The validator reads that manifest from
Git and verifies every receipt field. It does not reconstruct a manifest from
successful episodes or infer which scenarios would make an old score pass.

```bash
python scripts/validate_result.py --all
python scripts/build_results_index.py --check
```

Historical registration changes which manifest is checked. The result schema,
door IDs/families, scenario assignments, seed coverage, suite separation,
success/damage consistency and aggregate counts are still validated. Missing
receipts, unavailable revisions, changed source manifests and altered original
result bytes all fail. The original scores remain untouched.

The index records `validation_manifest` for each row. Historical completeness
and lock-state groups use the run's source manifest; the current supplementary
pet-door exclusion still filters the published subset. This is a recomputed
historical summary, not a new evaluation of current mechanics. New result files
use the current manifest and current eligibility rules. `--submission` always
checks current scenario coverage and eligibility, including when given a
registered historical file; history cannot bypass submission requirements.

CI checks out full Git history. For a shallow local checkout, fetch the recorded
commits or run `git fetch --unshallow` before validation. The validator never
silently substitutes current assets when an original revision is unavailable.

The registry can be reproduced into a fresh file for comparison:

```bash
python scripts/freeze_result_provenance.py \
  results/g1_locomotion.json results/random.json \
  results/scripted_hand.json results/scripted_hand_human.json \
  --output /tmp/historical-baselines-check.json
```

Only runs with an explicit clean working tree, a full existing commit, and
matching source dataset metadata can be frozen this way. A dirty working tree
needs separate captured-asset evidence; this mechanism rejects it. Historical
registration is an audit action, not a way to submit a new current leaderboard
claim. New evaluations should use new filenames so the shipped originals stay
immutable.
