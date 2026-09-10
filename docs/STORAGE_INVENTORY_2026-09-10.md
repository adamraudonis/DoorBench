# Local storage inventory — September 10, 2026

Measured with `du -sh` during active development; values are rounded and can change while collectors run. These directory totals are not an exact measure of unique APFS physical allocation.

| Location | Reported size |
| --- | ---: |
| Main `DoorBench` checkout | 32 GiB |
| Active `/private/tmp/doorbench-isaac-integration` worktree | 9.4 GiB |
| Adjacent `DoorBench-runs` archive directory | 8.0 GiB |

The main checkout includes 14 GiB in `.claude/worktrees` and 16 GiB in `out`. These are subsets of its 32 GiB, not additional totals. The two `out/reference-planned-corpus-v1` and `-v2` directories account for 3.2 and 2.8 GiB. Other generated exports include `planned-release` (1.8 GiB), `publication` (1.1 GiB), and `huggingface-release` (982 MiB). The filesystem had 33 GiB available at inspection.

## Retention rules

Detailed state and contact trajectories are useful for independent audits and diagnosing failed grasps. They do not all need permanent local retention. Preserve active source trajectories, pending audit inputs, and the qualified continuation chain. Archive completed superseded experiments with independently verified hashes and restore receipts before evicting their local raw data. Keep compact configuration, provenance, failure summaries, and audit results locally.

The current collector checks a combined 20 GiB logical evidence budget across its source worktree's `out` directory and `DoorBench-runs`, deduplicating hard links. It also reserves 10 GiB of filesystem free space plus the incoming transfer. These checks do **not** cap the entire project, old shared worktrees, remote storage, or other processes writing data. A collector refusing a transfer does not stop the remote experiment.

Do not remove shared worktrees or older corpora solely because of size. First identify ownership, unique uncommitted work, active readers, whether outputs are reproducible, and whether any required archive is verified. No shared files or active trial inputs were deleted as part of this inventory.
