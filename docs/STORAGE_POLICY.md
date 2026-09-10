# Development evidence storage

Full-resolution trajectories are needed for selected audited results, debugging contacts, and actual training datasets. They are **not needed locally for every failed development attempt**. Catalogue assets and installation environments are separate from research evidence.

## What accumulated

The September10 audit found repeated full-state/contact recordings across many uniquely named experiments. Native diagnostics retained both compressed working chunks and final JSON exports of the same records. Isaac retained periodic partial snapshots alongside final recordings; collection preserved those copies locally. Worktrees and saved run directories added separate copies of outputs, source bundles and dependencies. This is excessive retention, not evidence of one file growing forever.

A point-in-time `du` inventory reported approximately19GiB in `DoorBench-runs`,7GiB in this integration checkout's output, and21GiB under the main checkout's `.claude/worktrees`, with additional older `/tmp/doorbench-*` checkouts. These scopes overlap through links/clones and other cleanup was active; do not sum them as exclusive physical usage or infer a historical100GB peak. The owner's other audit reported56GB for the broader project. Detailed local inventory is `out/storage-audit-20260910.json`.

The large data is mostly per-physics-step contact/state archives, not videos alone. In the measured run folders, files named `trajectory.npz` accounted for approximately0.8GB, while compressed JSON/contact records and raw transition directories occupied much more. Names alone do not identify a dispensable file: retain necessary audited source states and model identities.

## Enforced limits

- Native acquisition/operation diagnostics now admit a run only when there is **10GiB host reserve plus16MiB per simulated second** available for evidence. This is a conservative estimate, not a measured guarantee for all robots. During physics the check repeats every simulated second; low space produces an incomplete/failed result and leaves room to export evidence.
- Native admission also checks a **20GiB retained-evidence budget**, including its output parent and this machine's `DoorBench-runs` when present, reserving the estimated new output. Collection checks20GiB within the archive root plus an incoming snapshot allowance. Hard-linked files count once; APFS clone sharing is not assumed. These are application checks, not OS quotas or controls over another agent's writes.
- The collector defaults to **10GiB free reserve plus the full incoming stable snapshot size**, allowing for atomic replacements. It waits when space is insufficient. It does not stop remote experiments or change teardown deadlines; attach a collector and ensure capacity before GPU dispatch.
- A successful `BoundedEvidence.export` reopens the compressed export, verifies its decoded SHA-256 against the bytes written from source records, then removes redundant working chunks. Later audits read the identical final JSON array. Interrupted or failed exports retain the chunks. This changes storage, not physical controls or acceptance thresholds.

These rules cover the active native operation driver and Isaac evidence collector. They do not retroactively cap legacy scripts, model caches, all worktrees, or every other application. Do not launch those outside the same budget discipline.

## Retain versus release

Keep the current qualified reference chain, model/contracts, checkpoints selected for training, final benchmark evidence and representative failures under investigation. For other closed experiments, retain a concise result, configuration/source identities and failure diagnosis. Full records may live in verified remote archives when reproducibility requires them; avoid keeping both remote archives and duplicate local copies indefinitely. Do not repeatedly restore whole old runs just to inspect their small reports.

Local raw copies may be offloaded only after archive upload **and independent download/hash verification**, or after byte-equivalent final export verification for working chunks. Never discard an incomplete run's only evidence, a source state needed by the next planner, or another agent's files. Keep a receipt and restore command. The existing private research release remains private.

The first cleanup under this audit removed1,439 exact archive-verified local copies:910,081,118 logical bytes. Measured free-space change was612,003,840 bytes; hard links/clones and concurrent activity mean logical sizes are not disk savings. Receipt: `DoorBench-runs/remote-archives/offload-verified-development-20260910.json`. Current reference dependencies and the restart-interrupted trial were excluded. The other cleanup agent owns broader old-worktree cleanup.

## Validation

31 focused tests pass for storage admission, hard-link accounting, transfer-size headroom, export byte identity, failure recovery, streaming records and final collector checksums. No new simulation or GPU run was launched for this storage change. Before resuming robotics work, bring retained evidence under budget and satisfy free-space admission; do not lower the reserve to force a run through.
