# Renew an owned GPU window

The deadline guards capture their deadline when they start. Editing the journal
alone does not extend the window. Use the explicit dry-run-first utility when an
authorized experiment needs more time on the same owned allocation:

```bash
python3 scripts/dexterous/renew_owned_guard.py \
  --pod-id YOUR_OWNED_POD_ID --hours 3 --output out/renewal-plan.json
```

Review the reported allocation ID, deadline and exact old guard process
identities. Then apply that plan within five minutes:

```bash
python3 scripts/dexterous/renew_owned_guard.py \
  --apply-plan out/renewal-plan.json --output out/renewal-result.json
```

The utility reads only the dedicated
`~/.runpod/doorbench_dexterous_pod.json` ownership journal, verifies the same
allocation through the API and SSH endpoint, and bounds the deadline to three
hours from the plan time and eight hours from original allocation creation.
It refuses a stale plan or a renewal within two minutes of teardown.

Fresh local and remote guards must acknowledge the exact allocation/deadline and
remain alive before the journal is atomically updated. Only the inventoried old
guard PIDs are signaled after checking their command, process start identity and
source hash. It never signals a process group or a simulation job. A failure
before the replacement acknowledgements leaves the old guards active. A later
failure is reported rather than claiming renewal; inspect the retained plan and
live guards before doing anything else.

The replacement guard source and private credentials live in protected local
`~/.runpod/doorbench-guards/` and remote `/workspace/.doorbench-guards/`
directories. Receipts, logs and plans omit credentials; do not copy their
`config.json` files into results or Git. Guards retain a captured allocation ID
and bounded teardown retries; the local guard marks only its matching journal
inactive after successful teardown. An unrelated/new allocation is never deleted
through a rewritten journal ID.

## Verified renewal

On September 8, 2026, allocation `ncuctlwa5acn1v` was renewed from **11:21:51 UTC**
to **13:59:03 UTC**. Old local/remote guards `22128`/`165` were replaced by
`12254`/`73880`, after both replacement handshakes. No GPU process was signaled.
The [credential-free result](../results/dexterous/2026-09-08/owned-guard-renewal.json)
records the identities, source hashes and acknowledgements. Fifteen tests cover
deadline bounds, wrong ownership, changed/reused process identities and mocked
captured-ID teardown with journal preservation.

Run Center and already-written experiment/connection JSON files are historical
display metadata; update the active experiment's displayed deadline from the
renewal result. They are not the controlling guard. Archive results before the
new deadline. Further work past the eight-hour allocation bound requires a new
owned allocation and fresh preparation checks.
