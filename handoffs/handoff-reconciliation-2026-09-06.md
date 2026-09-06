# Handoff reconciliation — September 6, 2026

Action is needed. The September 5 HTML and `CONTINUE_HERE.md` describe an older
generator. Their 1000/1000 sign-off is not evidence for the current mechanical
rebuild. This review compares fetched remote branches with master `7d36d7921`
and integration `6aa86047c` on `codex/mechanical-master-integration`.

## Branch disposition

| Branch | Reviewed tip | Disposition |
|---|---|---|
| `wf6-fix-overhead` | `1f4c08362` | Preserve as reference; do not merge wholesale. Current sectional and rolling mechanisms have separate builders and intact-header tests. The older curtain telescopes overlapping courses using joint equalities; it would replace the newer articulated curtain and physical hoist. Recover missing enclosure coverage against current geometry. |
| `wf6-fix-declared` | `e8ad6b9c9` | Partially superseded, with useful missing coverage. Current marine, hatch, stop, shaft and pocket assemblies overlap its changes. Its generic `spec_realized` gate is absent from integration. Port the contract intent after reconciling current names and actual mechanisms; do not restore obsolete geometry merely to satisfy name checks. |
| `worktree-agent-aee1f839bfa6c01d1` | `031900b1f` | Taxonomy page remains unfinished and unintegrated. Contains hierarchy UI, taxonomy report generation and tests. Retain for later completion after mechanical release blockers. |
| `worktree-agent-a0650dd8b52f7f671` | `55a575bb0` | Physics playground remains unfinished and unintegrated. Contains WASM loading, simulation and parameter UI. Its passive-law implementation needs comparison with current runtime mechanisms before claiming native equivalence. |

The four drafts identified as superseded in the previous handoff remain untouched.
No historical branch, generated asset or rendered image was merged during this review.

## Newly recovered evidence

The frozen `catalogue-mechanics-v5` run completed all 1000 receipts:
**776 passed, 217 failed checks, 7 execution errors**. Failures overlap:
attachment 164, closer return 27, clearance 18, running clearance 16, settle 11.
All seven execution errors are roll-up doors: DB184, DB258, DB313, DB419, DB636,
DB754 and DB888. Six workers exited with signal 11; DB184 recorded a native
constraint-allocation error. These need reproduction and root-cause analysis,
not a passing badge or a blind retry.

This run exported JSON and MJCF only. It did not establish USD validity,
Isaac parity, human feasibility or benchmark task completion. Its frozen source
differs from the reviewed integration in `geometry/paired_holds.py` and `qa.py`;
later threshold and operation-timing fixes therefore need fresh receipts.
The committed [receipt inventory](../docs/review/handoff-reconciliation/2026-09-06.json)
preserves all outcomes, source hashes and the exact limitation.

I also ran the old declared-branch contract against all 1000 frozen models,
without installing it in production QA. It flagged 425 doors: 211 missing-stop,
132 missing-extra, 100 operator-face, 14 multiplicity and 9 operator-presence
findings. These are **triage findings, not confirmed defect counts**. For example,
the old checker does not recognize the current `floor_post` contract. Its rules
also depend on geometry names. Full diagnostic output remains in
`out/handoff-reconciliation/legacy-spec-diagnostic.json` in the integration worktree.

## Taken-over work, in priority order

1. Reproduce roll-up execution errors and triage attachment failures against
   current source. Recheck failures affected by the two post-snapshot fixes.
   Continue the documented mechanical repairs, including vertical-rod retention,
   sectional operation and ship holdback sequencing. Preserve failing evidence.
2. Recover spec-to-geometry and enclosure contract coverage from the two old
   branches. Validate actual parts, approach-side access and full motion; geometry
   names alone are insufficient. Keep physically connected current assemblies.
3. Run release checks on one frozen, reconciled source revision. Then rerun
   full/canonical Isaac parity with source hashes and an armed teardown timer.
   The old 42 disagreements are a historical seed list, not a current count.
   The completed four-case G1 demo does not refresh all-door parity.
4. Publish consistent models, metadata and native evidence. Current Pages run
   `34046023058` fails 12 viewer checks against the published asset snapshot;
   previous run `34011154271` has the same failures. Do not weaken those checks
   to deploy. Mechanical source remains on the integration branch.
5. Resume taxonomy and playground after core mechanical correctness. The latter
   must implement current passive mechanics or explicitly limit its supported
   models; a generic `mj_step` loop alone is not runtime equivalence.

The broader component review remains in the integration branch under
`docs/review/mechanical-foundations/README.md`. Its status is incomplete.
No appearance rerenders or new GPU rentals were needed for this handoff review.
