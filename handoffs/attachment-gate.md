# "Nothing floats" attachment gate + deficiency-review command (task G7)

Read `handoffs/README.md` first. Branch to resume: `worktree-agent-ab282d0acdf4fb6b2` (3 real commits + a
snapshot: `doorbench/attachment.py` gate wired into QA as `checks["attachment"]`, `scripts/attachment_report.py`,
`scripts/deficiency_review.py`, `docs/review/deficiency/`, geometry fixes in `doorbench/geometry/*.py`).
The last agent was mid-way through "round 3" of fixes: run the report first to see what still fails.

## Why (owner's words)

"there is a floating cylinder which is meant to be a door stop ... Please ensure nothing floats."
"Please make some agent that can check for obvious deficiencies like this and floating parts."

## Goal

A deterministic gate (like `doorbench/clearance.py`, which builds the full-tier MuJoCo model with every geom
collidable, parent filter disabled, and sweeps joints) that fails a door when anything is unsupported or dead:

1. Intra-body connectivity: a body's geoms form one cluster (gaps <= 2 mm).
2. Body attached at rest: every non-world body within 3 mm of its parent's geoms (or of its equality partner /
   the world geoms it is mounted to).
3. Static geometry attached: jamb plates, keepers, stops, shoes, brackets, bumpers, tracks within 3 mm of frame,
   wall, floor, ceiling or another static geom.
4. Attached through motion: sweep the primary and every mechanism joint; `connect`/`weld` partners stay within
   1 mm; linkage tips stay at their anchors; rollers stay on their tracks.
5. Mechanisms actuate: every joint with a range moves when its mechanism is driven (a closer hinge that never
   moves when the door swings, a latch bolt that never retracts when the handle turns).
6. Degenerate content: zero-size geoms, bodies with mass but no geoms, geoms without material, duplicate geoms,
   mesh bounding box inconsistent with declared size, mirrored meshes upside-down.

Allow-lists only with a justification in `model.meta["attachment_allow"]`; tolerances as commented constants.
`signed_off` must require the gate. `scripts/attachment_report.py --workers 8` gives the dataset-wide report
(`--top N`, `--json`). `scripts/deficiency_review.py` runs all gates + renders every door (closed / mid / open,
3 cameras + hardware close-up) into contact sheets under `docs/review/deficiency/` and writes
`docs/DEFICIENCY_REVIEW.md` (counts by rule/family, top offenders, checklist). Document in `docs/REVIEW_AGENT.md`.

## Boundaries

Fix every legitimate finding in `doorbench/geometry/*.py` EXCEPT closer / operator / gas-strut mechanisms, which
another brief rebuilds (`closer-mechanisms.md`): list those findings in your report instead. Small per-door briefs
exist for db0024 (floating stop) and db0079 (rail too short): if you fix them here, say so.

## Done when

`python scripts/attachment_report.py --workers 8` -> 1000/1000 (closer findings excepted and listed);
regenerate -> 1000 signed off; clearance 1000/1000; `tests/test_attachment.py` (family representatives +
synthetic failing fixtures proving each rule fires) green; `docs/DEFICIENCY_REVIEW.md` produced from the final
dataset; before/after images for db0024 and db0012 under `docs/review/deficiency/`.
