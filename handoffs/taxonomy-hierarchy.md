# Taxonomy audit + hierarchy view on the site (task V6)

Read `handoffs/README.md` first. Branch to resume: `worktree-agent-aee1f839bfa6c01d1` (2 real commits + a
snapshot: hierarchy metadata in `doorbench/taxonomy.py` (motion classes, family cards, contexts, variants,
`build_hierarchy`), `docs/TAXONOMY.md` with audit findings T-01..T-35, WIP `viewer/src/Hierarchy.tsx` and edits
to `App.tsx`, `Catalogue.tsx`, `Families.tsx`, `styles.css`, `types.ts`). Read `docs/TAXONOMY.md` first.

## Why (owner's words)

"Doubly check the taxonomy of doors. Maybe even make a new hierarchy view that shows them all and their
relationship and grouping etc."

## Goal

1. Finish the audit in `docs/TAXONOMY.md`: hierarchy tree with counts (motion class -> family -> context ->
   kinematics), a card per family (what it is, real examples, kinematics, typical hardware, what makes it hard
   for a robot), findings with severity, and proposals. Do NOT change the sampler's seeded draws (the 1000 door
   ids/specs must stay byte-identical: regenerate and `git status assets/` must be clean); dataset-changing
   fixes are proposals for the next release.
2. `scripts/taxonomy_report.py` -> a committed JSON the viewer loads (per-node counts, representative door ids +
   thumbnails, shared-mechanism relationships: families x closers / latches / locks / operators).
3. "Hierarchy" page in the site nav: collapsible tree with counts, thumbnail strip per node (click -> catalogue
   filtered / door page), kinematics + hardware badges, and a relationships panel (heat matrix or chord SVG of
   families x mechanism kinds). Responsive, keyboard accessible, no per-door fetches. Cross-link with the
   existing "Door types" page (`Families.tsx`).
4. Tests: every door in exactly one leaf; counts add to 1000; every family in `taxonomy.py` appears in
   `docs/TAXONOMY.md`.

## Done when

`cd viewer && npm run typecheck && npm run build && npm test` clean; screenshots
`docs/media/hierarchy_{tree,relations}.png` (dev server + browser); tests green; dataset unchanged.
