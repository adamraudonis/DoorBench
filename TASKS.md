# DoorBench task board

Living tracker for everything in flight. One line per task, newest requests at the bottom of each section.
Status: `todo` · `doing` · `review` · `done` · `blocked`. Owner `main` = the coordinating session; `agent:<name>` = a
Fable subagent working in its own git worktree (merged by main).

## Quality gates (blocking for every release)

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| G1 | Force-driven QA sign-off (load all tiers, settle, hold, actuate, latch return, relatch, closer return, URDF/USD) | done | main | 1000/1000 pass; `doorbench/qa.py` |
| G2 | Deterministic kinematic clearance gate: sweep every joint of every door with all geometry collidable, flag any interpenetration > 2 mm | done | main | `doorbench/clearance.py`, `scripts/clearance_report.py`; 1000/1000 clean; wired into `qa.json` (`checks.clearance`) and the manifest; viewer shows it in the QA section |
| G3 | Lock/latch mechanism realism inspection (every operator/latch/lock type rendered close-up and reviewed; missing barrels, guides, keepers, strikes, housings) | todo | agent:locks | first finding: db0006 heavy slide bolt has no barrel/guides/keeper (user report) |
| G4 | pytest: MuJoCo import of every family + QA smoke test | todo | agent:mujoco | `tests/test_mujoco_import.py` |

## Dataset / physics

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| D1 | 1000 doors, 30 families, MJCF/URDF/USD, physics with provenance | done | main | |
| D2 | Regenerate + redeploy after G2/G3 land | todo | main | `scripts/generate_dataset.py`, Pages workflow |
| D3 | Photoreal Blender renders (Poly Haven CC0) | todo | — | later stage |

## Benchmark

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| B1 | DoorEnv + LabelTracker (touched / actuated / unlatched / opened / traversed / damaged …) | done | main | `doorbench/benchmark/` |
| B2 | Scenario + reward spec per door: start zone (randomisable), approach point, pass plane, goal zone, reward events (touch handle, unlatch, open, traverse, close), time budget / expected transit time; scenarios: open-and-traverse, open-then-close, hold-open-for-human, wait-for-human, close-behind | todo | agent:bench | emitted into `spec.json["benchmark"]` + `docs/BENCHMARK.md` |
| B3 | Viewer "Show evaluation" overlay (default off): start/goal zones, reward markers, human path, expected transit time | todo | agent:bench | `viewer/src/DoorView.tsx` |
| B4 | MuJoCo physics demo video: door opened by a programmatic hand, rendered to mp4 | todo | agent:mujoco | `scripts/demo_mujoco.py` |
| B5 | Real humanoid in the loop: Unitree G1/H1 (MuJoCo Menagerie, BSD) walking through a DoorBench door with an off-the-shelf pretrained policy (unitree_rl_gym sim2sim); GPU rental only if CPU inference is too slow | todo | agent:robot | licence check first; report feasibility + video |

## Viewer / site

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| V1 | Catalogue with thumbnails + filters, families page, 3D view with joint sliders + physics panel | done | main | live at https://adamraudonis.github.io/DoorBench/ |
| V2 | "Open / close" must actuate the operator first (latch retracts) and open joined leaves together (dutch joining bolt); slider on a latched leaf auto-unlatches; clearance badge in the panel | todo | agent:bench | |
| V3 | Fix resize feedback loop, HUD stacking, camera framing | done | main | |
| V4 | Info icon (ⓘ) with a plain-language explanation on every physics-panel row; fix units (slide operators in mm, translational springs in N) | todo | agent:bench | user request |

## Process

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| P1 | Reflection on why the interpenetrations were not caught earlier (QA was force-driven and only saw collision geometry; never swept joint ranges; visual-only parts never checked; sign-off from distant thumbnails) | done | main | README "Why gate 1 exists" |
| P2 | Keep this file current; merge agent worktrees; final regeneration + push | doing | main | |
