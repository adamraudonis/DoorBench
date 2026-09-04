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
| G4 | pytest: MuJoCo import of every family + QA smoke test | done | agent:mujoco | `tests/test_mujoco_import.py`: 30 family representatives + 20 seeded random doors; all MJCF tiers + scene.xml + door.urdf load, 500 free steps (no warnings, finite), `run_qa` hold / actuate / locked / relatch / closer, `DoorEnv` 200-step hand episode + labels, clearance gate on 10 doors; 267 tests, ~5 s (`pytest -q tests/`) |

## Dataset / physics

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| D1 | 1000 doors, 30 families, MJCF/URDF/USD, physics with provenance | done | main | |
| D2 | Regenerate + redeploy after G2/G3 land | todo | main | `scripts/generate_dataset.py`, Pages workflow |
| D3 | Photoreal Blender renders (Poly Haven CC0) | todo | — | later stage |
| D4 | Mass consistency gate: simulated moving mass of every door must match `spec.physics.mass.total_kg` (found by agent:mujoco: 315/1000 differ by >25 %, e.g. dutch, pivot glass, turnstiles, revolving up to 10-20x) | todo | main | fix inertial assignment in geometry builders; add to QA as `checks.mass` |
| D5 | Keypad locks on doors without a keypad (db0233, db0086: `keypad_code_*` engaged, plain lever) | todo | main | spec post-processing: keypad lock ⇒ keypad operator |

## Benchmark

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| B1 | DoorEnv + LabelTracker (touched / actuated / unlatched / opened / traversed / damaged …) | done | main | `doorbench/benchmark/` |
| B2 | Scenario + reward spec per door: start zone (randomisable), approach point, pass plane, goal zone, reward events (touch handle, unlatch, open, traverse, close), time budget / expected transit time; scenarios: open-and-traverse, open-then-close, hold-open-for-human, wait-for-human, close-behind | todo | agent:bench | emitted into `spec.json["benchmark"]` + `docs/BENCHMARK.md` |
| B3 | Viewer "Show evaluation" overlay (default off): start/goal zones, reward markers, human path, expected transit time | todo | agent:bench | `viewer/src/DoorView.tsx` |
| B4 | MuJoCo physics demo video: door opened by a programmatic hand, rendered to mp4 | done | agent:mujoco | `scripts/demo_mujoco.py`: hand drives operator joints / keypad keys / handwheel through `DoorEnv`, synthetic robot base passes, HUD with joint states + LabelTracker labels; 7 doors (lever+closer, patio slider hook lock, garage sectional, revolving, panic pair, vault, keypad) in `docs/media/demo_<id>.mp4|gif`, README section. Side fixes in `benchmark/labels.py`: multi-operator + all lock/latch joints from IR roles (pairs, vaults, hooks, dogs), touch marking for programmatic hands, overload must be sustained (joint-stop impulses are not damage), `DoorEnv.close()` |
| B5 | Real humanoid in the loop: Unitree G1/H1 (MuJoCo Menagerie, BSD) walking through a DoorBench door with an off-the-shelf pretrained policy (unitree_rl_gym sim2sim); GPU rental only if CPU inference is too slow | done | main |  done on CPU, no GPU needed: `robot_demo/run_g1_door.py` (Menagerie G1 + unitree_rl_gym `motion.pt`, both BSD-3, pinned in `robot_demo/LICENSES.md`, fetched by `setup.sh`); 4 videos in `docs/media/g1_door_*.{mp4,gif}`: open doorway db0119, automatic slider db0990, saloon push-through db0123 (240 N peak, no damage), latched push door db0705 (latch holds, 423 N, honest failure); 15–19x real time physics+policy; README section + `robot_demo/README.md`; latched/lever doors are out of scope for a locomotion-only policy |

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
