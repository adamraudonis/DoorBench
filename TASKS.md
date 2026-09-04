# DoorBench task board

Living tracker for everything in flight. One line per task, newest requests at the bottom of each section.
Status: `todo` · `doing` · `review` · `done` · `blocked`. Owner `main` = the coordinating session; `agent:<name>` = a
Fable subagent working in its own git worktree (merged by main).

## Quality gates (blocking for every release)

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| G1 | Force-driven QA sign-off (load all tiers, settle, hold, actuate, latch return, relatch, closer return, URDF/USD) | done | main | 1000/1000 pass; `doorbench/qa.py` |
| G2 | Deterministic kinematic clearance gate: sweep every joint of every door with all geometry collidable, flag any interpenetration > 2 mm | done | main | `doorbench/clearance.py`, `scripts/clearance_report.py`; 1000/1000 clean; wired into `qa.json` (`checks.clearance`) and the manifest; viewer shows it in the QA section |
| G3 | Lock/latch mechanism realism inspection (every operator/latch/lock type rendered close-up and reviewed; missing barrels, guides, keepers, strikes, housings) | review | agent:locks | `scripts/hardware_review.py` + `docs/HARDWARE_REVIEW.md` + `docs/review/`: 112 models reviewed (46 ok, 27 cosmetic, 25 unrealistic, 6 broken, 8 n/a); all 31 unrealistic/broken rebuilt (barrel/slide/drop bolts with plates+guides+keepers, hasp/staple/padlock, fork & MagnaLatch, Suffolk, handleset, cremone, rim exit latch in its case, keypads/readers/cylinders, chain, rim & night-latch cases, multipoint, electric bolt) + 17 cosmetic; 998/1000 signed off & clearance-clean (same two brace-vs-jamb doors as before). Left: padlocks on garage/hatch families, slide_bolt lock on sliding doors/pet doors, electric-strike faceplate, cold-storage latch body |
| G4 | pytest: MuJoCo import of every family + QA smoke test | done | agent:mujoco | `tests/test_mujoco_import.py`: 30 family representatives + 20 seeded random doors; all MJCF tiers + scene.xml + door.urdf load, 500 free steps (no warnings, finite), `run_qa` hold / actuate / locked / relatch / closer, `DoorEnv` 200-step hand episode + labels, clearance gate on 10 doors; 267 tests, ~5 s (`pytest -q tests/`) |
| G5 | Self-closing / power-operating mechanisms perfectly modelled as real articulated linkages with tunable physics (pinion spring + sweep/latch/backcheck valves through the arm linkage; gas struts and pneumatic tubes as hinged cylinder + slide; floor springs / spring hinges at the pivot; operators as motor on the pinion), reduced models calibrated to the full mechanism, `linkages` block in model.json, QA `closer_linkage` / `closer_closes` gates | doing | agent:closers | user 2026-09-04: db0012 closer arms floating and not actuating; "power and mechanics needs to be perfectly modeled with tunable physics parameters". Viewer solver found 5 rising-hinge cold-storage doors (db0188, db0432, db0549, db0585, db0937) whose planar closer loop cannot close (12-13 mm at full open: the leaf rises with the hinge while the shoe stays on the frame) - needs a closer/shoe geometry fix here |
| G6 | Viewer solves closed-loop linkages kinematically (two-bar closer arms, telescoping struts) from model.json `linkages` so mechanisms actuate in the 3D view | done | agent:viewer-linkages | the MuJoCo model closes the loop with a `connect` equality; the viewer only animated door joints, so arms rode rigidly with the leaf and floated off the shoe |
| G7 | Deterministic attachment gate (no floating parts: intra-body connectivity, body attached to parent / anchor at rest and through the sweep, static geoms attached to frame/wall/floor, mechanisms actuate, degenerate/duplicate/mirrored geoms) + one-command deficiency-review agent (`scripts/deficiency_review.py`, docs/DEFICIENCY_REVIEW.md) | doing | agent:attachment | user 2026-09-04: "make some agent that can check for obvious deficiencies like this and floating parts" |

## Dataset / physics

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| D1 | 1000 doors, 30 families, MJCF/URDF/USD, physics with provenance | done | main | |
| D2 | Regenerate + redeploy after G2/G3 land | todo | main | `scripts/generate_dataset.py`, Pages workflow |
| D3 | Photoreal Blender renders (Poly Haven CC0) | todo | — | later stage |
| D4 | Mass consistency gate: simulated moving mass of every door must match `spec.physics.mass.total_kg` (found by agent:mujoco: 315/1000 differed by >25 %, e.g. dutch, pivot glass, turnstiles, revolving up to 10-20x) | done | main | `build.py` reconciles the leaf bodies' mass to the spec (total minus modelled hardware, by volume); `checks.mass` in qa.json; 1000/1000 within 5 % |
| D5 | Keypad locks on doors without a keypad (db0233, db0086: `keypad_code_*` engaged, plain lever) | done | main | spec post-processing swaps in `lever_keypad` / `knob_keypad_deadbolt` |

## Benchmark

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| B1 | DoorEnv + LabelTracker (touched / actuated / unlatched / opened / traversed / damaged …) | done | main | `doorbench/benchmark/` |
| B4 | MuJoCo physics demo video: door opened by a programmatic hand, rendered to mp4 | done | agent:mujoco | `scripts/demo_mujoco.py`: hand drives operator joints / keypad keys / handwheel through `DoorEnv`, synthetic robot base passes, HUD with joint states + LabelTracker labels; 7 doors (lever+closer, patio slider hook lock, garage sectional, revolving, panic pair, vault, keypad) in `docs/media/demo_<id>.mp4|gif`, README section. Side fixes in `benchmark/labels.py`: multi-operator + all lock/latch joints from IR roles (pairs, vaults, hooks, dogs), touch marking for programmatic hands, overload must be sustained (joint-stop impulses are not damage), `DoorEnv.close()` |
| B5 | Real humanoid in the loop: Unitree G1/H1 (MuJoCo Menagerie, BSD) walking through a DoorBench door with an off-the-shelf pretrained policy (unitree_rl_gym sim2sim); GPU rental only if CPU inference is too slow | done | main |  done on CPU, no GPU needed: `robot_demo/run_g1_door.py` (Menagerie G1 + unitree_rl_gym `motion.pt`, both BSD-3, pinned in `robot_demo/LICENSES.md`, fetched by `setup.sh`); 4 videos in `docs/media/g1_door_*.{mp4,gif}`: open doorway db0119, automatic slider db0990, saloon push-through db0123 (240 N peak, no damage), latched push door db0705 (latch holds, 423 N, honest failure); 15–19x real time physics+policy; README section + `robot_demo/README.md`; latched/lever doors are out of scope for a locomotion-only policy |
| B2 | Scenario + reward spec per door: start zone (randomisable), approach point, pass plane, goal zone, reward events (touch handle, unlatch, open, traverse, close), time budget / expected transit time; scenarios: open-and-traverse, open-then-close, hold-open-for-human, wait-for-human, close-behind | done | agent:bench | `doorbench/benchmark/scenarios.py` → `spec.json["benchmark"]` (+ manifest summary); 8 scenario types incl. unlock_and_traverse / locked_recognize / knock_and_wait; seeded assignment: open_and_traverse 761 · unlock 141 · locked_recognize 98 · open_then_close 294 · close_only 115 · hold_open 42 · wait_for_human 29 · knock 10; `DoorEnv.reset(scenario=)`, kinematic human (mocap capsule), `reward()` / `success`; `tests/test_benchmark_scenarios.py`; formulas in `docs/BENCHMARK.md` |
| B3 | Viewer "Show evaluation" overlay (default off): start/goal zones, reward markers, human path, expected transit time | done | agent:bench | `viewer/src/evaluation.ts` + Evaluation panel section, scenario selector, person timeline scrubber, deep links `#/door/<id>?eval=1&scenario=…&t=…`; screenshots in `docs/media/` |
| B6 | Segregate human interaction: `core` suite (no person; open/close/unlock/locked-recognize) is the default for the runner, DoorEnv, viewer and every published table; `human` suite (hold open / wait / knock) is an advanced opt-in (`--suite human`) with its own table | doing | main | user request 2026-09-04: benchmark must be runnable without any human interaction |

## Viewer / site

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| V1 | Catalogue with thumbnails + filters, families page, 3D view with joint sliders + physics panel | done | main | live at https://adamraudonis.github.io/DoorBench/ |
| V2 | "Open / close" must actuate the operator first (latch retracts) and open joined leaves together (dutch joining bolt); slider on a latched leaf auto-unlatches; clearance badge in the panel | done | agent:bench | `viewer/src/doorLogic.ts` (pure, `bun test` against model.json: knob+latch, dutch joined/free, pair active leaf, locked stall, slide-bolt gate); operator → leaf → release phases; locked joints (range < 0.006) never driven; "latch retracted" toast; QA section lists `clearance` + failure pairs |
| V4 | Info icons (ⓘ, hover + click, accessible) on every physics / QA / evaluation row with plain-language explanation, unit and derivation; fix units for linear operators (mm, N, N/m) | done | agent:bench | `viewer/src/glossary.ts`; units follow the operator joint type from model.json |
| V3 | Fix resize feedback loop, HUD stacking, camera framing | done | main | |

## Isaac Lab (needs an NVIDIA GPU on Linux; this Mac cannot run Isaac Sim)

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| I1 | Static USD validation of all 1000 `door.usda` (schemas, articulation root, joints, drives, masses, mesh references) with usd-core here; Isaac Sim headless import validation script for the GPU box | review | agent:isaac | USD exporter rewritten (`doorbench/export/usd.py`): default prim = door root, static `Env` + `Articulation` with a fixed `base` link so every door is ONE PhysX tree (pairs / wall buttons / gate pins were multi-root before), PhysX applied schemas, `PhysxJointAxisAPI` Coulomb friction efforts (+ legacy coefficient scaled by the joint reaction, not the raw torque), mimic joints as proper `rel referenceJoint`, `/PhysicsScene` outside the default prim; new canonical `door_rl.usda` (identical 8 links / 7 joints for all 1000 doors, `doorbench:rl` meta) for multi-door spawning; QA gate `usd_rl_opens`. `scripts/isaaclab/validate_usd_static.py`: **1000/1000 pass** (both files, 21.7k colliders, 5.3k mesh refs, frames/limits/drives/friction checked against model.json) → `assets/usd_validation.json`. `validate_usd_isaacsim.py` written, NOT executed (no GPU) |
| I2 | Isaac Lab extension `doorbench_isaaclab`: multi-door scene (one door USD per env via multi-asset spawning), `DoorBench-Open-G1-v0` manager-based RL task (obs/rewards/terminations from the benchmark spec), RSL-RL PPO train/play scripts, one-command install | review | agent:isaac | `isaaclab/doorbench_isaaclab` (pip install -e): `multi_door_cfg` (MultiUsdFileCfg, round-robin/random), `DoorBench-Open-Hand-v0` (6-DoF gantry hand) + `DoorBench-Open-G1-v0` (+ Play variants), `DoorState` labels = benchmark events, `DoorMechanismAction` (spring targets, latch↔operator coupling, closer asymmetry, automatic doors), PPO cfgs, `scripts/isaaclab/{train,play,eval_all_doors}.py`, `tests/test_isaaclab_ext.py` (pass here), offline API-name check vs Isaac Lab v2.3.0 (130 symbols). NOT executed on a GPU — see `isaaclab/STATUS.md` |
| I3 | One-command cloud setup (Isaac Sim container + Isaac Lab + DoorBench) for Lambda/RunPod/any Ubuntu GPU box; hero-shot script rendering hundreds of envs with different doors in one 3D scene | review | agent:isaac | `isaaclab/cloud/{setup,validate,train,play,hero,eval}.sh`, `Dockerfile` (nvcr.io/nvidia/isaac-sim:5.1.0 + Isaac Lab v2.3.0), `cloud/README.md` (RunPod/Lambda instances + $/h), `scripts/isaaclab/record_hero.py` (512 envs × 512 doors → `docs/media/isaaclab_hero.{png,mp4}`), `docs/ISAAC_LAB.md`, `isaaclab/README.md`. Untested (no GPU) |
| I4 | Run I1-I3 live on a GPU box: validation report, short training run, hero screenshot/video into README + site | doing | main | RunPod Secure Cloud pod `doorbench-isaaclab` (L40S 46 GB, driver 580, Ubuntu 22.04, 150 GB volume, $1.09/h) created 2026-09-04 17:13 UTC on the user's account; base install (Isaac Sim 5.1 pip + Isaac Lab main) running; terminate when done |

## Benchmark runs & submissions

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| R1 | Benchmark runner: `doorbench benchmark run --policy ...` evaluates any policy (small Python interface) over all 1000 doors × scenarios × seeds in MuJoCo, parallel on CPU, writes a `results/*.json` with per-door outcomes + aggregate score | todo | agent:benchrun | |
| R2 | Baseline policies + full runs published: random, scripted heuristic hand, G1 locomotion-only; "N / 1000 doors successful" tables in README + a Results page on the site (per-door badge in the catalogue) | todo | agent:benchrun | |
| R3 | Submission flow for researchers: `docs/SUBMITTING.md`, result JSON schema + validator, PR-based leaderboard with a CI check | todo | agent:benchrun | |

## Process

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| P1 | Reflection on why the interpenetrations were not caught earlier (QA was force-driven and only saw collision geometry; never swept joint ranges; visual-only parts never checked; sign-off from distant thumbnails) | done | main | README "Why gate 1 exists" |
| P2 | Keep this file current; merge agent worktrees; final regeneration + push | doing | main | |
