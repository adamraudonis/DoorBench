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
| D4 | Mass consistency gate: simulated moving mass of every door must match `spec.physics.mass.total_kg` (found by agent:mujoco: 315/1000 differed by >25 %, e.g. dutch, pivot glass, turnstiles, revolving up to 10-20x) | done | main | `build.py` reconciles the leaf bodies' mass to the spec (total minus modelled hardware, by volume); `checks.mass` in qa.json; 1000/1000 within 5 % |
| D5 | Keypad locks on doors without a keypad (db0233, db0086: `keypad_code_*` engaged, plain lever) | done | main | spec post-processing swaps in `lever_keypad` / `knob_keypad_deadbolt` |

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

## Isaac Lab (needs an NVIDIA GPU on Linux; this Mac cannot run Isaac Sim)

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| I1 | Static USD validation of all 1000 `door.usda` (schemas, articulation root, joints, drives, masses, mesh references) with usd-core here; Isaac Sim headless import validation script for the GPU box | todo | agent:isaac | `scripts/isaaclab/validate_usd_static.py`, `scripts/isaaclab/validate_usd_isaacsim.py` |
| I2 | Isaac Lab extension `doorbench_isaaclab`: multi-door scene (one door USD per env via multi-asset spawning), `DoorBench-Open-G1-v0` manager-based RL task (obs/rewards/terminations from the benchmark spec), RSL-RL PPO train/play scripts, one-command install | todo | agent:isaac | `isaaclab/` |
| I3 | One-command cloud setup (Isaac Sim container + Isaac Lab + DoorBench) for Lambda/RunPod/any Ubuntu GPU box; hero-shot script rendering hundreds of envs with different doors in one 3D scene | todo | agent:isaac | `isaaclab/cloud/` |
| I4 | Run I1-I3 live on a GPU box: validation report, short training run, hero screenshot/video into README + site | blocked | main | needs a GPU instance from the user (no cloud credentials on this machine) |

## Benchmark runs & submissions

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| R1 | Benchmark runner: `doorbench benchmark run --policy ...` evaluates any policy (small Python interface) over all 1000 doors × scenarios × seeds in MuJoCo, parallel on CPU, writes a `results/*.json` with per-door outcomes + aggregate score | review | agent:benchrun | `doorbench/benchmark/policy.py` (Policy: `reset(door_info)`, `act(obs) -> {torques, base_velocity, badge, done}`; robot embodiments via `make_env` + `ctrl`), `scenarios.py` (default / traverse / traverse_close / hold_and_pass, success from labels: traversed without damage, closed for close scenarios), `runner.py` (DoorEnv hand with per-joint torque limits incl. far-side / unreachable operators = 0, synthetic base that only crosses the plane when the opening is clear and >= 0.45 m wide, automatic-door sensor, per-episode wall timeout, crash-tolerant process pool, seed 0 nominal / seeds >= 1 domain-randomised, events with timestamps, aggregate per family / difficulty / task / scenario / lock state), CLI `doorbench benchmark run|list-scenarios|list-policies` with `--dry-run`; `results/schema.json` + `scripts/validate_result.py`; `tests/test_runner.py` (12 tests). 0.2 s wall per full-tier episode; 1000 doors x 3 seeds in 76 s on 8 workers |
| R2 | Baseline policies + full runs published: random, scripted heuristic hand, G1 locomotion-only; "N / 1000 doors successful" tables in README + a Results page on the site (per-door badge in the catalogue) | review | agent:benchrun | `doorbench/benchmark/baselines/{random_policy,scripted_hand,g1_locomotion}.py`; full runs committed: `results/random.json` 55/1000, `results/scripted_hand.json` 860/1000, `results/g1_locomotion.json` 155/1000 (see README "Baseline results"); `results/index.json` + `results/README.md` generated by `scripts/build_results_index.py`; site: `viewer/src/Results.tsx` (`#/results`: leaderboard, per-family bars, per-task / lock / difficulty tables, 1000-cell per-door grid), `ResultBadges.tsx` (per-door chips in the catalogue; TODO for main: wire `<BaselineBadges id={id} compact={false} />` into `DoorView.tsx` next to the QA chip); Pages workflow copies `results/` into the site. Dataset findings from the runs are listed in the README section |
| R3 | Submission flow for researchers: `docs/SUBMITTING.md`, result JSON schema + validator, PR-based leaderboard with a CI check | review | agent:benchrun | `docs/SUBMITTING.md` (implement a Policy, run all 1000 doors x >= 3 seeds, validate, PR `results/<team>_<policy>.json`; rules: simulator + tier + commit, no asset edits), `.github/workflows/validate-results.yml` (schema + submission rules on changed files, `build_results_index.py --check`), `scripts/build_results_index.py` |
| R4 | Dataset / task-assignment defects exposed by the baseline runs (the scripted oracle hand cannot solve 140 doors; ids via `python scripts/validate_result.py` + `results/scripted_hand.json`, per-door grid on the site) | todo | main | (a) 22 accordion / bifold / revolving doors whose panels or wings rub the head jamb or drum header (e.g. db0927, db0030, db0779: 40 contacts `jamb_head`/`panel_*_slab`, `wall_header`/`wing_0_stile`) and do not move under 150 N m: the QA skips actuation for free-swing families, so nobody checked they open; (b) 38 exit doors with `operator.sides = push_side` but `robot.is_push = false`: the robot stands on the pull side with only a pull handle and tasks `open_and_traverse` / `open_only` / `hold_and_pass` / `peek` / `unlock_open_traverse` are unachievable (assign `locked_recognize` or put the robot on the push side); (c) 15 pet doors with tasks `push_through` / `traverse_open` (opening 0.16-0.39 m, a humanoid cannot pass); (d) 6 rotors (revolving / turnstiles) with task `traverse_open`: `DoorEnv.reset` cannot 'start open' an unlimited or +-0.05 rad rotor; (e) 11 `jam_stuck` doors with `locked_recognize` open under a 25 N m push (jam friction too low); (f) deadbolt doors whose thumbturn mesh collides with the deadbolt housing so the bolt stops at 50 % throw (db0912; the QA `actuate_opens` skips doors whose operator is a push plate / pull); (g) `keyed_cylinder` / `privacy_button` / `card_reader` / `keypad_*` with `robot_side_release = true` open under operator torque without ever setting `lock_released` (no backlash, nothing to release), and `electric_bolt` (db0146) limits the leaf range to 2 mm with no release path even after `badge()`; (h) `peek` on powered automatic doors is unachievable (the operator opens fully); (i) pet flaps / hatches reset 'open' for `traverse_open` / `close` fall shut on their own and slam (the runner ignores damage before the first touch, but the reset pose is unstable); (j) in `door_simple.xml` keypad keys are absent, so keypad codes cannot be entered in the `simple` tier |

## Process

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| P1 | Reflection on why the interpenetrations were not caught earlier (QA was force-driven and only saw collision geometry; never swept joint ranges; visual-only parts never checked; sign-off from distant thumbnails) | done | main | README "Why gate 1 exists" |
| P2 | Keep this file current; merge agent worktrees; final regeneration + push | doing | main | |
