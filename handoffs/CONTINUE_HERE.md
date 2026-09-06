# DoorBench — continue here

Written 2026-09-05 as a handoff. A formatted version of this document is `handoffs/handoff.html` (open it in a
browser), also published at https://claude.ai/code/artifact/3f02a5b2-84b3-45fc-b7cc-94d4a3b4b2bc Master is at `955531917`, pushed, and regenerates clean. Read this file, then
`docs/LOG.md` (the engineering log: every defect class, how it was found, what the fix was).

---

## 1. Where things stand

| | |
|---|---|
| Doors | 1000, all signed off |
| Deterministic gates | 26, all inside `signed_off` (list below) |
| Tests | 1656 passing, 22 skipped |
| USD validation | 1000/1000 both kinds (`door.usda`, `door_rl.usda`) |
| MuJoCo behavioural reference | 1000/1000 doors pass every applicable phase |
| Isaac Sim / PhysX parity | **stale — must be re-run**, see §3 |

Verify all of that yourself in about four minutes:

```bash
cd <repo> && export PYTHONPATH=$PWD
rm -rf assets/doors assets/hardware assets/manifest.json && .venv/bin/python scripts/generate_dataset.py --out assets --workers 8
.venv/bin/python scripts/clearance_report.py --workers 8      # clearance + running_clearance, both 1000/1000
.venv/bin/python scripts/attachment_report.py --workers 8     # 1000/1000
.venv/bin/python -m pytest -q tests                            # 1656 passed
```

The 26 gates in `doorbench/qa.py`: `clearance`, `running_clearance`, `attachment`, `no_jam`, `mass`, `settle`,
`hold`, `free_opens`, `actuate_opens`, `latch_returns`, `relatch`, `closer_returns`, `locked_holds`,
`operator_returns`, `operator_holds`, `keypad_code_works`, `all_latches_release`, `rod_points_hold`,
`sliding_track_support`, `linkage_feasibility`, `task_achievable`, `pair_swing`, `baby_gate_headroom`,
`urdf_loads`, `usd_opens`, `usd_rl_opens`.

---

## 2. The method, in one paragraph

Every gate on that list exists because a human or an agent looked at a door and saw something obviously wrong that
no automated check caught. The working loop is: **find one defect → prove its scope over all 1000 doors → fix the
generator → add a deterministic gate so the class cannot return → regenerate and confirm 1000/1000.** Two
independent finders drive it. A *second physics engine* (Isaac Sim/PhysX against MuJoCo) catches anything where the
two engines disagree about what the model means. A *vision review* (`scripts/vision_review.py`) renders each door
as a twelve-panel sheet captioned with what the spec says is there, and asks an agent what looks wrong; it found
defects in 97 of 122 doors, every one of which passed all gates. Neither finder replaces the other, and both
produce false positives that must be killed by measurement, not argument.

---

## 3. Do these first

### 3a. Two finished branches are unmerged, and they conflict with each other

The last workflow was stopped mid-flight. Its four fix agents all finished and pushed; the verifier never ran. Two
branches are merged into master already (mass, kinematics). Two are **not**:

| branch | what it does | state |
|---|---|---|
| `wf6-fix-overhead` | roll-up curtains coil instead of rising as a rigid slab out of their guides; the 2–2.5 m hole in the wall above every sectional garage door is closed; the garage opener is bolted to something. Adds a `guided_travel` gate. | 3 commits, unmerged |
| `wf6-fix-declared` | draws the hardware every spec declares (156 extras, 35 hold-open stops, 129 both-face operators, dog counts) and adds `checks["spec_realized"]`: the spec is a contract and the geometry must satisfy it. See `docs/SPEC_REALIZED.md` on the branch. | 3 commits, unmerged |

**They conflict in one place**: both rewrote the garage builder in `doorbench/geometry/other.py` (one hunk, around
the `garage_slide_lock` block). `wf6-fix-overhead`'s version is the larger rewrite and contains the slide-lock
handling; `wf6-fix-declared`'s is a narrower `if` on the engaged lock. Merge overhead first, then declared, and
reconcile that hunk by hand keeping overhead's structure plus declared's spec-realization intent. Everything else
auto-merges. Conflicts in `results/parity/mujoco*.json` are generated files: take either side and re-run
`scripts/parity_reference_mujoco.py --doors all --workers 8 --force`.

Neither branch has been adversarially verified. Both change geometry dataset-wide. Verify before trusting.

### 3b. Isaac parity results are stale

`qa.json.isaac_parity` and the manifest badges are currently empty, because the mass fix changed the physics of 219
doors after the last GPU run. The last good run (round 4, on the pre-mass dataset) was full **925/1000** strict
parity, rl 879/1000, 935 doors badged ok. To refresh, follow `docs/RUNPOD.md`:

```bash
python scripts/runpod_pod.py create && python scripts/runpod_pod.py wait && python scripts/runpod_pod.py bootstrap
# wait for "BOOTSTRAP DONE" in /workspace/bootstrap.log (~25 min), then on the pod:
#   source isaaclab/cloud/env.sh && tmux new -d -s p 'bash isaaclab/cloud/parity.sh --force > logs/p.log 2>&1'
# ~70 min, then back locally:
scp ...:/workspace/DoorBench/results/parity/isaac_{full,rl}.json results/parity/
python scripts/parity_compare.py && python scripts/isaaclab/parity_report.py && python scripts/merge_isaac_results.py
python scripts/runpod_pod.py terminate      # ~$2.50 for the whole cycle at $1.09/h
```

**Always arm a teardown timer**; a forgotten pod bills at $1.09/h. There is no pod running now and none on the
account. Total spend across the whole exercise was about $20.

**Known trap**: regenerating the dataset rewrites `qa.json`, which drops the parity badges. Re-run
`scripts/merge_isaac_results.py` after every regeneration. Making `generate_dataset.py` carry the block forward when
the door's inputs hash is unchanged would remove this footgun and is worth doing.

### 3c. Parity classes still open (from round 4)

42 doors disagreed. Named, with a likely cause each, in `docs/ISAAC_PARITY.md`: `PHYSX_HOLD_FAIL` on 36 sliding
singles and swing pairs (a latch that holds in MuJoCo lets go in PhysX), `SETTLE_DRIFT` on 16 cold-storage
rising-hinge doors, `EXPORT_WELD` on 6, `RL_CANON` on 7, `VELOCITY_EXPLOSION` on 3.

---

## 4. Open work, roughly by value

1. **Closer mechanisms as real linkages** (`TASKS.md` G5, brief in `handoffs/closer-mechanisms.md`). Today the
   closing torque is applied at the door hinge and the arm linkage is kinematic decoration that the viewer solves.
   The real mechanism puts the spring and the hydraulic valves on the pinion and lets the torque curve emerge
   through the linkage's changing mechanical advantage. This is the largest remaining physics gap.
2. **Run the vision review over all 1000 doors.** Only 122 were reviewed. The tool is built and mock-tested but has
   never run live: it needs an `ANTHROPIC_API_KEY`. Estimated **$49.56** for one request per door, **$24.78**
   through the Batches API. Everything it found in the 122 is listed in `docs/review/vision/triage.md`, including
   the classes nobody has fixed yet.
3. **Moving-vs-moving clearance.** `running_clearance` only measures moving parts against static geometry. Two
   leaves of a pair, adjacent fold panels, a leaf against a moving mullion are unguarded. The machinery is already
   in `doorbench/clearance.py` (drop the static/moving split in `Clearance.gaps`); seals and astragals that
   legitimately compress need a documented budget.
4. **Taxonomy hierarchy page** and **browser physics playground** (`TASKS.md` V6, V7; briefs in `handoffs/`). Both
   were requested and never started; partial work sits on `worktree-agent-aee1f839bfa6c01d1` and
   `worktree-agent-a0650dd8b52f7f671`.
5. **Older audit findings** (`TASKS.md` G10, `docs/review/physics_audit.md`): QA and the benchmark environment
   install different closer damping, and sign-off has no mandatory-checklist rule, so a door passes with whichever
   checks happened to run. A required-checks manifest per family would close that.

Four `worktree-agent-*` branches predate the current work and are **superseded** — their content was reimplemented
properly later. Confirm before deleting: `a919977c25b0ad6a2` (closers), `ab282d0acdf4fb6b2` (attachment),
`a3c65620742f28a27` (vision), `a68f30762fa4dd816` (operators/keypads).

---

## 5. Defect classes already fixed — do not reintroduce

Each has a gate now. This list is the fastest way to understand what "realistic" means in this project.

| class | scope when found | gate |
|---|---|---|
| Parts passing through each other | dataset-wide | `clearance` |
| **Parts attached to nothing** | **642 / 1000 doors, 1681 findings** | `attachment` |
| Zero running clearance (touching but not penetrating: fine in MuJoCo, jams in PhysX, wrong on a real door) | 139 doors, 229 pairs | `running_clearance` |
| Multi-leaf doors 2–8× too light (per-leaf mass used as whole-door mass) | 219 doors | `mass` |
| A releasable lock modelled by clamping the joint range, making the benchmark task impossible | 24 doors | `task_achievable` |
| Accordion folds kinematically locked by a coupling-sign vs joint-range mismatch | all 12 | coupling-range check in `clearance` |
| Revolving wings jamming on the wall header | 10 / 15 | `no_jam` |
| Sliding rails too short, rollers leaving the track | 168 doors | `sliding_track_support` |
| Multi-latch doors where only one latch actuated, and dogs that never held | 13 doors | `all_latches_release`, `rod_points_hold` |
| Handles that never sprang back; wheels that sprang back but must not | 607 operator joints | `operator_returns`, `operator_holds` |
| Keypad codes decorative (the lever opened the door with no code) | 28 doors | `keypad_code_works` |
| Double-egress pairs both swinging the same way | 10 / 10 | `pair_swing` |
| Closer loops geometrically impossible to close | 5 doors | `linkage_feasibility` |

---

## 6. Operational notes that will save you hours

* **Work in a git worktree, never the main checkout.** Another agent works in
  `~/Desktop/Projects/DoorBench` directly and switches branches; commits made there land wherever it happens to
  be. `git worktree add .claude/worktrees/<name> master`.
* **Never commit `assets/`** from an agent branch: it is generated, it is 7000+ files, and it creates conflicts in
  every merge. The maintainer regenerates and commits once. Avoid `git commit -a` and `git add -A`, which sweep it in.
* **Regenerate after every merge.** Several times a branch landed new geometry with stale committed assets and the
  test suite went red for reasons unrelated to the change.
* Run Python as `PYTHONPATH=$PWD .venv/bin/python` from your worktree root; the editable install points at the main
  checkout, and `PYTHONPATH` makes your worktree win.
* **`isolation: 'worktree'` in the Workflow tool fails when the session's cwd is not a git repo.** Pre-create
  worktrees with `git worktree add` and give agents explicit paths.
* Agents should **commit and push early and often**. Two whole batches of parallel agents were killed by usage
  limits mid-flight; the ones that had committed lost nothing.
* When a process looks slow with 0 % GPU and 100 % CPU, **get a stack before theorising**:
  `faulthandler.dump_traceback_later` found an Isaac Lab callback looping `render()` forever in headless mode after
  hours of wrong guesses. `py-spy` cannot ptrace inside a RunPod container.
* **Half of a visual review's first-pass findings will be the renderer**, not the doors: reflective materials
  mirroring the skybox, black doors rendering as silhouettes, invisible clear glazing, a camera fitted to the
  bounding sphere so "close-ups" are not close. Fix the camera before believing the finding.

---

## 7. Where everything is written down

| file | what it holds |
|---|---|
| `docs/LOG.md` | the engineering log: every defect class, how it was found, root cause, lesson |
| `TASKS.md` | the task board, with status per row |
| `docs/ISAAC_PARITY.md` | the parity gate: protocol, tolerances, per-door results, open classes |
| `docs/VISION_REVIEW.md` + `docs/review/vision/triage.md` | the visual review, its findings, and the false positives |
| `docs/RUNPOD.md` | reproducible GPU setup, costs, and a troubleshooting table of everything that broke |
| `docs/PHYSICS.md`, `docs/BENCHMARK.md`, `docs/DATASET_FORMAT.md` | the physics derivations, the benchmark, the file format |
| `handoffs/*.md` | per-task briefs, each self-contained |
