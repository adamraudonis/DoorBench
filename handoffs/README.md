# Handoff briefs

Each file in this folder is a self-contained task for an independent agent (human or AI). The tasks were
started by agents that were stopped mid-way; their partial work is committed on the branch named in each brief
and pushed to `origin`. Read this file first, then the brief.

## Setup (once, ~2 minutes)

```bash
git clone https://github.com/adamraudonis/DoorBench.git && cd DoorBench
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"      # or: uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
cd viewer && npm install && cd ..                                   # only for tasks that touch viewer/
git fetch origin && git checkout <branch from the brief>            # resume the partial work
git merge master                                                    # bring in everything merged since the branch started
```

Everything below runs from the repo root with `.venv/bin/python` (the briefs write `python` for short).

## The project in one paragraph

DoorBench is a dataset + benchmark of 1000 procedurally generated, physically accurate articulated doors
(30 families: swing, sliding, garage, gates, hatches, revolving, ship watertight, ...) for humanoid robots.
`doorbench/spec.py` samples a spec per door, `doorbench/geometry/*.py` build the bodies/joints/geoms (an IR
in `doorbench/ir.py`), `doorbench/physics.py` derives every physical parameter from real hardware standards,
`doorbench/export/{mjcf,urdf,usd}.py` write MJCF / URDF / USD, `doorbench/qa.py` runs deterministic quality
gates in MuJoCo (a door is only "signed off" when every gate passes), `doorbench/benchmark/` is the RL
environment + scenarios + runner, `viewer/` is the React/three.js site (https://adamraudonis.github.io/DoorBench/),
`TASKS.md` is the task board. `docs/` has the design docs (BENCHMARK.md, PHYSICS docs, HARDWARE_REVIEW.md, ...).

## Commands you will use

```bash
rm -rf assets/doors assets/hardware assets/manifest.json && python scripts/generate_dataset.py --out assets --workers 8   # regenerate all 1000 doors (~50 s); prints "done: 1000 doors, N signed off"
python scripts/clearance_report.py --workers 8      # gate 1: no part passes through another (must print 1000/1000 clean)
python -m pytest -q tests                            # all tests (a few hundred, ~20 s)
python -m doorbench show db0012_swing_single         # print one door's spec / physics / qa
cd viewer && npm run typecheck && npm run build && npm test   # viewer checks
```

Render a door from Python (MuJoCo offscreen, used by every review script):
`scripts/hardware_review.py` shows how (load `assets/doors/<id>/scene.xml`, set joint qpos, `mujoco.Renderer`).

## Rules (the owner is strict about these)

1. **Realism is the bar.** Every report from the owner so far ("parts pass through each other", "the lock
   doesn't exist", "the arm floats", "the rail is too short") was a real defect. If a human would call it
   obviously wrong, it is a bug. Model the real mechanism, not a decoration.
2. **Deterministic gates, not eyeballing.** Whenever you fix a class of defect, add a check to `doorbench/qa.py`
   (or a module it calls) so it cannot come back, and make `signed_off` depend on it.
3. **All 1000 doors must stay signed off**: regenerate and confirm `1000 signed off`, `clearance 1000/1000`,
   tests green, before you report. Never hand-edit files under `assets/` — they are generated.
4. **Do not commit `assets/`** (the maintainer regenerates and commits the dataset). Commit code, docs, tests,
   and small media under `docs/media/` or `docs/review/` (JPEG, keep folders under ~15 MB).
5. **Do not edit `TASKS.md` or `README.md`** — put the numbers in your report; the maintainer integrates.
6. Commit early and often on your branch (`git push origin <branch>`); if you get cut off, the next agent
   continues from your last commit.
7. Keep changes surgical and inside the files listed in the brief; other briefs run in parallel.

## Report format (paste at the end)

- What the model / feature is now (one paragraph, no hedging).
- Numbers: signed off N/1000, clearance N/1000, new gate N/1000, tests passed, viewer build status.
- Doors you rendered to prove it (paths under `docs/media/` or `docs/review/`).
- What you could not finish and why.
- Branch + final commit hash.

## Briefs

| file | scope | size | branch |
|---|---|---|---|
| [closer-mechanisms.md](closer-mechanisms.md) | every self-closing / power-operating mechanism as a real linkage with tunable physics | large | `worktree-agent-a919977c25b0ad6a2` |
| [attachment-gate.md](attachment-gate.md) | "nothing floats" gate + one-command deficiency review | large | `worktree-agent-ab282d0acdf4fb6b2` |
| [vision-review.md](vision-review.md) | photograph every door, vision-LLM common-sense review | medium | `worktree-agent-a3c65620742f28a27` |
| [operator-spring-return.md](operator-spring-return.md) | twist handles snap back with a spring | medium | `worktree-agent-a68f30762fa4dd816` |
| [keypad-codes.md](keypad-codes.md) | keypad codes physically work (press buttons in order) | medium | `worktree-agent-a68f30762fa4dd816` |
| [door-db0168-ship-dogs.md](door-db0168-ship-dogs.md) | ship watertight doors: all dogs actuate | small | new branch |
| [door-db0079-sliding-track.md](door-db0079-sliding-track.md) | sliding rails long enough; rollers stay on the track | small | new branch |
| [door-db0024-floating-stop.md](door-db0024-floating-stop.md) | wall/floor door stops must be mounted, nothing floats | small | new branch |
| [door-cold-storage-rising-hinge-closer.md](door-cold-storage-rising-hinge-closer.md) | 5 rising-hinge doors whose closer loop cannot close | small | new branch |
| [taxonomy-hierarchy.md](taxonomy-hierarchy.md) | taxonomy audit + hierarchy page on the site | medium | `worktree-agent-aee1f839bfa6c01d1` |
| [physics-playground.md](physics-playground.md) | MuJoCo-WASM playground to tune constants live in the browser | large | `worktree-agent-a0650dd8b52f7f671` |

Small briefs are independent and good first tasks. The two `worktree-agent-a68f30762fa4dd816` briefs share a
branch: do them one after the other or split the branch.
