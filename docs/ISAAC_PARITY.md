# Isaac parity gate

One behavioural protocol, run on every door in **MuJoCo** (the reference physics, CPU) and in **Isaac Sim / PhysX**
(GPU), compared per door and per phase.  The point is to catch problems with the doors themselves — export mapping
gaps, physics parameters that PhysX interprets differently, geometry that behaves differently under convex
decomposition — systematically, instead of discovering them through training.

Code: `doorbench/parity/protocol.py` (protocol as data + pure functions), `doorbench/parity/mujoco_runner.py`,
`scripts/parity_reference_mujoco.py` (CPU), `scripts/isaaclab/isaac_parity.py` + `isaaclab/cloud/parity.sh` (GPU),
`scripts/parity_compare.py` (verdicts), `tests/test_parity_protocol.py`.

## Running it

```bash
# 1. MuJoCo reference, all 1000 doors (~1 min with 8 workers); resumable, exact reproduction of qa.json is asserted
PYTHONPATH=$PWD python scripts/parity_reference_mujoco.py --doors all --workers 8
#    -> results/parity/mujoco.json (+ mujoco_summary.json; full 30 Hz curves in results/parity/cache/, not committed)

# 2. PhysX side on the GPU box (needs results/parity/mujoco.json for identical per-door inputs)
bash isaaclab/cloud/parity.sh                             # both USD kinds, all doors, batches of 20
bash isaaclab/cloud/parity.sh --limit 40 --which full      # first probe
bash isaaclab/cloud/parity.sh --hz 240 --iters 32,8 --tag _dt240   # solver-sensitivity rerun of grade B/C doors
#    -> results/parity/isaac_full.json, results/parity/isaac_rl.json (partial results after every batch, resumable)

# 3. Verdicts
PYTHONPATH=$PWD python scripts/parity_compare.py          # -> results/parity/compare.json, compare_summary.json
```

## The protocol

Phases follow `doorbench.qa.run_qa` (the dataset sign-off) expressed in simulated time, so a 500 Hz MuJoCo run and a
120 Hz PhysX run apply the same schedule.  All joint values are DoorBench coordinates (MuJoCo `q`; USD `q` +
`doorbench:zero_offset`).  Curves are recorded at 30 Hz; metrics and pass/fail come from one shared code path.

| phase | drive | duration | expectation (from `qa.door_flags` + the RL slot metadata) |
|---|---|---|---|
| settle | none | 1 s | primary drift < 0.05 rad / 0.01 m, no MuJoCo warnings, initial penetration > -12 mm |
| hold | adaptive push on the primary joint: `min(2(bias + friction + preload) + 60 N·m \| 80 N, 800 \| 4000)` | 1 s (holding) / ≤ 6 s (free) | `hold` (< 2°/15 mm) for latched / locked doors, `free_opens` (> 10°/5 cm) otherwise; free-swing families informational |
| operate | thumbturn 2 N·m (t < 1.2 s), aux bolts 3 N·m / 60 N, dogs 14 N·m, operator 4 / 8 / 10 / 14 N·m or 120 N from 0.6 s, push from 1.2 s while q < 50° | 6.4 s | `opens` (> min(20°, ½ max_open) / 5 cm; chain guards inside the slack window); RL: `stays_closed` when the release parts are welded engaged |
| release | none; primary joint pinned | 0.8 s | `bolt_returns` (< 6 mm) |
| relatch | −min(½ push, 1.5·static + 40) for 6 s, then +push 1 s | 7 s | `relatches` (closed < 2°, re-push < 2.5°) |
| closer | none, from min(60°, 0.8 max_open) | 12 s | `closes` (< 6°) |
| locked | operator 6 N·m / 150 N + push | 2 s | `locked_holds` |

Per-door inputs (`results/parity/mujoco.json` → `doors.<id>.inputs`) carry the forces measured in MuJoCo at
`qpos0` (gravity bias, Coulomb friction, spring preload), the thresholds, the couplings (one-sided latch tendon,
mimic equalities, welds, loop closures, MJCF servos) and the expected outcome per phase for `mjcf`, `usd_full`,
`usd_rl`.  The Isaac runner never recomputes a force; it only maps MJCF joint names to the joints of the file it runs.

### What the PhysX runner emulates (and records in `emulations_used`)

* **spring targets restored every step** — Isaac Lab zero-initialises position targets, which erased every USD spring
  preload in the first probe (closer doors opened under 60 N·m, levers sagged under gravity); the runner writes
  `doorbench:target_si` each step and applies efforts only through `set_joint_effort_target`
* **latch clamp (+ target)** — the one-sided MJCF tendon `bolt ≥ scale · operator` has no PhysX counterpart; the latch
  joint state is clamped to the tendon minimum every step (`write_joint_state_to_sim`) and, by default
  (`--latch-mode clamp+target`), the latch drive target follows that minimum while the tendon pulls — otherwise the
  300 N/m latch spring re-extends a 0.04 kg bolt by ~2.5 mm within one 1/120 s step and the recorded retraction
  chatters below the tendon minimum (the strike gap is 3 mm); `--latch-mode clamp` keeps the pure clamp
* **batch layout** — the doors of a batch sit on a centred 20 × 14 m grid (`--spacing`) on a ground plane sized to
  the grid: gate leaves sweep / slide up to 8.2 m from their origin and fences / floor-hatch decks extend up to
  9.9 m, so the 6 m grid of the first probe let neighbouring doors collide; batches group doors with the same phase
  schedule (`--no-group` to keep the `--doors` order) so a batch does not step 12 s of `closer` for one door
* **servo emulated** — MJCF position actuators of automatic doors (`ctrl = 0`) as clipped feed-forward effort
* **weld pinned hold** (opt-in `--emulate-weld`) — mag locks / delayed egress are MuJoCo `<weld>` equalities not
  exported to USD; by default the door is left free and the verdict reports `EXPORT_WELD_MISSING`

## Verdict and discrepancy codes

`compare_door` first compares the pass/fail status of every applicable phase, then the metrics of agreeing phases
against tolerances (hold 0.01 rad / 3 mm; opened within 20 % or 0.1 rad / 5 cm, `t_open` within 30 % or 0.3 s,
operator travel 10 %, bolt retraction 15 % of throw; release 2 mm / 0.2 s; relatch 1°; closer 2° / 30 % of closing
time; per-joint settle drift 0.02 rad / 5 mm).

| code | meaning |
|---|---|
| `OK` | every applicable phase agrees within tolerance |
| `PHYSX_NO_OPEN` | MuJoCo opens (free push or operator + push), PhysX does not |
| `PHYSX_HOLD_FAIL` | MuJoCo holds (latch / lock / locked handle), PhysX opens |
| `EXPORT_WELD_MISSING` | the hold relied on a MuJoCo weld (env-released lock) that is not in the USD |
| `LATCH_NO_RETURN`, `RELATCH_FAIL`, `CLOSER_NO_RETURN` | phase-specific PhysX failures with a MuJoCo pass |
| `SETTLE_DRIFT` | a joint moves during the free settle in one simulator only (e.g. lost spring preload) |
| `LIMIT_VIOLATION` | PhysX leaves an authored joint range that MuJoCo respects (MuJoCo's soft limits overshoot by a few degrees under hard pushes; only a PhysX overshoot > 2× MuJoCo's counts) |
| `NAN`, `LOAD_FAIL`, `STRUCTURE_FAIL` | non-finite state; spawn / inspection error; joint set, limits, gains or spring targets differ from model.json |
| `METRIC_DELTA` | statuses agree but a metric is outside tolerance (quantitative) |
| `RL_CANON` | door_rl.usda behaves differently by construction (welded lock parts, empty operator slot); informational |
| `MUJOCO_FAIL` | the reference itself fails a phase qa.json passed (protocol bug or nondeterminism, not a PhysX bug) |
| `INFO_DISAGREE` | disagreement on an informational phase (free-swing families, roller / magnetic catches) |

Grades per door and kind: **A** all phases agree within tolerance, **B** statuses agree but a metric is off
(`METRIC_DELTA`, `SETTLE_DRIFT`, `INFO_DISAGREE`), **C** a status disagreement or a limits / NaN failure,
**X** not comparable.  A door's grade is the worst of `full` and `rl`.

## Status

* MuJoCo reference: 1000/1000 doors pass every applicable phase and reproduce their qa.json metrics (`qa_push`,
  `hold_displacement`, `actuate_displacement`, `closer_final_angle`, ...) to 1e-3.  Verified independently on a
  seeded 61-door sample (2 per family): bit-identical records across worker counts, resumes and machines.
* Informational phases that fail in MuJoCo (`mujoco_summary.json` → `informational_fails`; families qa.py never
  pushed, so they are reported, not graded) — these are **door bugs the reference surfaced, not protocol bugs**:
  * accordion, 12/12: the panel couplings alternate `panel_i = ∓2·panel_0` but every panel hinge is authored with range
    `[-π, 0]`, so the even panels sit on their limit and the whole fold is kinematically locked (65 N·m moves the lead
    hinge 0.0009 rad; `qfrc_constraint` absorbs the full push, contacts carry no force)
  * revolving, 8/15: a wing stile touches `wall_header` at q0 (gap 0) and jams against it as the rotor turns
    (8.6 kN contact normal force; 3 doors do not move at all, 5 crawl < 0.12 rad in 6 s; the other 7 turn normally)
  * bifold, 3/30: the panel tops rub on `wall_header` (20–40 N normal force, zero gap) and the fold crawls ~0.1 rad in 6 s
  * cold-storage roller relatches (5): correct — a roller latch does not hold a re-push
* Behaviour that is *by construction* in both simulators and worth knowing when reading the metrics: closer doors run
  with the symmetric MJCF damping (`damping_opening` + air), because the asymmetric `damping_closing` / backcheck live
  only in `DoorEnv`'s passive callback and the USD carries them only as `doorbench:damping_closing` attributes — so
  `closer_t_close` is 0.6–1.8 s (median 1.07 s over 263 doors) against `closing_time_est_s` of 2–5 s and `slam` never
  fires; turnstile rotors have no indexing detent (`ratchet_deg` is spec-only), so a 68 N·m push spins a full-height
  rotor 1.5 rev/s; the qa.py push counts a closer's spring preload twice (`bias` = |qfrc_bias − qfrc_passive| already
  contains it), e.g. 311 N·m on db0012 — kept, because the gate must reproduce qa.json; doors with `rest_angle_deg`
  (10 stall doors) pass `hold` / `operate` trivially since they start open.
* PhysX side: written against Isaac Lab 2.3.2, **not executed on this machine** (no NVIDIA GPU).  First run:
  `bash isaaclab/cloud/parity.sh --limit 40`, then `scripts/parity_compare.py`.  Expect roughly 5-7 min per batch of
  20 doors (up to ~4200 physics steps per batch with a per-door Python loop) — about 8-10 h for 1000 doors × 2 kinds;
  `--retry-errors` re-runs doors whose record is a spawn / batch error, everything else is resumable.
* Known limits of the PhysX emulation (to verify on the first GPU run): joint Coulomb friction is authored twice in
  the USD (`physxJointAxis:*:staticFrictionEffort` and the legacy `physxJoint:jointFriction` coefficient; Isaac Lab
  exposes only the latter as `joint_friction_coeff`, recorded in `structure.friction_coeff_readback`); mimic-joint
  gearing units for revolute→prismatic couplings; `PhysxJointAxisAPI` friction efforts being honoured at all.
* Planned: write `isaac_parity` into each door's qa.json, a viewer badge, and fixing the export gaps the gate finds
  (latch tendon and welds as native PhysX constraints where possible).
