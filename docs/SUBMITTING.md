# Running the benchmark and submitting a result

DoorBench evaluates a policy on all 1000 doors, each on the **scenarios it lists** (`spec.json["benchmark"]`, see
[BENCHMARK.md](BENCHMARK.md)), over several seeds, in MuJoCo.  The whole loop is in this repository: a small policy
interface, a parallel runner, a JSON result format with a schema + validator, and a PR-based leaderboard
(`results/`, shown on the [Results page](https://adamraudonis.github.io/DoorBench/#/results) of the site).

## 0. Suites: `core` is the benchmark, `human` is the advanced add-on

Scenarios come in two suites and **are never mixed**:

| suite | scenarios | what is in the scene | how it is run |
|---|---|---|---|
| **core** (default) | `open_and_traverse`, `open_then_close`, `close_only`, `unlock_and_traverse`, `locked_recognize` | the door and the robot, nobody else | `doorbench benchmark run ...` - every door lists at least one core scenario (its primary one), so the core suite covers all 1000 doors; **the headline "N / 1000 doors" number is the core suite** |
| **human** (advanced, opt-in) | `hold_open_for_human`, `wait_for_human`, `knock_and_wait` | a kinematic simulated person the robot must hold the door for, yield to, or knock before | `--suite human` (79 doors list one of these) or `--suite all`; reported in its own table, never part of the core number |

Every episode in a result file carries its `suite`; the aggregate has one table per suite (`aggregate.core`,
`aggregate.human`); the validator rejects a table that mixes the two.  The core suite needs no human-interaction
code at all - a policy that never looks at `obs["human_xy"]` is a complete core submission.

## 1. Install

```bash
git clone https://github.com/adamraudonis/DoorBench.git && cd DoorBench
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[all]"                  # mujoco, numpy, ... (the dataset is in assets/)
doorbench benchmark list-scenarios          # the 8 scenario types and their suite
doorbench benchmark list-policies           # random | scripted_hand | g1_locomotion
```

Try a baseline on a few doors first (about 0.3 s of wall time per episode):

```bash
doorbench benchmark run --policy scripted_hand --doors family:vault --seeds 1 --workers 4 --out /tmp/vault.json
doorbench benchmark run --policy random --doors sample:50:0 --seeds 3 --dry-run       # prints door x scenario list, runs nothing
doorbench benchmark run --policy scripted_hand --suite human --doors first:200 --seeds 1 --out /tmp/human.json
```

## 2. Implement a policy

A policy is a Python class with `reset(door_info, env=None)` and `act(obs) -> action`
(`doorbench/benchmark/policy.py` documents every field; `doorbench/benchmark/baselines/` has three worked examples).

```python
# my_policy.py
from doorbench.benchmark.policy import Policy

class LeverAndPush(Policy):
    name = "leverpush"                  # leaderboard name: [A-Za-z0-9_.-]
    description = "press the operator, push the leaf, walk through when clear"
    control_dt = 0.01                   # act() is called every 10 ms; torques are held in between

    def reset(self, info, env=None):
        self.scenario = info["scenario"]               # e.g. "open_and_traverse" (info["suite"] == "core")
        self.op = info["operator_joints"][:1]          # e.g. ["leaf_handle_hinge"]
        self.leaf = info["primary_joint"]              # e.g. "leaf_hinge"
        self.push = info["torque_limits"][self.leaf]   # the hand's strength on this door

    def act(self, obs):
        if self.scenario == "locked_recognize" and obs["t"] > 4.0:
            return {"declare_locked": True}            # "this door is locked": ends the episode
        torques = {j: 4.0 for j in self.op}            # N*m on a lever / knob (clamped by torque_limits)
        if obs["t"] > 1.0:
            torques[self.leaf] = 0.5 * self.push        # push the leaf
        walk = obs["flags"]["door_clear_now"]          # the opening is wide enough for the base right now
        return {"torques": torques, "base_velocity": [0.0, 1.0 if walk else 0.0]}
```

What the policy sees and does (reference embodiment: DoorEnv's programmatic hand + a synthetic base):

* `door_info` (at reset): the scenario (`scenario`, `suite`, `scenario_spec` with its start zone, handle targets,
  pass plane, goal, rewards, success criteria and time budget), the sampled `start` pose, the door's full
  `spec.json`, `model.json` meta, every robot-interactive joint with its role / type / range, the lock state (and
  the keypad code when the robot is allowed to know it), the per-joint torque limits.  Lock parts on the far side of
  the door and operators the robot cannot reach (an exit device seen from the pull side, a keypad or the trim it
  releases on the far face of the door) have a limit of 0.
* `obs` (every `control_dt`): sim time, primary / secondary joint position and velocity, every interactive joint
  `{q, dq}`, grip / push site positions, the base position, the label flags reached so far (`touched`,
  `operator_actuated`, `latch_released`, `lock_released`, `door_opened`, `door_open_clear`, `door_clear_now`,
  `passed_through`, `closed_after`, `slammed`, `damaged`), the reward events fired so far and the return,
  `success` (whether the scenario criterion holds right now) and - human suite only - `human_xy`.
* `action`: `torques` (generalized forces on named joints, N*m / N), `base_velocity` (m/s, <= 1.5),
  `badge` (present a credential where the door has one), `knock` (knock on the closed leaf), `declare_locked`
  (`locked_recognize`: fires `recognized_locked` and ends the episode), `done` (stop early).
* The base is a point that walks with the commanded velocity from the scenario's seeded start pose; it can only
  cross the wall plane while the opening is clear **right now** (hinge >= 60 deg, slide >= 0.55 m or the travel,
  overhead >= 1.9 m) and is >= 0.45 m wide.  In the human suite it collides with the person when it comes within
  0.52 m of them.

Policies that bring their own robot (`embodiment = "robot"`, `make_env()` attaching a robot MJCF, actions as
`ctrl` arrays) are supported too; `doorbench/benchmark/baselines/g1_locomotion.py` is the worked example.

Load it by module or by file path:

```bash
doorbench benchmark run --policy my_pkg.policies:LeverAndPush --doors first:20 --seeds 1
doorbench benchmark run --policy ./my_policy.py:LeverAndPush --doors first:20 --seeds 1
```

## 3. Run the full benchmark

```bash
# core suite (the leaderboard): all 1000 doors, every core scenario each door lists, 3 seeds
doorbench benchmark run --policy ./my_policy.py:LeverAndPush \
    --doors all --seeds 3 --tier full --workers 8 \
    --label "team X, RTX 4090, MuJoCo 3.12" \
    --out results/teamx_leverpush.json

# human suite (optional, advanced): the 79 doors with a human scenario, own file
doorbench benchmark run --policy ./my_policy.py:LeverAndPush --suite human --seeds 3 --workers 8 \
    --out results/teamx_leverpush_human.json
```

* `--doors all` and `--seeds 3` (seeds 0, 1, 2) are required for the leaderboard; leave `--scenarios` unset so
  every scenario the door lists is evaluated (1409 core episodes per seed over the 1000 doors).  Seed 0 is the
  nominal door; seeds >= 1 randomise hinge friction, damping, closer stiffness and masses
  (`DoorEnv.reset(randomize=True)`) and draw a different start pose from the start zone.  A door counts as
  **solved** only if every scenario x every seed succeeded.
* Success is each scenario's own criterion (`DoorEnv.success`, the `success` list in the scenario block):
  `opened & traversed & !damage` for `open_and_traverse`, `... & closed_behind & [latched_behind] & !slam` for
  `open_then_close`, `closed & [latched] & !damage & !slam` for `close_only`, `unlock & opened & traversed & !damage`
  for `unlock_and_traverse`, `!opened & !damage & !hardware_misuse` for `locked_recognize`.  Each scenario carries
  its own time budget (`5 * ceil((3 * expected_transit + 10) / 5)` s, typically 20-60 s).
* The whole core run over 1000 doors x 3 seeds takes a few minutes with the reference hand on 8 CPU cores; a
  full-tier MuJoCo episode averages 0.2-0.5 s.  Robot embodiments are slower (the G1 baseline: ~2-3 s / episode).
* Every episode is deterministic for a given door, scenario, seed and policy; `--wall-timeout` guards against
  runaway policies (such episodes are reported as `timeout`).  `--suite all` puts both suites in one file, in
  separate tables.

The result JSON has one entry per episode (outcome, the scenario criteria and which held, timestamped label events,
reward events and return, time-to-traverse, damage events, peak forces, energy) and one aggregate table per suite
(overall, per family, per scenario, per difficulty, per lock state; human-collision rate in the human table).

## 4. Validate

```bash
python scripts/validate_result.py results/teamx_leverpush.json              # schema + internal consistency (suites never mixed)
python scripts/validate_result.py --submission results/teamx_leverpush.json # + the leaderboard rules
python scripts/build_results_index.py                                       # regenerates results/index.json, results/README.md and the README "Baseline results" tables
```

## 5. Submit (pull request)

1. Fork, add `results/<team>_<policy>.json` (the file name must contain `policy.name`; a human-suite run goes in
   `results/<team>_<policy>_human.json`), run `python scripts/build_results_index.py` and commit
   `results/index.json` + `results/README.md` (+ the regenerated README tables) with it.
2. Open a PR titled `results: <team> <policy>`.  In the description, say what the policy is (learned / scripted /
   robot model, training data, compute), and link code if you can.
3. CI (`.github/workflows/validate-results.yml`) validates the file against `results/schema.json`, applies the
   rules below, and checks that the index was regenerated.  Once merged, the site shows your row (core table; human
   table if you ran it), the per-scenario and per-family bars, and a per-door badge in the catalogue.

### Rules

* **Core suite: all 1000 doors, every core scenario each door lists, >= 3 seeds, each scenario's own time budget**,
  evaluated with `doorbench benchmark run` (or a runner that produces the same JSON and follows the same embodiment
  rules).  A human-suite run must cover all 79 doors that list a human scenario, on every human scenario they list.
* Report the **simulator and version** (`run.simulator`, `run.simulator_version`), the **tier** (`full` is the
  reference; `simple` and `minimal` are accepted and shown), and the **DoorBench commit hash** the assets came
  from (`benchmark.commit`).  Results are comparable within a simulator + tier column.
* **No edits to the door assets** (`assets/`), the environment (`doorbench/benchmark/env.py`, `labels.py`,
  `scenarios.py`) or the success criteria.  Policies may read anything in `door_info` but must not touch the
  simulator state directly except through the documented action (robot embodiments: through their own actuators).
* Torque limits, base speed and the clear-opening rule of the reference embodiment are part of the benchmark.
  Robot embodiments must name the robot model and the policy checkpoint (`policy.extra`), which must be
  obtainable (URL or commit).
* One row per (policy, embodiment, simulator, tier) and suite.  Re-submissions replace the previous file.
* Be honest about what the policy is: a scripted oracle that reads joint names from the spec is welcome on the
  board, labelled as such (the shipped `scripted_hand` is one).

Questions and problems: open an issue.
