# Running the benchmark and submitting a result

DoorBench evaluates a policy on all 1000 doors, each with its own task (`spec.task`), several seeds, in MuJoCo.
The whole loop is in this repository: a small policy interface, a parallel runner, a JSON result format with a
schema + validator, and a PR-based leaderboard (`results/`, shown on the
[Results page](https://adamraudonis.github.io/DoorBench/#/results) of the site).

## 1. Install

```bash
git clone https://github.com/adamraudonis/DoorBench.git && cd DoorBench
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[all]"                  # mujoco, numpy, ... (the dataset is in assets/)
doorbench benchmark list-scenarios
doorbench benchmark list-policies           # random | scripted_hand | g1_locomotion
```

Try a baseline on a few doors first (about 0.2 s of wall time per episode):

```bash
doorbench benchmark run --policy scripted_hand --doors family:vault --seeds 1 --workers 4 --out /tmp/vault.json
doorbench benchmark run --policy random --doors sample:50:0 --seeds 3 --dry-run       # prints the door list, runs nothing
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
        self.op = info["operator_joints"][:1]          # e.g. ["leaf_handle_hinge"]
        self.leaf = info["primary_joint"]              # e.g. "leaf_hinge"
        self.push = info["torque_limits"][self.leaf]   # the hand's strength on this door

    def act(self, obs):
        torques = {j: 4.0 for j in self.op}            # N*m on a lever / knob (clamped by torque_limits)
        if obs["t"] > 1.0:
            torques[self.leaf] = 0.5 * self.push        # push the leaf
        walk = obs["flags"]["door_open_clear"]         # the opening is wide enough for the base
        return {"torques": torques, "base_velocity": [0.0, 1.0 if walk else 0.0]}
```

What the policy sees and does (reference embodiment: DoorEnv's programmatic hand + a synthetic base):

* `door_info` (at reset): the door's full `spec.json`, `model.json` meta, every robot-interactive joint with its
  role / type / range, the lock state (and the keypad code when the robot is allowed to know it), the per-joint
  torque limits, approach / goal points.  Lock parts on the far side of the door and operators the robot cannot
  reach (an exit device seen from the pull side) have a limit of 0.
* `obs` (every `control_dt`): sim time, primary / secondary joint position and velocity, every interactive joint
  `{q, dq}`, grip / push site positions, the base position, and the label flags reached so far
  (`touched`, `operator_actuated`, `latch_released`, `lock_released`, `door_opened`, `door_open_clear`,
  `passed_through`, `damaged`).
* `action`: `torques` (generalized forces on named joints, N*m / N), `base_velocity` (m/s, <= 1.5),
  `badge` (present a credential where the door has one), `done` (stop early, e.g. "this door is locked").
* The base is a point that walks with the commanded velocity; it can only cross the wall plane while the opening
  is clear (hinge >= 60 deg, slide >= 0.55 m or 95 % of travel, overhead >= 1.9 m) and is >= 0.45 m wide.

Policies that bring their own robot (`embodiment = "robot"`, `make_env()` attaching a robot MJCF, actions as
`ctrl` arrays) are supported too; `doorbench/benchmark/baselines/g1_locomotion.py` is the worked example.

Load it by module or by file path:

```bash
doorbench benchmark run --policy my_pkg.policies:LeverAndPush --doors first:20 --seeds 1
doorbench benchmark run --policy ./my_policy.py:LeverAndPush --doors first:20 --seeds 1
```

## 3. Run the full benchmark

```bash
doorbench benchmark run --policy ./my_policy.py:LeverAndPush \
    --doors all --seeds 3 --scenarios default --tier full --workers 8 \
    --label "team X, RTX 4090, MuJoCo 3.12" \
    --out results/teamx_leverpush.json
```

* `--doors all` and `--seeds 3` (seeds 0, 1, 2) are required for the leaderboard.  Seed 0 is the nominal door;
  seeds >= 1 randomise hinge friction, damping, closer stiffness and masses (`DoorEnv.reset(randomize=True)`) and
  jitter the base start.  A door counts as **solved** only if every seed succeeded.
* `--scenarios default` evaluates each door's own task with a 20 s budget (40 s for delayed-egress doors).  Other
  scenarios (`traverse`, `traverse_close`, `hold_and_pass`) may be added but do not replace `default`.
* The whole run over 1000 doors x 3 seeds takes a few minutes with the reference hand on 8 CPU cores; a
  full-tier MuJoCo episode averages 0.1-0.5 s.  Robot embodiments are slower (the G1 baseline: ~1.5 s / episode).
* Every episode is deterministic for a given door, seed and policy; `--wall-timeout` guards against runaway
  policies (such episodes are reported as `timeout`).

The result JSON has one entry per episode (outcome, timestamped events, time-to-traverse, damage events, peak
forces, energy) and an aggregate (overall, per family, per difficulty, per task, per lock state).

## 4. Validate

```bash
python scripts/validate_result.py results/teamx_leverpush.json              # schema + internal consistency
python scripts/validate_result.py --submission results/teamx_leverpush.json # + the leaderboard rules
python scripts/build_results_index.py                                       # regenerates results/index.json, results/README.md and the README "Baseline results" tables
```

## 5. Submit (pull request)

1. Fork, add `results/<team>_<policy>.json` (the file name must contain `policy.name`), run
   `python scripts/build_results_index.py` and commit `results/index.json` + `results/README.md` with it.
2. Open a PR titled `results: <team> <policy>`.  In the description, say what the policy is (learned / scripted /
   robot model, training data, compute), and link code if you can.
3. CI (`.github/workflows/validate-results.yml`) validates the file against `results/schema.json`, applies the
   rules below, and checks that the index was regenerated.  Once merged, the site shows your row, the per-family
   bars, and a per-door badge in the catalogue.

### Rules

* **All 1000 doors, >= 3 seeds, the `default` scenario**, evaluated with `doorbench benchmark run` (or a runner
  that produces the same JSON and follows the same embodiment rules).
* Report the **simulator and version** (`run.simulator`, `run.simulator_version`), the **tier** (`full` is the
  reference; `simple` and `minimal` are accepted and shown), and the **DoorBench commit hash** the assets came
  from (`benchmark.commit`).  Results are comparable within a simulator + tier column.
* **No edits to the door assets** (`assets/`), the environment (`doorbench/benchmark/env.py`, `labels.py`),
  the scenarios, or the success predicates.  Policies may read anything in `door_info` but must not touch the
  simulator state directly except through the documented action (robot embodiments: through their own actuators).
* Torque limits, base speed and the clear-opening rule of the reference embodiment are part of the benchmark.
  Robot embodiments must name the robot model and the policy checkpoint (`policy.extra`), which must be
  obtainable (URL or commit).
* One row per (policy, embodiment, simulator, tier).  Re-submissions replace the previous file.
* Be honest about what the policy is: a scripted oracle that reads joint names from the spec is welcome on the
  board, labelled as such (the shipped `scripted_hand` is one).

Questions and problems: open an issue.
