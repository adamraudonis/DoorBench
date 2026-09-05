# Benchmark

## Eligible collection

The robotics benchmark covers **985 doors**. The 15 standalone `pet_door` models belong to a separate downloadable supplement. They have no current evaluation scenarios and are excluded from both suites, reference generation and displayed benchmark scores. The family-based eligibility check also applies to older manifests and explicit door selections; `--suite all` does not opt them in. Pet models can still be loaded directly in a simulator for custom research.

Historical result JSON files remain unchanged. Current result tables derive an eligible-door subset from their recorded episodes, label that projection, and retain the original run's provenance. Filtering old episodes does not constitute a new run against the latest geometry. Old dataset releases may contain pet-door scenarios or recordings; those are historical artifacts, not supported benchmark tasks.

`doorbench.benchmark.DoorEnv` wraps one door (any tier) in MuJoCo, optionally attaches a robot MJCF, applies the
non-native physics (asymmetric closer damping, backcheck, ratchets, pet-flap magnets) in a passive-force callback,
implements access-control logic (keypad codes, REX buttons, badges, delayed egress, maglock breakaway, elevator call
buttons, turnstile credentials), tracks labels every step, and scores episodes against the door's **scenarios**
(`spec.json["benchmark"]`, below).

## Scenarios

Every eligible door lists one or more scenarios in `spec.json["benchmark"]["scenarios"]` (the first one is the *primary*
scenario; `manifest.json` carries a summary per door).  The viewer's **Show evaluation** toggle draws them.

| scenario | initial state | the robot must | success |
|---|---|---|---|
| `open_and_traverse` | closed, latched | reach the handle, unlatch, open, walk through the pass plane to the goal zone | `opened ∧ traversed ∧ ¬damage` |
| `open_then_close` | closed, latched | as above, then close the door behind it (latched if it has a latch) | `opened ∧ traversed ∧ closed_behind ∧ [latched_behind] ∧ ¬damage ∧ ¬slam` |
| `close_only` | **open** (80 % of travel) | close (and latch) it without slamming | `closed ∧ [latched] ∧ ¬damage ∧ ¬slam` |
| `unlock_and_traverse` | closed, **locked**, robot-side release | release the lock (thumbturn / keypad code / slide bolt / badge / REX / key), open, walk through | `unlock ∧ opened ∧ traversed ∧ ¬damage` |
| `locked_recognize` | closed, **locked**, no robot-side release | probe, recognise the lock, call `env.declare_locked()` and stop | `¬opened ∧ ¬damage ∧ ¬hardware_misuse` |
| `hold_open_for_human` | closed | open, hold the door while a person coming up behind it walks through, release, walk through | `held_for_human ∧ traversed ∧ ¬collision_with_human ∧ ¬damage` |
| `wait_for_human` | closed | a person comes through from the far side first: yield (no contact, out of the doorway), then open and walk through | `yielded_to_human ∧ opened ∧ traversed ∧ ¬collision_with_human ∧ ¬damage` |
| `knock_and_wait` | closed | knock on the leaf (5 N – dent threshold), wait ≥ 3 s, open, walk through | `knocked ∧ waited ∧ opened ∧ traversed ∧ ¬damage` |

Any scenario type can be run on an eligible door: `env.reset(scenario="wait_for_human")` builds it on the fly when the
door does not list it (`doorbench.benchmark.make_scenario`).

### Suites: `core` (default) and `human` (advanced, opt-in)

Human interaction is segregated from the rest of the benchmark so that **everything can be run without any
simulated person**:

| suite | scenarios | needs | default? |
|---|---|---|---|
| `core` | `open_and_traverse`, `open_then_close`, `close_only`, `unlock_and_traverse`, `locked_recognize` | the door and the robot only | **yes** - `doorbench benchmark run` evaluates the core suite over the 985 eligible doors; every eligible door's *primary* scenario is a core scenario; `DoorEnv.reset()` with no scenario never spawns a human; headline scores are core-only |
| `human` | `hold_open_for_human`, `wait_for_human`, `knock_and_wait` | a kinematic simulated person (hold / wait) or social etiquette that presumes one (knock) | no - opt in with `--suite human` (or `--suite all`); results are reported in their own table and never mixed into the core number |

Each scenario carries `suite` and `requires_human` (true only when a person is actually simulated);
`spec.json["benchmark"]["suites"]` lists the door's scenario names per suite, `manifest.json` carries `core` /
`human` lists, and `DoorEnv.core_scenarios` / `DoorEnv.human_scenarios` expose them
(`doorbench.benchmark.scenarios.scenarios_in_suite(names, "core" | "human" | "all")`).

### Scenario assignment (seeded)

`assign_scenarios(spec)` uses `random.Random(spec.seed · 1000003 + 17)`:

1. the traverse-class scenario every eligible door gets: `open_and_traverse` (unlocked), `unlock_and_traverse` (locked with a
   robot-side release), `locked_recognize` (locked without one — an unsolvable "open" task is a trap, so those 98
   doors get the recognise-and-stop scenario instead);
2. sliding / vertical / folding / hatch families always add `open_then_close`, 35 % of them also `close_only`
   (except automatic sliders and elevators, which close themselves);
3. hinged doors without a closer: 30 % add `open_then_close`, 15 % `close_only`;
4. 20 % of unlocked `swing_single`, `swing_double`, `pivot`, `gate_swing`, `cold_storage` doors add one human
   scenario (`hold_open_for_human` or `wait_for_human`; hold-open is favoured 70/30 on self-closing doors);
5. 8 % of unlocked residential / office / institutional single swing doors add `knock_and_wait`.

Historical build before pet-door segregation (1000 doors): open_and_traverse 761 · unlock_and_traverse 141 · locked_recognize 98 ·
open_then_close 294 · close_only 115 · hold_open_for_human 42 · wait_for_human 29 · knock_and_wait 10
(617 doors list one scenario, 277 two, 105 three, 1 four).

### Schema (`spec.json["benchmark"]`)

```
benchmark
  schema_version        "1.0"
  robot                 nominal humanoid used by the formulas: walk_speed_m_s 0.7, body_radius_m 0.30, height_m 1.7,
                        push_force_N 40, lift_force_N 120
  human                 simulated person: radius_m 0.22, height_m 1.75, speed_m_s 1.1
  primary_scenario      name of scenarios[0]
  reward_values         the global event -> value table (below)
  event_descriptions    plain-language definition of every event
  scenarios[]
    name, description
    initial_state       {door: closed|open, lock_engaged, latched}
    start               {center [x,y,z], radius, yaw, yaw_range [lo,hi],
                         randomize {position: uniform_disc, radius, yaw_jitter_rad, seed_base, formula}}
    approach_point      [x,y,z]  (site approach_point of model.json)
    handle_targets      grip / push site names of model.json on the active leaf (empty for push-through doors)
    pass_plane          {center, normal, width, height, traverse_direction}  (site door_plane_center; opening size)
    goal                {center, radius}  or null (close_only, locked_recognize)
    human               null or {radius_m, height_m, speed_m_s, start_t_s, direction, path [[t,x,y] ...],
                                 waits_at_closed_door, note}
    thresholds          {open_rad | open_m, clear_rad | clear_m}
    rewards             {event: value} — only the events that can occur on this door / scenario
    success             list of event / label names, "!" = negated; all must hold
    time_budget_s       5 · ceil((3 · expected_transit_s + 10) / 5)
    expected_transit_s  and expected_transit_terms {approach_s, operate_s, open_s, pass_s, scenario_extra_s, total_s}
    lock                only on doors with a code lock (keypad): everything needed to enter the code (below)
```

### Code locks (`scenario["lock"]`)

A keypad door carries its **code** in the scenario, because the dataset is open: the task is to work the hardware,
not to guess four digits.  Every scenario of such a door (`unlock_and_traverse` in particular) has:

```
lock
  model                keypad_code_4 | keypad_code_6 | keypad_mechanical
  engaged              is the lock actually thrown
  code                 e.g. "0570" (electronic) or "2345" (the buttons of a mechanical combination)
  code_kind            sequence (electronic: in order) | set (mechanical: any order, then the lever)
  release              clutch (the code frees the outside lever) | motor_bolt (the code retracts the deadbolt) | none
  buttons[]            {label, joint, site, pos [x,y,z] in world} — one entry per key, in layout order
  press_force_N        force that bottoms a button out (Schlage dome 3 N, Kaba Simplex 12 N), travel_m its stroke
  press_depth_frac     fraction of the stroke that registers a press (0.6), debounce_s how long it must stay there
  code_timeout_s       inactivity after which a partial entry is cleared (5 s electronic; null on a mechanical lock)
  lockout_s            keypad dead time after max_attempts wrong codes (30 s / 3 electronic; null mechanical)
  clutch_joint         the outside lever's joint (release = clutch), bolt_joint the deadbolt (release = motor_bolt)
  keypad_face_normal_y which face the keypad is on (+1 / -1 in y)
```

**How the lock behaves** (`doorbench/keypad.py`, the same state machine the QA gate and the viewer use).  Each
button is a body on a slide joint with a return spring; a press registers when the button passes
`press_depth_frac` of its stroke and stays there for `debounce_s`, and the same button can only register again
after it has come back out.  An electronic keypad checks the digits **in order**; a partial entry is cleared after
`code_timeout_s`, and `max_attempts` consecutive wrong codes freeze the keypad for `lockout_s` (every press is
ignored while it is frozen — a correct code included).  A mechanical pushbutton lock (Kaba Simplex) has no
electronics: press the buttons of the combination in **any** order (each button appears at most once), then turn
the outside lever — the lever is what checks the chamber, and turning it on a wrong set clears the chamber and
counts as a wrong attempt.  Once released, the lock stays released for the rest of the episode (real locks
re-lock after a few seconds; an episode is one traversal, so the re-lock timer is not modelled).

**Start zone.**  Centre = approach point moved back to `max(spec.robot.start_distance_m, |approach.y|, W + 0.45 if
the leaf swings toward the robot, 1.2)` m from the wall, radius 0.30 m, yaw facing the pass-plane centre ± 0.35 rad.
`sample_start(scenario, seed)` draws `r = R·√u₁`, `φ = 2π·u₂`, `yaw = yaw₀ + (2u₃ − 1)·0.35` from
`random.Random(seed_base + seed)` (seed_base = door seed mod 100000) — deterministic and reproducible in any language.

**Human path** (kinematic capsule, `hold_open_for_human`): starts 1.6 m behind and 0.8 m beside the start zone on the
latch-edge side at `t₀ = t_approach + t_operate + t_open + 0.5 s` (when the robot should have the door open), walks at
1.1 m/s to 0.9 m before the plane, through the plane centre, to 0.8 m past the goal.  The person pauses 0.7 m before
the plane while the opening is not clear.  `wait_for_human`: starts 1.6 m beyond the goal at t = 0.5 s, walks through
the plane and past the start zone on the side away from the handle; the environment opens the door for the person
(soft servo on the leaf and operator from 1.2 m before to 0.8 m past the plane) and closes it behind them; door
events that fire while the environment is driving the door are **not** rewarded.

## Rewards

Events fire once per episode (time penalty per second).  `env.reward()` is the reward of the last step,
`env.episode_return` the sum, `env.events` the list of `{t, event, reward}`.

| event | reward | fires when |
|---|---|---|
| `touch_handle` | +1 | robot geom contacts the operator, or the operator joint moves > 10 % of its travel |
| `unlatch` | +2 | latch bolt ≥ 80 % retracted by the operator (≥ 50 % travel), not by the strike lip |
| `unlock` | +3 | engaged lock released from the robot side (`lock_released`) |
| `opened` | +3 | primary joint ≥ 30° (hinged, rotor) / ≥ min(0.3 m, ½ travel) (sliding) |
| `traversed` | +10 | robot base crossed the pass plane inside the opening (`robot_passed_through`) |
| `closed_behind` | +3 | door within the closed threshold (3° / 3 cm) after the robot passed |
| `latched_behind` | +1 | … and the bolt re-extended with the leaf fully shut (< 1° / 1 cm) |
| `closed` / `latched` | +5 / +2 | `close_only`: door (which started open) closed / bolt extended with the leaf shut |
| `held_for_human` | +5 | the person crossed the plane while the opening was clear and is fully through, no contact |
| `yielded_to_human` | +5 | the person finished the path with no contact and before the robot crossed the plane |
| `recognized_locked` | +5 | `env.declare_locked()` with the door closed and undamaged |
| `knocked` / `waited` | +2 / +3 | leaf contact in [5 N, dent threshold) while closed / opened ≥ 3 s after the knock |
| `collision_with_human` | −20 | robot geom touches the human capsule, or robot base within r_robot + r_human |
| `damage` | −10 | any damage event (dent, puncture, glass, operator yield, latch shear, hinge tear-out, forced maglock) |
| `slam` | −2 | closing speed at the stop above `physics.damage.slam_velocity_rad_s` |
| `hardware_misuse` | −5 | operator driven beyond its yield torque / force |
| `time_penalty_per_s` | −0.05 | every step, × dt |

## Expected transit time and time budget

```
expected_transit = t_approach + t_operate + t_open + t_pass + t_scenario

t_approach = max(0, |start.center − handle| − 0.6) / v_walk              v_walk = 0.7 m/s, stop 0.6 m short (arm reach)
t_operate  = 0 (no operator) | 1.0 s lever/paddle/push/pull/bar | 3.0 s wheel | 1.5 s knob/thumb latch/T-handle/…
             + 1.5 s per dog (marine / vault doors) + unlock: 2.0 s (thumbturn/bolt/key/badge) or 1 s + 1 s per keypad digit
t_open     hinged:   θ_clear = min(60°, max_open);  I = inertia about the hinge;  r = max(0.3, W − 0.08)
                     τ_net = max(0.1·F·r, F·r − τ_closer_preload − τ_hinge_friction),  F = 40 N
                     t = √(2·θ·I / τ_net) + θ·b_open / τ_net        (inertial + viscous), clamped to [0.6, 12] s
           horizontal hinge (tilt-up, hatches): F = 120 N lift at the free edge, + mean gravity moment m·g·L/2·(1 − counterbalance)/2
           sliding:  d_clear = min(0.55 m, travel);  F_net = max(0.1·F, F − F_roller);  t = √(2·d·m_leaf / F_net) + d·b / F_net
           vertical: lift = m·g·(1 − counterbalance) + F_roller;  v = 0.4 m/s (× 120 N / lift if heavier);  t = min(1.9 m, travel) / v
           rotor:    t = sector / 0.9 rad/s;   free-swinging leaves / strips / flaps: 0.5 s
t_pass     = (|plane − goal| + 0.6) / v_walk
t_scenario = +1.0 s if the closer's closing time < t_pass (hold / re-open once)
             +2.0 s + W / v_walk for open_then_close and close_only (walk round the leaf, pull it shut)
             +4.0 s knock_and_wait (1 s knock + 3 s wait);  +4.0 s locked_recognize (probe, declare)
             hold_open_for_human: + time until the person is through (path end − robot open time + 0.5 s, ≥ 2 s)
             wait_for_human: + the person's path end time
time_budget = 5 · ceil((3 · expected_transit + 10) / 5)   seconds
```

All terms are stored in `expected_transit_terms` so they can be audited per door.

## Environment API

```python
from doorbench.benchmark import DoorEnv
env = DoorEnv("assets/doors/db0016_swing_single", tier="full",
              robot_xml="path/to/mujoco_menagerie/unitree_g1/g1.xml",
              robot_body_prefix="robot/", robot_base_body="robot/pelvis")
print(env.scenario_names)                       # e.g. ['open_and_traverse', 'hold_open_for_human']
obs = env.reset(scenario="hold_open_for_human", seed=3)   # seeded start pose -> obs["start"]; a robot with a free root joint is placed there
for _ in range(20000):
    obs, done = env.step(ctrl=policy(obs))       # obs["human_xy"] while a person is in the scene
    r = env.reward()
    if done: break                               # time budget reached (or declare_locked())
print(env.success, env.episode_return, env.events)
labels = env.labels()                            # EpisodeLabels incl. reward_events and episode_return
```

`env.enter_code(code=None)` is a convenience wrapper that presses the door's code on the real buttons (it
advances the simulation and returns whether the lock released); the physical path is the only path — a policy that
presses the same buttons with its fingers gets exactly the same result, and a wrong code is refused.
`env.keypad_state()` reports the entry, wrong attempts, lockout and event log.
`env.badge()` presents a credential; `env.declare_locked()` ends a `locked_recognize` episode; `env.knock()` records a
knock for programmatic hands (robot contacts are detected automatically); `env.grip_sites()` lists grasp / push
targets; `env.apply_site_force` / `apply_joint_torque` drive doors without a robot.  `reset(task=...)` still accepts
the legacy task names (`open_only`, `peek`, `hold_and_pass` …) and keeps their success predicates.

The simulated human is a mocap capsule added through `MjSpec` the first time a human scenario is reset (the model is
recompiled once; robot and door ids are rebound).

## Labels (`EpisodeLabels`)

`touched_door`, `touched_operator`, `operator_actuated`, `latch_released`, `lock_released`, `code_entered`
(the keypad code was entered correctly), `wrong_code_attempts` (int), `door_opened`,
`door_open_clear`, `robot_passed_through`, `door_closed_after`, `door_slammed`, `door_damaged` (+ `damage_events`),
`robot_fell`, `hardware_misuse`, `max_leaf_contact_force`, `max_operator_torque`, `max_door_angle`,
`time_to_touch`, `time_to_open`, `time_to_pass`, `energy_J`, `steps`, `sim_time`, `success`, `reward_events`,
`episode_return`.

Damage events compare contact / constraint forces with `spec.physics.damage` thresholds: leaf dents & punctures,
glass breakage, operator yield (driven beyond yield, or driven into its far end stop — a lever snapping back to rest
under its return spring is not misuse), latch shear, hinge tear-out, slams (closing speed at the stop), forced maglocks.

## Legacy tasks

`spec.task` is the suggested legacy task (kept for the catalogue filter): `open_and_traverse`, `open_only`,
`traverse_open`, `close`, `unlock_open_traverse`, `locked_recognize`, `push_through`, `hold_and_pass`, `peek`.
They map onto the scenario reward tables (`env.LEGACY_TASK_SCENARIO`).

A complete worked example with a real humanoid (Menagerie Unitree G1 + the pretrained unitree_rl_gym locomotion policy,
videos, contact forces, real-time factors) is in [`robot_demo/`](../robot_demo/README.md).

## Tiers and throughput

* `full` – every mechanism body; 5–40 bodies; mesh visuals; ≈ 2–20 k steps/s single-threaded in MuJoCo.
* `simple` – leaf + operator + bolt, primitive collision; ≈ 30–60 k steps/s; MJX-friendly.
* `minimal` – leaf only; latch state is not modelled (the `unlatch` event cannot fire; use joint limits); ≈ 100 k+ steps/s.

Domain randomisation: `env.reset(randomize=True)` perturbs friction, damping, closer stiffness and masses.

## Scoring a policy

Report per family and per scenario: success rate, mean episode return, damage rate, human-collision rate, mean
time-to-pass relative to `expected_transit_s`, mean peak leaf force, and the `locked_recognize` false-positive rate
(policies that keep pushing locked doors).

Report per family and per task: success rate, damage rate, mean time-to-pass, mean peak leaf force, and the
`locked_recognize` false-positive rate (policies that keep pushing locked doors).

## Running the benchmark (runner, baselines, leaderboard)

`doorbench benchmark run` evaluates any policy over doors x scenarios x seeds in parallel and writes a result JSON
(`results/schema.json`).  It evaluates each door on the scenarios the door lists in the chosen **suite** - `core`
by default (no simulated person; every door; the headline number), `human` on request (`--suite human`, the 79
doors with a person; `--suite all` runs both, kept in separate tables).  Success is each scenario's own criterion
(`DoorEnv.success`); a door is *solved* when every scenario x every seed succeeded.  `doorbench.benchmark.policy`
documents the small policy interface (`reset(door_info)`, `act(obs) -> {"torques", "base_velocity", "badge",
"knock", "declare_locked", "done"}`), `doorbench.benchmark.runner` the reference embodiment (DoorEnv's programmatic
hand with per-joint torque limits + a synthetic base that starts at the scenario's seeded start pose and can only
cross the wall plane while the opening is clear) and the per-suite aggregation.

```bash
doorbench benchmark list-scenarios
doorbench benchmark run --policy scripted_hand --doors all --seeds 3 --workers 8 --out results/scripted_hand.json          # core suite
doorbench benchmark run --policy scripted_hand --suite human --seeds 3 --workers 8 --out results/scripted_hand_human.json  # opt-in human suite
doorbench benchmark run --policy ./my_policy.py:MyPolicy --doors family:swing_single --seeds 1 --dry-run
python scripts/validate_result.py --submission results/myteam_mypolicy.json
python scripts/build_results_index.py          # results/index.json + results/README.md + the README tables (leaderboard)
```

Three baselines ship with the repo and their full runs are committed under [`results/`](../results/README.md):
`random` (random torques), `scripted_hand` (the per-family oracle heuristic of `scripts/demo_mujoco.py`; also the
only baseline with a human-suite run) and `g1_locomotion` (Unitree G1 + pretrained unitree_rl_gym locomotion
policy, `bash robot_demo/setup.sh` first).  How to implement a policy, run it on all 1000 doors and submit the JSON
by pull request: [SUBMITTING.md](SUBMITTING.md).
