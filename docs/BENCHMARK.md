# Benchmark

`doorbench.benchmark.DoorEnv` wraps one door (any tier) in MuJoCo, optionally attaches a robot MJCF, applies the
non-native physics (asymmetric closer damping, backcheck, ratchets, pet-flap magnets) in a passive-force callback,
implements access-control logic (keypad codes, REX buttons, badges, delayed egress, maglock breakaway, elevator call
buttons, turnstile credentials) and tracks labels every step.

## Tasks

| task | start state | success |
|---|---|---|
| `open_and_traverse` | closed, latched | opened and robot base crossed the door plane, no damage |
| `open_only` | closed | opened past the clearance angle / travel |
| `traverse_open` | open | passed through without touching the door |
| `close` | open | closed and latched (no slam) |
| `unlock_open_traverse` | locked, robot-side release available | lock released, opened, passed |
| `locked_recognize` | locked, no robot-side release | no opening, no damage, no hardware misuse |
| `push_through` | free-swinging (saloon, strips, flap, turnstile, revolving) | passed through |
| `hold_and_pass` | self-closing door | opened, held, passed before it closed, no slam |
| `peek` | closed | opened between 10° and the clearance angle, held, did not pass |

Each door's `spec.task` is a suggested task consistent with its lock state; any task can be requested at `reset()`.

## Labels (`EpisodeLabels`)

`touched_door`, `touched_operator`, `operator_actuated`, `latch_released`, `lock_released`, `door_opened`,
`door_open_clear`, `robot_passed_through`, `door_closed_after`, `door_slammed`, `door_damaged` (+ `damage_events`),
`robot_fell`, `hardware_misuse`, `max_leaf_contact_force`, `max_operator_torque`, `max_door_angle`,
`time_to_touch`, `time_to_open`, `time_to_pass`, `energy_J`, `steps`, `sim_time`, `success`.

Damage events compare contact / constraint forces with `spec.physics.damage` thresholds: leaf dents & punctures,
glass breakage, operator yield, latch shear, hinge tear-out, slams (closing speed at the stop), forced maglocks.

## Using a robot

```python
from doorbench.benchmark import DoorEnv
env = DoorEnv("assets/doors/db0016_swing_single", tier="full",
              robot_xml="path/to/mujoco_menagerie/unitree_g1/g1.xml",
              robot_body_prefix="robot/", robot_base_body="robot/pelvis")
obs = env.reset(task="open_and_traverse")
for _ in range(4000):
    obs, done = env.step(ctrl=policy(obs))
    if done: break
labels = env.labels()
```
The robot is attached 1.5 m in front of the door (`approach_point`).  `env.badge()` presents a credential;
`env.unlocked_by_env` can be set by task logic.  `env.grip_sites()` lists grasp / push targets on the hardware.

A complete worked example with a real humanoid (Menagerie Unitree G1 + the pretrained unitree_rl_gym locomotion policy,
videos, contact forces, real-time factors) is in [`robot_demo/`](../robot_demo/README.md).

## Tiers and throughput

* `full` – every mechanism body; 5–40 bodies; mesh visuals; ≈ 2–20 k steps/s single-threaded in MuJoCo.
* `simple` – leaf + operator + bolt, primitive collision; ≈ 30–60 k steps/s; MJX-friendly.
* `minimal` – leaf only; latch state is not modelled (use joint limits); ≈ 100 k+ steps/s.

Domain randomisation: `env.reset(randomize=True)` perturbs friction, damping, closer stiffness and masses.

## Scoring a policy

Report per family and per task: success rate, damage rate, mean time-to-pass, mean peak leaf force, and the
`locked_recognize` false-positive rate (policies that keep pushing locked doors).
