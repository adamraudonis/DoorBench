# Humanoid in the loop: Unitree G1 walking through DoorBench doors

For the Isaac Sim version and a researcher policy plug-in, use the [step-by-step Isaac G1 guide](../docs/ISAAC_G1_DEMO.md). The results below describe the original MuJoCo demo.

A real, off-the-shelf humanoid simulation driving a DoorBench door in plain MuJoCo on a CPU:

* **Robot**: MuJoCo Menagerie [`unitree_g1/g1.xml`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)
  (29-dof G1 rev 1.0, BSD-3-Clause), merged into the door scene with `mujoco.MjSpec.attach` through DoorBench's
  `DoorEnv` (so closer / ratchet / lock logic and the `LabelTracker` labels apply unchanged).
* **Policy**: the pretrained sim2sim locomotion policy shipped with
  [`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym) (`deploy/pre_train/g1/motion.pt`, TorchScript
  LSTM + MLP actor, 47-d observation, 12 leg actions at 50 Hz over a 500 Hz PD loop, BSD-3-Clause). The
  observation / action construction is a straight port of `deploy/deploy_mujoco/deploy_mujoco.py`.
* **Compute**: Apple M4 Mac mini, CPU only, one torch thread. No GPU was needed at any point.

Commit hashes and licences: [LICENSES.md](LICENSES.md). Nothing third-party is committed; `setup.sh` fetches it.

| open doorway (`db0119`) | automatic sliding door (`db0990`) |
|---|---|
| ![](../docs/media/g1_door_db0119.gif) | ![](../docs/media/g1_door_db0990.gif) |
| **saloon pair, pushed open (`db0123`)** | **latched push door: locomotion alone cannot open it (`db0705`)** |
| ![](../docs/media/g1_door_db0123.gif) | ![](../docs/media/g1_door_db0705.gif) |

Each clip is `door iso camera | follow camera`; mp4 versions sit next to the GIFs in `docs/media/`.

## Setup and exact commands

```bash
# from the repo root, inside the project venv (mujoco >= 3.1 already installed by `pip install -e ".[all]"`)
bash robot_demo/setup.sh                      # clones Menagerie (sparse: unitree_g1, unitree_h1) + unitree_rl_gym at pinned commits, installs torch/pyyaml/imageio
pip install -e ".[robot]"                     # alternative to the pip step in setup.sh

python robot_demo/run_g1_door.py --door db0119_swing_single --door-open-frac 1.0 --task traverse_open   # 1. open doorway
python robot_demo/run_g1_door.py --door db0990_automatic_sliding                                        # 2. sensor-opened slider
python robot_demo/run_g1_door.py --door db0123_saloon                                                   # 3. push through a saloon pair
python robot_demo/run_g1_door.py --door db0705_swing_single --duration 10                               # 4. closed, latched push door (negative case)

python robot_demo/run_g1_door.py --door db0123_saloon --robot rlgym --no-video   # cross-check with unitree_rl_gym's own g1_12dof.xml
python robot_demo/run_g1_door.py --help
```

Outputs: `docs/media/g1_door_<id>.mp4` + `.gif` (1280x480 side-by-side, 30 fps mp4 / 12 fps gif) and
`robot_demo/results/g1_door_<id>.json` (labels, timings, contact forces, 50 Hz base trajectory + door joint + command).
The script adds the repo root to `sys.path`, so no `PYTHONPATH` is needed.

## What each video shows

Robot starts at the door's `approach_point` (0, -1.5, 0) facing +y, stands for 1 s, then walks with a 0.5 m/s
forward command; a P-controller on the yaw-rate command steers it toward the door centre line and on to `goal_point`
(0, +1.5). Numbers below are from `robot_demo/results/*.json` (deterministic; MuJoCo 3.12, torch 2.14).

| door | what happens | passed door plane | peak robot-door contact | door state | RTF physics+policy | RTF incl. video |
|---|---|---|---|---|---|---|
| `db0119_swing_single` lever + mortise latch, 1.067 m, started fully open (100°) | walks straight through the opening without touching leaf or frame; `traverse_open` success | t = 5.28 s | 0 N | stays at 100° | 18.7x | 1.5x |
| `db0990_automatic_sliding` bi-parting storefront glass, microwave motion sensor (1.8 m range, 0.3 m/s open, 2 s hold) | sensor emulation fires at t = 0.82 s when the robot is inside the range, both leaves slide to full travel (0.98 m) by t ≈ 4 s, robot passes with no contact; `open_and_traverse` success | t = 5.28 s | 0 N | 0.98 m (full) | 19.2x | 1.5x |
| `db0123_saloon` pine pair, 0.6 m leaves, double-acting spring hinges 5 N·m/rad, worn | robot walks into the leaves at t = 4.8 s, torso + arms push them to 48° (0.84 rad); it is slowed for ~4 s while the springs load up, keeps its balance, breaks through and reaches the goal; leaves swing back (rest at -7° because coulomb friction beats the spring near zero); `push_through` success | t = 5.43 s | 240 N on `leaf_a_slab` (dent threshold 700 N, no damage) | max 0.84 rad, re-closed | 16.7x | 1.7x |
| `db0705_swing_single` porcelain knob, mortise latch, push side, closed | robot walks into the closed leaf at t = 4.8 s; the latch holds (max leaf angle 0.12°), the robot keeps stepping in place against the door for the rest of the episode and stays upright; `open_and_traverse` fails, as it must: the policy has no arms | never | 423 N on the slab (no damage) | 0.002 rad | 14.7x | 1.6x |

`upright_at_end` is true and `robot_fell` false in all four runs. The cross-check with unitree_rl_gym's own 12-dof
model on the saloon door also passes (peak 358 N, 35x real time without video).

## Real-time factor on CPU

Per run (11–14 s of simulation): physics + policy + labels take 0.6–0.8 s of wall time, i.e. **15–19x real time**
with the 29-dof Menagerie model and a full-tier door (105 geoms, mesh collision), **35x** with the 12-dof model.
Policy inference is 0.13–0.17 ms per 50 Hz step. Offscreen rendering of two 640x480 views at 30 fps dominates
(6.5–7.5 s per run), so a complete run with video is 7–8 s of wall time (RTF ≈ 1.5x). GPU rental was therefore not
needed; it would only become relevant for *training* a loco-manipulation policy on the door tasks.

## How the pieces fit (`run_g1_door.py`)

* `G1DoorEnv(DoorEnv)` overrides `_merge_with_robot`: reads the `stand` keyframe (waist + arm hold pose), deletes the
  robot's own lights / keyframes, converts the 12 leg `<position>` actuators to torque motors (`MjsActuator.set_to_motor`,
  torques are still clamped by each joint's `actuatorfrcrange`), then attaches the robot spec at a frame placed on the
  `approach_point` site with a 90° yaw. Waist and arms stay position-servoed (kp = 500) at the keyframe pose.
* `G1Policy`: config + TorchScript policy from unitree_rl_gym; `torque()` is the 500 Hz PD law, `act()` builds the
  47-d observation (base angular velocity in the base frame, gravity direction in the base frame, scaled command,
  joint position offsets from the default pose, scaled joint velocities, previous action, sin/cos of a 0.8 s gait
  phase) and runs the network every 10 physics steps. Reloading the module resets the LSTM state.
* `AutoDoorController`: for `automatic_*` families, drives the door's `<position>` actuators like a presence sensor
  (open while the base is within `spec.kinematics.actuator.sensor_range_m` of `door_plane_center`, hold
  `hold_open_s`, ramp at `open_speed_m_s` / `close_speed_m_s`). `DoorEnv` itself has no sensor logic yet.
* `Recorder`: `mujoco.Renderer` offscreen, door `iso` camera + a free camera tracking the pelvis, `imageio` H.264 mp4
  and an ffmpeg palette GIF.

## Limitations

* **Locomotion only.** The policy drives the legs; arms are parked. It can traverse open doorways, sensor-operated
  doors and free-swinging push doors, but it cannot reach for a lever, knob, bar, keypad or bolt. Every latched or
  locked DoorBench door (the large majority of the 1000) is therefore out of scope for this controller; the `db0705`
  clip shows the honest failure mode. Those tasks need a whole-body / loco-manipulation policy trained on the doors
  (the `simple` / `minimal` tiers are meant for that), which is exactly what the benchmark is for.
* The upper body is held by the Menagerie position servos rather than being welded as in the policy's training
  model; behaviour matched the 12-dof model in the cross-check, but the policy was never trained with a 29-dof body.
* Heading is a hand-written P-controller on the yaw-rate command (the policy has no perception); lateral velocity
  commands are not used.
* The automatic-door sensor is emulated by the script; `wave_to_open` / `push_button` sensor kinds are treated as
  presence sensors. Breakout (`breakout_force_N`) is not exercised.
* Videos are single deterministic runs, not statistics. `env.reset(randomize=True)` is available for sweeps.
* H1 (`unitree_h1`, `deploy/pre_train/h1/motion.pt`) is cloned by `setup.sh` but not wired up; the G1 worked first
  try so it was not needed.
