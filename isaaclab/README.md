# DoorBench × Isaac Lab

Train a policy to open **hundreds of different DoorBench doors at once** in NVIDIA Isaac Lab: every environment of the
vectorised scene holds a different door (`MultiUsdFileCfg`), the rewards are the benchmark events, and the agent is
either a 6-DoF gantry hand (fast, validates the door mechanics) or the Unitree G1 humanoid.

> **Status:** written and statically validated on a Mac without an NVIDIA GPU. The USD side is verified (1000/1000
> doors pass the pxr validator, see [`STATUS.md`](STATUS.md)); the Isaac Lab side is syntax/API-checked against
> Isaac Lab v2.3.0 but **has not been executed**. Expect small fixes on the first GPU run — the scripts are thin and
> mirror Isaac Lab's own train/play scripts to keep that cheap.

## Ultra-short path (fresh Ubuntu 22.04 GPU box)

```bash
git clone https://github.com/adamraudonis/DoorBench.git && cd DoorBench
bash isaaclab/cloud/setup.sh                   # Isaac Sim 5.1 (pip) + Isaac Lab v2.3.0 + DoorBench, ~15 min
source isaaclab/cloud/env.sh
bash isaaclab/cloud/validate.sh                # headless import of all 1000 doors -> assets/usd_validation_isaacsim.json
bash isaaclab/cloud/train.sh --task DoorBench-Open-Hand-v0 --num_envs 1024 --max_iterations 300
bash isaaclab/cloud/hero.sh                    # 512 doors in one scene -> docs/media/isaaclab_hero.{png,mp4}
bash isaaclab/cloud/eval.sh logs/rsl_rl/doorbench_hand/<run>/model_300.pt   # all 1000 doors x 3 seeds -> results/*.json
```

Container alternative: [`cloud/Dockerfile`](cloud/Dockerfile) (`nvcr.io/nvidia/isaac-sim:5.1.0` base).
Instance recommendations and prices: [`cloud/README.md`](cloud/README.md).

## What is in here

```
isaaclab/
  doorbench_isaaclab/           pip install -e isaaclab   (inside the Isaac Lab python)
    __init__.py                 gym registrations: DoorBench-Open-Hand-v0, DoorBench-Open-G1-v0 (+ -Play-v0)
    doors.py                    dataset index, curated subsets (easy-100 ...), selection strings
    assets.py                   door_cfg(id) / multi_door_cfg(ids) (MultiUsdFileCfg), HAND_CFG, g1_cfg()
    door_task_env_cfg.py        ManagerBasedRLEnvCfg for both agents (scene, obs, actions, rewards, terminations, events)
    mdp/door_state.py           per-env door metadata + benchmark labels (touch, unlatch, open, traverse, slam, damage ...)
    mdp/actions.py              DoorMechanismAction: spring targets, latch<-operator coupling, closer asymmetry, automatic doors
    mdp/{observations,rewards,terminations,events}.py
    agents/rsl_rl_ppo_cfg.py    PPO runner configs
    data/gantry_hand.usda       the hand agent (scripts/isaaclab/make_hand_usd.py)
  cloud/                        setup.sh, validate.sh, train.sh, play.sh, hero.sh, eval.sh, Dockerfile, README.md
  STATUS.md                     exactly what was verified here and what awaits the GPU run
scripts/isaaclab/
  validate_usd_static.py        pxr-only validator (runs anywhere) -> assets/usd_validation.json
  validate_usd_isaacsim.py      headless Isaac Sim import / settle / actuate test -> assets/usd_validation_isaacsim.json
  train.py / play.py            RSL-RL PPO (Isaac Lab's scripts + --doors)
  record_hero.py                hero screenshot + video of N envs with N different doors
  eval_all_doors.py             checkpoint -> per-door success JSON over the whole dataset
  make_hand_usd.py, check_api_names.py
```

## The two USDs per door

* `door.usda` — full fidelity (every mechanism body, mimic joints for couplings, PhysX joint friction efforts).
  Use with `door_cfg(door_id, canonical=False)` for single-door work; its joint names are the door's own.
* `door_rl.usda` — **canonical articulation** for vectorised RL: every door has the same 8 links and 7 joints
  (`door_slide`, `door_hinge`, `operator_hinge`, `operator_slide`, `latch_slide`, `leaf2_slide`, `leaf2_hinge`; unused
  slots are locked). PhysX articulation views (and therefore Isaac Lab's `Articulation` over many envs) require
  homogeneous articulations, so this is what makes "a different door in every env" possible. The `doorbench:rl`
  attribute on the root prim tells the environment which slots are live, the thresholds, grip points and sites.

Details: [`../docs/ISAAC_LAB.md`](../docs/ISAAC_LAB.md).

## Tasks

| id | agent | obs | actions | notes |
|---|---|---|---|---|
| `DoorBench-Open-Hand-v0` | 6-DoF gantry hand (x,y,z,yaw,pitch,roll; palm sphere + finger bar) | hand joints, tip↔handle, tip↔goal, door state, task one-hot, event flags, last action (43) | relative joint-position targets (6) | trains in minutes; the door mechanics are the hard part |
| `DoorBench-Open-G1-v0` | Unitree G1 (`isaaclab_assets` G1_MINIMAL) at the approach point | G1 proprioception (as Isaac Lab's velocity task) + handle/goal in the base frame + door state (≈ 100) | joint position targets (scale 0.5) | loco-manipulation; locomotion regularisers from Isaac Lab's G1 task |

Rewards (both): touch handle +1 · unlatch +2 · open past 10° / 0.10 m +3 · clear (60° / 0.55 m) +2 · traverse the pass
plane +10 · close behind (close task) +5 · damage −10 · slam −5 · operator overload −0.5 · time −0.01/step, plus dense
shaping (reach the handle, opening fraction, progress toward the goal). Terminations: time-out, task success, hand
too far / G1 torso contact or fall. Door subset: `--doors easy-100` (default; unlocked swing / sliding / saloon /
automatic doors with levers, plates, pulls and touch bars) · `easy-300` · `all` · `family:...` · explicit ids.

Because the event bonuses fire once per episode, the RSL-RL log `Episode_Reward/traverse` ÷ 10 is the traverse
rate, `Episode_Reward/damage` ÷ −10 the damage rate, etc.
