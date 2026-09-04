# DoorBench

**1000 physics-grounded, fully articulated doors for training and benchmarking humanoid robots in simulation.**

[Catalogue & 3D viewer](https://adamraudonis.github.io/DoorBench/) · [Dataset format](docs/DATASET_FORMAT.md) · [Physics model](docs/PHYSICS.md) · [Benchmark](docs/BENCHMARK.md) · [Simulator integration](docs/INTEGRATION.md)

Every door is a working mechanism, not a decorated slab: spring latches that hold and re-latch when slammed,
deadbolts driven by thumbturns, panic bars, keypads with physical keys, dogs and vault bolt-work, closers sized to
EN 1154, hinge friction from a bearing-load model, and masses calibrated to manufacturer door-weight tables.
Each door ships as **MJCF** (MuJoCo), **URDF** and **USD** (Isaac Sim / Isaac Lab) in three fidelity tiers, with an
auditable `spec.json` that lists every physical parameter, its formula and its source.

## What is in the box

| | |
|---|---|
| Doors | 1000, procedurally generated with a balanced design of experiments (seeded, reproducible); **1000 / 1000 pass the automated sign-off** (load in all tiers, settle, hold while latched, open when actuated, re-latch, closer return, URDF + USD load) |
| Kinematic families | 30: swing (single / pair / dutch / saloon / pivot), sliding (pocket, barn, patio, shoji, bypass), bifold, accordion, revolving, tripod & full-height turnstiles, garage (sectional, tilt-up, roll-up), pet doors, floor & ceiling hatches, marine watertight, vault, blast, gates (swing, sliding, baby), toilet stalls, strip curtains, cold storage, automatic sliding & swing, elevator |
| Operators | 59: levers, knobs, paddles, pulls, push plates, panic touch bars & crossbars, thumb latches, wheels, dogs, slide bolts, hooks, lift pins, keypads, card readers, handlesets, cremone bolts, T-handles … |
| Locks | 27 incl. privacy, keyed, single/double deadbolts, chain, swing-bar guard, padlock, 4/6-digit keypads, mechanical pushbutton, card reader, maglock, electric strike, delayed egress, interlock, jammed |
| Slab constructions | 72: hollow core, particleboard & SCL cores, solid hardwoods, fire-rated mineral cores, 14–24 ga hollow metal, storefront aluminium, frameless tempered glass, fiberglass, uPVC, shoji/fusuma, cold-storage panels, chain-link, wrought iron, PVC strips, vault composite … |
| Closers | 17 (EN 1154 sized surface/concealed/floor-spring, spring hinges, pneumatic, gate, gas strut, hold-open, automatic operators) |
| Conditions | new · normal · worn · old/dry · rusty · swollen · sagging · damaged · well oiled |
| Formats | MJCF full/simple/minimal · URDF full/simple/minimal · USD |
| Benchmark | 9 tasks, 20+ per-episode labels (touched, actuated, unlatched, unlocked, opened, passed through, closed, slammed, damaged, fell …), MuJoCo environment with lock/access-control logic |

## Quick start

```bash
git clone https://github.com/adamraudonis/DoorBench.git && cd DoorBench
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[all]"

# regenerate the whole dataset (≈ 30 s with 8 workers, deterministic) or just use assets/ from the repo
python scripts/generate_dataset.py --out assets --workers 8

# look at one door in MuJoCo
python -m mujoco.viewer --mjcf assets/doors/db0002_swing_single/scene.xml

# run the benchmark environment with a programmatic "hand"
python - <<'EOF'
from doorbench.benchmark import DoorEnv
env = DoorEnv("assets/doors/db0002_swing_single", tier="full")
env.reset()
for _ in range(800):
    env.apply_joint_torque(env.meta["operator_joint"], 3.0)   # turn the knob
    env.apply_joint_torque(env.meta["primary_joint"], 30.0)   # push the door
    env.step()
print(env.labels().to_dict())
EOF

# browse the catalogue + 3D viewer locally
cd viewer && bun install && bun run dev      # http://localhost:5173
```

## Humanoid in the loop

An off-the-shelf humanoid walks through DoorBench doors in plain MuJoCo on a CPU: the MuJoCo Menagerie **Unitree G1**
(BSD-3) driven by the pretrained **unitree_rl_gym** sim2sim locomotion policy (BSD-3), merged into the door scene with
`MjSpec.attach` through `DoorEnv`. Physics + policy run at 15–19x real time on an M4; no GPU involved.

| open doorway (`db0119`) | automatic sliding door, opened by its sensor (`db0990`) |
|---|---|
| ![](docs/media/g1_door_db0119.gif) | ![](docs/media/g1_door_db0990.gif) |
| **saloon pair pushed open, 240 N peak on the leaf, no damage (`db0123`)** | **latched push door: 423 N, latch holds, locomotion alone cannot open it (`db0705`)** |
| ![](docs/media/g1_door_db0123.gif) | ![](docs/media/g1_door_db0705.gif) |

```bash
bash robot_demo/setup.sh                                     # fetch Menagerie G1 + unitree_rl_gym policy (pinned commits)
python robot_demo/run_g1_door.py --door db0123_saloon        # -> docs/media/g1_door_db0123.{mp4,gif}, robot_demo/results/*.json
```
Setup, per-run numbers (time-to-pass, contact forces, real-time factors) and limitations: [robot_demo/README.md](robot_demo/README.md).
A locomotion policy cannot operate levers, knobs or bolts; latched and locked doors need the loco-manipulation policies this
benchmark is built to train.

## How doors are built

1. **Taxonomy & sampler** (`doorbench/taxonomy.py`, `spec.py`): 30 families with sub-contexts (residential interior,
   fire egress, hospital, storefront …). For each family a balanced sampler cycles through every level of every
   discrete dimension (slab, panel style, operator, latch, lock, closer, hinge, stop, seal, condition, handing, push/pull)
   and jitters continuous ones (sizes, spring adjustment, colours).
2. **Physics** (`physics.py`): mass breakdown, hinge/roller friction, closer spring & damping (EN 1154), latch/lock
   parameters, ADA/IBC compliance flags and damage thresholds, each with formula + source.
3. **Geometry** (`geometry/`): procedural primitives + a shared library of procedural hardware meshes; every
   mechanism is a real body with a joint (bolts, thumbturns, pads, keys, dogs, closer arms with loop closure).
4. **Export** (`export/`): MJCF, URDF, USD from one simulator-agnostic IR (`ir.py`).
5. **Sign-off** (`qa.py`): each door is loaded in MuJoCo (all tiers), settled, pushed while latched (must hold),
   its operator actuated (must open; chained doors open only to the slack limit; locked doors must not open),
   released (latch must re-extend), slammed (must re-latch), its closer tested (must return), and its URDF/USD
   checked. `qa.json` records every metric; the catalogue shows the result. Current build: 1000 / 1000 signed off.

## Data layout

```
assets/doors/<id>/  spec.json  model.json  door.xml door_simple.xml door_minimal.xml scene.xml  door.urdf  door.usda  qa.json  thumb_*.jpg
assets/hardware/    shared hardware meshes (obj + usdc)
assets/manifest.json
```
See [docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md).

## Prior work

DoorGym (2019) randomised three handle types on one door; PartNet-Mobility / SAPIEN provide doors as cabinet parts
without tuned physics; Infinigen-Sim (2025) generates procedural door geometry but no friction or material physics;
ArtVIP (ICLR 2026) has 10 mocap-tuned doors; NVIDIA's DoorMan trained humanoids on procedural Isaac Lab doors that
were not released. DoorBench aims to be the first large, mechanism-complete, physics-audited, multi-format door corpus.

## License

Code and generated assets: MIT. Physics catalogues cite public standards and manufacturer data; textures referenced
for photoreal renders are Poly Haven CC0.

## Citation

See [CITATION.cff](CITATION.cff).
