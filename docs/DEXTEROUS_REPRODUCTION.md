# Reproduce this experiment on another cluster

This is the migration record for the simulated dexterous-humanoid project. The current executable training adapter is **native MuJoCo, H1 with two Shadow Hands, privileged body reaching**. It has not solved a door and it is not an Isaac Sim implementation. A different robot requires a new embodiment adapter and fresh validation; a checkpoint cannot simply be renamed or loaded into it.

The project objective and acceptance protocol are in [the approved plan](DEXTEROUS_PLAN.md). Current results and limitations are in [the execution record](DEXTEROUS_HUMANOID.md).

## Reproduction contract

Keep these independent:

| Component | Preserve across clusters | Replace for a different robot / Isaac Sim |
|---|---|---|
| Task | Door source/spec hashes, start conditions, intended procedure, required aperture, trial seeds | Simulator implementation of the same physical mechanism |
| Embodiment | Explicit identity, units, audit and limits | Robot asset, joint/actuator mapping, tendons, contact shapes, cameras, tactile surfaces |
| Policy interface | Images, finite local touch, proprioception, previous commands; no door-state oracle | Sensor dimensions and encoder/action adapters, with a declared new interface version |
| Training | Stage, rewards, curricula, seeds, optimizer and checkpoint provenance | Vectorized backend, rollout collection and batch sizing |
| Evaluation | Frozen eligibility, scenarios, missing-trial/failure accounting, independent physical checks | Backend instrumentation; validate equivalence before comparing scores |
| Infrastructure | Artifact layout and complete run record | Scheduler, mounts, image, GPUs, launch mechanism and teardown |

RunPod allocation scripts are convenience infrastructure. Neither the environment nor the training command requires RunPod, SSH, a particular GPU model, or account credentials.

## Native run, from a clean machine

1. Check out the exact source revision from the run manifest in an isolated checkout. Create a Python virtual environment. Current verified versions are Python 3.10.12 on the GPU host, MuJoCo 3.12.0, stable-baselines3 2.7.0 and gymnasium 1.2.2. Install a PyTorch build compatible with the node's driver; record the exact result rather than assuming a container tag describes the active interpreter.
2. Install DoorBench (`python -m pip install -e .`) and the optional training packages: MuJoCo, PyTorch, stable-baselines3, gymnasium, scipy, Pillow, imageio, imageio-ffmpeg and pytest. Reproduction bundles include exact installed package versions. EGL/OpenGL libraries are needed for camera rendering on a headless host.
3. Run `python scripts/dexterous/setup_robot.py`. This fetches and checks the pinned HumanoidBench revision. Keep upstream assets and weights outside Git; retain upstream licenses. Rebuild robot XML on the destination because generated XML resolves mesh paths on that host.
4. Generate the native development door and run the checks:

```bash
python scripts/generate_dataset.py --out out/dexterous/assets --ids db0055_swing_single --workers 1 --formats mjcf,json --no-thumbs
python -m pytest -q tests/test_dexterous_humanoid.py tests/test_dexterous_protocol.py tests/test_dexterous_contacts.py
```

5. Inspect the portable configuration, then launch it. Configuration paths resolve against the repository; absolute paths can target cluster-mounted asset storage.

```bash
python scripts/dexterous/run_experiment.py --config configs/dexterous/h1-shadow-reach-v1.json --output out/dexterous/training/my-run --dry-run
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/dexterous/run_experiment.py --config configs/dexterous/h1-shadow-reach-v1.json --output out/dexterous/training/my-run --envs 8 --device cuda
```

Use a new output directory for every experiment. Resume weights with `--checkpoint /path/to/latest.zip`; the manifest records this input and session transitions are distinguished from cumulative checkpoint transitions. On another architecture, retrain rather than reuse incompatible actuator outputs.

6. Evaluate outside training, using development seeds never used for optimization. These are **development validation seeds**, not the final benchmark holdout. Run the same seeds against the frozen upstream controller by omitting `--checkpoint`.

```bash
python scripts/dexterous/evaluate_reach.py --upstream out/dexterous/upstream/humanoid-bench --robot out/dexterous/robot/h1-shadow.xml --door out/dexterous/assets/doors/db0055_swing_single --checkpoint out/dexterous/training/my-run/final.zip --output out/dexterous/evaluation/my-run --episodes 30 --seed 10000
```

`render_trial.py` converts the recorded native states to video. It does not generate or improve motion. Preserve uncut failures alongside successes. Final opening/traversal evaluation will require its own frozen protocol; this reaching evaluator cannot supply that score.

## What every run retains

- `experiment.json` / `config.json`: requested experiment and resolved training arguments.
- `manifest.json`: capture time, source revision and file hashes, input/model/checkpoint identities, Python/packages, CPU/GPU/driver information. A late capture must be labeled as late; it is not launch-time provenance.
- `source.tar.gz`: the actual relevant Python/shell sources, including uncommitted code if present; no generated assets or account configuration.
- `requirements.txt`: installed package names/versions, without credential-bearing direct package URLs.
- `progress.json`, `history.jsonl`, `run.log`: live state and learning history; `failed.json` or `completed.json` for terminal status.
- `latest.zip`, `final.zip`: optimizer/policy state. Copy artifacts off ephemeral storage before shutdown.
- Evaluation reports, per-trial states/controls, wide videos and hand close-ups. Checkpoint success requires runtime evidence, not a replay alone.

For stronger reproducibility on the second cluster, retain its container digest, scheduler job specification, driver/CUDA versions, CPU allocation, GPU topology, storage paths and distributed-process layout. Do not archive credentials or process environments. Matching seeds does not guarantee bitwise identity across devices or physics engines; compare behavior and tolerance-based physical metrics.

## Isaac Sim / massive-node migration gates

The following are pending implementation, not completed support:

1. **Freeze the destination robot.** Record all joint ranges, actuator effort/speed limits, coupling, collision exclusions, body mass/inertia, hand pad geometry and sensor placement. Identify wrist/palm frames explicitly; upstream H1 `left_hand`/`right_hand` sites are forearm sites, not palm centers.
2. **Port one complete plant.** Import the robot and one lever door, preserving contact geometry, tendons/coupling, passive return, latch behavior and units. DoorBench Python callbacks and MuJoCo plugins do not execute automatically in PhysX. Verify forces, release travel, articulation axes and collision response before adding RL.
3. **Implement tactile observations.** Reproduce declared local force arrays from solved contacts without passing object IDs to the actor. Test sign, frame, clipping, finite resolution and temporal synchronization. Debug force overlays must never appear in policy RGB frames.
4. **Run parity fixtures.** Free fall, standing/contact support, individual actuator steps, finger flexion/limits, thumb opposition, lever contact and release. Compare trajectories, impulses, energy and sensitivity to timestep; document justified tolerances and remaining disagreements.
5. **Scale after correctness.** Profile state-only, RGB and RGB+tactile separately, increasing environment count with memory headroom. Measure useful transitions/sec, resets, solver convergence and rendering latency. A 4x4 video layout is not the training batch size. For multiple GPUs, define process/world ownership and checkpoint synchronization explicitly.
6. **Repeat the small-door curriculum.** Demonstrate unsupported-body grasp/open/traverse from the required start region. Then expand families and distill the sensor-only student. A native MuJoCo body-reaching score is not evidence of Isaac opening performance.

The first alternative GPU-physics probe used MuJoCo Warp 3.12.0 on the unchanged model. It rejected the 53 `mjSENS_PLUGIN` tactile sensors. This matches the [upstream unsupported-feature documentation](https://github.com/google-deepmind/mujoco_warp). Do not remove touch or required mechanics and call the result an equivalent port. A supported sensor implementation and callback parity are prerequisites for that route too.

## Operational lessons already found

- The borrowed body checkpoint crouches and can fall on larger reaches. Good hand-position error alone must not pass an upright reaching task.
- Native H1 is +X forward. DoorBench scenario yaw already uses that convention; adding a quarter-turn breaks the interface.
- Tactile plugin visualization ignores the ordinary site-visibility switch. Hide its render decorations explicitly while preserving touch data.
- NumPy booleans in episode metadata need conversion before JSON logging. A logging failure must stop being displayed as a live run.
- Older Git versions interpret `sparse-checkout set --cone` unexpectedly. Use `sparse-checkout init --cone`, then `sparse-checkout set ...`; verify root packaging files exist.
- The model has 61 actuators, 69 articulated joints and 448 three-axis taxels. Preserve these audited facts rather than relying on a robot's marketing DOF count.
- The current GPU performs neural-network updates; native physics runs on CPUs. Report this honestly when interpreting GPU utilization or extrapolating cost.
- The first long run exhausted MuJoCo's default 16 MiB constraint arena during a fall (349 contacts, 1,143 constraints). The dexterous adapter now requests 128 MiB per environment, without removing collisions or changing solver tolerances. Include this allocation in node memory planning. A failed subprocess also needs bounded cleanup so its parent cannot hang in the vector-environment shutdown path.
