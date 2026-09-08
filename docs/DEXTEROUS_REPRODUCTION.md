# Reproduce this experiment on another cluster

This is the migration record for the simulated dexterous-humanoid project. Native MuJoCo experiments use **H1 with two Shadow Hands** for privileged body reaching and initialized tactile grasping. The same robot now runs in live Isaac Sim/PhysX, with a verified import, motor adapter and one-click environment setup. A complete door-opening policy has not yet passed validation. A different robot requires a new embodiment adapter and fresh validation; a checkpoint cannot simply be renamed or loaded into it.

The project objective and acceptance protocol are in [the approved plan](DEXTEROUS_PLAN.md). Current results and limitations are in [the execution record](DEXTEROUS_HUMANOID.md) and [experiment ledger](DEXTEROUS_EXPERIMENTS.md).

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

5. Inspect the portable configuration, then launch it. Configuration paths resolve against the repository; absolute paths can target cluster-mounted asset storage. The commands below reproduce the historical v1 setup. For new body-training experiments add `"standing_weight": 4.0` and `"continue_after_success": true` to a copied configuration; v3 tests those settings while resuming v2. The ledger explains the early-success termination problem. Use the original captured configuration when reproducing a historical result.

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

The current allocation is one L40S, 16 vCPUs and 94 GB reported RAM, at $1.09/hour. Its requested image is `runpod/pytorch:1.2.0-rc.162-cu1281-torch271-ubuntu2204`; the training virtual environment subsequently installed different packages, so use each run's requirements rather than inferring PyTorch from that tag. `cluster.json` is a sanitized, explicitly late infrastructure capture. The API did not expose an immutable container digest; that remains a reproduction limitation for this first allocation and must be captured at provisioning on the second cluster.

The first run's durable local artifact archive is `~/Desktop/Projects/DoorBench-runs/2026-09-07-dexterous/`, outside the temporary Git worktree. It contains checkpoints, source bundles, exact input poses/preloads, evaluation states, reports and relevant upstream licenses. These model/trajectory artifacts are not committed to Git and have not been published as a released policy. Copy this archive to the second cluster in addition to checking out the source branch.

After copying a run, execute `python scripts/dexterous/verify_run.py /mounted/run`. This checks archived source and bundled input checksums without extracting code. It does not verify external assets or demonstrate physics equivalence. Generated robot XML contains destination-specific mesh paths, so its byte hash may change after rebuilding; compare the upstream revision, asset contents and the model audit before attributing differences to training.

## Initialized tactile grasp experiment

This is a separate skill, not a complete opening policy. The body is free in gravity, but its arm/body position-motor targets are held while the hand learns contact correction. The input pose starts at the handle. Its 438 actor inputs are 24 right-hand joint positions, 24 velocities, 128 three-axis taxels and six previous actions. Six residual actions adjust five thumb targets and a shared four-finger curl. Exact object state and privileged contact identities are used only by the reward/audit. This small action adapter must be replaced or expanded for general manipulation.

Retain `inputs/pose.npz` and `inputs/best.json` from the training manifest bundle. These are optimized inputs, not files recreated by installing the repository. The selected seed was fitted with bounded IK and a collision penalty; the preload was selected by a short native-physics search. Re-running those optimizers may find a different candidate and constitutes a new experiment. The exact input hashes define a repeat of the original run.

With those inputs mounted on the destination, run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/dexterous/train_grasp.py --robot out/dexterous/robot/h1-shadow.xml --door out/dexterous/assets/doors/db0055_swing_single --seed-pose /mounted/run/inputs/pose.npz --preload-json /mounted/run/inputs/best.json --output out/dexterous/training/grasp-repeat --envs 6 --steps 200000 --device cpu
python scripts/dexterous/evaluate_grasp.py --robot out/dexterous/robot/h1-shadow.xml --door out/dexterous/assets/doors/db0055_swing_single --seed-pose /mounted/run/inputs/pose.npz --preload-json /mounted/run/inputs/best.json --checkpoint out/dexterous/training/grasp-repeat/final.zip --output out/dexterous/evaluation/grasp-repeat --episodes 30 --seed 20000
```

Repeat evaluation without `--checkpoint` in a new output directory to compare against constant optimized motor preload on the same seeds. The gate is one continuous second of physically loaded opposition: four fingers on one side and thumb on the other, upright torso and sufficient pelvis height. Evaluation retains every trial, including failures. Small hand-initialization perturbations are development validation only; success here does not establish robust acquisition, lever release, opening or traversal. Check wide body views and hand close-ups as well as contact traces.

The first fitting/search experiments predated automatic source capture. Their outcomes are historical diagnostics, not complete launch-time reproduction bundles. Training captures its actual input pose/preload and sources before starting.

## Second-run handoff checklist

Before launching a large Isaac Sim job, fill in a new experiment record with the destination robot asset revision and license, Isaac Sim/Isaac Lab and container versions, scheduler launch file, GPU/CPU allocation, sensor/action interface version, exact dataset manifest, and the upstream checkpoint or fresh-training decision. Preserve the previous experiment unchanged. Do not assume an H1/Shadow policy can control a different joint layout.

Run the migration fixtures below on one environment, then a small batch, before scaling. Archive the fixture reports and an uncut single-door video with both wide and hand views. Profile the real sensor pipeline and training update together; select the batch by useful throughput with memory headroom rather than GPU count alone. Keep reset failures, numerical warnings and failed tasks in the report. Only then begin the full curriculum and frozen evaluation. The readiness fixtures listed below have passed on one L40S; they do not replace the remaining tactile-policy and task-success gates.

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
- Native MuJoCo physics runs on CPUs, with neural-network updates on the GPU. Isaac readiness uses CUDA PhysX and GPU rendering; development opening trials explicitly record their CPU/CUDA physics device. Report each run's actual configuration when interpreting GPU utilization or extrapolating cost.
- The first long run exhausted MuJoCo's default 16 MiB constraint arena during a fall (349 contacts, 1,143 constraints). The dexterous adapter now requests 128 MiB per environment, without removing collisions or changing solver tolerances. Include this allocation in node memory planning. A failed subprocess also needs bounded cleanup so its parent cannot hang in the vector-environment shutdown path.

## Isaac preparation now available (September 8, 2026)

Use the [one-click launcher](ISAAC_ONE_CLICK.md) for the pinned Isaac 5.1 / Lab 2.3.2 environment. The installed desktop launcher passed at **2026-09-08 01:43:02 UTC** and its cached-runtime reuse passed again at **2026-09-08 02:01:41 UTC**. These L40S runs checked live CUDA physics, rendering, a commanded wrist motion, finite native-derived motor limits, motor-command delivery, standing and the clock. Independent native kinematics agreed with all imported robot link poses within 0.912 micrometers and 0.000022 degrees; mass differed by 0.00495 grams. Tactile-policy parity, robust opening and traversal remain separate work.

The importer-compatible MJCF is a generated copy. Explicit joint axes/types are necessary: nested defaults produced a wrong shoulder axis, and scalar-type conversion must preserve the free base. Duplicate mesh declarations get unique temporary filenames because Isaac 5.1's concurrent converter races when both thumbs share one source basename. Inertias and all 61 actuator transmissions are exported explicitly; zero-stiffness motor tendons are represented by the motor matrix, not by invented passive springs.

Always step physics with `sim.step(render=False)` and call `sim.render()` separately at camera cadence. In this runtime, passing `render=True` to the combined step advances toward the rendering timestep; calling it every twentieth 2 ms motor update therefore inserted extra physics steps. Early videos from opening attempts 006–008 are **invalid controller evaluations** and do not establish a robot opening result. Every later run checks the simulator clock against the commanded step count.

Hand contact audits use PhysX's rigid-contact tensor view on the actual imported rigid-body paths, filtered to the lever body. The event-report callback did not yield usable contacts in these runs. The tensor measurements are force audits, not yet an equivalent finite taxel observation pipeline for the learned tactile actor.

The MJCF importer also omitted robot friction materials. Preparation now exports an explicit native contact contract and binds it through instanced collision meshes. The current H1/Shadow asset has uniform sliding friction 1.0 and three-dimensional contacts. Both PhysX friction coefficients are set to 1.0 with maximum combination, matching the native equal-priority mixing rule; the live adapter reads coefficients back from the solver and refuses silent defaults. The adapter rejects heterogeneous materials, nonzero priorities or torsional contact dimensions until an appropriate mapping is supplied. See [MuJoCo contact mixing](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters) and [NVIDIA material binding and combination](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html#configure-rigid-body-s-material-properties).

Reset fitting must check **all** robot joints, not just the arm being optimized. Early workspace candidates 009/021/022 exceeded an ankle stop and are rejected. The geometric screen also checks unused limbs at every planned waypoint; a waist-assisted candidate initially put the idle left hand through the door. An input that passes this screen still needs live contact, motor, stability and close-up review.

The readiness receipt and source manifest are retained under `out/isaac-launch/readiness-004/`; the exact remote paths are recorded in its `connection.json`. The launcher's source bundle excludes generated assets and checkpoints. Copy the receipt, source archive, input files, traces and videos into durable run storage before the owned allocation's deadline.
