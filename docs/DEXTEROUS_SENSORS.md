# Finite vision/tactile interface for the Isaac student

**Status, September 8, 2026:** implemented sensor emulation and an actor packet boundary; 19 CPU tests pass, including real MuJoCo contact/plugin parity and USD collider discovery. Exporting the current H1/dual-Shadow model yields 53 tactile mounts, 448 three-axis taxels (1,344 scalar channels), 69 joint sensors and 61 actions. **The standalone live PhysX fixture subsequently passed 10/10 checks at 2026-09-08 07:43:28 UTC. The first H1 capture passed numeric stream and opening audits, but native camera usefulness failed visual review. A fixed wider/downward camera repeat passed placement and sensor checks but failed the existing opening threshold; both results are retained below. No sensor-only policy is claimed.** This work implements part of stage 3 of [the execution plan](DEXTEROUS_NEXT_STEPS.md), independently of the privileged opening teacher.

## Actor contract

The new `doorbench.sensors.v2` interface exposes copied numeric arrays only:

| Field | Meaning and units |
|---|---|
| `joint_position`, `joint_velocity` | Robot's own ordered encoders; rad and rad/s for H1 |
| `imu_gyro`, `imu_accelerometer` | Local IMU angular velocity and specific force; rad/s and m/s² |
| `tactile` | Finite sensor-local force bins, N; no contact-point list |
| `rgb_left`, `rgb_right` | RGB images, uint8, declared resolution; no debug overlays |
| `previous_action` | Previous issued normalized robot command, clipped to [-1, 1] |
| `sensor_time_s`, `sensor_valid` | Acquisition time and validity in the fixed `SENSOR_KEYS` order |

`ActorObservationBuilder` rejects unexpected fields, dimensions and nonfinite readings. It copies inputs and outputs so the policy cannot mutate simulator buffers. No root position/orientation, exact projected gravity, door joints, handle pose, object IDs, counterpart labels, world contacts, goal transform or evaluator state enters the packet. In particular, the earlier native `gravity_body` channel is intentionally absent: any future attitude estimate must be computed from declared measurements and independently audited. This version cannot load the old 438-input tactile hand checkpoint unchanged.

The evaluator owns simulator objects and calls the student with the packet alone. The teacher and critic may receive privileged state through separate code paths during training; never merge their dictionaries into actor observations. The whole runtime controller stack still needs source and behavioral audits: a packet validator cannot detect hidden global simulator access inside arbitrary policy code.

Each stream has configurable clipping, additive Gaussian noise, dropout, delivery delay and maximum age. Defaults use zero noise/delay for parity fixtures; training must explicitly record nonzero perturbations. Missing, dropped and stale samples are zero with `sensor_valid=false`. Delayed samples retain their original acquisition timestamps. Each episode resets queues and RNGs, preventing old contacts/images from leaking across resets. Noise uses independent seeded streams.

## Local tactile geometry

The native model uses angular `touch_grid` sensors. Each mount has a fixed robot-body transform and finite azimuth/elevation bins. Forces sum in local **z, x, y** channel order; the sensor looks along local negative z. Layout is sensor order, then channel, vertical bin, horizontal bin. The implemented `fov_degrees` are angular half extents: the native `(180,90)` setting covers the sphere. Gamma values other than zero and torque channels are rejected rather than silently approximated. The algorithm is independently implemented and checked against a solved native fixture; its conventions were verified against [MuJoCo's plugin implementation](https://github.com/google-deepmind/mujoco/blob/3.3.7/plugin/sensor/touch_grid.cc).

Export the actual robot's layout; do not hand-author approximate finger frames:

```bash
python scripts/dexterous/export_sensor_layout.py \
  --robot out/dexterous/robot/h1-shadow.xml \
  --output out/dexterous/sensors/h1-shadow.json
python -m pytest -q tests/test_dexterous_sensors.py
```

The export records the robot XML hash, joint/action order, sensor dimensions and exact compiled local mount transforms. It is generated outside tracked assets. On another robot, generate and audit a new layout and action adapter. If the importer merges a fixed body, compose its transform explicitly; do not silently substitute a similarly named frame.

## PhysX integration

`doorbench/dexterous/isaac_sensors.py` provides `PhysXTaxelAdapter`, `mounts_from_layout`, `enable_tactile_reporting`, `enqueue_robot_sensors` and `enqueue_camera`.

Detailed PhysX points require counterpart filters; an unfiltered contact view supplies only net forces. The adapter therefore discovers **every** collidable participant from USD physics APIs, including static shapes, robot bodies and mesh instances. No caller-supplied door/handle filter is accepted. The complete counterpart dimension is erased before local binning. Normal and friction forces are binned at their respective reported locations; those locations need not coincide. Buffer exhaustion fails the run. These behaviors follow the [PhysX tensor API contract](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.RigidContactView). Contact buffers are copied before the friction getter because the backend may [reuse count/index buffers](https://github.com/isaac-sim/IsaacSim/issues/122).

Integration order:

1. Load the fixed robot layout, resolve body names to audited imported rigid paths, and create mounts. Match every joint by the exported joint order; pass those indices to `enqueue_robot_sensors`.
2. Before `sim.reset()`, apply `enable_tactile_reporting(stage, mounts)`. Enable `/physics/disableContactProcessing=false` as in the existing initialized demo. Declare robot-mounted policy cameras and a local IMU, with visualization disabled. The [Lab 2.3.2 IMU](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/sensors/imu/imu.py) supplies `ang_vel_b` and `lin_acc_b`; use only these channels, with the gravity bias configured for specific force.
3. After reset, construct `PhysXTaxelAdapter(sim.physics_sim_view, stage, mounts)`. Construction checks sensor row order and complete counterpart coverage. Reset packet queues at every episode boundary.
4. After each actual physics step, update articulation/IMU buffers, call `adapter.read(physics_dt=actual_dt)`, then `enqueue_robot_sensors(...)`. The force conversion uses the physics interval, never camera or control cadence. Reading the most recent step currently supplies an instantaneous sample, not an interval-integrated tactile exposure.
5. Render separately with `sim.render()` at the camera cadence. Enqueue the pixels with the simulation time of the rendered state and the time they become available. Never restamp an old image as fresh or pass diagnostic gold-material/force-overlay footage as policy RGB. Read the actor packet at control cadence.

This first adapter copies tensor buffers to CPU and performs NumPy binning. It is deliberately a correctness implementation, **not a large-batch performance path**. A later batched GPU kernel must match these fixtures and avoid cross-environment contacts; increasing batch size does not establish that equivalence.

## Required live GPU gates

Before labeling any rollout sensor-only, retain the following checks on the pinned Isaac 5.1/Lab 2.3.2 environment:

- Press and slide one pad against a static surface and a dynamic body. Compare local force sign/magnitude against independent aggregate contact forces; rotate/translate the whole fixture and check identical local bins. Confirm static colliders, articulations, instances and self contact all appear. Record contact/friction buffer counts and no-capacity-exhaustion status.
- Verify sensor transforms against native link/site FK and each imported link; no world-to-local quaternion convention mismatch. Check every counterpart path actually resolves; fail if a static collision prim needs a different native representation.
- Verify stationary IMU specific force is approximately +9.81 m/s² upward, free-fall specific force approximately zero, and gyro signs under known rotations. Keep exact orientation strictly in the evaluator. Reset differentiation buffers before scoring.
- Use a moving visual fixture to validate RGB acquisition time, delivery delay, stale-frame flags and the physics clock. Confirm no sensor overlays or annotated debug images in either camera.
- Rename/reorder counterpart prims and show unchanged actor arrays for an equivalent physical scene. Test contact with an unexpected object; no semantic routing should be added to make it register.
- Run one environment, then a small batch with separate environment ownership. Profile physics, copies, taxel construction and camera delivery separately before replacing the NumPy path or scaling.

CPU passing tests establish numerical/API-boundary behavior, not live Isaac coverage, the physical equivalence of contact solvers, correct cameras, or task success. Record those remaining gates in the run manifest and publish failures alongside passes.

## Standalone live smoke command

On the prepared GPU environment, run this before attaching the adapter to H1:

```bash
source isaaclab/cloud/env.sh
python scripts/dexterous/isaac_sensor_smoke.py \
  --headless --enable_cameras --device cuda:0 \
  --output out/sensor-smoke/fixture-001
```

The fixture contains two free pads: one settles on static ground, the other on a dynamic block. It checks local support-force signs/magnitudes, friction opposing an imposed initial sliding velocity, stationary/free-fall IMU, finite arrays, camera delivery timestamps and the exact physics clock. It retains traces, RGB arrays/images, source hashes, progress and a pass/fail report. The two image channels intentionally duplicate one fixture camera and are not a robot policy configuration. The fixture does not establish H1 camera placement, complete-body touch, self-contact, cross-engine dynamics or task success. Use a new output directory for every attempt, including failures.


### Live fixture evidence

The [first fixture report](../results/dexterous/2026-09-08/isaac-sensor-fixture.json) passed all ten checks on CUDA with Isaac Sim 5.1.0.0 and PyTorch 2.7.0+cu128. Static and dynamic support measured 9.8103 N and 9.8142 N for the 1 kg pads; sliding local shear reached -5.9157 N. Stationary IMU specific force was approximately 9.8103 m/s², free-fall readings passed the declared tolerance, camera acquisition age was 20–58 ms with delay enabled, and the physics clock error was 9.50e-8 s. Static ground and all dynamic counterpart paths resolved.

The uncut numeric trace, RGB arrays/frames, logs, source files and package record are archived outside Git at `~/Desktop/Projects/DoorBench-runs/2026-09-08-isaac-sensors/fixture-001/` and its parent directory. The RGB image was visually inspected; both free pads and their support are visible. This is a two-pad fixture, not a humanoid task result.

## Opt-in H1 capture beside the initialized teacher

The native layout export now includes the two fixed eye-camera calibrations and the IMU site. Pass `--sensor-layout /path/to/h1-shadow-sensors.json` to the existing `scripts/dexterous/isaac_opening.py` command. This creates all tactile mounts, a local IMU and independent 128×128 eye cameras; captures actor arrays and image timestamps under `sensors/`; and disables the diagnostic gold material override for these policy pixels. The previous-action field records bounded motor force normalized by each native motor force range. All teacher computations and motor limits remain as before.

This is instrumentation beside a privileged teacher, not a sensor-only controller. It needs an actual H1 run, the existing independent opening audit, camera close-up inspection and a source check before being accepted. The serialized sensor packet contains no task geometry, while the independent teacher/evaluator trace intentionally still does. Keep these archives separate during training.

### Fixed camera revision: placement verified, opening repeat failed

Native eye images from the first H1 capture largely miss the working hand and handle. The explicit `configs/dexterous/h1-manipulation-cameras.json` profile keeps both physical eye positions and tilts their fixed optical axes down 45 degrees with a 100-degree field of view. It never receives a task target or moves independently of the robot body. Native replay and subsequent live Isaac inspection show the working hand, lever and forearm, including natural occlusion. At 128×128, this establishes useful placement, not detailed finger/contact visual quality; the hand and lever occupy a small pixel region.

The [second live H1 report](../results/dexterous/2026-09-08/isaac-h1-manipulation-camera-capture.json) retains 5,000 finite packets and 250 frames per eye, with 16/16 sensor checks passing. **The physical opening repeat failed:** final opening was 0.449 rad (25.7°), below the unchanged 0.7 rad gate, so only 17/18 opening checks passed. All shared model, controller, motor and reference source hashes match the first capture; only the camera layout changed. This comparison does not establish the cause of physical variation.

Trace review exposes a robustness gap in the initialized teacher: both attempts start releasing near 0.257 rad and withdraw near 0.49 rad, before reaching the success threshold. The first door then coasts open; the second stalls. A completed-opening controller must resolve this dependency on post-release motion. The failure, wide video, corrected eye images, numeric streams and provenance are preserved in `h1-002-evidence.tar.gz` beside the original archive. No sensor-only task success or reliable opening rate follows from either attempt.

Export with `export_sensor_layout.py --camera-profile configs/dexterous/h1-manipulation-cameras.json`, or pass `run_handle_demo.py --sensors --sensor-camera-profile configs/dexterous/h1-manipulation-cameras.json` on the ready environment. The generated layout retains original native calibration, the explicit profile and its hash. `preview_sensor_cameras.py` renders fixed camera views from recorded native states for review; it is a replay, not a new task rollout. Its calibration tests verify that forward/down direction, body mount and tactile channels remain correct.

The ready-environment wrapper now exposes the same capture with one flag:

```bash
source isaaclab/cloud/env.sh
python scripts/isaac/run_handle_demo.py --sensors
```

It exports the native layout, runs the initialized teacher, and executes both the existing opening audit and the independent sensor-archive audit. This command still does not train or evaluate a sensor-only controller.


### H1 capture: numeric passes, camera-view failure retained

The [H1 capture report](../results/dexterous/2026-09-08/isaac-h1-sensor-capture.json) records 5,000 full-body sensor packets and 250 frames per eye over ten seconds. The independent opening audit passed 18/18 checks: final opening 95°, maximum torso tilt 1.805°, and loaded opposition throughout the measured lever-turn window. The separate archive audit passed 16/16 checks, including all 1,344 finite tactile channels, physical right-hand contact, fixed numeric fields, normalized previous motor forces and causal image timestamps. These are separate instrumentation and initialized-opening results, not sensor-driven control.

**Visual review failed the native camera configuration for manipulation.** Both forward-facing 45° eye images mostly show a nearby wall/door crop; they do not provide a useful view of the manipulating hand and handle. The inspected images remain in the archive. Training must wait for a declared fixed wider/downward calibration or suitable wrist camera and another visual review. A numeric packet pass does not override this failure.

The complete source, wide video, traces, actor archives, camera samples and both audits are retained in the 20.7 MB `h1-001-evidence.tar.gz` archive under `~/Desktop/Projects/DoorBench-runs/2026-09-08-isaac-sensors/`. Its checksum is in the linked report.

## Adjacent live walking fixture

`scripts/dexterous/isaac_h1_walking.py` ports the separately documented [H1 walking actor and phase braking](DEXTEROUS_LOCOMOTION.md) to the unchanged prepared Isaac articulation. `export_h1_walking_reset.py` derives its reset from the native collision soles. The first declared schedule is one second at zero command, eight seconds forward at 0.4 m/s, then five seconds at zero command with one-second phase fade. The live fixture retains native motor/servo bounds, 20 ms policy and 2 ms physics clocks, and tests a strict final-second stop (<0.02 m/s, <1 cm excursion, >30 N per foot). The [live run passed all 14 checks](ISAAC_H1_WALKING.md) and was visually inspected. This is an empty-plane motor-skill test with perfect attitude input, not vision/tactile door control.
