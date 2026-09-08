# Finite vision/tactile interface for the Isaac student

**Status, September 8, 2026:** implemented sensor emulation and an actor packet boundary; 19 CPU tests pass, including real MuJoCo contact/plugin parity and USD collider discovery. Exporting the current H1/dual-Shadow model yields 53 tactile mounts, 448 three-axis taxels (1,344 scalar channels), 69 joint sensors and 61 actions. **No live PhysX sensor parity, sensor-only Isaac policy, or performance result is claimed.** This work implements part of stage 3 of [the execution plan](DEXTEROUS_NEXT_STEPS.md), independently of the privileged opening teacher.

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
