# H1 walking and quiet stopping in Isaac Sim

The unchanged prepared H1/dual-Shadow robot completed one live CUDA PhysX forward-and-stop development trial on September 8, 2026 UTC. **All 14 declared checks passed.** This is an empty-plane motor skill, not door traversal, a vision/tactile controller, or a robustness score.

| Measure | Live result | Declared gate |
|---|---:|---:|
| Net travel | 2.6455 m | >1.6 m |
| Peak torso tilt | 2.6205° | <12° |
| Final-second maximum horizontal speed | 0.00869 m/s | <0.02 m/s |
| Final-second position excursion | 3.980 mm | <10 mm |
| Final-second minimum support per foot | 85.73 N | >30 N |
| Maximum joint-stop penetration | 0.0104 mrad | <20 mrad |
| Sampled self/nonfoot ground penetration | 0 / 0 mm | <3 mm |
| Motor delivery error | 3.76e-6 Nm | <1e-4 Nm |

Both feet swung during walking, all 61 native motor force bounds were retained, and the robot stayed freely simulated. No root pose writes, foot supports, external wrenches or door forces occur after reset. Contacts are inspected at 50 Hz; joint/motor checks run at 500 Hz. The [machine-readable report](../results/dexterous/2026-09-08/isaac-h1-walking.json) records exact source identifiers, completion time from the report file, numerical limits and archive checksum.

The schedule is one second of zero velocity command, eight seconds forward at 0.4 m/s, then five seconds at zero command. After stopping is requested, the two gait-phase channels fade over one second, as in the separately measured [native H1 development work](DEXTEROUS_LOCOMOTION.md). The official pinned H1 actor uses leg encoders, body gyro, perfect projected gravity, commanded velocity and its own previous actions/phase. It receives no world position, linear velocity, door geometry, images or touch. Perfect attitude remains an idealized observation.

The full 53.239896 kg robot, Shadow hands, 69 articulated joints and 61 native motors are retained. A native geometry export positions the collision soles at reset; the measured root height is 1.035016 m. Leg torques use the official actor's target/PD law and are converted through the native affine servo equation with original control and force caps. Native passive damping, friction and armature are preserved by the audited Isaac motor adapter. The original walking policy's fixed upper-body deployment example is not substituted for this articulated plant.

## Tooling and evidence

- `scripts/dexterous/export_h1_walking_reset.py`: native-only reset derivation, without training solver dependencies.
- `scripts/dexterous/isaac_h1_walking.py`: live fixture, video, per-sample states/forces/contacts, pass/fail report and Run Center progress.
- `scripts/dexterous/setup_h1_walking.py`: download and verify the pinned official checkpoint and its retained license.

The complete 10.7 MB evidence archive is `~/Desktop/Projects/DoorBench-runs/2026-09-08-isaac-walking/walking-001-evidence.tar.gz`. Its extracted `sensor-smoke-runs/walking-001/` directory contains the uncut video, trace, actual executed source, manifest and frames at the start, walk and stop. Those three views were personally inspected. The initial reset exporter failed before any Isaac run because it unnecessarily imported OSQP through a training module; that dependency was removed, with the setup failure retained. The pod environment was not changed.

Later tooling adds automatic live pipeline metadata and before/after readback of mass, joint limits, effort caps and contact materials/offsets. Those extra readback checks are **not retroactively counted** in the first run. Repeat the full gate after changing the robot, sensor/actuator interface or simulator.

Actual doorway clearance, approach-to-grasp coordination and transitions to/from a lowered manipulation posture remain separate gates. The native raised-arm stopping failures remain in their existing report and are not replaced by this standard-arm Isaac success.
