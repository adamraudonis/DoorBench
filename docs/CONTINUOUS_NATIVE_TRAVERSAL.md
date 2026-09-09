# Uninterrupted native opening and traversal

The September 9 development trials `continuous-opening-traversal-001` and `002` complete approach, opposed grasp, lever operation, door opening, release, arm stow, walking through and a quiet finish in **127.990 simulated seconds**. The repeat passes **47/47 recorded task checks and 16/16 independent archive checks** on `db0055_swing_single`, with byte-identical dynamics throughout. This is a slow, privileged MuJoCo teacher for the H1 with Shadow Hands, not an Isaac success or a learned vision/tactile policy.

The first run's stronger independent aggregate remains **failed** because its inherited recorder did not save complete MuJoCo warning counters. The repeat at **02:34 UTC** adds that logging and passes. Both the [first-run limitation](evidence/continuous-native-traversal-001.json) and [complete repeat evidence](evidence/continuous-native-traversal-002.json) are retained.

| Measurement | Actual result |
|---|---:|
| Physics steps, each 2 ms | 63,995 |
| Opening-to-continuation handoff | 75.168 s |
| Aperture at handoff | 1.22987 rad |
| Palm load at handoff, after a qualifying 0.5 s hold | 3.53019 N |
| Position and velocity continuity | Every interval matches exactly |
| Motor delivery error / cap excess | 0 / 0 |
| Final-second minimum robot collision-bound Y | 0.91135 m beyond the door plane |
| Final-second maximum horizontal speed | 0.02686 m/s |
| Final-second horizontal excursion | 0.01130 m |
| Invalid right-hand distal patches after transfer begins | 0 |
| Maximum physical joint-limit overshoot | 0.01243 rad, below the unchanged 0.02 bound |

The controller uses one plant and clock. It records the actual 69-joint state, velocities, contacts and delivered forces at the transition; it does not load a saved pose. All 61 original motors and eight passive finger loopbacks remain. The first 75 seconds are byte-identical to the independently audited opening trial. The new opening handoff requires continuous measured palm support; it preserves the earlier aperture-crossing receipt rather than hiding its contact dip. Terminal palm-load checks from the opening-only runner are retained as false because the full sequence intentionally releases the panel.

An earlier initialized diagnostic from the same opening endpoint separately passed 26/26 task checks and 6/6 independent archive checks over 69 seconds. That diagnostic motivated the continuation; it is not spliced into the uninterrupted result.

## Reproduce and inspect

The exact executable arguments are in `out/continuous-opening-traversal-001-launch.json` and `out/continuous-opening-traversal-002-launch.json`. The run manifests capture source commits `b169b5768` and `9222e04d9`, respectively, with frozen source, robot, door, motor contract, reference and checkpoint. Complete local archives and SHA-256 manifests live under `~/Desktop/Projects/DoorBench-runs/2026-09-09/`, including matching `-source` directories. `continuous-native-inputs-001` preserves all 749 model-asset/input mappings and their original bytes; its manifest maps original paths to content-addressed files. Rebinding on another host still requires source-design, compiled-geometry and physical checks. Generated evidence is kept out of Git.

Run these from the recorded environment:

```bash
python scripts/dexterous/audit_continuous_native.py --run "$RUN"
python scripts/dexterous/audit_walking_release.py --run "$RUN"
python scripts/dexterous/audit_panel_phase_moment.py --run "$RUN" --output "$NEW_AUDIT_DIR"
python scripts/dexterous/render_continuous_opening.py --run "$RUN" \
  --output "$NEW_VIDEO_DIR" --azimuth 140 --cutaway-walls
```

The cutaway hides only wall visuals in a recorded-state replay so the robot remains visible after crossing. It changes no physics and is labeled in the video. Hand close-ups are available with `--view right-hand` or `--view left-hand`.

`continuous-opening-traversal-002` stores full warning-counter capture separately to preserve exact raw-prefix comparison. All 256 raw chunks match through the terminal state at 127.990 seconds, beyond the required 127.5-second comparison. Independent contact reconstruction finds no invalid distal patches, and all 2,682 selected panel-control intervals pass the target/force audit. This is a same-start reproducibility check; varied-start repeatability, Isaac parity, speed/naturalness improvements, sensor-only learning and catalogue expansion remain in [the active plan](DEXTEROUS_NEXT_STEPS.md).
