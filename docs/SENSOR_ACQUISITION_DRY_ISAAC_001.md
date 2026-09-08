# Corrected-friction Isaac grasp: failed

The 19-second test launched on September 8, 2026 at **20:14:28 UTC** completed
9,500 physical steps. It passes 21/23 grouped checks and all 14 original mechanics
checks, but **fails sustained opposed grasp and distal-contact anatomy**. The
69 original passive joint coefficients and armatures remain exactly unchanged
through all 9,500 intervals under the explicit `backend-dry-v2` adapter.

This is a scripted canonical joint route with sensor-feedback balance. Encoders,
IMU and touch are used; captured RGB is unused. It does not establish learned
control, opening, traversal or robustness. [Result](../results/dexterous/2026-09-08/sensor-acquisition-dry-isaac-001.json).

| Measurement | Actual result |
|---|---:|
| Longest qualified five-pad hold | 0.006 s; required 0.5 s |
| Invalid loaded patches | 1,987 |
| Final index / middle / ring distal load | 0.084 / 0.287 / 0.366 N |
| Final little-finger distal / invalid middle-link load | 0.640 / 0.472 N |
| Final thumb distal load | 1.523 N |
| Peak torso tilt | 0.3701° |
| Maximum motor-coordinate tracking error | 0.02747 rad |
| Palm target error | 5.971 mm |

The little-finger middle segment contacts near the lever end instead of the
required distal surface. The minimum final-half-second index load is zero. An
attractive hand pose cannot override either failure. The actual final wide and
hand close-up images were personally inspected.

## What the corrected friction resolved

The [isolated passive-joint fixture](ISAAC_PASSIVE_JOINT_PROFILE.md) exposed an
oscillation in the legacy explicit friction approximation. Correcting that law
greatly reduces the free-space J1/J2 drift and improves the final finger split.
For example, the final index motor-sum difference from the native success is only
−0.000090 rad, and its split difference is −0.03443 rad. The original failed
legacy-profile grasp remains available; neither result is relabeled.

The remaining geometry discrepancy is largely whole-body placement. At the
final recorded instant, the Isaac palm differs from native by
**[3.203, 1.451, 1.218] mm**. Detached robot-only forward kinematics reproduces
the recorded Isaac hand bodies within 0.873 micrometers and 1.91 microradians
over ten sampled frames. Substituting only the native root into that detached
calculation leaves a 0.398 mm palm discrepancy; substituting only native arm and
torso angles leaves 3.430 mm. These are diagnostic substitutions, never runtime
pose writes or physical replays.

Unlike the legacy run, substituting only native passive finger splits now
*increases* the material-point gaps. Enforcing a native finger split is therefore
not the next correction. The next candidate compensates for attained body pose
while following an explicitly declared static palm route, using the robot's own
sensor-based estimate. Contact and physical qualification remain separate.

## Evidence

All **141 output files and 594 frozen source/input files** were copied off-pod
and verified against the remote byte hashes. Independent reduction reproduces
all 9,500 raw contact intervals, floor loads, hand counts, the failed report and
the original passive properties. [Audit and pose evidence](evidence/sensor-acquisition-dry-isaac-001.json).

The canonical archive is
`DoorBench-runs/2026-09-08-shadow-loopback/sensor-acquisition-dry-isaac-001`.
Run `scripts/dexterous/audit_isaac_sensor_balance.py --run ARCHIVE --robot ROBOT_XML --output NEW_RECEIPT`
to recheck the original report. The saved-pose comparison uses
`scripts/dexterous/compare_recorded_acquisition_poses.py`; it never steps physics.
