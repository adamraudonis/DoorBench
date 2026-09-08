# First sensor-feedback Isaac grasp: failed

The frozen 19-second Isaac test completed all 9,500 physical steps with all 14
original mechanics checks passing. It **did not acquire a valid sustained grasp**.
The independently corrected evaluator passes 21 of 23 grouped checks; the two
failures are loaded distal-pad anatomy and the final opposed five-pad hold.
This is a fixed joint-route experiment with sensor-feedback balance, not a
learned or vision-conditioned policy, opening, or traversal result.

The archive is `DoorBench-runs/2026-09-08-shadow-loopback/sensor-acquisition-balance-isaac-001`.
All 138 run files and 566 frozen source/input files independently match the remote
manifest (`b1d66b1fadc580fdf89284ad6ad1575d9ee3dde26786e210c2b5e2b43f758cb0`).
The original failed `balance-report.json` is unchanged, SHA256
`06a06ba773059d112673f8068788b0b3e9205d8b29d7f7bdd5c2ebe7262c4240`.

## Evidence reconstruction correction

The original evaluator rejected 1,262 rows, first at 15.246 seconds, because it
compared two different arithmetic calculations. The producer computes patch
force × normal, axis-sum, matrix difference and norm using copied PhysX float32
arrays. JSON preserves each scalar but loses the arithmetic dtype. The evaluator
recomputed a sequential float64 sum and incorrectly required its final norm to
match the recorded float32 norm within 1e-8 N.

Every saved force, normal and matrix scalar is exactly representable as float32.
Replaying the original ordered float32 calculation reproduces all 9,500 recorded
error scalars exactly. The maximum independent float64 pair-force error is
3.04494314374089e-7 N; the maximum producer float32 error is
4.768380676978268e-7 N. Both are below the unchanged 1e-3 N force-consistency
threshold. The largest disagreement between arithmetic methods is
2.0794511255327013e-7 N.

The fix reproduces source arithmetic for the 1e-8 N receipt comparison, requires
exact float32 source scalars, and separately retains the float64 1e-3 N physical
check. It changes no contact, force, anatomy, opposition, duration or hold
threshold. Native evidence retains its explicitly different geometry epoch and
does not claim an independent PhysX pair matrix.

The original archived evaluator independently reproduces the original failed
report. The corrected evaluation is a separately named retrospective report:
`sensor-acquisition-balance-isaac-001-recomputed-report.json`. It correctly
validates 9,500 rows through 19.0 seconds instead of reporting 8,238 valid rows
through 18.994 seconds. Original and corrected contact audits both reconstruct
all floor loads, hand-contact counts and unintended-contact counts exactly.

## Physical failure retained

The raw reconstruction reproduces every live pad classification and qualified
force exactly. There are 1,549 loaded invalid patches: 1,440 on the little
finger's middle link and 109 on its distal link. The first is a side contact at
15.246 seconds with body-local +Y position and only 0.0577 inward radial normal
alignment. The last middle-link contact also lies on body-local +Y/side with
0.1082 radial alignment. These are not a basis for relabeling the result as a
valid broader volar grasp.

| Digit | Final qualified pad load | Minimum in final 0.5 seconds |
|---|---:|---:|
| Index | 0 N | 0 N |
| Middle | 0 N | 0 N |
| Ring | 0 N | 0 N |
| Little | 2.1415 N | 0 N |
| Thumb | 0.4871 N | 0.2072 N |

The little finger additionally has an invalid middle-link load of 2.2458 N at
the final step. The longest valid opposed five-pad interval is only 0.002 s.
Motor-coordinate tracking error is 0.02213 rad and palm endpoint error is
5.418 mm, but maximum nominal individual-joint error is 0.20124 rad. The passive
J1/J2 split therefore needs an actual-pose comparison with the native success;
matching motor sums does not establish matching finger posture.

The final whole-body and hand close-up images were personally inspected. The
body is upright; the hand reaches the lever, but the image alone cannot prove
which surfaces carry load. The raw contact evidence above determines failure.

Reproduce arithmetic diagnosis with
`scripts/dexterous/diagnose_physx_pair_arithmetic.py --run ARCHIVE --output NEW_RECEIPT`.
Recompute all contacts and the scoped qualification with
`scripts/dexterous/audit_isaac_sensor_balance.py --run ARCHIVE --robot ROBOT_XML --output NEW_RECEIPT`.
Its report-comparison flag intentionally differs when using the corrected
evaluator against the unchanged original report. The receipt in
`docs/evidence/sensor-acquisition-isaac-001.json` binds the separate analyses.

## Saved-pose comparison: drift begins before contact

The native trajectory is a **full scene** qpos array: its three door coordinates
precede the robot free root. The comparison resolves every joint name against
that model and requires exact agreement with the native named reset. It does
not index that array with robot-only qpos addresses. The Isaac reset agrees to
its original float32 storage precision.

At ten sampled epochs, detached robot-only FK reproduces the recorded Isaac
right-hand body positions within 0.669 micrometers and orientations within
2.18 microradians. This excludes a gross name/order or rigid-frame mismatch for
those samples. It does not qualify collision-mesh or friction parity.

| Time | Finger | Native J1 / J2 | Isaac J1 / J2 | Motor-sum difference | Split difference |
|---|---|---:|---:|---:|---:|
| 1 s, hands clear | Index | 0.12360 / 0.12352 | 0.09382 / 0.15420 | 0.00090 rad | −0.06046 rad |
| 1 s, hands clear | Middle | 0.12262 / 0.12257 | 0.09305 / 0.15299 | 0.00085 rad | −0.05999 rad |
| 1 s, hands clear | Ring | 0.07267 / 0.07266 | 0.03729 / 0.10913 | 0.00108 rad | −0.07184 rad |
| 8 s, hands clear | Index | 0.23330 / 0.29200 | 0.09201 / 0.43618 | 0.00290 rad | −0.28547 rad |
| 19 s | Index | 0.36091 / 0.49022 | 0.39133 / 0.46629 | 0.00650 rad | +0.05436 rad |
| 19 s | Middle | 0.47021 / 0.58462 | 0.51463 / 0.54831 | 0.00812 rad | +0.08072 rad |
| 19 s | Ring | 0.52758 / 0.66232 | 0.58209 / 0.61723 | 0.00942 rad | +0.09960 rad |

Differences are Isaac minus native; split means J1−J2. The one-sided J1≤J2
mechanism and individual joint limits pass in both runs. That limit does not
select the same state along the motor's J1+J2 coordinate in both engines.

The final native verified contact patches define fixed, force-weighted material
points on the distal links. Mapping those same material points through the
actual Isaac poses places the index, middle and ring points respectively
0.291, 0.431 and 0.298 mm outside the actual lever cylinder. The little finger
and thumb points are touching. The measured native-versus-Isaac world positions
of all five material points differ by 2.6–3.9 mm.

An explicitly detached FK counterfactual preserves the actual Isaac body, every
other joint, and each finger's actual motor sum while substituting only the
native J1−J2 split. Those three point gaps become −0.382, −0.568 and −0.907 mm.
This supports a split-dependent loss of contact. It is not an executed motion,
contact-force result or permission to enforce equality between the joints.
These are distances of identified material points, not a surface-distance
optimization or collision resolution.

The early free-space drift makes actuator/friction parity a priority before
retuning contact control. The root investigation independently identified the
adapter's smooth `frictionloss*tanh(velocity/.001)` law as a candidate difference
from native static friction; a separate measured fixture is needed to establish
that cause. This pose audit neither changes that law nor modifies any plant.

Both final-hand close-ups were inspected: the native
`scripted-acquisition-hand-az150-0475.png` and Isaac `hand-frame-09000.png`.
Their viewpoints differ, so the numeric body-frame comparison provides the
alignment evidence. Source hashes and the full sampled comparison are bound in
`docs/evidence/sensor-acquisition-isaac-001-poses.json`. Reproduce with
`scripts/dexterous/compare_recorded_acquisition_poses.py --native NATIVE_ARCHIVE --isaac ISAAC_ARCHIVE --robot ROBOT_XML --door DOOR_DIRECTORY --output NEW_RECEIPT`.
