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
