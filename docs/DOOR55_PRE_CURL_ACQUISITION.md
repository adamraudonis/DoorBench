# Native Door55 grasp acquisition, 2026-09-08

The frozen `door55-precurl-v2` reference acquired an opposed grasp from an actual
contact-free hand position with the door closed and handle at rest. Every strict
2 ms gate passed in a 10.6 s native MuJoCo run, including a final uninterrupted
0.5 s of five volar-pad contacts. This is **grasp acquisition only**: opening,
traversal, Isaac qualification and a vision/tactile-only policy remain separate
work. The controller uses privileged kinematics and contact geometry.

| Measurement | Nominal native run |
|---|---:|
| Initial right-hand contacts with any body | 0 |
| Maximum loopback inequality violation | 0.788 mrad |
| Maximum individual joint-limit violation | 2.739 mrad |
| Maximum non-foot penetration | 0.505 mm |
| Maximum torso tilt | 0.262° |
| Final palm target error | 0.494 mm |
| Minimum final pad clearance from lever cylinder ends | 3.895 mm |
| Final FF / MF / RF / LF / TH pad loads | 2.135 / 1.924 / 1.314 / 1.796 / 7.743 N |

The robot uses [versioned Shadow loopback mechanics](SHADOW_LOOPBACK_MECHANICS.md),
the original motor strength and a free root. There are no runtime robot pose
writes, direct door commands, hand-object welds or external supporting forces.
The body remains in the original bent-knee stance with an upright torso. This
run does not yet start from the learned walking-stop pose.

## What changed and why

The old lateral route first placed the little finger's **tip** on the lever at
4.208 s, around 47.6% of the path. It subsequently rolled onto the finger's dorsal
surface. The palm tracked within 0.9 mm, but the real sum-driven tendon split and
proximal-joint lag consumed the planned finger clearance. Its LFJ1 reached
−27.44 mrad, despite satisfying the newly restored J1 ≤ J2 inequality.

The replacement approaches from above, with a 150 mm vertical / 30 mm outward
offset, and curls distal links before lowering the proximal joints. In the
backward geometric release schedule, distal opening uses the square of phase;
proximal opening uses its square root. The nominal joint penalty is 0.03. A 12 mm
lever-clearance objective tapers into the final contact pose. The path is screened
at 401 waypoints and four interior points per segment. An additional screen
perturbs J1/J2 by −/+50 mrad, preserving their motor sum, and J3 by ±40 mrad;
all and individual fingers retain at least 2 mm lever clearance through 85% of
the approach. This robustness screen is geometric, not a physical success claim.

Two failures were retained while completing the route:

- `precurl-execution-001` reached valid pad opposition, but the starting FF/MF/RF
  tips overlapped the door panel by up to 0.107 mm. The strict start gate correctly
  rejected it. An additional 30 mm outward palm translation, tapering to zero by
  85%, removes those initial panel contacts.
- `precurl-execution-002` had a clear start and all five volar radial contacts,
  but two loaded thumb patches were only 0.846–0.848 mm inside the lever cylinder
  end, below the existing 1 mm margin. Translating the entire grasp 3 mm toward
  the lever center supplies real margin; the threshold was not relaxed.

`precurl-execution-003` is the first full acquisition pass. The frozen reference
is [reference.json](../configs/dexterous/door55-precurl-v2/reference.json), with
[protocol and qualification receipt](../configs/dexterous/door55-precurl-v2/protocol.json).
Its numeric joint sequence is unchanged from the tested route; only JSON
whitespace was compacted. The robot XML hash is recorded in that receipt.

## Reproduce

Generate the versioned robot and Door55 assets, activate the native Python
environment, then run from the repository root:

```bash
python scripts/dexterous/screen_precurl_route.py \
  --robot out/dexterous/robot/h1-shadow-loopback-v2.xml \
  --door out/dexterous/assets/doors/db0055_swing_single \
  --output out/dexterous/precurl-screen.json

python scripts/dexterous/reproduce_precurl_acquisition.py \
  --robot out/dexterous/robot/h1-shadow-loopback-v2.xml \
  --door out/dexterous/assets/doors/db0055_swing_single \
  --output out/dexterous/precurl-native-001
```

Use a fresh output directory. The reproduction wrapper freezes all tested
controller flags. The run captures source/dependency/asset hashes, native states
and controls, initial plus every-2-ms physical evidence, and strict pad audit.
Check both `report.json` and `strict-grasp-audit.json`; positive digit loads alone
are not sufficient. The static screen additionally requires no right-hand contact
with **any** body through 85% of the route, catching shallow panel contacts that
ordinary penetration thresholds intentionally tolerate.

Full local evidence, including all three nominal trials and the exact geometric
construction scripts, is under `/tmp/doorbench-shadow-loopback/` and the durable
copy `~/Desktop/Projects/DoorBench-runs/2026-09-08-shadow-loopback/`. The nominal
pass is `precurl-execution-003/`; its `hand-inspection.jpg` includes three camera
angles at the beginning, middle and final state, with the thumb colored blue.

Before repetitions, the route and controller were frozen with three seeds
17, 29 and 43. Each perturbs only initial right-arm/wrist/hand joint angles by
uniform ±5 mrad, clips individual limits, and applies a reset-only J1 ≤ J2
projection preserving motor sums. The root, legs, target route and controller
remain unchanged. These are small initial-condition tests, not a broad robustness
or sensor-noise benchmark; all outcomes must be retained.
