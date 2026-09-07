# Anatomical hand: one-door inspection prototype

An articulated stick figure pre-shapes its hand, grasps the lever, unlatches a
passive door, pulls it open and holds it. **This is opening and hold only.**
Release, traversal and natural human motion remain unvalidated. Attempts at
withdrawal still produced snagging and excessive forces; they are not included
in the accepted demonstration.

[Phone video](https://github.com/adamraudonis/DoorBench/releases/download/anatomical-hand-20260907/doorbench-anatomical-hand-phone.mp4)
· [Local orbitable viewer](http://127.0.0.1:5184)
· [Exact measurements and checks](review/physical-human/prototype-checks.json)

## What changed

The original two-link thumb and rectangular palm were inadequate. The hand now
uses anatomical joint offsets, axes and limits from
[MyoHand](https://github.com/MyoHub/myo_sim/blob/eb327acbae0fad12279495040607f5235d962328/myo_sim/models/arm/README_hand.md):
five metacarpals, distinct finger proportions, a thumb metacarpal plus two
phalanges, and two thumb CMC axes. There are **20 digit degrees of freedom per
hand**, two wrist axes and separate forearm rotation. The three-axis twisting
wrist was removed.

The **21 landmarks per hand** follow the
[COCO-WholeBody layout used by Sapiens](https://github.com/facebookresearch/sapiens/blob/main/pose/configs/_base_/datasets/coco_wholebody.py).
These are source-model joint centres and authored tip landmarks, not predictions
from Sapiens or captured human motion. Bone rods show the kinematic skeleton,
not anatomical bone surfaces. [Source, modifications and Apache-2.0 license](../scripts/physical_human/anatomy/README.md).

Real lever-grip photographs were inspected for the diagonal approach, finger
pre-shape and thumb placement. They are visual guides, not metric 3D ground
truth; links are in the provenance document. The working hand approaches above
the lever so the thumb base clears it. The standing opening angle is limited to
keep the wrist in a more comfortable posture.

## Recorded result

Native MuJoCo **3.12.0**, CPU, **2026-09-07 01:36:38–01:36:44 UTC**.

| Case | Maximum opening |
|---|---:|
| Physical hand | **43.41°** |
| Hand contact disabled | **0.00°** |
| Latch blocked | **0.36°** of strike clearance |

All nine opening/hold acceptance checks passed. Maximum foot drift was
**0.23 mm**, hand/environment penetration **0.64 mm**, and skeleton self-contact
penetration **0.14 mm**. No non-hand body contact moved the door. Peak summed
hand contact was **91.5 N**; this is a simulator measurement, not a validated
human force profile.

| Actual left-thumb angle | Observed range | Model limits |
|---|---:|---:|
| CMC flexion | −5.43 to −0.44° | −44.69 to +40.11° |
| CMC abduction | −12.87 to +10.94° | −28.65 to +44.69° |
| MCP flexion | −42.51 to −3.39° | −45.00 to +40.00° |
| IP flexion | −21.57 to −2.45° | −75.00 to +25.00° |

The audit checks **achieved poses at every 1 ms step**, not just controller
commands. Maximum joint-limit excess across the actor was **0.00063°**, well
below the **0.5° numerical tolerance**. Maximum geometric wrist bend was
**41.13°**; the prototype rejection threshold is **55°**. These thresholds are
engineering acceptance criteria, not population-wide medical limits.

Tests also deliberately inject an overextended thumb, an excessive wrist bend
and a fast digit rotation, and require rejection. A mirrored-pose test checks
all 21 landmarks across several articulated poses; this catches incorrectly
reflected hinge axes. The native causal tests require contact and an unblocked
latch for opening. **Five tests pass.**

## Scope and physical model

- Custom 0.81 × 2.07 m, 19 kg lever door; not a catalogue-wide human baseline.
- Floating pelvis and native floor contacts. No foot anchors, hand welds, door
  actuator, mocap body, external root force or mechanism-pose writes.
- Torque-limited servos and bounded arm IK follow authored targets. Finger
  pre-shaping reduces the collision that occurred with a fully straight hand.
- Original tissue envelopes contact the environment. Thinner bone capsules
  enforce hand/hand and hand/body separation. Skin deformation, finger-to-finger
  tissue pressure, muscles and calibrated inertias are not modeled.
- 48 native touch sensors. UI force readings sum hand/door contact-pair magnitudes;
  they are not resultant pull force. Raw touch sensors can count self-contact.

An anatomical skeleton and passing checks do **not** establish human ground
truth. Motion style, release, locomotion, robustness and biomechanical calibration
still need validation before retargeting or benchmarking this as a human baseline.

## Reproduce

```sh
python -m pip install -e '.[reference]'
python scripts/physical_human/prototype.py --out out/physical-human/normal
python scripts/physical_human/prototype.py --out out/physical-human/no-touch --no-touch
python scripts/physical_human/prototype.py --out out/physical-human/blocked --latch-blocked
python -m pytest tests/test_physical_human_prototype.py -q
```

No GPU is required. `--anchors` is a labeled debugging option, off in the
accepted run. The blocked and no-contact runs are diagnostic negative controls.

Install the viewer dependencies, package and serve the evidence:

```sh
cd viewer
bun install
cd ..
python scripts/physical_human/export.py out/physical-human/normal \
  --web out/physical-human-demo --three viewer/node_modules/three
python -m http.server 5184 --bind 127.0.0.1 --directory out/physical-human-demo
```

The viewer offers bone-only and translucent-contact-envelope views, a ghosted
door, actual thumb/wrist angle readings, surface contact dots, force by digit,
orbit controls and slow motion. It replays native states; it does not simulate
physics in the browser. Reset the scene to its `initial` keyframe and run the
controller to reproduce the rollout; loading XML alone does not run it.

`trajectory.npz` contains 50 Hz timestamps, qpos, qvel, motor targets/forces,
touch sensors and `hand_keypoints[frame, left/right, 21, world_XYZ_metres]`.
The report includes full-rate extrema, joint-angle ranges, warnings, provenance
and checks. Narrow 1 kHz force peaks can exceed the values in 50 Hz replay rows.

Create the phone video (Pillow, imageio and imageio-ffmpeg required):

```sh
python scripts/physical_human/phone_video.py out/physical-human/normal \
  --out out/physical-human/doorbench-anatomical-hand-phone.mp4
```

It shows the same six-second native recording at real time and half speed.
The second pass ghosts the door and changes the camera to expose the thumb.
It writes H.264/yuv420p with faststart, a JPEG poster and a SHA-256 receipt,
and refuses v2 runs that fail the opening/hold acceptance checks.
